"""Preserve exact within-segment Jacobian blocks in a causal Qwen3.5 model.

Generalizes the validated same-position rule to contiguous token segments.
Primal outputs are unchanged. Cross-segment gradient edges are suppressed.
With a causal model, this preserves each diagonal SEGMENT block of the full
Jacobian. No random probes. Equivalent recurrent fp32 arithmetic is used for
delta-rule backward; BF16 rounding differs slightly from native chunk backward.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from exact_self_gradients import previous_states


@contextmanager
def block_position_gradients(width,offset,history_cache=None):
    assert width>=1 and 0<=offset<width
    original_conv=m.causal_conv1d_fn
    original_delta=m.torch_chunk_gated_delta_rule
    original_attention=m.eager_attention_forward
    delta_index=0

    class Attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx,q,k,v,mask,scale):
            scores=q@k.transpose(-1,-2)*scale
            if mask is not None:scores=scores+mask
            p=scores.softmax(-1,dtype=torch.float32).to(q.dtype)
            ctx.save_for_backward(q,k,v,p);ctx.scale=scale
            return p@v

        @staticmethod
        def backward(ctx,z):
            q,k,v,p=ctx.saved_tensors
            dp=z@v.transpose(-1,-2)
            ds=p*(dp-(dp*p).sum(-1,keepdim=True))*ctx.scale
            groups=(torch.arange(q.shape[-2],device=q.device)+offset)//width
            keep=groups[:,None]==groups[None,:]
            return ds@k,(ds*keep).transpose(-1,-2)@q,(p*keep).transpose(-1,-2)@z,None,None

    class Conv(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,w,b):
            pre=F.conv1d(x.to(w.dtype),w.unsqueeze(1),bias=b,padding=w.shape[-1]-1,groups=x.shape[1])[:,:,:x.shape[-1]]
            ctx.save_for_backward(pre,w);ctx.dtype=x.dtype
            return F.silu(pre).to(x.dtype)

        @staticmethod
        def backward(ctx,z):
            pre,w=ctx.saved_tensors
            with torch.enable_grad():
                pre=pre.detach().requires_grad_(True)
                g,=torch.autograd.grad(F.silu(pre),pre,z.to(pre.dtype))
            g=g.float();w=w.float();T=g.shape[-1]
            groups=(torch.arange(T,device=g.device)+offset)//width
            result=torch.zeros_like(g)
            for lag in range(min(w.shape[-1],T)):
                n=T-lag
                keep=groups[:n]==groups[lag:]
                result[:,:,:n]+=g[:,:,lag:]*w[:,-1-lag][None,:,None]*keep
            return result.to(ctx.dtype),None,None

    class Delta(torch.autograd.Function):
        @staticmethod
        def forward(ctx,q,k,v,g,beta):
            nonlocal delta_index
            output,_=original_delta(q,k,v,g=g,beta=beta,initial_state=None,
                output_final_state=False,use_qk_l2norm_in_kernel=True)
            key_index=delta_index;delta_index+=1
            if history_cache is not None and key_index in history_cache:
                signature,history=history_cache[key_index]
                if not all(torch.equal(x[:1],saved) for x,saved in zip((k,v,g,beta),signature)):
                    raise RuntimeError('Cached recurrent history inputs changed')
            else:
                history=previous_states(q,k,v,g,beta,include_final=True)
                if history_cache is not None:
                    history_cache[key_index]=([x[:1].detach().clone() for x in (k,v,g,beta)],history)
            ctx.save_for_backward(q,k,v,g,beta,history)
            return output

        @staticmethod
        def backward(ctx,z):
            q0,k0,v0,g0,b0,history=ctx.saved_tensors
            # Exact adjoint: the state cotangent in a width-w segment is a
            # sum of at most w rank-one terms. Keeping this algebraic sum
            # avoids materializing B copies of KxV states. No rank truncation.
            B,T,H,K=q0.shape;V=v0.shape[-1]
            qraw=q0[:1].transpose(1,2).float();kraw=k0[:1].transpose(1,2).float()
            value=v0[:1].transpose(1,2).float();decay=g0[:1].transpose(1,2).float().exp()
            beta=b0[:1].transpose(1,2).float()
            iq=torch.rsqrt(qraw.square().sum(-1,keepdim=True)+1e-6)
            ik=torch.rsqrt(kraw.square().sum(-1,keepdim=True)+1e-6)
            q=qraw*iq/(K**.5);k=kraw*ik
            z=z.transpose(1,2).float()
            dq=torch.zeros(B,H,T,K,device=z.device);dk=torch.zeros_like(dq)
            dv=torch.zeros(B,H,T,V,device=z.device)
            dg=torch.zeros(B,H,T,device=z.device);db=torch.zeros_like(dg)
            blocks=(T+offset+width-1)//width
            base=torch.arange(blocks,device=z.device)*width-offset
            left=[];right=[]
            for j in reversed(range(width)):
                positions=base+j;valid=(positions>=0)&(positions<T)
                index=positions.clamp(0,T-1)
                qt=q[:,:,index];kt=k[:,:,index];bt=beta[:,:,index,None];dt=decay[:,:,index,None]
                go=z[:,:,index]*valid[None,None,:,None]
                before=history[:,:,index];after=history[:,:,index+1]
                # M = decay * previous state, shared by all replica samples.
                M=before*dt[...,None]
                memory=torch.einsum('ahnkv,ahnk->ahnv',M,kt)
                residual=value[:,:,index]-memory
                delta=residual*bt
                q_gradient=torch.einsum('ahnkv,bhnv->bhnk',after,go)
                # Add the present output's rank-one state cotangent.
                left.append(qt);right.append(go)
                delta_gradient=torch.zeros_like(go)
                key_gradient=torch.zeros_like(q_gradient)
                g_gradient=torch.zeros_like(go[...,0])
                products=[]
                for l,r in zip(left,right):
                    dot=(kt*l).sum(-1,keepdim=True)
                    products.append(dot)
                    delta_gradient+=dot*r
                    key_gradient+=l*(delta*r).sum(-1,keepdim=True)
                    ml=torch.einsum('ahnkv,ahnk->ahnv',M,l)
                    g_gradient+=(ml*r).sum(-1)
                memory_gradient=-bt*delta_gradient
                key_gradient+=torch.einsum('ahnkv,bhnv->bhnk',M,memory_gradient)
                g_gradient+=(memory*memory_gradient).sum(-1)
                rawq=qraw[:,:,index];rawk=kraw[:,:,index]
                q_gradient=iq[:,:,index]/(K**.5)*(q_gradient-rawq*iq[:,:,index].square()*(rawq*q_gradient).sum(-1,keepdim=True))
                key_gradient=ik[:,:,index]*(key_gradient-rawk*ik[:,:,index].square()*(rawk*key_gradient).sum(-1,keepdim=True))
                dst=positions[valid]
                dq[:,:,dst]=q_gradient[:,:,valid];dk[:,:,dst]=key_gradient[:,:,valid]
                dv[:,:,dst]=(bt*delta_gradient)[:,:,valid]
                dg[:,:,dst]=g_gradient[:,:,valid]
                db[:,:,dst]=(delta_gradient*residual).sum(-1)[:,:,valid]
                # Backpropagate each exact rank-one term to the prior state.
                left=[dt*(l-bt*kt*dot) for l,dot in zip(left,products)]
            return tuple(g.transpose(1,2).to(original.dtype) for g,original in zip((dq,dk,dv,dg,db),(q0,k0,v0,g0,b0)))

    def conv(x,w,bias=None,activation=None,**kwargs):
        if activation not in ('silu','swish'):raise ValueError(activation)
        return Conv.apply(x,w,bias)
    def delta(q,k,v,g,beta,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
        if initial_state is not None or output_final_state or not use_qk_l2norm_in_kernel:raise ValueError('Uncached normalized delta only')
        if not any(x.requires_grad for x in (q,k,v,g,beta)):
            return original_delta(q,k,v,g=g,beta=beta,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
        return Delta.apply(q,k,v,g,beta),None
    def attention(module,q,k,v,attention_mask,scaling,dropout=0.,**kwargs):
        if dropout:raise ValueError('Eval only')
        k=m.repeat_kv(k,module.num_key_value_groups);v=m.repeat_kv(v,module.num_key_value_groups)
        return Attention.apply(q,k,v,attention_mask,scaling).transpose(1,2).contiguous(),None
    m.causal_conv1d_fn=conv;m.torch_chunk_gated_delta_rule=delta;m.eager_attention_forward=attention
    try:yield
    finally:
        m.causal_conv1d_fn=original_conv;m.torch_chunk_gated_delta_rule=original_delta;m.eager_attention_forward=original_attention

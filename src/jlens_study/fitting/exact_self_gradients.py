"""Exact diagonal-position Jacobians for Qwen3.5, without random probes.

For causal block Jacobians A,B, (AB)[t,t]=A[t,t]B[t,t]. Removing
cross-position gradient edges at token mixers therefore preserves the full
model's diagonal position blocks. Primal outputs are left unchanged.
Parameters must be frozen; cache-free eval only. Delta-rule local derivatives
use its equivalent recurrent formulation in fp32 (ordinary roundoff applies).
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as m


class SelfAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,scale):
        scores=q@k.transpose(-1,-2)*scale
        if mask is not None: scores=scores+mask
        p=scores.softmax(-1,dtype=torch.float32).to(q.dtype)
        ctx.save_for_backward(q,k,v,p);ctx.scale=scale
        return p@v

    @staticmethod
    def backward(ctx,z):
        q,k,v,p=ctx.saved_tensors
        dp=z@v.transpose(-1,-2)
        ds=p*(dp-(dp*p).sum(-1,keepdim=True))*ctx.scale
        dq=ds@k
        dk=ds.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)*q
        dv=p.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)*z
        return dq,dk,dv,None,None


class SelfConv(torch.autograd.Function):
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
        return (g*w[:,-1][None,:,None]).to(ctx.dtype),None,None


def previous_states(q,k,v,g,beta,include_final=False):
    # Fit batches are replicas of one prompt. Store just one copy of history.
    if q.shape[0]>1:
        if not torch.equal(q,q[:1].expand_as(q)):
            raise ValueError('Expected replicated prompt batch')
    k=k[:1].transpose(1,2).float();v=v[:1].transpose(1,2).float()
    g=g[:1].transpose(1,2).float();beta=beta[:1].transpose(1,2).float()
    k=m.l2norm(k,dim=-1,eps=1e-6)
    state=torch.zeros(k.shape[0],k.shape[1],k.shape[-1],v.shape[-1],device=k.device)
    history=[]
    for t in range(k.shape[2]):
        history.append(state)
        decayed=state*g[:,:,t].exp()[...,None,None]
        kt=k[:,:,t];vt=v[:,:,t]
        delta=(vt-(decayed*kt.unsqueeze(-1)).sum(-2))*beta[:,:,t,None]
        state=decayed+kt.unsqueeze(-1)*delta.unsqueeze(-2)
    if include_final:history.append(state)
    return torch.stack(history,dim=2)


def make_self_delta(original):
    class SelfDelta(torch.autograd.Function):
        @staticmethod
        def forward(ctx,q,k,v,g,beta):
            output,_=original(q,k,v,g=g,beta=beta,initial_state=None,
                output_final_state=False,use_qk_l2norm_in_kernel=True)
            history=previous_states(q,k,v,g,beta)
            ctx.save_for_backward(q,k,v,g,beta,history)
            return output

        @staticmethod
        def backward(ctx,z):
            q0,k0,v0,g0,b0,history=ctx.saved_tensors
            with torch.enable_grad():
                inputs=[x.detach().requires_grad_(True) for x in (q0,k0,v0,g0,b0)]
                q,k,v,g,beta=[x.transpose(1,2).float() for x in inputs]
                q=m.l2norm(q,dim=-1,eps=1e-6)/(q.shape[-1]**.5)
                k=m.l2norm(k,dim=-1,eps=1e-6)
                # Compute M^T q and M^T k without storing B copies of M.
                mq=torch.einsum('bhtkv,bhtk->bhtv',history,q)*g.exp()[...,None]
                mk=torch.einsum('bhtkv,bhtk->bhtv',history,k)*g.exp()[...,None]
                local=mq+(q*k).sum(-1,keepdim=True)*beta[...,None]*(v-mk)
                local=local.transpose(1,2).to(q0.dtype)
                gradients=torch.autograd.grad(local,inputs,z)
            return tuple(gradients)
    return SelfDelta


@contextmanager
def same_position_gradients():
    original_conv=m.causal_conv1d_fn
    original_delta=m.torch_chunk_gated_delta_rule
    original_attention=m.eager_attention_forward
    Delta=make_self_delta(original_delta)
    def conv(x,w,bias=None,activation=None,**kwargs):
        if activation not in ('silu','swish'): raise ValueError(activation)
        return SelfConv.apply(x,w,bias)
    def delta(q,k,v,g,beta,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
        if initial_state is not None or output_final_state or not use_qk_l2norm_in_kernel:
            raise ValueError('Only uncached normalized delta-rule supported')
        return Delta.apply(q,k,v,g,beta),None
    def attention(module,q,k,v,attention_mask,scaling,dropout=0.,**kwargs):
        if dropout: raise ValueError('Eval only')
        k=m.repeat_kv(k,module.num_key_value_groups)
        v=m.repeat_kv(v,module.num_key_value_groups)
        return SelfAttention.apply(q,k,v,attention_mask,scaling).transpose(1,2).contiguous(),None
    m.causal_conv1d_fn=conv
    m.torch_chunk_gated_delta_rule=delta
    m.eager_attention_forward=attention
    try: yield
    finally:
        m.causal_conv1d_fn=original_conv
        m.torch_chunk_gated_delta_rule=original_delta
        m.eager_attention_forward=original_attention

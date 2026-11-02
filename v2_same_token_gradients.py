"""Exact causal diagonal blocks, retaining v2's native SDPA forward.

For lower-triangular token Jacobians, diag(AB)=diag(A)diag(B).
Thus removing cross-token derivative edges at each causal mixer preserves
the complete model's same-token Jacobians. No random probes or frozen Q/K.
The recurrent adjoint uses FP32 algebra; BF16 rounding differs from the native
chunk backward. Every forward value uses the original v2 operations.
"""
from contextlib import contextmanager
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from exact_block_gradients import block_position_gradients


@contextmanager
def same_token_sdpa_gradients():
    original = m.ALL_ATTENTION_FUNCTIONS['sdpa']

    class SDPA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, mask, module, scaling, kwargs):
            output, _ = original(module, q, k, v, mask, scaling=scaling, **kwargs)
            groups = module.num_key_value_groups
            kr = m.repeat_kv(k, groups).float()
            vr = m.repeat_kv(v, groups).float()
            qf = q.float()
            scale = scaling if scaling is not None else q.shape[-1] ** -.5
            scores = qf @ kr.transpose(-1, -2) * scale
            if mask is not None:
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, float('-inf'))
                else:
                    scores = scores + mask.float()
            elif kwargs.get('is_causal', getattr(module, 'is_causal', True)) and q.shape[-2] > 1:
                causal = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
                scores = scores.masked_fill(~causal, float('-inf'))
            p = scores.softmax(-1)
            ctx.save_for_backward(qf, kr, vr, p)
            ctx.groups = groups
            ctx.scale = scale
            ctx.dtypes = (q.dtype, k.dtype, v.dtype)
            return output

        @staticmethod
        def backward(ctx, z):
            q, k, v, p = ctx.saved_tensors
            z = z.transpose(1, 2).float()
            dp = z @ v.transpose(-1, -2)
            ds = p * (dp - (dp*p).sum(-1, keepdim=True)) * ctx.scale
            dq = ds @ k
            dk = ds.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * q
            dv = p.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * z
            if ctx.groups > 1:
                B, H, T, D = dk.shape
                dk = dk.reshape(B, H//ctx.groups, ctx.groups, T, D).sum(2)
                dv = dv.reshape(B, H//ctx.groups, ctx.groups, T, D).sum(2)
            return dq.to(ctx.dtypes[0]), dk.to(ctx.dtypes[1]), dv.to(ctx.dtypes[2]), None, None, None, None

    def attention(module, q, k, v, attention_mask, scaling=None, dropout=0., **kwargs):
        if dropout or kwargs.get('cache') is not None or kwargs.get('position_bias') is not None:
            raise ValueError('Only uncached Qwen3.5 eval attention supported')
        return SDPA.apply(q, k, v, attention_mask, module, scaling, kwargs), None

    with block_position_gradients(1, 0):
        m.ALL_ATTENTION_FUNCTIONS.register('sdpa', attention)
        try:
            yield
        finally:
            m.ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

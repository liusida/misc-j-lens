"""Value-preserving RelP backward rules used by the Qwen R-Lens fitter."""

from types import MethodType
from typing import Any

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm


class _IdentityActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        ratio = torch.where(
            values != 0,
            activated / values,
            torch.full_like(values, 0.5),
        )
        ctx.save_for_backward(ratio)
        return activated

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        (ratio,) = ctx.saved_tensors
        return gradient * ratio, None


class _HalfProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(left, right)
        return left * right

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = ctx.saved_tensors
        return 0.5 * gradient * right, 0.5 * gradient * left


def install_qwen35_relp_rules(model: torch.nn.Module) -> dict[str, Any]:
    """Install R-Lens RelP rules without changing any forward values."""
    norms = mlps = 0
    for name, module in model.named_modules():
        residual_norm = (
            name.endswith(".input_layernorm")
            or name.endswith(".post_attention_layernorm")
            or name == "model.norm"
        )
        if isinstance(module, Qwen3_5RMSNorm) and residual_norm:
            def relp_norm(current: Qwen3_5RMSNorm, values: torch.Tensor) -> torch.Tensor:
                inverse_rms = torch.rsqrt(
                    values.float().pow(2).mean(-1, keepdim=True) + current.eps
                ).detach()
                return (
                    values.float() * inverse_rms * (1 + current.weight.float())
                ).type_as(values)

            module.forward = MethodType(relp_norm, module)
            norms += 1
        elif isinstance(module, Qwen3_5MLP):
            def relp_mlp(current: Qwen3_5MLP, values: torch.Tensor) -> torch.Tensor:
                gate_values = current.gate_proj(values)
                gate = _IdentityActivation.apply(
                    gate_values, current.act_fn(gate_values)
                )
                return current.down_proj(
                    _HalfProduct.apply(gate, current.up_proj(values))
                )

            module.forward = MethodType(relp_mlp, module)
            mlps += 1
    return {
        "ln_rule": True,
        "identity_rule": True,
        "half_rule": True,
        "norm_modules": norms,
        "mlp_modules": mlps,
    }

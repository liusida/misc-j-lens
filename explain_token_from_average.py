# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gb10-load-llm",
#   "torch>=2.7",
#   "transformers>=5.0",
# ]
# ///
"""Decompose a one-prompt J-Lens token logit across averaged positions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "anthropic-jacobian-lens"))
from jlens import from_hf  # noqa: E402


def exact_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token for this tokenizer: {ids}")
    return ids[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="j_lens_output/one_prompt_layer.pt")
    parser.add_argument("--query-position", type=int, default=51)
    parser.add_argument("--tokens", nargs="+", default=[" cities", " city"])
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    data = torch.load(args.artifact, map_location="cpu", weights_only=False)
    model_name = data["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = from_pretrained_to_cuda(
        AutoModelForCausalLM,
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    model = from_hf(hf_model, tokenizer)

    valid = data["valid_positions"].long()
    raw = data["raw_grads_before_position_mean"].to("cuda").float()
    h = data["source_hidden"][args.query_position].to("cuda").float()
    n = len(valid)

    # contribution[p, i] = (G[:, source_position_p, :] @ h)[i] / n
    # einsum avoids materializing another copy of the full raw tensor.
    contributions = torch.einsum("itj,j->ti", raw[:, valid, :], h) / n
    transported = contributions.sum(dim=0)
    saved_transport = data["j_prompt_after_position_mean"].to("cuda") @ h
    transport_error = (transported - saved_transport).abs().max().item()

    norm = model._final_norm
    lm_head = model._lm_head
    weight = norm.weight.float()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    inv_rms = torch.rsqrt(transported.square().mean() + eps)
    normed = transported * inv_rms * weight
    model_logits_bf16 = lm_head(normed.to(lm_head.weight.dtype)).float()
    logits_fp32 = normed @ lm_head.weight.float().T

    print(
        f"Query h: layer {data['layer']}, position {args.query_position}, "
        f"token={data['tokens'][args.query_position]!r}"
    )
    print(f"J average positions: {valid.tolist()} (N={n})")
    print(f"max |sum_t (G_t h/N) - J_prompt h| = {transport_error:.3e}")
    top_values, top_ids = logits_fp32.topk(10)
    print("Top decoded tokens:")
    for rank, (value, token_id) in enumerate(zip(top_values, top_ids, strict=True), 1):
        text = tokenizer.decode([int(token_id)])
        print(f"  {rank:2d}. {text!r:18s} logit={value.item():+.8f}")

    for token_text in args.tokens:
        token_id = exact_token_id(tokenizer, token_text)
        unembed = lm_head.weight[token_id].float()

        # With the denominator computed from the complete transported vector,
        # RMSNorm is linear in each component. These values therefore sum
        # exactly to the complete token logit (assuming the usual bias-free head).
        additive = (contributions * weight * inv_rms) @ unembed

        # Leave-one-out recomputes RMSNorm after removing each contribution.
        without = transported.unsqueeze(0) - contributions
        without_inv_rms = torch.rsqrt(without.square().mean(dim=1) + eps)
        without_normed = without * without_inv_rms[:, None] * weight
        without_logits = without_normed @ unembed
        leave_one_out = logits_fp32[token_id] - without_logits

        order = additive.abs().argsort(descending=True)
        rank = int((logits_fp32 > logits_fp32[token_id]).sum().item()) + 1
        print(f"\nToken {token_text!r} (id={token_id})")
        print(f"  vocabulary rank:                {rank}")
        print(f"  model BF16 logit:               {model_logits_bf16[token_id].item():+.8f}")
        print(f"  FP32 reference logit:           {logits_fp32[token_id].item():+.8f}")
        print(f"  sum of additive contributions:  {additive.sum().item():+.8f}")
        print("  largest |contribution| positions:")
        print("    pos token           additive_logit   leave-one-out_delta")
        for index in order[: args.top].tolist():
            position = int(valid[index])
            token = data["tokens"][position]
            print(
                f"    {position:02d}  {token!r:14s} "
                f"{additive[index].item():+15.8f} "
                f"{leave_one_out[index].item():+19.8f}"
            )

    print("\nInterpretation:")
    print("  additive_logit uses the complete y's RMS denominator and sums exactly.")
    print("  leave-one-out_delta recomputes RMSNorm and need not sum across positions.")


if __name__ == "__main__":
    main()

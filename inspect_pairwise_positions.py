# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gb10-load-llm",
#   "torch>=2.7",
#   "transformers>=5.0",
# ]
# ///
"""Compute individual target/source-position Jacobians for two positions."""

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
from jlens.hooks import ActivationRecorder  # noqa: E402
from j_lens_one_prompt import DEFAULT_PROMPT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--source-position", type=int, default=16)
    parser.add_argument("--target-positions", type=int, nargs="+", default=[16, 17])
    parser.add_argument("--vjp-batch-size", type=int, default=8)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output", type=Path, default=Path("j_lens_output/pairwise_16_16_17_16.pt")
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = from_pretrained_to_cuda(
        AutoModelForCausalLM,
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    hf_model.config.use_cache = False
    model = from_hf(hf_model, tokenizer)
    input_ids = model.encode(args.prompt, max_length=128)
    tokens = [tokenizer.decode([int(x)]).replace("\n", "\\n") for x in input_ids[0]]
    target_layer = model.n_layers - 1
    d_model = model.d_model
    batch_size = args.vjp_batch_size

    print(f"source t={args.source_position}: {tokens[args.source_position]!r}")
    for target_position in args.target_positions:
        print(f"target t'={target_position}: {tokens[target_position]!r}")

    matrices: dict[tuple[int, int], torch.Tensor] = {}
    with ActivationRecorder(
        model.layers,
        at=[args.layer, target_layer],
        start_graph_at=args.layer,
    ) as recorder, torch.enable_grad():
        model.forward(input_ids.expand(batch_size, -1))
        source = recorder.activations[args.layer]
        target = recorder.activations[target_layer]
        batch_indices = torch.arange(batch_size, device=target.device)
        cotangent = torch.zeros_like(target)
        total_passes = len(args.target_positions) * ((d_model + batch_size - 1) // batch_size)
        pass_number = 0

        for target_number, target_position in enumerate(args.target_positions):
            matrix = torch.empty((d_model, d_model), dtype=torch.bfloat16)
            for start in range(0, d_model, batch_size):
                stop = min(start + batch_size, d_model)
                n = stop - start
                pass_number += 1
                cotangent.zero_()
                cotangent[
                    batch_indices[:n],
                    target_position,
                    start + batch_indices[:n],
                ] = 1
                final_pass = pass_number == total_passes
                (gradient,) = torch.autograd.grad(
                    target,
                    source,
                    grad_outputs=cotangent,
                    retain_graph=not final_pass,
                )
                matrix[start:stop].copy_(
                    gradient[:n, args.source_position, :].detach().cpu()
                )
                if pass_number % 64 == 0 or final_pass:
                    print(f"backward pass {pass_number}/{total_passes}")
            matrices[(target_position, args.source_position)] = matrix

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "prompt": args.prompt,
            "tokens": tokens,
            "source_layer": args.layer,
            "target_layer": target_layer,
            "matrices": matrices,
        },
        args.output,
    )

    print("\nEach matrix is [output coordinate, input coordinate].")
    for (target_position, source_position), matrix in matrices.items():
        fp32 = matrix.float()
        print(
            f"G_{{{target_position},{source_position}}}: "
            f"target={tokens[target_position]!r}, source={tokens[source_position]!r}"
        )
        print(f"  shape={list(matrix.shape)}, Frobenius norm={fp32.norm().item():.8f}")
        print("  top-left 5x5:")
        print(fp32[:5, :5])
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

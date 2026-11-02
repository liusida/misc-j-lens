# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gb10-load-llm",
#   "torch>=2.7",
#   "transformers>=5.0",
# ]
# ///
"""Compute one prompt's contribution to a Jacobian lens for one Qwen layer.

This follows the reproduction pseudocode in the J-Lens paper:

    G[i, t, :] = d sum_{valid t'} h_target[t', i] / d h_layer[t, :]
    J_prompt[i, :] = mean_{valid t} G[i, t, :]
    J_layer = mean_prompt J_prompt       # only one prompt in this demo

Causality makes gradients for t' < t zero, so summing over all final positions
is equivalent to summing over the allowed t' >= t positions. The paper's
pseudocode sums target positions and averages source positions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = (
    "Paris is the capital and largest city of France. It stands on the Seine "
    "and has been a major center of finance, diplomacy, commerce, culture, "
    "fashion, and gastronomy since the seventeenth century. Visitors often "
    "recognize the city by the Eiffel Tower and"
)

OFFICIAL_REPO = Path(__file__).resolve().parent / "anthropic-jacobian-lens"
if not OFFICIAL_REPO.is_dir():
    raise RuntimeError(
        f"Missing official reference checkout at {OFFICIAL_REPO}. Run: "
        "git clone https://github.com/anthropics/jacobian-lens.git "
        "anthropic-jacobian-lens"
    )
sys.path.insert(0, str(OFFICIAL_REPO))

from jlens import from_hf  # noqa: E402
from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Block-output index, matching Anthropic's official implementation",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--vjp-batch-size",
        type=int,
        default=8,
        help="Number of final residual coordinates differentiated in parallel",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=SKIP_FIRST_N_POSITIONS,
        help="Leading positions excluded from source and target averages",
    )
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("j_lens_output"))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def readable_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id]).replace("\n", "\\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This GB10 demonstration requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Loading model via gb10-load-llm: {args.model}")
    model = from_pretrained_to_cuda(
        AutoModelForCausalLM,
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    model.eval()
    model.config.use_cache = False
    # Use Anthropic's official HF adapter and hook locations. Its constructor
    # freezes parameters; only the recorded residual activation needs grads.
    lens_model = from_hf(model, tokenizer)
    input_ids = lens_model.encode(args.prompt, max_length=args.max_seq_len)
    token_ids = input_ids[0]
    tokens = [readable_token(tokenizer, int(token_id)) for token_id in token_ids]
    position_mask = valid_position_mask(len(tokens), skip_first=args.skip_first)
    valid_positions_cpu = position_mask.nonzero(as_tuple=True)[0]

    print(f"Prompt: {args.prompt!r}")
    print(f"Tokens ({len(tokens)}):")
    for position, (token_id, token) in enumerate(zip(token_ids.tolist(), tokens, strict=True)):
        print(f"  t={position:02d} id={token_id:6d} token={token!r}")

    if not 0 <= args.layer < lens_model.n_layers - 1:
        raise ValueError(
            f"--layer must be in [0, {lens_model.n_layers - 2}], got {args.layer}"
        )
    target_layer = lens_model.n_layers - 1
    sequence_length = input_ids.shape[1]
    d_model = lens_model.d_model
    valid_positions = valid_positions_cpu.to(lens_model.input_device)
    print(
        f"Valid positions after skip_first={args.skip_first} and final-position "
        f"exclusion: {valid_positions_cpu.tolist()}"
    )
    print(
        "Running differentiable forward pass with official ActivationRecorder "
        f"(source block L{args.layer}, target block L{target_layer})..."
    )

    # raw_grads[i, t, j] is the unaveraged datum in the paper pseudocode:
    # d(sum_{t'} final[t', i]) / d(source[t, j]).
    # It can be large: D*T*D bf16 values (about 128 MiB for D=2048,T=16).
    raw_grads_cpu = torch.empty(
        (d_model, sequence_length, d_model), dtype=torch.bfloat16, device="cpu"
    )
    num_batches = math.ceil(d_model / args.vjp_batch_size)
    started = time.monotonic()
    with (
        ActivationRecorder(
            lens_model.layers,
            at=[args.layer, target_layer],
            start_graph_at=args.layer,
        ) as recorder,
        torch.enable_grad(),
    ):
        # This is exactly the official batching strategy: each replicated
        # prompt batch element carries one output-coordinate cotangent.
        replicated_ids = input_ids.expand(args.vjp_batch_size, -1)
        lens_model.forward(replicated_ids)
        source = recorder.activations[args.layer]
        final = recorder.activations[target_layer]
        batch_indices = torch.arange(args.vjp_batch_size, device=final.device)
        cotangent = torch.zeros_like(final)

        for batch_index, start in enumerate(
            range(0, d_model, args.vjp_batch_size), 1
        ):
            stop = min(start + args.vjp_batch_size, d_model)
            batch = stop - start
            cotangent.zero_()
            cotangent[
                batch_indices[:batch, None],
                valid_positions[None, :],
                start + batch_indices[:batch, None],
            ] = 1
            (gradient,) = torch.autograd.grad(
                outputs=final,
                inputs=source,
                grad_outputs=cotangent,
                retain_graph=stop < d_model,
                create_graph=False,
            )
            raw_grads_cpu[start:stop].copy_(gradient[:batch].detach().to("cpu"))
            elapsed = time.monotonic() - started
            print(
                f"VJP batch {batch_index:3d}/{num_batches}: rows "
                f"{start:4d}:{stop:4d} ({elapsed:.1f}s elapsed)"
            )

        source_hidden_cpu = source[0].detach().cpu()
        target_hidden_cpu = final[0].detach().cpu()

    # The first averaging stage: within this prompt, over source token position.
    # Accumulate in fp32 so the displayed equality is not obscured by bf16 sums.
    per_position_fp32 = raw_grads_cpu[:, valid_positions_cpu, :].float()
    j_prompt = per_position_fp32.mean(dim=1)
    # The second stage would average j_prompt across prompts. With one prompt:
    j_layer = torch.stack([j_prompt], dim=0).mean(dim=0)
    max_mean_error = (j_layer - per_position_fp32.mean(dim=1)).abs().max().item()

    # Demonstrate a lens readout at the final source token.
    source_activation = source_hidden_cpu[-1].to(lens_model.input_device)
    mapped = j_layer.to(device=source_activation.device, dtype=source_activation.dtype) @ source_activation
    logits = lens_model.unembed(mapped.unsqueeze(0))[0].float()
    top_values, top_ids = logits.topk(args.top_k)
    lens_top_tokens = [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": readable_token(tokenizer, int(token_id)),
            "logit": float(value),
        }
        for rank, (value, token_id) in enumerate(
            zip(top_values.cpu(), top_ids.cpu(), strict=True), 1
        )
    ]

    artifact = {
        "model": args.model,
        "layer": args.layer,
        "prompt": args.prompt,
        "token_ids": token_ids.detach().cpu(),
        "tokens": tokens,
        "valid_positions": valid_positions_cpu,
        "source_hidden": source_hidden_cpu,
        "target_hidden": target_hidden_cpu,
        "raw_grads_before_position_mean": raw_grads_cpu,
        "j_prompt_after_position_mean": j_prompt,
        "j_layer_after_prompt_mean": j_layer,
        "lens_source_position": sequence_length - 1,
        "lens_top_tokens": lens_top_tokens,
    }
    artifact_path = args.output_dir / "one_prompt_layer.pt"
    torch.save(artifact, artifact_path)

    summary = {
        "formula_raw": "G[i,t,j] = d sum_{valid tprime} h_target[tprime,i] / d h_layer[t,j]",
        "formula_position_mean": "J_prompt[i,j] = mean_{valid t} G[i,t,j]",
        "formula_prompt_mean": "J_layer[i,j] = mean_prompt J_prompt[prompt,i,j]",
        "model": args.model,
        "layer": args.layer,
        "target_layer": target_layer,
        "prompt": args.prompt,
        "skip_first": args.skip_first,
        "valid_positions": valid_positions_cpu.tolist(),
        "shape_raw_grads_i_t_j": list(raw_grads_cpu.shape),
        "shape_j_prompt_i_j": list(j_prompt.shape),
        "raw_dtype_on_disk": str(raw_grads_cpu.dtype),
        "means_accumulated_as": "torch.float32",
        "max_reconstruction_error": max_mean_error,
        "lens_source_position": sequence_length - 1,
        "lens_source_token": tokens[-1],
        "lens_top_tokens": lens_top_tokens,
        "artifact": str(artifact_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print("\nAveraging audit")
    print(f"  raw G shape [output_dim, all source positions, input_dim]: {raw_grads_cpu.shape}")
    print(f"  positions included in mean:                              {valid_positions_cpu.tolist()}")
    print(f"  mean over valid source positions -> J_prompt shape:      {j_prompt.shape}")
    print(f"  mean over one prompt -> J_layer shape:                {j_layer.shape}")
    print(f"  max reconstruction error:                             {max_mean_error:.3e}")
    print(f"Lens readout for final token {tokens[-1]!r}:")
    for item in lens_top_tokens:
        print(f"  {item['rank']:2d}. {item['token']!r:20s} logit={item['logit']:+.4f}")
    print(f"Saved full tensors to {artifact_path}")
    print(f"Saved human-readable summary to {summary_path}")


if __name__ == "__main__":
    main()

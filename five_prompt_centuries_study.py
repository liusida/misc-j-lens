# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gb10-load-llm",
#   "torch>=2.7",
#   "transformers>=5.0",
# ]
# ///
"""Fit a five-prompt L12 J-Lens and fully attribute the ' centuries' logit."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
OFFICIAL_REPO = ROOT / "anthropic-jacobian-lens"
sys.path.insert(0, str(OFFICIAL_REPO))

from jlens import from_hf  # noqa: E402
from jlens.fitting import jacobian_for_prompt, valid_position_mask  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from j_lens_one_prompt import DEFAULT_PROMPT  # noqa: E402


PROMPTS = [
    DEFAULT_PROMPT,
    (
        "The history of astronomy spans many cultures and centuries. Ancient "
        "observers recorded the motions of the Sun, Moon, planets, and stars, "
        "while later scholars developed mathematical models that gradually "
        "changed humanity's understanding of its place in the universe."
    ),
    (
        "Forests regulate climate, protect watersheds, and provide habitats "
        "for countless species. A mature forest contains layers ranging from "
        "the shaded floor to the canopy, with fungi, insects, birds, and large "
        "mammals participating in interconnected cycles of nutrients and energy."
    ),
    (
        "Railway networks transformed travel and trade during industrialization. "
        "Engineers constructed bridges, tunnels, stations, and long tracks that "
        "connected inland towns with ports. The resulting movement of people "
        "and goods helped cities grow and regional markets become integrated."
    ),
    (
        "A computer program is a sequence of instructions expressed in a "
        "language that a machine can execute. Large systems are usually divided "
        "into modules with clear interfaces, allowing programmers to test each "
        "part, locate errors, and change behavior without rewriting everything."
    ),
]


def one_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected one token for {text!r}, got {ids}")
    return ids[0]


def decoded_tokens(tokenizer, ids: torch.Tensor) -> list[str]:
    return [tokenizer.decode([int(token)]).replace("\n", "\\n") for token in ids]


def main() -> None:
    model_name = "Qwen/Qwen3.5-2B"
    source_layer = 12
    query_position = 17
    word = " centuries"
    skip_first = 16
    dim_batch = 8
    output_dir = ROOT / "j_lens_output" / "five_prompt_centuries"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = from_pretrained_to_cuda(
        AutoModelForCausalLM,
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    hf_model.config.use_cache = False
    model = from_hf(hf_model, tokenizer)
    target_layer = model.n_layers - 1
    token_id = one_token_id(tokenizer, word)

    query_ids = model.encode(PROMPTS[0], max_length=128)
    query_tokens = decoded_tokens(tokenizer, query_ids[0])
    print(
        f"Query activation: h_L{source_layer},t={query_position}, "
        f"token={query_tokens[query_position]!r}"
    )
    print(f"Readout token: {word!r}, id={token_id}")

    # Stage 1: fit exactly one layer, retaining each per-prompt matrix.
    per_prompt: list[torch.Tensor] = []
    prompt_metadata: list[dict] = []
    for prompt_index, prompt in enumerate(PROMPTS):
        print(f"\nFitting prompt {prompt_index + 1}/{len(PROMPTS)}")
        jacobians, seq_len, n_valid = jacobian_for_prompt(
            model,
            prompt,
            [source_layer],
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=128,
            skip_first=skip_first,
        )
        per_prompt.append(jacobians[source_layer])
        prompt_metadata.append(
            {"prompt_index": prompt_index, "seq_len": seq_len, "n_valid": n_valid}
        )
        print(f"  seq_len={seq_len}, valid={n_valid}")
    j_bar = torch.stack(per_prompt).mean(dim=0)
    torch.save(
        {
            "J": j_bar.half(),
            "per_prompt_J": [item.half() for item in per_prompt],
            "prompts": PROMPTS,
            "source_layer": source_layer,
            "target_layer": target_layer,
            "skip_first": skip_first,
        },
        output_dir / "five_prompt_lens.pt",
    )

    # Capture query activations for every position.
    with ActivationRecorder(model.layers, at=[source_layer]) as recorder:
        with torch.no_grad():
            model.forward(query_ids)
        query_hidden = recorder.activations[source_layer][0].float()
    h_query = query_hidden[query_position]
    j_cuda = j_bar.to("cuda")
    transported_all = query_hidden @ j_cuda.T
    transported = transported_all[query_position]

    norm = model._final_norm
    norm_weight = norm.weight.float()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    unembed_all = model._lm_head.weight.float()
    unembed_word = unembed_all[token_id]
    inv_rms = torch.rsqrt(transported.square().mean() + eps)
    effective_output_direction = unembed_word * norm_weight * inv_rms
    logits_query = (transported * inv_rms * norm_weight) @ unembed_all.T
    word_logit = logits_query[token_id]
    word_rank = int((logits_query > word_logit).sum().item()) + 1

    # The input-space J-Lens vector for this word. Its dot with h is the logit.
    lens_vector = j_cuda.T @ effective_output_direction
    dimension_contributions = h_query * lens_vector

    # Compare every activation position against the same word readout. Each
    # position has its own RMS denominator, exactly as normal lens application.
    inv_rms_all = torch.rsqrt(transported_all.square().mean(dim=1) + eps)
    logits_all = (
        transported_all * inv_rms_all[:, None] * norm_weight
    ) @ unembed_all.T
    word_logits_all = logits_all[:, token_id]
    word_ranks_all = 1 + (logits_all > word_logits_all[:, None]).sum(dim=1)
    cosine_all = torch.nn.functional.cosine_similarity(
        query_hidden, lens_vector.unsqueeze(0), dim=1
    )

    # Stage 2: exact scalar decomposition over every prompt/source/target pair.
    # Since effective_output_direction is held fixed from the complete output,
    # all rows add exactly to the FP32 reference token logit.
    rows: list[dict] = []
    for prompt_index, prompt in enumerate(PROMPTS):
        ids = model.encode(prompt, max_length=128)
        tokens = decoded_tokens(tokenizer, ids[0])
        mask = valid_position_mask(ids.shape[1], skip_first=skip_first)
        valid_cpu = mask.nonzero(as_tuple=True)[0]
        valid = valid_cpu.to("cuda")
        n_valid = len(valid_cpu)
        print(f"Attributing prompt {prompt_index + 1}/{len(PROMPTS)}")
        with ActivationRecorder(
            model.layers,
            at=[source_layer, target_layer],
            start_graph_at=source_layer,
        ) as recorder, torch.enable_grad():
            model.forward(ids)
            source = recorder.activations[source_layer]
            target = recorder.activations[target_layer]
            for target_number, target_position_tensor in enumerate(valid):
                target_position = int(target_position_tensor)
                cotangent = torch.zeros_like(target)
                cotangent[0, target_position] = effective_output_direction.to(
                    target.dtype
                )
                last = target_number == n_valid - 1
                (gradient,) = torch.autograd.grad(
                    target,
                    source,
                    grad_outputs=cotangent,
                    retain_graph=not last,
                )
                # Each valid source t <= t' contributes a scalar
                # a^T G_{t',t} h_query / (P*N_p).
                scalar_by_source = gradient[0].float() @ h_query
                for source_position_tensor in valid_cpu:
                    source_position = int(source_position_tensor)
                    if source_position > target_position:
                        continue
                    contribution = scalar_by_source[source_position] / (
                        len(PROMPTS) * n_valid
                    )
                    rows.append(
                        {
                            "prompt_index": prompt_index,
                            "source_layer": source_layer,
                            "source_position": source_position,
                            "source_token": tokens[source_position],
                            "target_layer": target_layer,
                            "target_position": target_position,
                            "target_token": tokens[target_position],
                            "contribution_to_centuries_logit": float(contribution),
                        }
                    )

    csv_path = output_dir / "centuries_all_pair_contributions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    position_rows = []
    for position, token in enumerate(query_tokens):
        position_rows.append(
            {
                "query_position": position,
                "query_token": token,
                "centuries_logit": float(word_logits_all[position]),
                "centuries_rank": int(word_ranks_all[position]),
                "cosine_with_centuries_lens_vector": float(cosine_all[position]),
            }
        )
    with (output_dir / "query_position_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(position_rows[0]))
        writer.writeheader()
        writer.writerows(position_rows)

    dim_order = dimension_contributions.abs().argsort(descending=True)
    dim_rows = [
        {
            "hidden_dimension": int(index),
            "h_value": float(h_query[index]),
            "lens_vector_value": float(lens_vector[index]),
            "product_contribution": float(dimension_contributions[index]),
        }
        for index in dim_order.cpu()
    ]
    with (output_dir / "h17_hidden_dimension_contributions.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dim_rows[0]))
        writer.writeheader()
        writer.writerows(dim_rows)

    total_pair_contribution = sum(row["contribution_to_centuries_logit"] for row in rows)
    top_pair_rows = sorted(
        rows, key=lambda row: abs(row["contribution_to_centuries_logit"]), reverse=True
    )[:20]
    top_dim_rows = dim_rows[:20]
    top_values, top_ids = logits_query.topk(20)
    decoded_top = [
        {
            "rank": rank,
            "token": tokenizer.decode([int(token)]),
            "token_id": int(token),
            "logit": float(value),
        }
        for rank, (value, token) in enumerate(
            zip(top_values.cpu(), top_ids.cpu(), strict=True), 1
        )
    ]
    summary = {
        "query": {
            "source_layer": source_layer,
            "position": query_position,
            "token": query_tokens[query_position],
        },
        "readout_token": word,
        "readout_token_id": token_id,
        "centuries_logit_fp32": float(word_logit),
        "centuries_vocabulary_rank": word_rank,
        "sum_all_pair_contributions": total_pair_contribution,
        "pair_reconstruction_error": abs(total_pair_contribution - float(word_logit)),
        "number_of_pair_rows": len(rows),
        "top_decoded_tokens": decoded_top,
        "top_pair_contributions": top_pair_rows,
        "top_hidden_dimension_contributions": top_dim_rows,
        "prompt_metadata": prompt_metadata,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    print("\nResult")
    print(f"  {word!r} logit: {float(word_logit):+.8f}")
    print(f"  vocabulary rank: {word_rank}")
    print(f"  pair rows: {len(rows)}")
    print(f"  sum of all pair contributions: {total_pair_contribution:+.8f}")
    print(
        f"  reconstruction error: "
        f"{abs(total_pair_contribution - float(word_logit)):.3e}"
    )
    print("  top decoded tokens:")
    for item in decoded_top[:10]:
        print(f"    {item['rank']:2d}. {item['token']!r:18s} {item['logit']:+.8f}")
    print("  largest pair contributions:")
    for row in top_pair_rows[:10]:
        print(
            f"    p{row['prompt_index']} "
            f"L{row['source_layer']}:{row['source_position']} "
            f"{row['source_token']!r} -> "
            f"L{row['target_layer']}:{row['target_position']} "
            f"{row['target_token']!r}: "
            f"{row['contribution_to_centuries_logit']:+.8f}"
        )
    print(f"Saved all outputs under {output_dir}")


if __name__ == "__main__":
    main()

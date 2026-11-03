# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch", "transformers>=5", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Offline CUDA experiment: downstream responses to hidden/token directions."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import from_pretrained_to_cuda
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "references/upstream/anthropic-jacobian-lens"))
from jlens import from_hf
from jlens.hooks import ActivationRecorder

PROMPT = ("Paris is the capital and largest city of France. It stands on the Seine and has been "
          "a major center of finance, diplomacy, commerce, culture, fashion, and gastronomy since "
          "the seventeenth century. Visitors often recognize the city by the Eiffel Tower and")

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--position", type=int, default=17)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--token", help="Override automatic top lens token, e.g. ' centuries'")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Perturbation norm as fraction of original activation norm")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/oscillation/output")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if a.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    revision = "980e34bc0f0460c437bb4fa0238a58b9b50158d3"
    folder = "workspace-lenses/v2/Qwen--Qwen3.5-2B/j"
    path = hf_hub_download("sida/llmdbg", f"{folder}/lens.safetensors",
                           repo_type="dataset", revision=revision, local_files_only=True)
    name = "Qwen/Qwen3.5-2B"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, name,
                                 dtype=torch.bfloat16, local_files_only=True)
    model = from_hf(hf, tokenizer)
    if not 0 <= a.layer < model.n_layers - 1:
        raise ValueError("Source must precede final block")
    ids = model.encode(a.prompt, max_length=512)
    if not 0 <= a.position < ids.shape[1]:
        raise ValueError("Invalid query position")
    tokens = [tokenizer.decode([int(i)]) for i in ids[0]]
    last = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[a.layer, last]) as rec:
        model.forward(ids)
        h = rec.activations[a.layer][0, a.position].float().clone()
        baseline = rec.activations[last][0].float().clone()
    J = load_file(path)[f"post.{a.layer}"].cuda().float()
    y = J @ h
    logits = model.unembed(y[None])[0].float()
    q = int(logits.argmax())
    if a.token is not None:
        word_ids = tokenizer.encode(a.token, add_special_tokens=False)
        if len(word_ids) != 1:
            raise ValueError("Requested token must encode to exactly one token")
        q = word_ids[0]
    # Qwen3.5 RMSNorm gain is zero-centered: 1 + weight.
    gain = 1 + model._final_norm.weight.float()
    output_word = gain * model._lm_head.weight[q].float()
    directions = {"hidden_state": h / h.norm(), "top_token": J.T @ output_word}
    directions["top_token"] /= directions["top_token"].norm()
    a.output.mkdir(parents=True, exist_ok=True)
    rows, saved, checks = [], {}, {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for row_index, (label, direction) in enumerate(directions.items()):
        def forward_delta(scale):
            def hook(module, inputs, output):
                tensor = output if torch.is_tensor(output) else output[0]
                changed = tensor.clone()
                changed[:, a.position] = (tensor[:, a.position].float() + scale * direction).to(tensor.dtype)
                return changed if torch.is_tensor(output) else (changed, *output[1:])
            handle = model.layers[a.layer].register_forward_hook(hook)
            try:
                with ActivationRecorder(model.layers, at=[last]) as rec:
                    model.forward(ids)
                    return rec.activations[last][0].float().clone()
            finally:
                handle.remove()
        # Central differences estimate K[t',t]v without requiring hybrid-kernel JVP support.
        estimates = []
        for fraction in [a.epsilon, a.epsilon / 2]:
            delta = fraction * float(h.norm())
            estimates.append((forward_delta(delta) - forward_delta(-delta)) / (2 * delta))
        response = estimates[-1][a.position:]
        relative_change = float((estimates[0]-estimates[1]).norm() / estimates[1].norm().clamp_min(1e-12))
        checks[label] = {"relative_change_when_halving_epsilon": relative_change,
                         "pre_source_response_norm": float(estimates[-1][:a.position].norm())}
        normalized = torch.nn.functional.normalize(response, dim=-1)
        cosine = normalized @ normalized.T
        reference = torch.nn.functional.normalize(J @ direction, dim=0)
        projection = response @ reference
        word_projection = response @ output_word
        lags = np.arange(len(response))
        # Vector Fourier response; no token choice needed. Large DC can mask other peaks.
        omega = torch.linspace(0, torch.pi, 257, device="cuda")
        phase = torch.exp(-1j * omega[:, None] * torch.arange(len(response), device="cuda"))
        spectrum = (phase @ response.to(torch.complex64) / len(response)).norm(dim=1)
        for lag in lags:
            t = int(lag) + a.position
            rows.append(dict(method=label, target_position=t, target_token=tokens[t],
                             response_norm=float(response[lag].norm()),
                             projection_on_average_transport=float(projection[lag]),
                             word_projection_before_rms=float(word_projection[lag])))
        axes[row_index, 0].plot(lags, projection.cpu(), marker=".")
        axes[row_index, 0].axhline(0, color="gray", lw=0.7)
        axes[row_index, 0].set(title=f"{label}: response along average transport", xlabel="Target lag", ylabel="Signed projection")
        image = axes[row_index, 1].imshow(cosine.cpu(), vmin=-1, vmax=1, cmap="coolwarm", origin="lower")
        axes[row_index, 1].set(title="Similarity of full response vectors", xlabel="Target lag", ylabel="Target lag")
        fig.colorbar(image, ax=axes[row_index, 1])
        axes[row_index, 2].plot(omega.cpu()/torch.pi, spectrum.cpu())
        axes[row_index, 2].set(title="Vector frequency response", xlabel="ω / π", ylabel="Norm of weighted mean")
        saved[label] = {"direction": direction.cpu(), "response": response.cpu(),
                        "pairwise_cosine": cosine.cpu(), "omega": omega.cpu(), "spectrum": spectrum.cpu()}
        print(label, checks[label], flush=True)
    fig.savefig(a.output / "responses.png", dpi=160)
    with (a.output / "responses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(model=name, lens_revision=revision, lens_file=path, prompt=a.prompt,
                   layer=a.layer, position=a.position, query_token=tokens[a.position],
                   decoded_token=tokenizer.decode([q]), token_rank=int((logits > logits[q]).sum())+1,
                   epsilon=a.epsilon, checks=checks,
                   note="Central finite differences in BF16; inspect epsilon sensitivity. Frequency peaks are descriptive, not significance tests.")
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    torch.save(dict(h=h.cpu(), transported_h=y.cpu(), baseline=baseline.cpu(), methods=saved), a.output / "responses.pt")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

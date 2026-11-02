# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors"]
# ///
"""Patch one block-output activation toward a token's J-Lens direction.

Run: uv run test_centuries_patch.py
Uses the cached sida/llmdbg v2 lens. No fitting is performed.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "anthropic-jacobian-lens"))
from jlens import from_hf
from jlens.hooks import ActivationRecorder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--position", type=int, default=17)
    parser.add_argument("--token", default=" centuries")
    parser.add_argument("--output", type=Path, default=ROOT / "j_lens_output/centuries_patch_v2_results.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    revision = "980e34bc0f0460c437bb4fa0238a58b9b50158d3"
    lens_folder = "workspace-lenses/v2/Qwen--Qwen3.5-2B/j"
    def cached(filename):
        return hf_hub_download(repo_id="sida/llmdbg", repo_type="dataset",
                               filename=f"{lens_folder}/{filename}",
                               revision=revision, local_files_only=True)
    manifest = json.loads(Path(cached("manifest.json")).read_text())
    lens_path = cached("lens.safetensors")
    if manifest["model_id"] != "Qwen/Qwen3.5-2B" or manifest["method"] != "j":
        raise ValueError("Unexpected lens model or method")
    print(f"Using cached v2 lens: {lens_path}; prompts={manifest['prompts']}", flush=True)
    name = "Qwen/Qwen3.5-2B"
    tokenizer = AutoTokenizer.from_pretrained(name)
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, name, dtype=torch.bfloat16)
    model = from_hf(hf, tokenizer)
    layer = 12
    J = load_file(lens_path)[f"post.{layer}"].cuda().float()
    word_ids = tokenizer.encode(args.token, add_special_tokens=False)
    if len(word_ids) != 1:
        raise ValueError(f"Expected one token: {word_ids}")
    word_id = word_ids[0]
    prompt = "Paris is the capital and largest city of France. It stands on the Seine and has been"
    full_ids = model.encode(prompt, max_length=128)
    # Stop at the queried token: generate its immediate future, without forcing
    # the rest of the original passage (which includes 'century').
    prefix = full_ids[:, :args.position + 1]
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        model.forward(prefix)
        original = recorder.activations[layer][0, args.position].float().clone()
    norm = model._final_norm
    # Qwen3.5 uses zero-centered RMSNorm: gain = 1 + weight.
    probe = J @ original
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    gain = 1.0 + norm.weight.float()
    manual = probe * torch.rsqrt(probe.square().mean() + eps) * gain
    actual = norm(probe.to(norm.weight.dtype)).float()
    if not torch.allclose(manual, actual, atol=0.1, rtol=0.03):
        raise RuntimeError("Final norm formula does not match module")
    direction = J.T @ (gain * model._lm_head.weight[word_id].float())
    unit = direction / direction.norm()
    aligned = original.norm() * unit
    results = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        replacement = (1-alpha)*original + alpha*aligned
        replacement = replacement / replacement.norm() * original.norm()

        def hook(module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            changed = hidden.clone()
            changed[:, args.position] = replacement.to(hidden.dtype)
            return changed if torch.is_tensor(output) else (changed, *output[1:])

        handle = model.layers[layer].register_forward_hook(hook)
        try:
            ids = prefix.clone()
            probabilities = []
            generated = []
            # Recompute prefix with use_cache=False. Patch only this one fixed
            # position on each replay; generated positions are never patched.
            for step in range(args.max_new_tokens):
                output = model.forward(ids)
                logits = model._lm_head(output.last_hidden_state[:, -1]).float()[0]
                probabilities.append(float(logits.softmax(-1)[word_id]))
                next_id = int(logits.argmax())
                generated.append(next_id)
                ids = torch.cat([ids, ids.new_tensor([[next_id]])], dim=1)
                if next_id == tokenizer.eos_token_id:
                    break
        finally:
            handle.remove()
        lens_logits = model.unembed((J @ replacement).unsqueeze(0)).float()[0]
        row = dict(alpha=alpha, cosine=float(torch.nn.functional.cosine_similarity(replacement, unit, dim=0)), norm=float(replacement.norm()), original_norm=float(original.norm()), lens_token_rank=int((lens_logits > lens_logits[word_id]).sum())+1, lens_token_logit=float(lens_logits[word_id]), generated_text=tokenizer.decode(generated), exact_token_count=generated.count(word_id), token_probability_by_step=probabilities)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(lens_repo="sida/llmdbg", lens_revision=revision, lens_file=f"{lens_folder}/lens.safetensors", lens_prompts=manifest["prompts"], prefix=tokenizer.decode(prefix[0]), layer=layer, position=args.position, token=args.token, results=results), indent=2, ensure_ascii=False)+"\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

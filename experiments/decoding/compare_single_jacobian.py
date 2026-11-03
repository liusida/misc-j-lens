# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors"]
# ///
"""Compare an unrelated same-position Jacobian with a cached trained lens."""
import csv
import json
import random
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from five_prompt_centuries_study import PROMPTS, from_hf, ActivationRecorder

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/runs/j/legacy/single_jacobian_comparison"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        "sida/llmdbg", "workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors",
        repo_type="dataset", revision="980e34bc0f0460c437bb4fa0238a58b9b50158d3",
        local_files_only=True,
    )
    with safe_open(path, framework="pt", device="cpu") as f:
        trained = f.get_tensor("post.12").float().cuda()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, "Qwen/Qwen3.5-2B",
                                dtype=torch.bfloat16, low_cpu_mem_usage=False)
    model = from_hf(hf, tokenizer)
    ids = model.encode(PROMPTS[0], max_length=128)
    position = random.Random(42).randrange(16, ids.shape[1] - 1)
    print(f"Seed 42: prompt 0 position {position}, token {tokenizer.decode([int(ids[0, position])])!r}", flush=True)
    D, B = model.d_model, 8
    K = torch.empty(D, D, device="cuda", dtype=torch.float32)
    with ActivationRecorder(model.layers, at=[12, 23], start_graph_at=12) as rec:
        model.forward(ids.expand(B, -1))
        source, target = rec.activations[12], rec.activations[23]
        cot = torch.zeros_like(target)
        for start in range(0, D, B):
            n = min(B, D-start)
            indices = torch.arange(n, device="cuda")
            cot.zero_()
            cot[indices, position, start+indices] = 1
            grad, = torch.autograd.grad(target, source, grad_outputs=cot, retain_graph=start+B<D)
            K[start:start+n] = grad[:n, position].float()
            if start % 512 == 0:
                print(f"Jacobian rows {start}/{D}", flush=True)
    del source, target, grad, cot, rec
    query_ids = model.encode(PROMPTS[1], max_length=128)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[12]) as rec:
        model.forward(query_ids)
        h = rec.activations[12][0].float()
        single_logits = model.unembed(h @ K.T).float()
        trained_logits = model.unembed(h @ trained.T).float()
        single_probs = single_logits.softmax(-1)
        trained_probs = trained_logits.softmax(-1)
        kl = (trained_probs * (trained_logits.log_softmax(-1) - single_logits.log_softmax(-1))).sum(-1)
        cos = torch.nn.functional.cosine_similarity(h @ K.T, h @ trained.T, dim=-1)
    rows = []
    for t, token in enumerate(query_ids[0]):
        s = single_logits[t].topk(5).indices.tolist()
        j = trained_logits[t].topk(5).indices.tolist()
        rows.append(dict(position=t, input_token=tokenizer.decode([int(token)]),
                         single_top5=json.dumps([tokenizer.decode([x]) for x in s]),
                         trained_top5=json.dumps([tokenizer.decode([x]) for x in j]),
                         top1_agreement=s[0]==j[0], top5_overlap=len(set(s)&set(j)),
                         transport_cosine=float(cos[t]), kl_trained_to_single=float(kl[t])))
    with (OUT/"comparison.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    torch.save(dict(K=K.cpu(), source_position=position, seed=42, source_layer=12,
                    target_layer=23, trained_key="post.12"), OUT/"single_jacobian.pt")
    summary=dict(seed=42, source_position=position, source_token=tokenizer.decode([int(ids[0,position])]),
                 source_prompt=PROMPTS[0], query_prompt=PROMPTS[1], trained_key="post.12",
                 trained_path=path, tokens=len(rows), top1_agreement=sum(r['top1_agreement'] for r in rows)/len(rows),
                 mean_top5_overlap=sum(r['top5_overlap'] for r in rows)/len(rows),
                 mean_transport_cosine=float(cos.mean()), mean_kl=float(kl.mean()))
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    for row in rows:
        print(row['position'], repr(row['input_token']), row['single_top5'], row['trained_top5'])


if __name__ == "__main__":
    main()

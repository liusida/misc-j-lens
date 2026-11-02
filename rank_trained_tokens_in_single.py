# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors"]
# ///
"""Locate trained-lens top tokens in the saved single-Jacobian readouts."""
import csv
import json
import statistics

from compare_single_jacobian import (
    OUT, PROMPTS, ActivationRecorder, AutoModelForCausalLM, AutoTokenizer,
    from_hf, from_pretrained_to_cuda, hf_hub_download, safe_open, torch,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    K = torch.load(OUT / "single_jacobian.pt", weights_only=True)["K"].cuda()
    path = hf_hub_download(
        "sida/llmdbg", "workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors",
        repo_type="dataset", revision="980e34bc0f0460c437bb4fa0238a58b9b50158d3",
        local_files_only=True,
    )
    with safe_open(path, framework="pt", device="cpu") as f:
        J = f.get_tensor("post.12").float().cuda()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, "Qwen/Qwen3.5-2B",
                                dtype=torch.bfloat16, low_cpu_mem_usage=False)
    model = from_hf(hf, tok)
    ids = model.encode(PROMPTS[1], max_length=128)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[12]) as rec:
        model.forward(ids)
        h = rec.activations[12][0].float()
        single = model.unembed(h @ K.T).float()
        logit_lens = model.unembed(h).float()
        trained = model.unembed(h @ J.T).float()
    rows = []
    for t, token in enumerate(ids[0]):
        q = int(trained[t].argmax())
        value = single[t, q]
        # BF16 readout can tie; give both bounds of the tied rank interval.
        rank = 1 + int((single[t] > value).sum())
        rank_end = int((single[t] >= value).sum())
        ll_value = logit_lens[t, q]
        ll_rank = 1 + int((logit_lens[t] > ll_value).sum())
        ll_rank_end = int((logit_lens[t] >= ll_value).sum())
        rows.append(dict(position=t, input_token=tok.decode([int(token)]),
                         trained_top1_token=tok.decode([q]), trained_top1_id=q,
                         rank_in_single=rank, tied_rank_end=rank_end,
                         rank_in_logit_lens=ll_rank, logit_lens_tied_rank_end=ll_rank_end,
                         logit_lens_top1_token=tok.decode([int(logit_lens[t].argmax())]),
                         single_logit=float(value), single_top1_token=tok.decode([int(single[t].argmax())])))
    with (OUT / "trained_top1_ranks_in_single.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ranks = [r['rank_in_single'] for r in rows]
    summary = dict(positions=len(rows), median_rank=statistics.median(ranks),
                   mean_rank=statistics.mean(ranks),
                   **{f"rank_le_{k}":sum(r <= k for r in ranks) for k in [1,5,10,100,1000]})
    ll_ranks = [r['rank_in_logit_lens'] for r in rows]
    summary['logit_lens'] = dict(median_rank=statistics.median(ll_ranks),
                                mean_rank=statistics.mean(ll_ranks),
                                **{f'rank_le_{k}':sum(r <= k for r in ll_ranks) for k in [1,5,10,100,1000]})
    summary['single_better_rank'] = sum(a < b for a,b in zip(ranks,ll_ranks))
    summary['logit_lens_better_rank'] = sum(a > b for a,b in zip(ranks,ll_ranks))
    summary['equal_rank'] = sum(a == b for a,b in zip(ranks,ll_ranks))
    (OUT / "trained_top1_rank_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    for r in rows:
        print(f"{r['position']:2d} {r['input_token']!r:18s} {r['trained_top1_token']!r:18s} single={r['rank_in_single']} logit_lens={r['rank_in_logit_lens']} LL_top1={r['logit_lens_top1_token']!r}")


if __name__ == "__main__":
    main()

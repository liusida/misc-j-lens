# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors"]
# ///
"""Independent single-Jacobian replication; no matrices are averaged."""
import csv
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from gb10_load_llm import from_pretrained_to_cuda
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'anthropic-jacobian-lens'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder

LAYERS = list(range(0, 22, 3))
SEED = 20261031
REVISION = '980e34bc0f0460c437bb4fa0238a58b9b50158d3'


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-lag', type=int, choices=[0,1], default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(SEED)
    corpus = json.loads((ROOT/'prompts.json').read_text())
    out = ROOT/('results' if args.target_lag == 0 else 'results_next')
    out.mkdir(exist_ok=True)
    cache = out/'jacobians'
    cache.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    model = from_hf(from_pretrained_to_cuda(
        AutoModelForCausalLM, 'Qwen/Qwen3.5-2B', dtype=torch.bfloat16,
        low_cpu_mem_usage=False), tokenizer)
    lens_path = hf_hub_download('sida/llmdbg',
        'workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors',
        repo_type='dataset', revision=REVISION, local_files_only=True)
    source_ids = [model.encode(p, max_length=128) for p in corpus['source']]
    rng = random.Random(SEED)
    positions = [sorted(rng.sample(range(16, x.shape[1]-1), 3)) for x in source_ids]
    atomic_json(out/'manifest.json', dict(seed=SEED, layers=LAYERS,
        positions=positions, prompts=corpus, trained_revision=REVISION,
        target_layer=23, target_lag=args.target_lag,
        selection='3 positions per source, >=16 and excluding last',
        ranking='competition rank: 1 + count strictly greater; BF16 ties possible'))
    with safe_open(lens_path, framework='pt', device='cpu') as f:
        trained = {l:f.get_tensor(f'post.{l}').float().cuda() for l in LAYERS}
    queries = []
    with torch.no_grad():
        for prompt in corpus['evaluation']:
            ids = model.encode(prompt, max_length=128)
            with ActivationRecorder(model.layers, at=LAYERS) as rec:
                model.forward(ids)
                hidden = {l:rec.activations[l][0].float().detach() for l in LAYERS}
            queries.append((ids[0].cpu(), hidden))
    # Precompute trained targets and identity-map ranks, identical for every J.
    reference = {}
    with torch.no_grad():
        for l in LAYERS:
            for e, (ids, hidden) in enumerate(queries):
                h = hidden[l]
                target = model.unembed(h @ trained[l].T).argmax(-1)
                logits = model.unembed(h).float()
                scores = logits.gather(1, target[:,None])
                ranks = 1+(logits > scores).sum(-1)
                reference[l,e] = (target, ranks)
    D, B = model.d_model, 8
    for l in LAYERS:
        for s, ids in enumerate(source_ids):
            for position in positions[s]:
                stem = f'L{l:02d}_source{s:02d}_pos{position:03d}'
                report = out/f'{stem}.csv'
                if report.exists():
                    continue
                checkpoint = cache/f'{stem}.pt'
                if checkpoint.exists():
                    J = torch.load(checkpoint, weights_only=True).cuda().float()
                else:
                    print(f'COMPUTE {stem}', flush=True)
                    J = torch.empty(D,D,device='cuda')
                    with ActivationRecorder(model.layers, at=[l,23], start_graph_at=l) as rec:
                        model.forward(ids.expand(B,-1))
                        source, target = rec.activations[l], rec.activations[23]
                        cot = torch.zeros_like(target)
                        for start in range(0,D,B):
                            idx = torch.arange(min(B,D-start),device='cuda')
                            cot.zero_()
                            cot[idx,position+args.target_lag,start+idx] = 1
                            grad, = torch.autograd.grad(target,source,grad_outputs=cot,
                                                       retain_graph=start+B<D)
                            J[start:start+len(idx)] = grad[:,position].float()
                    torch.save(J.cpu(), checkpoint)
                    del source,target,cot,grad,rec
                rows = []
                with torch.no_grad():
                    for e,(token_ids,hidden) in enumerate(queries):
                        q,ranks_ll = reference[l,e]
                        logits = model.unembed(hidden[l] @ J.T).float()
                        scores = logits.gather(1,q[:,None])
                        ranks = 1+(logits > scores).sum(-1)
                        for t in range(len(token_ids)):
                            rows.append(dict(layer=l,source_prompt=s,source_position=position,
                                source_token=tokenizer.decode([int(ids[0,position])]),
                                target_position=position+args.target_lag,
                                target_token=tokenizer.decode([int(ids[0,position+args.target_lag])]),
                                evaluation_prompt=e,position=t,
                                input_token=tokenizer.decode([int(token_ids[t])]),
                                trained_top_token=tokenizer.decode([int(q[t])]),
                                trained_top_id=int(q[t]),single_rank=int(ranks[t]),
                                logit_lens_rank=int(ranks_ll[t]),exclude_early=t>=16))
                temp = report.with_suffix('.tmp')
                with temp.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                temp.replace(report)
                print(f'SAVED {stem} ({len(rows)} rows)',flush=True)
                del J
    summarize(out)


def summarize(out):
    import statistics
    summaries=[]
    for path in sorted(out.glob('L*.csv')):
        with path.open() as f:
            rows=list(csv.DictReader(f))
        for early in [False,True]:
            selected=[r for r in rows if not early or int(r['position'])>=16]
            # Each passage receives equal weight regardless of length.
            groups={e:[r for r in selected if int(r['evaluation_prompt'])==e] for e in range(10)}
            result=dict(jacobian=path.stem,layer=int(rows[0]['layer']),exclude_first16=early)
            for method,col in [('single','single_rank'),('logit','logit_lens_rank')]:
                result[method+'_mrr']=statistics.mean(
                    statistics.mean(1/int(r[col]) for r in group) for group in groups.values())
                result[method+'_median_rank']=statistics.median(int(r[col]) for r in selected)
                for k in [10,100,1000]:
                    result[f'{method}_top{k}']=statistics.mean(
                        statistics.mean(int(r[col])<=k for r in group) for group in groups.values())
            summaries.append(result)
    atomic_json(out/'summary.json',summaries)
    by_layer=[]
    for l in LAYERS:
        for early in [False,True]:
            group=[r for r in summaries if r['layer']==l and r['exclude_first16']==early]
            by_layer.append(dict(layer=l,exclude_first16=early,jacobians=len(group),
                single_wins_mrr=sum(r['single_mrr']>r['logit_mrr'] for r in group),
                median_mrr_improvement=statistics.median(r['single_mrr']-r['logit_mrr'] for r in group),
                median_rank_improvement=statistics.median(r['logit_median_rank']-r['single_median_rank'] for r in group)))
    atomic_json(out/'layer_summary.json',by_layer)
    print('COMPLETE: 240 independent Jacobians',flush=True)


if __name__ == '__main__':
    main()

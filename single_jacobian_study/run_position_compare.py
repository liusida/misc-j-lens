# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors"]
# ///
"""Compare same-token Jacobians at first / middle / last positions.

Matched-token design (default --filler special):
  the measured vocabulary id T is identical across roles; only left context
  changes. Under causal attention, tokens after T do not affect h_T, so
  "middle" uses a medium-length special-token prefix and an optional suffix
  for sequence shape, while "last" uses a longer prefix with T at the end.

Natural mode (--filler text): T is still matched, but the prefix is the
leading tokens of a natural sentence (truncated), not specials.

Metrics match run.py: agreement with trained J-Lens vs logit lens on the
shared evaluation passages.
"""
import argparse
import csv
import json
import statistics
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
N_SOURCES = 15
# Prefix lengths put T at first / mid / late absolute indices.
PREFIX = {'first': 0, 'middle': 31, 'last': 63}
SUFFIX_MIDDLE = 31  # cosmetic for causal same-pos J; does not affect h_T
SPECIAL_DEFAULT = '<|endoftext|>'


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def pick_content_tokens(model, tokenizer, sentences, n):
    """Unique single-token content ids from sentence starts (fallback: any pos)."""
    seen, texts, tids, origins = set(), [], [], []
    for sentence in sentences:
        ids = model.encode(sentence, max_length=128)[0].tolist()
        for tid in ids:
            if tid in seen:
                continue
            text = tokenizer.decode([tid])
            # skip pure specials as content tokens
            if text.startswith('<|') and text.endswith('|>'):
                continue
            seen.add(tid)
            texts.append(text)
            tids.append(int(tid))
            origins.append(sentence)
            if len(texts) >= n:
                return texts, tids, origins
    raise RuntimeError(f'Need {n} content tokens, got {len(texts)}')


def build_sequence(device, dtype, content_id, role, filler_ids, text_prefix_ids=None):
    """Return (input_ids[1,T], position_of_content)."""
    prefix_len = PREFIX[role]
    if text_prefix_ids is not None:
        prefix = list(text_prefix_ids[:prefix_len])
        # pad/truncate with filler if natural prefix shorter/longer than needed
        if len(prefix) < prefix_len:
            if not filler_ids:
                raise RuntimeError('Natural prefix too short and no filler to pad')
            pad = filler_ids * ((prefix_len - len(prefix) + len(filler_ids) - 1) // len(filler_ids))
            prefix = pad[:prefix_len - len(prefix)] + prefix
        prefix = prefix[-prefix_len:] if prefix_len else []
    else:
        if prefix_len and not filler_ids:
            raise RuntimeError('filler required for non-zero prefix')
        prefix = (filler_ids * ((prefix_len + len(filler_ids) - 1) // max(len(filler_ids), 1)))[:prefix_len] if prefix_len else []
    seq = prefix + [content_id]
    position = len(prefix)
    if role == 'middle' and SUFFIX_MIDDLE:
        if not filler_ids:
            raise RuntimeError('filler required for middle suffix')
        suffix = (filler_ids * ((SUFFIX_MIDDLE + len(filler_ids) - 1) // len(filler_ids)))[:SUFFIX_MIDDLE]
        seq = seq + suffix
    ids = torch.tensor([seq], device=device, dtype=dtype)
    return ids, position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filler', choices=['special', 'text', 'none'], default='special',
                        help='special: <|endoftext|> context; text: natural sentence prefix; '
                             'none: only first role (length-1), for smoke tests')
    parser.add_argument('--special-token', default=SPECIAL_DEFAULT,
                        help='Special token string used as filler context')
    parser.add_argument('--n-sources', type=int, default=N_SOURCES)
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory (default results_position_{filler})')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(SEED)
    corpus = json.loads((ROOT / 'prompts.json').read_text())
    sentences = list(corpus['source']) + list(corpus.get('source_one_token_sentences', []))
    out_name = args.out or f'results_position_{args.filler}'
    out = ROOT / out_name
    out.mkdir(exist_ok=True)
    cache = out / 'jacobians'
    cache.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    model = from_hf(from_pretrained_to_cuda(
        AutoModelForCausalLM, 'Qwen/Qwen3.5-2B', dtype=torch.bfloat16,
        low_cpu_mem_usage=False), tokenizer)
    lens_path = hf_hub_download(
        'sida/llmdbg',
        'workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors',
        repo_type='dataset', revision=REVISION, local_files_only=True)

    texts, token_ids, origins = pick_content_tokens(
        model, tokenizer, sentences, args.n_sources)
    probe = model.encode('The', max_length=8)
    device, dtype = probe.device, probe.dtype

    filler_ids = []
    if args.filler in ('special', 'text') or args.filler == 'none':
        # always resolve special ids for padding when needed
        filler_ids = tokenizer.encode(args.special_token, add_special_tokens=False)
        if len(filler_ids) != 1:
            raise RuntimeError(
                f'--special-token must be one token id, got {filler_ids} for {args.special_token!r}')
    roles = ['first'] if args.filler == 'none' else ['first', 'middle', 'last']

    # Natural prefixes: leading tokens of each origin sentence, excluding the content token slot.
    text_prefixes = []
    if args.filler == 'text':
        for origin, tid in zip(origins, token_ids):
            ids = model.encode(origin, max_length=128)[0].tolist()
            # use tokens before first occurrence of tid when possible
            if tid in ids and ids.index(tid) > 0:
                text_prefixes.append(ids[:ids.index(tid)])
            else:
                text_prefixes.append(ids[1:] if len(ids) > 1 else [])

    cases = []
    for s, (text, tid, origin) in enumerate(zip(texts, token_ids, origins)):
        for role in roles:
            prefix_src = text_prefixes[s] if args.filler == 'text' else None
            ids, position = build_sequence(
                device, dtype, tid, role, filler_ids, text_prefix_ids=prefix_src)
            assert int(ids[0, position]) == tid, (role, s, position, tid)
            cases.append(dict(
                source=s, role=role, text=text, token_id=tid, origin=origin,
                position=position, seq_len=int(ids.shape[1]), ids=ids))

    atomic_json(out / 'manifest.json', dict(
        seed=SEED, layers=LAYERS, filler=args.filler,
        special_token=args.special_token, special_token_id=filler_ids[0] if filler_ids else None,
        prefix_lengths=PREFIX, suffix_middle=SUFFIX_MIDDLE,
        n_sources=args.n_sources, roles=roles,
        prompts=dict(
            content_tokens=texts, content_token_ids=token_ids, origins=origins,
            evaluation=corpus['evaluation'],
            cases=[{k: v for k, v in c.items() if k != 'ids'} for c in cases]),
        trained_revision=REVISION, target_layer=23, target_lag=0,
        note='Causal LM: tokens after the measured position do not affect same-position J. '
             'middle vs last therefore differ by left-context length, not by the suffix.',
        ranking='competition rank: 1 + count strictly greater; BF16 ties possible'))

    with safe_open(lens_path, framework='pt', device='cpu') as f:
        trained = {l: f.get_tensor(f'post.{l}').float().cuda() for l in LAYERS}
    queries = []
    with torch.no_grad():
        for prompt in corpus['evaluation']:
            ids = model.encode(prompt, max_length=128)
            with ActivationRecorder(model.layers, at=LAYERS) as rec:
                model.forward(ids)
                hidden = {l: rec.activations[l][0].float().detach() for l in LAYERS}
            queries.append((ids[0].cpu(), hidden))
    reference = {}
    with torch.no_grad():
        for l in LAYERS:
            for e, (ids, hidden) in enumerate(queries):
                h = hidden[l]
                target = model.unembed(h @ trained[l].T).argmax(-1)
                logits = model.unembed(h).float()
                scores = logits.gather(1, target[:, None])
                ranks = 1 + (logits > scores).sum(-1)
                reference[l, e] = (target, ranks)

    D, B = model.d_model, 8
    for case in cases:
        s, role, position, ids = case['source'], case['role'], case['position'], case['ids']
        for l in LAYERS:
            stem = f'L{l:02d}_source{s:02d}_{role}_pos{position:03d}'
            report = out / f'{stem}.csv'
            if report.exists():
                continue
            checkpoint = cache / f'{stem}.pt'
            if checkpoint.exists():
                J = torch.load(checkpoint, weights_only=True).cuda().float()
            else:
                print(f'COMPUTE {stem} token={case["text"]!r} seq_len={case["seq_len"]}',
                      flush=True)
                J = torch.empty(D, D, device='cuda')
                with ActivationRecorder(model.layers, at=[l, 23], start_graph_at=l) as rec:
                    model.forward(ids.expand(B, -1))
                    source, target = rec.activations[l], rec.activations[23]
                    cot = torch.zeros_like(target)
                    for start in range(0, D, B):
                        idx = torch.arange(min(B, D - start), device='cuda')
                        cot.zero_()
                        cot[idx, position, start + idx] = 1
                        grad, = torch.autograd.grad(
                            target, source, grad_outputs=cot, retain_graph=start + B < D)
                        J[start:start + len(idx)] = grad[:, position].float()
                torch.save(J.cpu(), checkpoint)
                del source, target, cot, grad, rec
            rows = []
            with torch.no_grad():
                for e, (token_ids_q, hidden) in enumerate(queries):
                    q, ranks_ll = reference[l, e]
                    logits = model.unembed(hidden[l] @ J.T).float()
                    scores = logits.gather(1, q[:, None])
                    ranks = 1 + (logits > scores).sum(-1)
                    for t in range(len(token_ids_q)):
                        rows.append(dict(
                            layer=l, source_prompt=s, role=role,
                            source_position=position, source_token=case['text'],
                            source_token_id=case['token_id'],
                            seq_len=case['seq_len'],
                            evaluation_prompt=e, position=t,
                            input_token=tokenizer.decode([int(token_ids_q[t])]),
                            trained_top_token=tokenizer.decode([int(q[t])]),
                            trained_top_id=int(q[t]),
                            single_rank=int(ranks[t]),
                            logit_lens_rank=int(ranks_ll[t]),
                            exclude_early=t >= 16))
            temp = report.with_suffix('.tmp')
            with temp.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            temp.replace(report)
            print(f'SAVED {stem} ({len(rows)} rows)', flush=True)
            del J
    summarize(out, roles)
    print(f'COMPLETE: {len(cases) * len(LAYERS)} Jacobians under filler={args.filler}',
          flush=True)


def summarize(out, roles):
    summaries = []
    for path in sorted(out.glob('L*.csv')):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        role = rows[0]['role']
        for early in [False, True]:
            selected = [r for r in rows if not early or int(r['position']) >= 16]
            groups = {e: [r for r in selected if int(r['evaluation_prompt']) == e]
                      for e in range(10)}
            result = dict(jacobian=path.stem, layer=int(rows[0]['layer']), role=role,
                          exclude_first16=early)
            for method, col in [('single', 'single_rank'), ('logit', 'logit_lens_rank')]:
                result[method + '_mrr'] = statistics.mean(
                    statistics.mean(1 / int(r[col]) for r in group)
                    for group in groups.values())
                result[method + '_median_rank'] = statistics.median(
                    int(r[col]) for r in selected)
                for k in [10, 100, 1000]:
                    result[f'{method}_top{k}'] = statistics.mean(
                        statistics.mean(int(r[col]) <= k for r in group)
                        for group in groups.values())
            summaries.append(result)
    atomic_json(out / 'summary.json', summaries)
    by_layer = []
    for l in LAYERS:
        for role in roles:
            for early in [False, True]:
                group = [r for r in summaries
                         if r['layer'] == l and r['role'] == role
                         and r['exclude_first16'] == early]
                if not group:
                    continue
                by_layer.append(dict(
                    layer=l, role=role, exclude_first16=early, jacobians=len(group),
                    median_single_mrr=statistics.median(r['single_mrr'] for r in group),
                    logit_mrr=group[0]['logit_mrr'],
                    single_wins_mrr=sum(r['single_mrr'] > r['logit_mrr'] for r in group),
                    median_mrr_improvement=statistics.median(
                        r['single_mrr'] - r['logit_mrr'] for r in group)))
    atomic_json(out / 'layer_summary.json', by_layer)


if __name__ == '__main__':
    main()

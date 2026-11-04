# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Next-token accuracy for matched-token first/middle/last Jacobians."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
REVISION = '980e34bc0f0460c437bb4fa0238a58b9b50158d3'
COLORS = {'first': '#2878B5', 'middle': '#6F9F3B', 'last': '#C73E3A'}
LABELS = {'first': 'First (pos 0)', 'middle': 'Middle', 'last': 'Last'}
STEM_RE = re.compile(r'^L(\d+)_source(\d+)_(first|middle|last)_pos(\d+)$')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results_position_special')
    parser.add_argument('--tag', default='special')
    args = parser.parse_args()
    results = ROOT / args.results
    out = ROOT / f'next_token_accuracy_position_{args.tag}'
    out.mkdir(exist_ok=True)
    blog = ROOT.parent / 'blog-post'

    prompts = json.loads((ROOT / 'prompts.json').read_text())['evaluation']
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    model = from_hf(from_pretrained_to_cuda(
        AutoModelForCausalLM, 'Qwen/Qwen3.5-2B', dtype=torch.bfloat16,
        low_cpu_mem_usage=False), tok)
    path = hf_hub_download(
        'sida/llmdbg',
        'workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors',
        repo_type='dataset', revision=REVISION, local_files_only=True)

    queries = []
    with torch.no_grad():
        for text in prompts:
            ids = model.encode(text, max_length=128)
            with ActivationRecorder(model.layers, at=LAYERS) as rec:
                model.forward(ids)
                queries.append((ids[0], {l: rec.activations[l][0].float() for l in LAYERS}))

    rows = []

    def evaluate(matrix, l, method, name, role):
        for e, (ids, hidden) in enumerate(queries):
            h = hidden[l][:-1]
            logits = model.unembed(h if matrix is None else h @ matrix.T)
            pred = logits.argmax(-1)
            top2 = logits.topk(2, dim=-1).indices
            top5 = logits.topk(5, dim=-1).indices
            for t in range(len(pred)):
                rows.append(dict(
                    layer=l, method=method, role=role, jacobian=name,
                    evaluation_prompt=e, position=t,
                    target_id=int(ids[t + 1]), predicted_id=int(pred[t]),
                    correct=int(pred[t] == ids[t + 1]),
                    correct_top2=int((top2[t] == ids[t + 1]).any()),
                    correct_top5=int((top5[t] == ids[t + 1]).any())))

    with torch.no_grad(), safe_open(path, framework='pt', device='cpu') as f:
        for l in LAYERS:
            evaluate(None, l, 'logit', 'identity', 'baseline')
            evaluate(f.get_tensor(f'post.{l}').float().cuda(), l, 'trained', 'trained', 'baseline')
            for checkpoint in sorted((results / 'jacobians').glob(f'L{l:02d}_*.pt')):
                m = STEM_RE.match(checkpoint.stem)
                if not m:
                    raise RuntimeError(f'Unexpected stem {checkpoint.stem}')
                role = m.group(3)
                evaluate(torch.load(checkpoint, weights_only=True).cuda().float(),
                         l, 'single', checkpoint.stem, role)
                print('EVALUATED', checkpoint.stem, flush=True)

    with (out / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for l in LAYERS:
        for method in ['single', 'trained', 'logit']:
            subset = [r for r in rows if r['layer'] == l and r['method'] == method]
            for name in sorted({r['jacobian'] for r in subset}):
                selected = [r for r in subset if r['jacobian'] == name]
                role = selected[0]['role']
                for early, label in [(False, 'all'), (True, 'ge16')]:
                    used = [r for r in selected if not early or r['position'] >= 16]
                    summary.append(dict(
                        layer=l, method=method, role=role, jacobian=name,
                        positions=label,
                        accuracy=float(np.mean([
                            np.mean([r['correct'] for r in used if r['evaluation_prompt'] == e])
                            for e in range(10)])),
                        accuracy_top2=float(np.mean([
                            np.mean([r['correct_top2'] for r in used if r['evaluation_prompt'] == e])
                            for e in range(10)])),
                        accuracy_top5=float(np.mean([
                            np.mean([r['correct_top5'] for r in used if r['evaluation_prompt'] == e])
                            for e in range(10)]))))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    plot_rows = [r for r in summary if r['positions'] == 'all']
    roles = ['first', 'middle', 'last']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    fig.subplots_adjust(left=.075, right=.98, bottom=.30, top=.78, wspace=.12)
    for ax, key, title in zip(
            axes, ['accuracy', 'accuracy_top2', 'accuracy_top5'],
            ['Top-1', 'Top-2', 'Top-5']):
        for role in roles:
            values = np.array([[r[key] * 100 for r in plot_rows
                                if r['layer'] == l and r['method'] == 'single'
                                and r['role'] == role]
                               for l in LAYERS])
            ax.plot(LAYERS, np.median(values, axis=1), 'o-', color=COLORS[role],
                    label=LABELS[role])
            ax.fill_between(LAYERS, np.quantile(values, .1, axis=1),
                            np.quantile(values, .9, axis=1), color=COLORS[role], alpha=.14)
        for method, color, label in [
                ('logit', '#D87526', 'Logit lens'),
                ('trained', '#8B4BA8', 'Trained J-Lens')]:
            y = [next(r[key] * 100 for r in plot_rows
                      if r['layer'] == l and r['method'] == method) for l in LAYERS]
            ax.plot(LAYERS, y, 's--', color=color, label=label)
        ax.set(xticks=LAYERS, xlabel='Source layer', title=title, ylim=(0, 100))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Next-token accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, .99),
               ncol=5, frameon=False, fontsize=9)
    fig.text(.53, .035,
             f'15 matched content tokens × 3 roles · filler={args.tag} · 10 held-out passages\n'
             'Shading: 10th–90th percentile across Jacobians · passages weighted equally',
             ha='center', fontsize=8)
    name = f'position-compare-{args.tag}-next-token-accuracy.png'
    fig.savefig(out / name, dpi=180, bbox_inches='tight')
    fig.savefig(blog / name, dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Print quick layer-21 comparison
    for role in roles:
        vals = [r['accuracy'] for r in plot_rows
                if r['layer'] == 21 and r['method'] == 'single' and r['role'] == role]
        print(f'L21 {role}: median top1={np.median(vals)*100:.1f}%', flush=True)
    for method in ['logit', 'trained']:
        v = next(r['accuracy'] for r in plot_rows if r['layer'] == 21 and r['method'] == method)
        print(f'L21 {method}: top1={v*100:.1f}%', flush=True)
    print('WROTE', blog / name, flush=True)


if __name__ == '__main__':
    main()

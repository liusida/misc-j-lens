# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""One figure: first / middle / last matched-token Jacobians vs logit lens."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
BLOG = ROOT.parent / 'blog-post'
LAYERS = list(range(0, 22, 3))
COLORS = {'first': '#2878B5', 'middle': '#6F9F3B', 'last': '#C73E3A'}
LABELS = {'first': 'First (pos 0)', 'middle': 'Middle', 'last': 'Last'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results_position_special')
    parser.add_argument('--tag', default='special',
                        help='Filename tag, e.g. special or text')
    args = parser.parse_args()
    out = ROOT / args.results
    rows = [r for r in json.loads((out / 'summary.json').read_text())
            if not r['exclude_first16']]
    roles = sorted({r['role'] for r in rows},
                   key=lambda x: ['first', 'middle', 'last'].index(x)
                   if x in ('first', 'middle', 'last') else 99)
    manifest = json.loads((out / 'manifest.json').read_text())

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    fig.subplots_adjust(left=.11, right=.98, bottom=.30, top=.78)
    for role in roles:
        values = np.array([[r['single_mrr'] for r in rows
                            if r['layer'] == l and r['role'] == role]
                           for l in LAYERS])
        color = COLORS.get(role, '#333333')
        ax.plot(LAYERS, np.median(values, axis=1), 'o-', color=color,
                label=LABELS.get(role, role))
        ax.fill_between(LAYERS, np.quantile(values, .1, axis=1),
                        np.quantile(values, .9, axis=1), color=color, alpha=.14)
    logit = [next(r['logit_mrr'] for r in rows if r['layer'] == l) for l in LAYERS]
    ax.plot(LAYERS, logit, 's--', color='#D87526', label='Logit lens')
    ax.set(xticks=LAYERS, xlabel='Source layer', ylabel='Mean reciprocal rank ↑',
           title='Matched-token Jacobians by absolute position', ylim=(0, 1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.55, .99), ncol=4, frameon=False, fontsize=9)
    pref = manifest.get('prefix_lengths', {})
    filler = manifest.get('filler')
    special = manifest.get('special_token')
    n = manifest.get('n_sources', '?')
    fig.text(
        .55, .035,
        f'{n} matched content tokens · filler={filler}'
        + (f' ({special})' if special else '')
        + f' · prefixes first/mid/last = {pref.get("first")}/{pref.get("middle")}/{pref.get("last")}\n'
        'Same measured token id across roles; causal same-pos J depends on left context only · '
        'shading: 10th–90th percentile',
        ha='center', fontsize=8)
    name = f'position-compare-{args.tag}-vs-logit-lens.png'
    fig.savefig(BLOG / name, dpi=180, bbox_inches='tight')
    fig.savefig(out / name, dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Compact layer table figure: median MRR by role
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    fig.subplots_adjust(left=.12, right=.98, bottom=.22, top=.82)
    x = np.arange(len(LAYERS))
    width = 0.25
    for i, role in enumerate(roles):
        med = [np.median([r['single_mrr'] for r in rows
                          if r['layer'] == l and r['role'] == role]) for l in LAYERS]
        ax.bar(x + (i - 1) * width, med, width=width, color=COLORS.get(role, '#333'),
               label=LABELS.get(role, role))
    ax.plot(x, logit, 's--', color='#D87526', label='Logit lens')
    ax.set(xticks=x, xticklabels=LAYERS, xlabel='Source layer',
           ylabel='Median MRR ↑', title='Does later absolute position help?',
           ylim=(0, 1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.55, .99), ncol=4, frameon=False, fontsize=9)
    name2 = f'position-compare-{args.tag}-bars.png'
    fig.savefig(BLOG / name2, dpi=180, bbox_inches='tight')
    fig.savefig(out / name2, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', BLOG / name, 'and', BLOG / name2)


if __name__ == '__main__':
    main()

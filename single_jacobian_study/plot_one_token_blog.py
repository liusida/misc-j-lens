# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Blog figures for one-token-prompt Jacobians (matched style to plot_blog.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
BLOG = ROOT.parent / 'blog-post'
layers = list(range(0, 22, 3))
agreement = json.loads((ROOT / 'results_one_token/summary.json').read_text())
accuracy = json.loads(
    (ROOT / 'next_token_accuracy_one_token/summary.json').read_text())
accuracy_all = [r for r in accuracy if r['positions'] == 'all']

fig, ax = plt.subplots(figsize=(9, 4))
fig.subplots_adjust(left=.11, right=.98, bottom=.27, top=.78)
rows = [r for r in agreement if not r['exclude_first16']]
values = np.array([[r['single_mrr'] for r in rows if r['layer'] == l] for l in layers])
ax.plot(layers, np.median(values, axis=1), 'o-', color='#2878B5',
        label='One-token Jacobian')
ax.fill_between(layers, np.quantile(values, .1, axis=1),
                np.quantile(values, .9, axis=1), color='#2878B5', alpha=.18)
ax.plot(layers,
        [next(r['logit_mrr'] for r in rows if r['layer'] == l) for l in layers],
        's--', color='#D87526', label='Logit lens')
ax.set(xticks=layers, xlabel='Source layer', ylabel='Mean reciprocal rank ↑',
       title='Agreement with trained J-Lens · length-1 source prompts', ylim=(0, 1))
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=.2)
fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.55, .99), ncol=2, frameon=False, fontsize=10)
fig.text(.55, .035,
         '30 independent matrices/layer · J = ∂h₂₃[0]/∂h_ℓ[0] on a single token\n'
         'Blue: median and 10th–90th percentiles · 10 held-out passages, weighted equally',
         ha='center', fontsize=8)
fig.savefig(BLOG / 'one-token-jacobian-vs-logit-lens.png', dpi=180, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig.subplots_adjust(left=.075, right=.98, bottom=.27, top=.78, wspace=.12)
for ax, key, title in zip(
        axes, ['accuracy', 'accuracy_top2', 'accuracy_top5'],
        ['Top-1', 'Top-2', 'Top-5']):
    values = np.array([[r[key] * 100 for r in accuracy_all
                        if r['layer'] == l and r['method'] == 'single']
                       for l in layers])
    ax.plot(layers, np.median(values, axis=1), 'o-', color='#2878B5',
            label='One-token Jacobian')
    ax.fill_between(layers, np.quantile(values, .1, axis=1),
                    np.quantile(values, .9, axis=1), color='#2878B5', alpha=.18)
    for method, color, label in [
            ('logit', '#D87526', 'Logit lens'),
            ('trained', '#8B4BA8', 'Trained J-Lens')]:
        y = [next(r[key] * 100 for r in accuracy_all
                  if r['layer'] == l and r['method'] == method) for l in layers]
        ax.plot(layers, y, 's--', color=color, label=label)
    ax.set(xticks=layers, xlabel='Source layer', title=title, ylim=(0, 100))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
axes[0].set_ylabel('Next-token accuracy (%)')
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.53, .99), ncol=3, frameon=False, fontsize=10)
fig.text(.53, .035,
         '30 independent length-1 matrices/layer · 10 held-out passages\n'
         'Blue: median and 10th–90th percentiles · passages weighted equally',
         ha='center', fontsize=8)
fig.savefig(BLOG / 'one-token-jacobian-next-token-accuracy.png',
            dpi=180, bbox_inches='tight')
plt.close(fig)
print('Wrote one-token blog figures.')

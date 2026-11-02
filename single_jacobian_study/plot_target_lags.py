# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "numpy"]
# ///
"""Matched current/next-position Jacobian comparison against trained readouts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'plots'
OUT.mkdir(exist_ok=True)
current=json.loads((ROOT/'results/summary.json').read_text())
layers=sorted({r['layer'] for r in current})


def render(exclude):
    fig,ax=plt.subplots(figsize=(9,4))
    fig.subplots_adjust(left=.11,right=.98,bottom=.27,top=.78)
    variants=[(current,'#2878B5','Single Jacobian')]
    for rows,color,label in variants:
        values=np.array([[r['single_mrr'] for r in rows if r['layer']==l and r['exclude_first16']==exclude] for l in layers])
        if values.shape!=(8,30):
            raise ValueError(f'Expected 30 matrices for each layer, got {values.shape}')
        ax.plot(layers,np.median(values,axis=1),'o-',color=color,label=label)
        ax.fill_between(layers,np.quantile(values,.1,axis=1),np.quantile(values,.9,axis=1),color=color,alpha=.18)
    baseline=[next(r['logit_mrr'] for r in current if r['layer']==l and r['exclude_first16']==exclude) for l in layers]
    ax.plot(layers,baseline,'s--',color='#D87526',label='Logit lens')
    ax.set(xticks=layers,ylim=(0,1),xlabel='Source layer',
           ylabel='Mean reciprocal rank ↑',
           title='Positions ≥16' if exclude else 'All positions')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.55,.99),ncol=2,frameon=False,fontsize=10)
    fig.text(.55,.035,'Reference: trained J-Lens top token · 30 matrices/layer\nBlue: median and 10th–90th percentiles · passages weighted equally',ha='center',fontsize=8)
    stem='target_lag_comparison_positions16' if exclude else 'target_lag_comparison_all'
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


render(False)
render(True)
print(f'Saved matched target-lag plots to {OUT}')

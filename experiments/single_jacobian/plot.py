# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "numpy"]
# ///
"""Plot independent Jacobian replication results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parents[1]/'artifacts/runs/single_jacobian'
OUT = ARTIFACTS/'plots'
OUT.mkdir(exist_ok=True)
rows = json.loads((ARTIFACTS/'results/summary.json').read_text())
layers = sorted({r['layer'] for r in rows})
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
blue, orange = '#2878B5', '#D87526'


def group(layer, exclude=False):
    return [r for r in rows if r['layer']==layer and r['exclude_first16']==exclude]


def save(fig, name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


fig, axes = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for exclude, color, label, offset in [(False,blue,'All positions',-.5),(True,orange,'Positions ≥16',.5)]:
    wins=[sum(r['single_mrr']>r['logit_mrr'] for r in group(l,exclude)) for l in layers]
    axes[0].bar(np.array(layers)+offset,wins,width=.95,color=color,label=label)
    for l,w in zip(layers,wins):
        axes[0].text(l+offset,w+.5,str(w),ha='center',fontsize=9)
axes[0].set(xticks=layers,ylim=(0,34),xlabel='Source layer (output, zero-based)',ylabel='Individual Jacobians beating logit lens / 30',title='How often does one Jacobian help?')
axes[0].legend(fontsize=9)
rng=np.random.default_rng(42)
for i,l in enumerate(layers):
    values=[r['single_mrr']-r['logit_mrr'] for r in group(l)]
    axes[1].scatter(l+rng.uniform(-.55,.55,len(values)),values,s=23,color=blue,alpha=.65)
    axes[1].plot([l-.65,l+.65],[np.median(values)]*2,color='black',lw=2)
axes[1].axhline(0,color='gray',ls='--',lw=1)
axes[1].set(xticks=layers,xlabel='Source layer',ylabel='MRR(single Jacobian) − MRR(logit lens)',title='Every dot is one independent Jacobian')
fig.suptitle('One same-position Jacobian, reused across unrelated passages\n30 matrices per layer · 10 held-out passages · target: final layer 23',fontsize=13)
save(fig,'overview')

fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for ax,exclude in zip(axes,[False,True]):
    values=np.array([[r['single_mrr'] for r in group(l,exclude)] for l in layers])
    ax.fill_between(layers,np.quantile(values,.1,axis=1),np.quantile(values,.9,axis=1),color=blue,alpha=.18,label='Single: 10th–90th percentile')
    ax.plot(layers,np.median(values,axis=1),'o-',color=blue,label='Single: median of 30 matrices')
    ax.plot(layers,[group(l,exclude)[0]['logit_mrr'] for l in layers],'s--',color=orange,label='Logit lens')
    ax.set(xticks=layers,xlabel='Source layer',ylabel='Mean reciprocal rank (higher is better)',title='Positions ≥16' if exclude else 'All positions',ylim=(0,1))
    ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Where does the trained J-Lens top token rank?\nReciprocal rank = 1/rank; evaluation passages weighted equally',fontsize=13)
save(fig,'rank_quality')

fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,k in zip(axes,[10,100,1000]):
    values=np.array([[r[f'single_top{k}'] for r in group(l)] for l in layers])
    ax.fill_between(layers,np.quantile(values,.1,axis=1)*100,np.quantile(values,.9,axis=1)*100,color=blue,alpha=.18)
    ax.plot(layers,np.median(values,axis=1)*100,'o-',color=blue,label='Single Jacobian: median')
    ax.plot(layers,[group(l)[0][f'logit_top{k}']*100 for l in layers],'s--',color=orange,label='Logit lens')
    ax.set(xticks=layers,xlabel='Source layer',ylabel='Evaluation positions (%)',title=f'Trained top token found in top {k:,}',ylim=(0,100))
    ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Top-token recovery · all positions\nShading: 10th–90th percentile across individual Jacobians',fontsize=13)
save(fig,'top_token_recovery')
print(f'Saved three figures as PNG and PDF under {OUT}')

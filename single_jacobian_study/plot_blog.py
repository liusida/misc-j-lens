# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Replot blog figures using all eligible evaluation positions, from saved data."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
BLOG=ROOT.parent/'blog-post'
layers=list(range(0,22,3))
agreement=json.loads((ROOT/'results/summary.json').read_text())
groups=defaultdict(lambda:defaultdict(list))
with (ROOT/'next_token_accuracy/predictions.csv').open() as f:
    for r in csv.DictReader(f):
        groups[int(r['layer']),r['method'],r['jacobian']][int(r['evaluation_prompt'])].append([int(r['correct']),int(r['correct_top2']),int(r['correct_top5'])])
accuracy=[dict(layer=l,method=m,jacobian=j,
               accuracy=float(np.mean([np.mean(v,axis=0)[0] for v in passages.values()])),
               accuracy_top2=float(np.mean([np.mean(v,axis=0)[1] for v in passages.values()])),
               accuracy_top5=float(np.mean([np.mean(v,axis=0)[2] for v in passages.values()])))
          for (l,m,j),passages in groups.items()]
(ROOT/'next_token_accuracy/summary_all.json').write_text(json.dumps(accuracy,indent=2)+'\n')

for metric,name,ylabel,title in [
    ('mrr','single-jacobian-vs-logit-lens','Mean reciprocal rank ↑','Agreement with trained J-Lens')]:
    fig,ax=plt.subplots(figsize=(9,4))
    fig.subplots_adjust(left=.11,right=.98,bottom=.27,top=.78)
    if metric=='mrr':
        rows=[r for r in agreement if not r['exclude_first16']]
        values=np.array([[r['single_mrr'] for r in rows if r['layer']==l] for l in layers])
        baselines=[('Logit lens','#D87526',[next(r['logit_mrr'] for r in rows if r['layer']==l) for l in layers])]
    else:
        values=np.array([[r['accuracy']*100 for r in accuracy if r['layer']==l and r['method']=='single'] for l in layers])
        baselines=[(label,color,[next(r['accuracy']*100 for r in accuracy if r['layer']==l and r['method']==method) for l in layers])
                   for method,label,color in [('logit','Logit lens','#D87526'),('trained','Trained J-Lens','#8B4BA8')]]
    ax.plot(layers,np.median(values,axis=1),'o-',color='#2878B5',label='Single Jacobian')
    ax.fill_between(layers,np.quantile(values,.1,axis=1),np.quantile(values,.9,axis=1),color='#2878B5',alpha=.18)
    for label,color,y in baselines:
        ax.plot(layers,y,'s--',color=color,label=label)
    ax.set(xticks=layers,xlabel='Source layer',ylabel=ylabel,title=title)
    ax.set_ylim(bottom=0)
    if metric=='mrr':ax.set_ylim(0,1)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.99),ncol=len(baselines)+1,frameon=False,fontsize=10)
    fig.text(.55,.035,'30 independent matrices/layer · 10 held-out passages\nBlue: median and 10th–90th percentiles · passages weighted equally',ha='center',fontsize=8)
    fig.savefig(BLOG/f'{name}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
fig.subplots_adjust(left=.075,right=.98,bottom=.27,top=.78,wspace=.12)
for ax,key,title in zip(axes,['accuracy','accuracy_top2','accuracy_top5'],['Top-1','Top-2','Top-5']):
    values=np.array([[r[key]*100 for r in accuracy if r['layer']==l and r['method']=='single'] for l in layers])
    ax.plot(layers,np.median(values,axis=1),'o-',color='#2878B5',label='Single Jacobian')
    ax.fill_between(layers,np.quantile(values,.1,axis=1),np.quantile(values,.9,axis=1),color='#2878B5',alpha=.18)
    for method,color,label in [('logit','#D87526','Logit lens'),('trained','#8B4BA8','Trained J-Lens')]:
        y=[next(r[key]*100 for r in accuracy if r['layer']==l and r['method']==method) for l in layers]
        ax.plot(layers,y,'s--',color=color,label=label)
    ax.set(xticks=layers,xlabel='Source layer',title=title,ylim=(0,100))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Next-token accuracy (%)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.99),ncol=3,frameon=False,fontsize=10)
fig.text(.53,.035,'30 independent matrices/layer · 10 held-out passages\nBlue: median and 10th–90th percentiles · passages weighted equally',ha='center',fontsize=8)
fig.savefig(BLOG/'single-jacobian-next-token-accuracy-all.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print('Updated both blog figures using all eligible positions.')

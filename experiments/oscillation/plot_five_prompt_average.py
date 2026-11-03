# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "matplotlib"]
# ///
"""Average response magnitudes over the five fitting prompts at common lags."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

folder=Path(__file__).resolve().parents[2]/'artifacts/runs/oscillation/training_five'
data=[torch.load(folder/f'prompt_{i}.pt',map_location='cpu',weights_only=True) for i in range(5)]
length=min(d['responses'].shape[1] for d in data)
# [prompt, direction, future lag], excluding the source's own position.
norms=np.stack([d['responses'][:,:length].norm(dim=-1).numpy() for d in data])[:,:,1:]
averaged=norms.mean(axis=0)
lag=np.arange(1,length)
random_cmap=LinearSegmentedColormap.from_list('blue_yellow',['#1546b3','#21a7c4','#f1cf22'])
token_cmap=LinearSegmentedColormap.from_list('red_pink',['#d00000','#f9b9cf'])
fig,axes=plt.subplots(1,2,figsize=(17,6),layout='constrained')
order=np.argsort(-averaged[:50].mean(axis=1))
for rank,index in reversed(list(enumerate(order))):
    for ax in axes: ax.plot(lag,averaged[index],color=random_cmap(rank/49),alpha=.4,lw=.8)
for k in reversed(range(10)):
    for ax in axes: ax.plot(lag,averaged[50+k],color=token_cmap(k/9),lw=1.2,alpha=.75,label=f'Token rank {k+1}' if k==0 else None)
curves={
    'Random-50 median':np.median(averaged[:50],axis=0),
    'Top-1 token direction':averaged[50],
    'Full-transport direction':averaged[60],
}
colors={'Random-50 median':'black','Top-1 token direction':'#d00000','Full-transport direction':'#8a2be2'}
fits={}
for name,y in curves.items():
    slope,intercept=np.polyfit(lag,np.log(y),1)
    predicted=np.exp(intercept+slope*lag)
    r2=1-np.sum((np.log(y)-np.log(predicted))**2)/np.sum((np.log(y)-np.log(y).mean())**2)
    fits[name]=dict(slope=float(slope),decay_length_tokens=float(-1/slope) if slope<0 else None,log_space_r_squared=float(r2),amplitude=float(np.exp(intercept)))
    for ax in axes:
        ax.plot(lag,y,color=colors[name],lw=2.3,label=name)
        ax.plot(lag,predicted,color=colors[name],ls='--',lw=1.5,label=f'Exp fit: R²={r2:.2f}')
for ax in axes:
    ax.set_xlabel('Future target lag from source position 17')
    ax.set_ylabel('Mean response magnitude across 5 prompts')
    ax.grid(alpha=.2)
axes[0].set_title('Linear scale')
axes[1].set_yscale('log');axes[1].set_title('Log scale: exponential decay would be straight')
axes[1].legend(fontsize=8,ncol=2)
fig.suptitle(f'First five fitting prompts: common future lags 1–{length-1}\nSolid = averaged measurements; dashed = exponential fits; source position excluded')
for extension in ['png','pdf']:fig.savefig(folder/f'five_prompt_average_decay.{extension}',dpi=180)
rows=[]
for k,t in enumerate(lag):
    row={'lag':int(t)}
    row.update({name:float(y[k]) for name,y in curves.items()})
    rows.append(row)
with (folder/'five_prompt_average_decay.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report=dict(prompts=5,common_future_positions=length-1,averaging='Arithmetic mean across prompts of per-position response norms, not norm of mean vectors. Token ranks select different tokens in each prompt. Random median taken after averaging each shared direction across prompts.',fit_method='OLS of log magnitude against lag; descriptive fit over common window',fits=fits)
(folder/'five_prompt_average_decay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

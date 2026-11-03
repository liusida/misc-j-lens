# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "matplotlib"]
# ///
"""Compare 5/10/25 prompt averages on one common future-lag window."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

folder=Path(__file__).resolve().parents[2]/'artifacts/runs/oscillation/training_five'
data=[torch.load(folder/f'prompt_{i}.pt',map_location='cpu',weights_only=True) for i in range(25)]
length=min(d['responses'].shape[1] for d in data)
values=np.stack([d['responses'][:,:length].norm(dim=-1).numpy() for d in data])[:,:,1:]
x=np.arange(1,length)
fig,axes=plt.subplots(1,3,figsize=(18,5.5),layout='constrained')
fitfig,fitaxes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
report={}
colors={'Random-50 median':'black','Top-1 token':'#d00000','Full transport':'#8a2be2'}
for count,style,alpha in [(5,':',.6),(10,'--',.8),(25,'-',1)]:
    mean=values[:count].mean(axis=0)
    curves={'Random-50 median':np.median(mean[:50],axis=0),'Top-1 token':mean[50],'Full transport':mean[60]}
    report[count]={}
    for ax,(name,y) in zip(axes,curves.items()):
        ax.semilogy(x,y,color=colors[name],ls=style,alpha=alpha,label=f'{count} prompts',lw=2)
        ax.set(title=name,xlabel='Future lag',ylabel='Average response magnitude (log)')
        ax.grid(alpha=.2);ax.legend()
        slope,intercept=np.polyfit(x,np.log(y),1)
        predicted=np.exp(intercept+slope*x)
        r2=1-np.sum((np.log(y)-np.log(predicted))**2)/np.sum((np.log(y)-np.log(y).mean())**2)
        report[count][name]=dict(decay_length=float(-1/slope) if slope<0 else None,log_r_squared=float(r2))
        if count==25:
            for fax in fitaxes:
                fax.plot(x,y,color=colors[name],label=name,lw=2)
                fax.plot(x,predicted,color=colors[name],ls='--',label=f'Exp fit R²={r2:.2f}')
for fax in fitaxes:
    fax.set(xlabel='Future lag',ylabel='Average response magnitude');fax.grid(alpha=.2)
fitaxes[0].set_title('25 prompts: linear scale')
fitaxes[1].set_yscale('log');fitaxes[1].set_title('25 prompts: log scale');fitaxes[1].legend(fontsize=8)
for f,title in [(fig,'5-, 10-, and 25-prompt averages'),(fitfig,'25-prompt average and exponential fits')]:
    f.suptitle(f'{title}; common future lags 1–{length-1}\nMean of response norms, source L12 position 17; lag 0 excluded')
for name,f in [('average_convergence_25',fig),('average_decay_25',fitfig)]:
    for ext in ['png','pdf']:f.savefig(folder/f'{name}.{ext}',dpi=180)
for ax in axes:
    ax.set_yscale('linear')
    ax.set_ylabel('Average response magnitude')
    ax.set_ylim(bottom=0)
fig.suptitle(f'5-, 10-, and 25-prompt averages: linear scale; common future lags 1–{length-1}\nMean of response norms, source L12 position 17; lag 0 excluded')
for ext in ['png','pdf']:fig.savefig(folder/f'average_convergence_25_linear.{ext}',dpi=180)
# Separate standalone linear-scale view of the 25-prompt fit.
fitaxes[1].set_yscale('linear')
fitaxes[1].set_title('25 prompts: linear scale')
fitaxes[1].set_ylim(bottom=0)
fitfig.delaxes(fitaxes[0])
fitaxes[1].set_position([0.08,0.14,0.86,0.72])
fitfig.set_layout_engine(None)
fitfig.set_size_inches(11,6)
fitfig.suptitle(f'25-prompt average and exponential fits: linear scale\nCommon future lags 1–{length-1}; source position excluded')
for ext in ['png','pdf']:fitfig.savefig(folder/f'average_decay_25_linear.{ext}',dpi=180)
result=dict(common_future_lags=length-1,fits=report,max_epsilon_error=max(max(d['epsilon_errors']) for d in data))
(folder/'average_decay_25.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

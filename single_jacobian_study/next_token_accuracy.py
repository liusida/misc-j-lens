# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Actual supplied next-token top-1 accuracy; independent same-position Js."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'anthropic-jacobian-lens'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder


def main():
    out=ROOT/'next_token_accuracy'
    out.mkdir(exist_ok=True)
    layers=list(range(0,22,3))
    prompts=json.loads((ROOT/'prompts.json').read_text())['evaluation']
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    model=from_hf(from_pretrained_to_cuda(AutoModelForCausalLM,'Qwen/Qwen3.5-2B',dtype=torch.bfloat16,low_cpu_mem_usage=False),tok)
    path=hf_hub_download('sida/llmdbg','workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors',repo_type='dataset',revision='980e34bc0f0460c437bb4fa0238a58b9b50158d3',local_files_only=True)
    queries=[]
    with torch.no_grad():
        for text in prompts:
            ids=model.encode(text,max_length=128)
            with ActivationRecorder(model.layers,at=layers) as rec:
                model.forward(ids)
                queries.append((ids[0],{l:rec.activations[l][0].float() for l in layers}))
    rows=[]
    def evaluate(matrix,l,method,name):
        for e,(ids,hidden) in enumerate(queries):
            h=hidden[l][:-1]
            logits=model.unembed(h if matrix is None else h@matrix.T)
            pred=logits.argmax(-1)
            top2=logits.topk(2,dim=-1).indices
            top5=logits.topk(5,dim=-1).indices
            for t in range(len(pred)):
                rows.append(dict(layer=l,method=method,jacobian=name,evaluation_prompt=e,position=t,
                    target_id=int(ids[t+1]),predicted_id=int(pred[t]),correct=int(pred[t]==ids[t+1]),
                    correct_top2=int((top2[t]==ids[t+1]).any()),
                    correct_top5=int((top5[t]==ids[t+1]).any())))
    with torch.no_grad(),safe_open(path,framework='pt',device='cpu') as f:
        for l in layers:
            evaluate(None,l,'logit','identity')
            evaluate(f.get_tensor(f'post.{l}').float().cuda(),l,'trained','trained')
            for checkpoint in sorted((ROOT/'results/jacobians').glob(f'L{l:02d}_*.pt')):
                evaluate(torch.load(checkpoint,weights_only=True).cuda().float(),l,'single',checkpoint.stem)
                print('EVALUATED',checkpoint.stem,flush=True)
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for l in layers:
        for method in ['single','trained','logit']:
            subset=[r for r in rows if r['layer']==l and r['method']==method]
            for name in sorted({r['jacobian'] for r in subset}):
                selected=[r for r in subset if r['jacobian']==name and r['position']>=16]
                accuracy=np.mean([np.mean([r['correct'] for r in selected if r['evaluation_prompt']==e]) for e in range(10)])
                summary.append(dict(layer=l,method=method,jacobian=name,accuracy=float(accuracy)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,4))
    fig.subplots_adjust(left=.11,right=.98,bottom=.27,top=.78)
    values=np.array([[r['accuracy'] for r in summary if r['layer']==l and r['method']=='single'] for l in layers])*100
    ax.plot(layers,np.median(values,axis=1),'o-',color='#2878B5',label='Single Jacobian')
    ax.fill_between(layers,np.quantile(values,.1,axis=1),np.quantile(values,.9,axis=1),color='#2878B5',alpha=.18)
    for method,color,label in [('logit','#D87526','Logit lens'),('trained','#8B4BA8','Trained J-Lens')]:
        y=[next(r['accuracy'] for r in summary if r['layer']==l and r['method']==method)*100 for l in layers]
        ax.plot(layers,y,'s--',color=color,label=label)
    ax.set(xticks=layers,xlabel='Source layer',ylabel='Next-token top-1 accuracy (%)',title='Positions ≥16 · actual next token in supplied passage')
    ax.set_ylim(bottom=0)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.99),ncol=3,frameon=False,fontsize=10)
    fig.text(.55,.035,'30 independent matrices/layer · 10 held-out passages\nBlue: median and 10th–90th percentiles · passages weighted equally',ha='center',fontsize=8)
    fig.savefig(out/'accuracy.png',dpi=180,bbox_inches='tight')
    import shutil
    shutil.copyfile(out/'accuracy.png',ROOT.parent/'blog-post/single-jacobian-next-token-accuracy-all.png')
    import subprocess
    subprocess.run(['uv','run',str(ROOT/'plot_blog.py')],check=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()

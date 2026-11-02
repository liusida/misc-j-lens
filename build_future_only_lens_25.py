# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors", "matplotlib"]
# ///
"""Future-only lens from matched normal minus self-only per-prompt matrices.

Retains the normal estimator's 1/N_source normalization, including a zero
future contribution for the last valid source. No reweighting by pair counts.
The subtraction inherits the BF16 differences recorded in self validation.
"""
import csv
import json
import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModelForCausalLM
from gb10_load_llm import from_pretrained_to_cuda
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'anthropic-jacobian-lens'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder
from test_swimming_patch import PROMPT


def cos(a,b):return float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out=ROOT/'j_lens_output/future_only_lens_25';out.mkdir(parents=True,exist_ok=True)
    paired=ROOT/'j_lens_output/paired_self_normal_25'
    selfdir=ROOT/'j_lens_output/exact_self_lens_25'
    config=json.loads((paired/'config.json').read_text())
    config.update(method='future-only = matched normal minus matched self-only',
        formula='mean_prompt mean_valid_source sum_valid_target_gt_source dh23[target]/dh12[source]',
        normalization='Same source-position count as normal and self-only; no division by number of future pairs',
        numerical_note='Subtraction inherits BF16 rounding differences in validated self-only backward')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'prompts.json').write_text((paired/'prompts.json').read_text())
    filename='workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors'
    path=hf_hub_download('sida/llmdbg',filename,repo_type='dataset',revision=config['reference_revision'],local_files_only=True)
    with safe_open(path,framework='pt',device='cpu') as f: reference=f.get_tensor('post.12').cuda().float()
    totals={label:torch.zeros_like(reference) for label in ['self','normal','future']}
    previous={};metrics=[]
    for p in range(1,26):
        self_J=torch.load(selfdir/f'prompt_{p:02d}_J.pt',map_location='cuda',weights_only=True)
        normal_J=torch.load(paired/f'normal_prompt_{p:02d}_J.pt',map_location='cuda',weights_only=True)
        future_J=normal_J-self_J
        torch.save(future_J.cpu(),out/f'prompt_{p:02d}_J.pt')
        row=dict(n_prompts=p)
        for name,J in [('self',self_J),('normal',normal_J),('future',future_J)]:
            totals[name]+=J
            mean=totals[name]/p
            row[f'{name}_cosine_to_v2']=cos(mean,reference)
            row[f'{name}_lim']=None if name not in previous else float((mean-previous[name]).norm()/previous[name].norm())
            previous[name]=mean.clone()
        row['future_normal_cosine']=cos(previous['future'],previous['normal'])
        row['future_self_cosine']=cos(previous['future'],previous['self'])
        row['self_normal_cosine']=cos(previous['self'],previous['normal'])
        metrics.append(row)
    future=previous['future']
    error=float((future-(previous['normal']-previous['self'])).abs().max())
    assert error<1e-5
    torch.save(dict(J=future.cpu(),n_prompts=25,source_layer=12,target_layer=23,config=config,
                    reconstruction_max_error=error),out/'lens.pt')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    fig,axes=plt.subplots(1,3,figsize=(17,4.8),layout='constrained')
    x=list(range(1,26))
    for label,color in [('self','#2166ac'),('normal','#b2182b'),('future','#762a83')]:
        axes[0].plot(x,[r[f'{label}_cosine_to_v2'] for r in metrics],'o-',ms=3,color=color,label=label)
        axes[1].plot(x,[r[f'{label}_lim'] for r in metrics],'o-',ms=3,color=color,label=label)
    for key,label in [('future_normal_cosine','future vs normal'),('future_self_cosine','future vs self'),('self_normal_cosine','self vs normal')]:
        axes[2].plot(x,[r[key] for r in metrics],'o-',ms=3,label=label)
    axes[0].set_title('Running fits vs. v2 reference');axes[0].set_ylabel('Matrix cosine similarity');axes[0].set_ylim(0,1)
    axes[1].set_title('Update to running mean after each prompt');axes[1].set_ylabel('lim = ||mean[n] − mean[n−1]||F / ||mean[n−1]||F')
    axes[2].set_title('Comparisons between lenses');axes[2].set_ylabel('Matrix cosine similarity');axes[2].set_ylim(-1,1)
    for ax in axes:ax.set_xlabel('Completed prompts');ax.set_xticks([1,5,10,15,20,25]);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Matched 25-prompt L12 → L23 lenses: self, normal, and future-only\nFuture = normal − self; identical source normalization')
    fig.savefig(out/'comparison.png',dpi=200);plt.close(fig)

    tokenizer=AutoTokenizer.from_pretrained(config['model'])
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,config['model'],dtype=torch.bfloat16,attn_implementation='eager')
    model=from_hf(hf,tokenizer)
    with torch.no_grad():
        ids=tokenizer(PROMPT,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
        with ActivationRecorder(model.layers,at=[12]) as rec:model.forward(ids);hidden=rec.activations[12][0].float()
        rows=[]
        for t,h in enumerate(hidden):
            row=dict(position=t,token=tokenizer.decode([int(ids[0,t])]))
            for label,J in [*previous.items(),('v2',reference)]:
                logits=model.unembed((J@h)[None]).float()[0]
                values,indices=logits.topk(10)
                row[label]=[dict(token=tokenizer.decode([int(i)]),logit=float(v)) for v,i in zip(values,indices)]
            rows.append(row)
        (out/'sport_readouts.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    print('FINAL',json.dumps(metrics[-1]),'reconstruction_max_error',error)
    for row in rows:
        if row['position'] in [16,19,25]:
            print('Position',row['position'],repr(row['token']))
            for label in ['self','normal','future','v2']:print(label,[r['token'] for r in row[label][:6]])
    print('Saved',out)


if __name__=='__main__':main()

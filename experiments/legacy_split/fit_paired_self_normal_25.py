# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors", "matplotlib"]
# ///
"""Matched exact self-only and ordinary Jacobian fits, reusing checkpoints.

The normal fit calls Anthropic jacobian_for_prompt unmodified. The self fit
uses validated causal diagonal-block derivatives. Both average positions then
prompts identically. lim = relative Frobenius change between consecutive
running-average matrices. Angular changes are also saved separately.
"""
import csv
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from gb10_load_llm import from_pretrained_to_cuda
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from jlens_study.fitting.exact_self_gradients import same_position_gradients

sys.path.insert(0,str(ROOT/'references/upstream/anthropic-jacobian-lens'))
sys.path.insert(0,str(ROOT/'experiments/token_patching'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder
from jlens.fitting import jacobian_for_prompt,valid_position_mask
from test_swimming_patch import PROMPT


def cosine(a,b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))


def exact_self_prompt(model,ids,B=64):
    D=model.d_model
    valid=valid_position_mask(ids.shape[1],skip_first=4).nonzero(as_tuple=True)[0].cuda()
    matrix=torch.empty(D,D,device='cuda')
    with same_position_gradients(),ActivationRecorder(model.layers,at=[12,23],start_graph_at=12) as rec:
        model.forward(ids.expand(B,-1))
        source,target=rec.activations[12],rec.activations[23]
        cotangent=torch.zeros_like(target);batch=torch.arange(B,device='cuda')
        for first in range(0,D,B):
            n=min(B,D-first);cotangent.zero_()
            cotangent[batch[:n,None],valid[None],first+batch[:n,None]]=1
            grad,=torch.autograd.grad(target,source,cotangent,retain_graph=first+B<D)
            matrix[first:first+n]=grad[:n,valid].float().mean(1)
    return matrix


def plot(metrics,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[r['n_prompts'] for r in metrics]
    fig,axes=plt.subplots(1,3,figsize=(17,4.8),layout='constrained')
    for label,color in [('self','#2166ac'),('normal','#b2182b')]:
        axes[0].plot(x,[r[f'{label}_cosine_to_v2'] for r in metrics],'o-',ms=3,color=color,label=label)
        axes[1].plot(x,[r[f'{label}_lim'] for r in metrics],'o-',ms=3,color=color,label=label)
    axes[0].set_ylabel('Matrix cosine to cached v2 post.12');axes[0].set_title('Running fits vs. v2 reference')
    axes[0].set_ylim(0,1)
    axes[2].plot(x,[r['self_normal_cosine'] for r in metrics],'o-',ms=3,color='#1b7837')
    axes[2].set_ylabel('Matrix cosine: running self vs. running normal');axes[2].set_title('Self-only vs. normal fit');axes[2].set_ylim(0,1)
    axes[1].set_ylabel('lim = ||mean[n] − mean[n−1]||F / ||mean[n−1]||F')
    axes[1].set_title('Update to running fitted matrix after each prompt')
    for ax in axes:
        ax.set_xlabel('Completed prompts');ax.set_xticks([1,5,10,15,20,25]);ax.grid(alpha=.25)
    axes[0].legend();axes[1].legend()
    fig.suptitle('Matched 25-prompt L12 → L23 Jacobian lenses\nSame corpus, mask, precision and averaging; only future-position derivatives differ')
    fig.savefig(out/'comparison.png',dpi=200);plt.close(fig)


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out=ROOT/'artifacts/runs/j/legacy/paired_self_normal_25';out.mkdir(parents=True,exist_ok=True)
    selfdir=ROOT/'artifacts/runs/j/legacy/exact_self_lens_25'
    prompts=json.loads((selfdir/'prompts.json').read_text())
    assert len(prompts)==25
    checks=json.loads((selfdir/'validation.json').read_text())
    (out/'prompts.json').write_text(json.dumps(prompts,indent=2,ensure_ascii=False))
    config=dict(model='Qwen/Qwen3.5-2B',source_layer=12,target_layer=23,skip_first=4,max_seq_len=128,
        dim_batch=64,dtype='bfloat16',accumulator_dtype='float32',n_prompts=25,
        lim_definition='norm(running_mean_n - running_mean_previous) / norm(running_mean_previous), Frobenius norm',
        lim_note='Relative change to the running fitted matrix. Angular change = 1 - cosine of successive running means is saved separately.',
        reference_repo='sida/llmdbg',reference_revision='980e34bc0f0460c437bb4fa0238a58b9b50158d3',self_validation=checks)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    tokenizer=AutoTokenizer.from_pretrained(config['model'])
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,config['model'],dtype=torch.bfloat16,attn_implementation='eager')
    model=from_hf(hf,tokenizer)
    path=hf_hub_download('sida/llmdbg','workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors',repo_type='dataset',revision=config['reference_revision'],local_files_only=True)
    with safe_open(path,framework='pt',device='cpu') as f: reference=f.get_tensor('post.12').cuda().float()
    sum_self=torch.zeros_like(reference);sum_normal=torch.zeros_like(reference)
    prev_mean_self=prev_mean_normal=None;metrics=[]
    for p,prompt in enumerate(prompts):
        started=time.monotonic()
        selfpath=selfdir/f'prompt_{p+1:02d}_J.pt';normalpath=out/f'normal_prompt_{p+1:02d}_J.pt'
        if selfpath.exists():
            Js=torch.load(selfpath,map_location='cuda',weights_only=True)
            print(f'Prompt {p+1}/25 self: reused exact checkpoint',flush=True)
        else:
            print(f'Prompt {p+1}/25 self: computing',flush=True)
            Js=exact_self_prompt(model,model.encode(prompt,max_length=128))
            torch.save(Js.cpu(),selfpath)
        if normalpath.exists():
            Jn=torch.load(normalpath,map_location='cuda',weights_only=True)
        else:
            print(f'Prompt {p+1}/25 normal: official unmodified jacobian_for_prompt',flush=True)
            result,_,_=jacobian_for_prompt(model,prompt,[12],target_layer=23,dim_batch=64,max_seq_len=128,skip_first=4)
            Jn=result[12].cuda();torch.save(result[12],normalpath)
        sum_self+=Js;sum_normal+=Jn
        mean_self=sum_self/(p+1);mean_normal=sum_normal/(p+1)
        row=dict(n_prompts=p+1,self_cosine_to_v2=cosine(mean_self,reference),normal_cosine_to_v2=cosine(mean_normal,reference),
            self_normal_cosine=cosine(mean_self,mean_normal),relative_self_normal_difference=float((mean_self-mean_normal).norm()/mean_normal.norm()),
            self_lim=None if prev_mean_self is None else float((mean_self-prev_mean_self).norm()/prev_mean_self.norm()),
            normal_lim=None if prev_mean_normal is None else float((mean_normal-prev_mean_normal).norm()/prev_mean_normal.norm()),
            self_angular_change=None if prev_mean_self is None else 1-cosine(mean_self,prev_mean_self),
            normal_angular_change=None if prev_mean_normal is None else 1-cosine(mean_normal,prev_mean_normal),
            elapsed_seconds=time.monotonic()-started)
        prev_mean_self=mean_self.clone();prev_mean_normal=mean_normal.clone();metrics.append(row)
        for label,J in [('self',mean_self),('normal',mean_normal)]:
            torch.save(dict(J=J.cpu(),n_prompts=p+1,source_layer=12,target_layer=23,config=config,method=label),out/f'{label}_lens.pt')
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        with (out/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(metrics)
        plot(metrics,out)
        print('COMPLETED',json.dumps(row),flush=True)
    # Decode original conversation with the final matched lenses and v2.
    with torch.no_grad():
        ids=tokenizer(PROMPT,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
        with ActivationRecorder(model.layers,at=[12]) as rec:model.forward(ids);hidden=rec.activations[12][0].float()
        rows=[]
        for t,h in enumerate(hidden):
            row=dict(position=t,token=tokenizer.decode([int(ids[0,t])]))
            for label,J in [('self',mean_self),('normal',mean_normal),('v2',reference)]:
                values,indices=model.unembed((J@h)[None]).float()[0].topk(10)
                row[label]=[dict(token=tokenizer.decode([int(i)]),logit=float(v)) for v,i in zip(values,indices)]
            rows.append(row)
        (out/'sport_readouts.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    print('DONE',out,flush=True)


if __name__=='__main__':main()

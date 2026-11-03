# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors", "matplotlib"]
# ///
"""Exact lag>=3 Jacobian average from 25 matched prompts, no random probes.

Let S_w sum the within-segment Jacobians over all w offsets of a width-w
partition. A lag-d pair has multiplicity max(w-d,0). Therefore S_3-S_2
is precisely the lag-0/1/2 sum. Far = normal - (S_3-S_2).
Every output coordinate and valid source position is computed. Uses validated
causal block-diagonal gradient rule; equivalent recurrent arithmetic has BF16
roundoff, as in the preceding exact self-only fit. Checkpoints are resumable.
"""
import argparse
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
from jlens_study.fitting.exact_block_gradients import block_position_gradients

sys.path.insert(0,str(ROOT/'references/upstream/anthropic-jacobian-lens'))
sys.path.insert(0,str(ROOT/'experiments/token_patching'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder
from jlens.fitting import valid_position_mask
from test_swimming_patch import PROMPT

PARTITIONS=[(3,0,1),(3,1,1),(3,2,1),(2,0,-1),(2,1,-1)]


def cosine(a,b):return float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))


def validate(model,ids):
    T=ids.shape[1];valid=valid_position_mask(T,skip_first=4).nonzero(as_tuple=True)[0].cuda()
    positions=[4,17,T-4]
    direction=torch.sin(torch.arange(model.d_model,device='cuda').float()/17).to(model._embed_tokens.weight.dtype)
    expected=[]
    with ActivationRecorder(model.layers,at=[12,23],start_graph_at=12) as rec:
        model.forward(ids);source,target=rec.activations[12],rec.activations[23]
        primal=target.detach().clone()
        for i,t in enumerate(positions):
            cot=torch.zeros_like(target);cot[0,t:t+3]=direction
            grad,=torch.autograd.grad(target,source,cot,retain_graph=i<len(positions)-1)
            expected.append(grad[0,t].detach().float())
    actual=torch.zeros_like(rec.activations[12][0],dtype=torch.float32)
    forward_errors=[];offblock_errors=[]
    for width,offset,sign in PARTITIONS:
        with block_position_gradients(width,offset),ActivationRecorder(model.layers,at=[12,23],start_graph_at=12) as rec:
            model.forward(ids);source,target=rec.activations[12],rec.activations[23]
            forward_errors.append(float((target.detach()-primal).abs().max()))
            cot=torch.zeros_like(target);cot[:,valid]=direction
            grad,=torch.autograd.grad(target,source,cot,retain_graph=True)
            actual+=sign*grad[0].detach().float()
            cot.zero_();cot[0,17]=direction
            grad,=torch.autograd.grad(target,source,cot)
            group=(torch.arange(T,device='cuda')+offset)//width
            offblock_errors.append(float(grad[:,group!=group[17]].abs().max()))
    checks=[dict(position=t,relative_error=float((actual[t]-e).norm()/e.norm()),cosine=cosine(actual[t],e)) for t,e in zip(positions,expected)]
    result=dict(forward_max_error=max(forward_errors),offblock_gradient_max=max(offblock_errors),near_three_position_checks=checks)
    print('VALIDATION',json.dumps(result),flush=True)
    if max(forward_errors)!=0 or max(offblock_errors)!=0 or any(r['relative_error']>.06 or r['cosine']<.998 for r in checks):
        raise RuntimeError('Near-band validation failed; refusing fit')
    return result


def fit_partition(model,ids,width,offset,B,history_cache=None):
    D=model.d_model
    valid=valid_position_mask(ids.shape[1],skip_first=4).nonzero(as_tuple=True)[0].cuda()
    result=torch.empty(D,D,device='cuda')
    with block_position_gradients(width,offset,history_cache=history_cache),ActivationRecorder(model.layers,at=[12,23],start_graph_at=12) as rec:
        model.forward(ids.expand(B,-1));source,target=rec.activations[12],rec.activations[23]
        cot=torch.zeros_like(target);batch=torch.arange(B,device='cuda')
        for first in range(0,D,B):
            n=min(B,D-first);cot.zero_()
            cot[batch[:n,None],valid[None],first+batch[:n,None]]=1
            grad,=torch.autograd.grad(target,source,cot,retain_graph=first+B<D)
            result[first:first+n]=grad[:n,valid].float().mean(1)
            if first%(B*32)==0:print(f'  width={width} offset={offset} rows {first}/{D}',flush=True)
    return result


def plot(metrics,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[r['n_prompts'] for r in metrics]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for name,color in [('self','#2166ac'),('normal','#b2182b'),('future','#762a83'),('far','#008837')]:
        axes[0].plot(x,[r[f'{name}_cosine_to_v2'] for r in metrics],'o-',ms=3,color=color,label=name if name!='far' else 'future lag ≥ 3')
        axes[1].plot(x,[r[f'{name}_lim'] for r in metrics],'o-',ms=3,color=color,label=name if name!='far' else 'future lag ≥ 3')
    axes[0].set_ylabel('Matrix cosine to v2 post.12');axes[0].set_ylim(0,1)
    axes[1].set_ylabel('lim = ||mean[n] − mean[n−1]||F / ||mean[n−1]||F')
    for ax in axes:ax.set_xlabel('Completed prompts');ax.set_xticks([1,5,10,15,20,25]);ax.grid(alpha=.25);ax.legend()
    fig.suptitle('L12 → L23: exclude self and the next two targets\nSame 25 prompts and source normalization; far = normal − lags 0, 1, 2')
    fig.savefig(out/'comparison.png',dpi=200);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--validate-fp32',action='store_true',help='Validate derivative formulas in fp32, then exit')
    parser.add_argument('--dim-batch',type=int,default=16)
    parser.add_argument('--prompt-limit',type=int,default=25)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out=ROOT/'artifacts/runs/j/legacy/far_future_lens_25';out.mkdir(parents=True,exist_ok=True)
    paired=ROOT/'artifacts/runs/j/legacy/paired_self_normal_25'
    prompts=json.loads((paired/'prompts.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,'Qwen/Qwen3.5-2B',dtype=torch.bfloat16,attn_implementation='eager')
    if args.validate_fp32:hf.float()
    model=from_hf(hf,tokenizer)
    check=validate(model,model.encode(prompts[0],max_length=128))
    (out/('validation_fp32.json' if args.validate_fp32 else 'validation.json')).write_text(json.dumps(check,indent=2)+'\n')
    if args.validate_fp32:
        if any(r['relative_error']>1e-4 for r in check['near_three_position_checks']):raise RuntimeError('FP32 validation failed')
        return
    if args.validate_only:return
    config=json.loads((paired/'config.json').read_text())
    config.update(method='exact far-future lag>=3',minimum_target_lag=3,partitions=PARTITIONS,block_validation=check,
        construction='normal minus (sum of width-3 segment diagonals over 3 offsets minus width-2 over 2 offsets)',
        normalization='Unchanged source count, including zero far contribution at last three valid sources',dim_batch=args.dim_batch)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'prompts.json').write_text(json.dumps(prompts,indent=2,ensure_ascii=False))
    path=hf_hub_download('sida/llmdbg','workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors',repo_type='dataset',revision=config['reference_revision'],local_files_only=True)
    with safe_open(path,framework='pt',device='cpu') as f: reference=f.get_tensor('post.12').cuda().float()
    old=json.loads((ROOT/'artifacts/runs/j/legacy/future_only_lens_25/metrics.json').read_text())
    sums=torch.zeros_like(reference);previous=None;metrics=[]
    for p,prompt in enumerate(prompts[:args.prompt_limit],1):
        start=time.monotonic();near=torch.zeros_like(reference);history_cache={}
        ids=model.encode(prompt,max_length=128)
        for width,offset,sign in PARTITIONS:
            cache=out/f'prompt_{p:02d}_width_{width}_offset_{offset}.pt'
            if cache.exists():J=torch.load(cache,map_location='cuda',weights_only=True)
            else:
                print(f'Prompt {p}/25 width={width} offset={offset}',flush=True)
                J=fit_partition(model,ids,width,offset,args.dim_batch,history_cache=history_cache)
                torch.save(J.cpu(),cache)
            near+=sign*J
        normal=torch.load(paired/f'normal_prompt_{p:02d}_J.pt',map_location='cuda',weights_only=True)
        far=normal-near;torch.save(far.cpu(),out/f'prompt_{p:02d}_J.pt')
        sums+=far;mean=sums/p
        row=dict(old[p-1]);row.update(far_cosine_to_v2=cosine(mean,reference),far_lim=None if previous is None else float((mean-previous).norm()/previous.norm()),seconds=time.monotonic()-start)
        previous=mean.clone();metrics.append(row)
        torch.save(dict(J=mean.cpu(),n_prompts=p,source_layer=12,target_layer=23,config=config),out/'lens.pt')
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        with (out/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(metrics)
        plot(metrics,out)
        print('COMPLETED',json.dumps(row),flush=True)
    with torch.no_grad():
        ids=tokenizer(PROMPT,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
        with ActivationRecorder(model.layers,at=[12]) as rec:model.forward(ids);hidden=rec.activations[12][0].float()
        rows=[]
        for t,h in enumerate(hidden):
            logits=model.unembed((mean@h)[None]).float()[0];values,indices=logits.topk(10)
            rows.append(dict(position=t,token=tokenizer.decode([int(ids[0,t])]),far=[dict(token=tokenizer.decode([int(i)]),logit=float(v)) for v,i in zip(values,indices)]))
        (out/'sport_readouts.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    print('DONE',out,flush=True)


if __name__=='__main__':main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors", "matplotlib"]
# ///
"""Exact self-position L12 Jacobian average, 25 fixed prompts, no sampling.

Computes every output coordinate and every valid source position. Uses causal
diagonal-block chain rule, checked against ordinary isolated-target autodiff.
Forward values are unchanged. Recurrence equivalence has floating-point error.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from gb10_load_llm import from_pretrained_to_cuda
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from exact_self_gradients import same_position_gradients

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'anthropic-jacobian-lens'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder
from test_swimming_patch import PROMPT


def validate(model,ids):
    # Ordinary full Jacobian-vector products at three source/target positions.
    positions=[4,17,ids.shape[1]-2]
    direction=torch.sin(torch.arange(model.d_model,device='cuda').float()/17).to(torch.bfloat16)
    ordinary=[]
    with ActivationRecorder(model.layers,at=[12,23],start_graph_at=12) as rec:
        model.forward(ids)
        source,target=rec.activations[12],rec.activations[23]
        final_reference=target.detach().clone()
        for i,t in enumerate(positions):
            cotangent=torch.zeros_like(target);cotangent[0,t]=direction
            g,=torch.autograd.grad(target,source,cotangent,retain_graph=i<len(positions)-1)
            ordinary.append(g[0,t].detach().float())
    with same_position_gradients(),ActivationRecorder(model.layers,at=[12,23],start_graph_at=12) as rec:
        model.forward(ids)
        source,target=rec.activations[12],rec.activations[23]
        forward_error=float((target-final_reference).abs().max())
        cotangent=direction[None,None].expand_as(target)
        grad,=torch.autograd.grad(target,source,cotangent,retain_graph=True)
        checks=[]
        for t,expected in zip(positions,ordinary):
            actual=grad[0,t].float()
            error=float((actual-expected).norm()/expected.norm())
            cosine=float(torch.nn.functional.cosine_similarity(actual,expected,dim=0))
            checks.append(dict(position=t,relative_error=error,cosine=cosine))
        # Ensure a single-target cotangent has zero gradients at other positions.
        cotangent=torch.zeros_like(target);cotangent[0,17]=direction
        g,=torch.autograd.grad(target,source,cotangent)
        outside=g.clone();outside[:,17]=0
        off_position_max=float(outside.abs().max())
    result=dict(forward_max_error=forward_error,off_position_gradient_max=off_position_max,diagonal_checks=checks)
    print('Validation:',json.dumps(result),flush=True)
    if forward_error!=0 or off_position_max!=0 or any(c['relative_error']>.04 or c['cosine']<.999 for c in checks):
        raise RuntimeError('Exact diagonal validation failed; refusing fit')
    return result


def plot_metrics(metrics,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    x=[r['n_prompts'] for r in metrics]
    axes[0].plot(x,[r['cosine_to_v2'] for r in metrics],'o-',color='#2166ac')
    axes[0].set_ylabel('Matrix cosine similarity to v2 post.12');axes[0].set_ylim(-1,1)
    axes[1].plot(x,[r['relative_update'] for r in metrics],'o-',color='#b2182b')
    axes[1].set_ylabel('Relative change from previous running mean')
    for ax in axes:
        ax.set_xlabel('Completed training prompts');ax.set_xticks([1,5,10,15,20,25]);ax.grid(alpha=.25)
    fig.suptitle('Exact same-position Jacobian lens, L12 → L23\nReference: cached v2 ordinary J-Lens (includes future targets)')
    fig.savefig(out/'convergence.png',dpi=200);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--dim-batch',type=int,default=8)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out=ROOT/'j_lens_output/exact_self_lens_25';out.mkdir(parents=True,exist_ok=True)
    prompts=json.loads((out/'prompts.json').read_text())
    assert len(prompts)==25
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,'Qwen/Qwen3.5-2B',dtype=torch.bfloat16,attn_implementation='eager')
    model=from_hf(hf,tokenizer)
    checks=validate(model,model.encode(prompts[0],max_length=128))
    (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    if args.validate_only:return
    (out/'prompts.json').write_text(json.dumps(prompts,indent=2,ensure_ascii=False))
    revision='980e34bc0f0460c437bb4fa0238a58b9b50158d3'
    filename='workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors'
    path=hf_hub_download('sida/llmdbg',filename,repo_type='dataset',revision=revision,local_files_only=True)
    with safe_open(path,framework='pt',device='cpu') as f: reference=f.get_tensor('post.12').cuda().float()
    total=torch.zeros(model.d_model,model.d_model,device='cuda')
    metrics=[];previous=None
    for p,prompt in enumerate(prompts):
        start_time=time.monotonic()
        saved=out/f'prompt_{p+1:02d}_J.pt'
        ids=model.encode(prompt,max_length=128)
        valid=torch.arange(4,ids.shape[1]-1,device='cuda')
        if saved.exists():
            per_prompt=torch.load(saved,map_location='cuda',weights_only=True)
            print(f'Resumed prompt {p+1}',flush=True)
        else:
            B=args.dim_batch;D=model.d_model
            per_prompt=torch.empty_like(total)
            with same_position_gradients(),ActivationRecorder(model.layers,at=[12,23],start_graph_at=12) as rec:
                model.forward(ids.expand(B,-1))
                source,target=rec.activations[12],rec.activations[23]
                cotangent=torch.zeros_like(target)
                batch=torch.arange(B,device='cuda')
                for first in range(0,D,B):
                    n=min(B,D-first);cotangent.zero_()
                    cotangent[batch[:n,None],valid[None],first+batch[:n,None]]=1
                    grad,=torch.autograd.grad(target,source,cotangent,retain_graph=first+B<D)
                    per_prompt[first:first+n]=grad[:n,valid].float().mean(1)
                    if first%(B*64)==0:print(f'Prompt {p+1}/25 rows {first}/{D}',flush=True)
            torch.save(per_prompt.cpu(),saved)
        total+=per_prompt
        mean=total/(p+1)
        cosine=float(torch.nn.functional.cosine_similarity(mean.flatten(),reference.flatten(),dim=0))
        update=None if previous is None else float((mean-previous).norm()/previous.norm())
        metrics.append(dict(n_prompts=p+1,cosine_to_v2=cosine,relative_update=update,frobenius_norm=float(mean.norm()),elapsed_seconds=time.monotonic()-start_time))
        previous=mean.clone()
        torch.save(dict(J=mean.cpu(),n_prompts=p+1,source_layer=12,target_layer=23,skip_first=4,max_seq_len=128,
            method='exact same-position blocks, no random probes',validation=checks),out/'lens.pt')
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        with (out/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
        plot_metrics(metrics,out)
        print(f'COMPLETED {p+1}/25 cosine_to_v2={cosine:.6f}, relative_update={update}, seconds={time.monotonic()-start_time:.1f}',flush=True)
    with torch.no_grad():
        ids=tokenizer(PROMPT,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
        with ActivationRecorder(model.layers,at=[12]) as rec:model.forward(ids);hidden=rec.activations[12][0].float()
        rows=[]
        for t,h in enumerate(hidden):
            row=dict(position=t,token=tokenizer.decode([int(ids[0,t])]))
            for label,J in [('exact_self_only',mean),('v2_reference',reference)]:
                logits=model.unembed((J@h)[None]).float()[0];values,indices=logits.topk(10)
                row[label]=[dict(token=tokenizer.decode([int(i)]),logit=float(v)) for v,i in zip(values,indices)]
            rows.append(row)
        (out/'sport_readouts.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    print('DONE',out,flush=True)


if __name__=='__main__':main()

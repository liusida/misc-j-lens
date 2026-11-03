# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "matplotlib", "numpy"]
# ///
"""Pairwise cosines of soccer readout directions from individual Jacobians.

All causal pairs, including positions excluded by usual lens fitting masks.
Computes G.T @ a directly by VJP, not full D x D matrices. The RMS denominator
is omitted because it is a positive scalar and cannot change cosine similarity.
"""
import csv
import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'references/upstream/anthropic-jacobian-lens'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder
from j_lens_one_prompt import DEFAULT_PROMPT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prompt-index',type=int,default=0,choices=range(5))
    parser.add_argument('--cmap',default='RdBu_r')
    args=parser.parse_args()
    from five_prompt_centuries_study import PROMPTS
    prompt=PROMPTS[args.prompt_index]
    title=['Paris','Astronomy','Forests','Railways','Computer programs'][args.prompt_index]
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    out=ROOT/'artifacts/runs/j/legacy'/('paris_soccer_pair_cosines' if args.prompt_index==0 else f'prompt{args.prompt_index}_soccer_pair_cosines')
    out.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,'Qwen/Qwen3.5-2B',dtype=torch.bfloat16)
    model=from_hf(hf,tokenizer)
    ids=model.encode(prompt,max_length=128)
    tokens=[tokenizer.decode([int(t)]) for t in ids[0]]
    word_ids=tokenizer.encode(' soccer',add_special_tokens=False)
    assert len(word_ids)==1
    a=((1+model._final_norm.weight.float())*model._lm_head.weight[word_ids[0]].float()).detach()
    vectors=[]; metadata=[]
    with ActivationRecorder(model.layers,at=[12,23],start_graph_at=12) as rec:
        model.forward(ids)
        source=rec.activations[12]; target=rec.activations[23]
        for tprime in range(len(tokens)):
            cotangent=torch.zeros_like(target)
            cotangent[0,tprime]=a.to(target.dtype)
            grad,=torch.autograd.grad(target,source,grad_outputs=cotangent,retain_graph=tprime<len(tokens)-1)
            for t in range(tprime+1):
                vectors.append(grad[0,t].detach().float().cpu())
                metadata.append(dict(index=len(metadata),source_position=t,source_token=tokens[t],source_layer=12,
                    target_position=tprime,target_token=tokens[tprime],target_layer=23,
                    same_position=t==tprime,label=f'{t}→{tprime}'+('*' if t==tprime else '')))
            print(f'Target {tprime}/{len(tokens)-1} complete',flush=True)
    V=torch.stack(vectors)
    norms=V.norm(dim=1)
    normalized=torch.nn.functional.normalize(V,dim=1)
    C=normalized @ normalized.T
    torch.save(dict(directions=V,cosines=C,norms=norms,pairs=metadata,prompt=prompt,
                    readout_token=' soccer',formula='G_target,source.T @ ((1+norm.weight)*W_U[soccer])'),out/'directions_and_cosines.pt')
    np.save(out/'cosines.npy',C.numpy())
    with (out/'pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[*metadata[0],'direction_norm'])
        writer.writeheader()
        writer.writerows(dict(row,direction_norm=float(norms[i])) for i,row in enumerate(metadata))
    # Target-major ordering groups each target's causal source positions.
    # PDF labels every pair; PNG uses group boundaries to remain readable.
    labels=[r['label'] for r in metadata]
    fig,ax=plt.subplots(figsize=(17,15),layout='constrained')
    im=ax.imshow(C.numpy(),vmin=-1,vmax=1,cmap=args.cmap,interpolation='nearest',rasterized=True)
    same=[i for i,r in enumerate(metadata) if r['same_position']]
    ticks=sorted(set(same+[max(0,i-1) for i in same]))
    ax.set_xticks(ticks,[labels[i] for i in ticks],rotation=90,fontsize=4)
    ax.set_yticks(ticks,[labels[i] for i in ticks],fontsize=4)
    ax.set_xlabel('Jacobian pair: source L12 → target L23 (* = same position)')
    ax.set_ylabel('Jacobian pair: source L12 → target L23 (* = same position)')
    ax.set_title(f'{title} prompt: individual “ soccer” Jacobian directions\n{len(metadata):,} causal pairs; target-major order; ticks show same and next position')
    fig.colorbar(im,ax=ax,label='Cosine similarity',shrink=.8)
    fig.savefig(out/'heatmap.png',dpi=220)
    plt.close(fig)
    # Large vector PDF: every tick is labeled, future pairs have no asterisk.
    fig,ax=plt.subplots(figsize=(48,48))
    im=ax.imshow(C.numpy(),vmin=-1,vmax=1,cmap=args.cmap,interpolation='nearest',rasterized=True)
    ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=1.5)
    ax.set_yticks(range(len(labels)),labels,fontsize=1.5)
    ax.tick_params(length=1,width=.2,pad=1)
    ax.set_xlabel('source L12 → target L23; * = same position; no * = future target')
    ax.set_ylabel('source L12 → target L23; * = same position; no * = future target')
    ax.set_title('Individual soccer Jacobian directions: cosine similarity (all pair labels)')
    fig.colorbar(im,ax=ax,shrink=.8)
    fig.savefig(out/'heatmap_all_labels.pdf',bbox_inches='tight',dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(48,48))
    im=ax.imshow(C.numpy(),vmin=-1,vmax=1,cmap=args.cmap,interpolation='nearest')
    ax.set_xticks(ticks,[labels[i] for i in ticks],rotation=90,fontsize=5)
    ax.set_yticks(ticks,[labels[i] for i in ticks],fontsize=5)
    ax.tick_params(length=1,width=.2,pad=1)
    ax.set_xlabel('Source L12 → target L23; * = same position; no * = next position')
    ax.set_ylabel('Source L12 → target L23; * = same position; no * = next position')
    ax.set_title(f'{title}: individual soccer Jacobian directions; target-major order')
    fig.colorbar(im,ax=ax,label='Cosine similarity',shrink=.8)
    fig.savefig(out/'heatmap_same_and_next_ticks.png',dpi=250,bbox_inches='tight')
    plt.close(fig)
    print('Saved',out,flush=True)
    print('Pairs:',len(metadata),'same:',len(same),'zero directions:',int((norms==0).sum()))


if __name__=='__main__':
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Compare individual Paris soccer directions against the cached v2 lens."""
import csv
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent


def main():
    folder=ROOT/'j_lens_output/paris_soccer_pair_cosines'
    data=torch.load(folder/'directions_and_cosines.pt',map_location='cpu',weights_only=False)
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    model=from_pretrained_to_cuda(AutoModelForCausalLM,'Qwen/Qwen3.5-2B',dtype=torch.bfloat16)
    ids=tokenizer.encode(' soccer',add_special_tokens=False)
    assert len(ids)==1
    # Text decoder location for this model's wrapper.
    import sys
    sys.path.insert(0,str(ROOT/'anthropic-jacobian-lens'))
    from jlens import from_hf
    adapter=from_hf(model,tokenizer)
    a=((1+adapter._final_norm.weight.float())*adapter._lm_head.weight[ids[0]].float()).detach()
    revision='980e34bc0f0460c437bb4fa0238a58b9b50158d3'
    path=hf_hub_download('sida/llmdbg','workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors',repo_type='dataset',revision=revision,local_files_only=True)
    with safe_open(path,framework='pt',device='cpu') as f:
        reference=f.get_tensor('post.12').cuda().float().T @ a
    reference=reference.cpu()
    similarities=torch.nn.functional.cosine_similarity(data['directions'],reference[None],dim=1).numpy()
    pairs=data['pairs']
    source=np.array([r['source_position'] for r in pairs])
    target=np.array([r['target_position'] for r in pairs])
    n=int(target.max())+1
    same=np.full(n,np.nan); next_=same.copy(); future=same.copy()
    for t in range(n):
        same[t]=similarities[(source==t)&(target==t)][0]
        mask=(source==t)&(target==t+1)
        if mask.any(): next_[t]=similarities[mask][0]
        mask=(source==t)&(target>t)
        if mask.any(): future[t]=similarities[mask].mean()
    rows=[dict(r,cosine_to_v2_reference=float(c)) for r,c in zip(pairs,similarities)]
    with (folder/'cosine_to_v2_reference.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(17,6),layout='constrained')
    ax.scatter(source,similarities,s=8,alpha=.18,color='gray',label='All individual source→target directions')
    ax.plot(range(n),same,'o-',ms=3,color='#b2182b',label='Same-position target (*)')
    ax.plot(range(n),next_,'o-',ms=3,color='#2166ac',label='Next-position target')
    ax.plot(range(n),future,'-',lw=2,color='#1b7837',label='Mean cosine over strictly future targets')
    tokens={r['source_position']:r['source_token'] for r in pairs}
    ax.set_xticks(range(n),[f'{i}: {tokens[i]!r}' for i in range(n)],rotation=90,fontsize=8)
    ax.set_ylim(-1,1);ax.axhline(0,color='black',lw=.7)
    ax.set_xlabel('Source token position in Prompt 0 (Paris), layer 12 post')
    ax.set_ylabel('Cosine similarity with v2 soccer direction at layer 12')
    ax.set_title('Individual Prompt 0 soccer Jacobian directions vs. the 100-prompt v2 J-Lens direction')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    output=folder/'cosine_to_v2_reference.png'
    fig.savefig(output,dpi=220)
    print(output)
    print('all-pair cosine range',similarities.min(),similarities.max())
    print('mean same',np.nanmean(same),'mean next',np.nanmean(next_),'mean all future',similarities[target>source].mean())


if __name__=='__main__':main()

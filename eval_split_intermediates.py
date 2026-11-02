# /// script
# dependencies = ["torch", "transformers", "gb10-load-llm", "safetensors", "matplotlib"]
# ///
"""Intermediate recovery on released Anthropic items; matched three lenses."""
import json
import argparse
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from gb10_load_llm import from_pretrained_to_cuda
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'anthropic-jacobian-lens'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder

SYN={'multiplication':['multiplication','multiply','times','×','*'],
     'addition':['addition','add','plus','+'],
     'subtraction':['subtraction','subtract','minus','-'],
     'division':['division','divide','divided','/','÷']}
NUM=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty']
KS=[1,2,5,10,20,50,100]

def candidates(tok,concept,arithmetic):
    forms=SYN.get(concept,[concept]) if arithmetic else [concept]
    if arithmetic and concept.isdigit() and int(concept)<len(NUM):
        forms=forms+[NUM[int(concept)]]
    ids=set()
    for form in forms:
        for variant in {form,form.lower(),form.capitalize()}:
            for prefix in ['', ' ']:
                encoded=tok.encode(prefix+variant,add_special_tokens=False)
                if len(encoded)==1: ids.add(encoded[0])
    return sorted(ids)

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-positions',action='store_true')
    args=parser.parse_args()
    base=ROOT/'j_lens_output/v2_split_lenses'
    out=ROOT/'j_lens_output'/('intermediate_eval_all_positions' if args.all_positions else 'intermediate_eval');out.mkdir(exist_ok=True)
    manifests={m:json.loads((base/m/'manifest.json').read_text()) for m in ['normal','same_token','future_only']}
    ref=manifests['normal']
    assert all(v['prompts']==ref['prompts'] and v['model_revision']==ref['model_revision'] for v in manifests.values())
    tok=AutoTokenizer.from_pretrained(ref['model_id'],revision=ref['model_revision'])
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,ref['model_id'],revision=ref['model_revision'],dtype=torch.bfloat16)
    model=from_hf(hf,tok)
    # Common fitted post layers only; final identity/zero layer excluded equally.
    maps={m:load_file(str(base/m/'lens.safetensors')) for m in manifests}
    layers=sorted(int(k.split('.')[1]) for k in maps['normal'] if k.startswith('post.'))
    maps={m:torch.stack([v[f'post.{l}'] for l in layers]).cuda().float() for m,v in maps.items()}
    records=[]
    for path in sorted((ROOT/'anthropic-jacobian-lens/data/evaluations').glob('*.json')):
        category=path.stem.replace('lens-eval-','')
        for index,item in enumerate(json.loads(path.read_text())['items']):
            text=item['prompt']
            if category=='poetry': text=text[:text.rfind('\n')+1]
            ids=model.encode(text,max_length=4096)
            with ActivationRecorder(model.layers,at=layers) as recorder:
                model.forward(ids)
                h=torch.stack([recorder.activations[l][0].float() for l in layers])
            targets=[candidates(tok,c,category=='order-ops') for c in item['intermediates']]
            row={'category':category,'name':item['name'],'position':ids.shape[1]-1,
                 'position_token':tok.decode([int(ids[0,-1])]),'intermediates':item['intermediates'],
                 'candidate_ids':targets,'ranks':{},'best_locations':{}}
            for method,matrix in maps.items():
                ranks=[None if not target else 10**9 for target in targets]
                locations=[None for target in targets]
                positions=list(range(ids.shape[1])) if args.all_positions else [ids.shape[1]-1]
                for li,layer in enumerate(layers):
                    for start in range(0,len(positions),32):
                        pos=positions[start:start+32]
                        z=h[li,pos] @ matrix[li].T
                        logits=model.unembed(z).float()
                        for ci,target in enumerate(targets):
                            if not target: continue
                            scores=logits[:,target].max(dim=1).values
                            rr=(logits>scores[:,None]).sum(dim=1)+1
                            best=int(rr.argmin());rank=int(rr[best])
                            if rank<ranks[ci]:
                                ranks[ci]=rank
                                position=pos[best]
                                locations[ci]={'layer':layer,'position':position,'token':tok.decode([int(ids[0,position])])}
                row['ranks'][method]=ranks
                row['best_locations'][method]=locations
            records.append(row)
            if index%10==0: print(category,index,flush=True)
        (out/'items.json').write_text(json.dumps(records,ensure_ascii=False))
    summary={}
    for category in sorted({r['category'] for r in records}):
        group=[r for r in records if r['category']==category]
        summary[category]={}
        for method in maps:
            rates=[]
            for k in KS:
                rates.append(sum(sum(rank is not None and rank<=k for rank in r['ranks'][method])/len(r['intermediates']) for r in group)/len(group))
            summary[category][method]=dict(zip(map(str,KS),rates))
    report={'config':{'k':KS,'layers':layers,'manifests':manifests,
        'matching':'single-token plain/space-prefixed original/lower/capitalized variants; arithmetic explicit synonyms and 0-20 words',
        'unencodable':'counted as misses','positions':('all positions in causal prefix' if args.all_positions else 'raw prompt last token')+'; poetry truncated after last newline',
        'comparison_note':'All-position searches more cells than the paper fixed-position protocol; higher scores may reflect additional search opportunities. No tokens after the designated readout boundary are included.',
        'metric':'equal item mean of fraction of concepts recovered at any common fitted post layer'},'pass_at_k':summary}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,(category,methods) in zip(axes.flat,summary.items()):
        for method,values in methods.items(): ax.plot(KS,list(values.values()),marker='o',label=method)
        ax.set(title=category,xscale='log',ylim=(0,1),xlabel='k',ylabel='pass@k');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(out/'pass_at_k.png',dpi=160)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

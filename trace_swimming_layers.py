# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors"]
# ///
"""Trace exact swimming-token J-Lens rank at position 19 across blocks."""
import json
import argparse
from pathlib import Path
import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from test_swimming_patch import PROMPT, ROOT, from_hf, ActivationRecorder


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--readout-token', default=' swimming')
    args=parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    ids=tokenizer(PROMPT,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
    word=tokenizer.encode(' swimming',add_special_tokens=False)
    assert len(word)==1
    word=word[0]
    readout_ids=tokenizer.encode(args.readout_token,add_special_tokens=False)
    if len(readout_ids)!=1:
        raise ValueError(f'Readout must be one token: {readout_ids}')
    readout=readout_ids[0]
    path=hf_hub_download('sida/llmdbg','workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors',repo_type='dataset',revision='980e34bc0f0460c437bb4fa0238a58b9b50158d3',local_files_only=True)
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,'Qwen/Qwen3.5-2B',dtype=torch.bfloat16)
    model=from_hf(hf,tokenizer)
    layers=list(range(17,24))
    generation=json.loads((ROOT/'j_lens_output/swimming_patch_v2_results.json').read_text())
    tail=[s['generated_id'] for s in generation['results'][0]['steps']]
    ids=torch.cat([ids,ids.new_tensor([tail])],dim=1)
    with ActivationRecorder(model.layers,at=layers) as rec:
        model.forward(ids)
        baseline={l:rec.activations[l][0].float().clone() for l in layers}
    with safe_open(path,framework='pt',device='cpu') as f:
        J=f.get_tensor('post.17').cuda().float()
        direction=J.T @ ((1+model._final_norm.weight.float())*model._lm_head.weight[word].float())
        replacement=baseline[17][19].norm()*direction/direction.norm()
        def hook(module,inputs,output):
            h=output if torch.is_tensor(output) else output[0]
            changed=h.clone(); changed[:,19]=replacement.to(h.dtype)
            return changed if torch.is_tensor(output) else (changed,*output[1:])
        handle=model.layers[17].register_forward_hook(hook)
        try:
            with ActivationRecorder(model.layers,at=layers) as rec:
                model.forward(ids)
                patched={l:rec.activations[l][0].float().clone() for l in layers}
        finally:
            handle.remove()
        rows=[]
        for l in layers:
            # Final block is already in the target basis; its lens is identity.
            J=f.get_tensor(f'post.{l}').cuda().float() if l<23 else None
            for position in range(19,ids.shape[1]):
                row=dict(layer=l,site='post',position=position,token=tokenizer.decode([int(ids[0,position])]))
                for label,states in [('baseline',baseline),('patched',patched)]:
                    h=states[l][position]
                    transported=J@h if J is not None else h
                    logits=model.unembed(transported.unsqueeze(0)).float()[0]
                    row[label]=dict(rank=int((logits>logits[readout]).sum())+1,logit=float(logits[readout]),probability=float(logits.softmax(-1)[readout]))
                rows.append(row)
            print('Finished layer',l,flush=True)
    out=ROOT/f'j_lens_output/{args.readout_token.strip()}_later_positions_v2.json'
    out.write_text(json.dumps(dict(prompt=PROMPT,steering_token=' swimming',readout_token=args.readout_token,rows=rows),indent=2)+'\n')
    print('Saved',out)
    for position in range(19,ids.shape[1]):
        entries=[r for r in rows if r['position']==position]
        print(position,repr(entries[0]['token']), 'baseline:',[r['baseline']['rank'] for r in entries], 'patched:',[r['patched']['rank'] for r in entries])


if __name__=='__main__':
    main()

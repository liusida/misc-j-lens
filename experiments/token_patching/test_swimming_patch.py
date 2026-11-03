# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors"]
# ///
"""Patch post.17 position 19 toward swimming using cached v2 J-Lens."""
import json
import sys
from pathlib import Path
import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'references/upstream/anthropic-jacobian-lens'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder

PROMPT = ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
          '<|im_start|>user\nName a sport.<|im_end|>\n'
          '<|im_start|>assistant\n<think>\n\n</think>')


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    ids = tokenizer(PROMPT, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
    token_table = [dict(position=i, token_id=int(t), token=tokenizer.decode([int(t)]))
                   for i,t in enumerate(ids[0])]
    print(json.dumps(token_table, ensure_ascii=False), flush=True)
    assert token_table[19]['token'] == '\n', 'Requested token 19 is not newline'
    revision = '980e34bc0f0460c437bb4fa0238a58b9b50158d3'
    path = hf_hub_download('sida/llmdbg', 'workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors',
                           repo_type='dataset', revision=revision, local_files_only=True)
    with safe_open(path, framework='pt', device='cpu') as f:
        J = f.get_tensor('post.17').cuda().float()
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, 'Qwen/Qwen3.5-2B', dtype=torch.bfloat16)
    model = from_hf(hf, tokenizer)
    with ActivationRecorder(model.layers, at=[17]) as recorder:
        model.forward(ids)
        original = recorder.activations[17][0,19].float().clone()
    word_ids = tokenizer.encode(' swimming', add_special_tokens=False)
    assert len(word_ids) == 1
    word = word_ids[0]
    direction = J.T @ ((1 + model._final_norm.weight.float()) * model._lm_head.weight[word].float())
    unit = direction / direction.norm()
    replacement = original.norm() * unit
    print('Measured original norm:', float(original.norm()), 'cosine:',
          float(torch.nn.functional.cosine_similarity(original, unit, dim=0)), flush=True)

    def hook(module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        changed = hidden.clone()
        changed[:,19] = replacement.to(hidden.dtype)
        return changed if torch.is_tensor(output) else (changed, *output[1:])

    def logits_at(tokens):
        return model._lm_head(model.forward(tokens).last_hidden_state).float()[0]

    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>')}
    results=[]
    for patched in [False, True]:
        handle=model.layers[17].register_forward_hook(hook) if patched else None
        try:
            tokens=ids.clone()
            steps=[]
            generated=[]
            for step in range(60):
                logits=logits_at(tokens)[-1]
                probs=logits.softmax(-1)
                chosen=int(logits.argmax())
                steps.append(dict(step=step+1, swimming_probability=float(probs[word]),
                                  swimming_rank=int((logits>logits[word]).sum())+1,
                                  generated_id=chosen, generated_token=tokenizer.decode([chosen])))
                generated.append(chosen)
                tokens=torch.cat([tokens,tokens.new_tensor([[chosen]])],dim=1)
                if chosen in stop_ids:
                    break
        finally:
            if handle: handle.remove()
        lens_logits=model.unembed((J @ (replacement if patched else original)).unsqueeze(0)).float()[0]
        row=dict(condition='patched' if patched else 'baseline',
                 generated_text=tokenizer.decode(generated), steps=steps,
                 lens_swimming_rank=int((lens_logits>lens_logits[word]).sum())+1,
                 norm=float((replacement if patched else original).norm()),
                 cosine=float(torch.nn.functional.cosine_similarity(replacement if patched else original,unit,dim=0)))
        results.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    # Measure each intervention on the same baseline continuation as a control.
    baseline_tail=[s['generated_id'] for s in results[0]['steps']]
    shared_ids=torch.cat([ids,ids.new_tensor([baseline_tail])],dim=1)
    controlled=[]
    for patched in [False,True]:
        handle=model.layers[17].register_forward_hook(hook) if patched else None
        try:
            logits=logits_at(shared_ids)[ids.shape[1]-1:ids.shape[1]-1+len(baseline_tail)]
            controlled.append(dict(condition='patched' if patched else 'baseline',
                swimming_probabilities=logits.softmax(-1)[:,word].cpu().tolist()))
        finally:
            if handle: handle.remove()
    output=ROOT/'artifacts/runs/j/legacy/swimming_patch_v2_results.json'
    output.write_text(json.dumps(dict(prompt=PROMPT, tokens=token_table, layer=17, position=19,
        lens_revision=revision, lens_file=path, results=results, same_baseline_continuation=controlled),indent=2,ensure_ascii=False)+'\n')
    print('Saved',output)


if __name__=='__main__':
    main()

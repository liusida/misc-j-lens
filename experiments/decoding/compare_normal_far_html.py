# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0"]
# ///
"""Decode the saved computer-science conversation with Normal and lag>=3 lenses."""
import ast
import json
import sys
from pathlib import Path
import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'references/upstream/anthropic-jacobian-lens'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder

@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    original = json.loads((ROOT / 'artifacts/comparisons/j_split_l12_legacy/readouts.json').read_text())
    out = ROOT / 'artifacts/comparisons/normal_far_legacy'
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, 'Qwen/Qwen3.5-2B', dtype=torch.bfloat16, attn_implementation='eager')
    model = from_hf(hf, tokenizer)
    ids = torch.tensor([[r['token_id'] for r in original['rows']]], device='cuda')
    with ActivationRecorder(model.layers, at=[12]) as recorder:
        model.forward(ids)
        hidden = recorder.activations[12][0].float()
    rows = [{k: r[k] for k in ['position', 'phase', 'token_id', 'token']} for r in original['rows']]
    paths = dict(normal=ROOT/'artifacts/runs/j/legacy/paired_self_normal_25/normal_lens.pt', far=ROOT/'artifacts/runs/j/legacy/far_future_lens_25/lens.pt')
    for name, path in paths.items():
        saved = torch.load(path, map_location='cuda', weights_only=True)
        assert saved['n_prompts'] == 25
        J = saved['J'].float()
        for first in range(0, len(rows), 8):
            logits = model.unembed(hidden[first:first+8] @ J.T).float()
            probs = logits.softmax(-1)
            values, indices = logits.topk(10, dim=-1)
            for b in range(len(values)):
                rows[first+b][name] = [dict(token_id=int(i), token=tokenizer.decode([int(i)]), logit=float(v), probability=float(probs[b,i])) for v,i in zip(values[b], indices[b])]
        print('Decoded', name, len(rows), 'positions', flush=True)
    data = {k: original[k] for k in ['model', 'layer', 'site', 'user_prompt', 'chat_prompt', 'generated_answer', 'generation', 'generation_truncated']}
    data.update(rows=rows, lens_paths={k:str(v) for k,v in paths.items()})
    (out/'readouts.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    tree = ast.parse((ROOT/'render_normal_future_horizontal.py').read_text())
    template = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='TEMPLATE' for t in node.targets))
    template = template.replace("'future'", "'far'").replace('.future{', '.far{').replace('Future-only', 'Far-future-only')
    template = template.replace('Far-future-only = normal − same-position, with identical source averaging; inherits the recorded BF16 numerical differences.', 'Far-future-only retains target positions at least 3 tokens after each source; excludes self and the next two positions. Both fits use the same 25 prompts and source normalization, with BF16 model arithmetic and FP32 accumulation.')
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    page = template.replace('__DATA__', payload)
    (out/'comparison.html').write_text(page)
    assert len(rows)==len(original['rows']) and all(len(r[k])==10 for r in rows for k in paths)
    print(out/'comparison.html', flush=True)

if __name__=='__main__':
    main()

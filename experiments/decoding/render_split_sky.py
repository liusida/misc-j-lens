# /// script
# dependencies = ["torch", "transformers", "gb10-load-llm", "safetensors"]
# ///
"""Apply split maps to saved sky activations and reuse its interactive renderer."""
import copy
import json
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from gb10_load_llm import from_pretrained_to_cuda

ROOT = Path(__file__).resolve().parents[2]
TABLE = Path('/home/liusida/research/llm-debugger/llmdbg/scripts/diagnostics/sky_layer_table')
sys.path.insert(0, str(TABLE))
from compact import write_html

@torch.inference_mode()
def main():
    folder = TABLE/'results/qwen35-2b'
    data = json.loads((folder/'readouts.json').read_text())
    split = ROOT/'artifacts/lenses/j/v2_split'
    out = ROOT/'artifacts/runs/j/legacy/split_sky_comparison'
    out.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(data['model'], revision=data['revision'])
    model = from_pretrained_to_cuda(AutoModelForCausalLM, data['model'], revision=data['revision'], dtype=torch.bfloat16)
    model.eval()
    decoder = model.model
    if not hasattr(decoder, 'norm'):
        decoder = decoder.language_model
    head = model.get_output_embeddings()
    data['lens_rows'] = {'j': data['rows']}
    data['lens_labels'] = {'j':'J (reference)', 'same':'Same-token', 'future':'Future-only'}
    data['lens_summaries'] = {}
    with safe_open(folder/'activations.safetensors', framework='pt', device='cpu') as capture:
        assert capture.get_tensor('token_ids').tolist() == [t['id'] for t in data['tokens']]
        for name, directory in [('same','same_token'), ('future','future_only')]:
            manifest = json.loads((split/directory/'manifest.json').read_text())
            assert manifest['model_revision'] == data['revision']
            data['lens_summaries'][name] = manifest
            rows = []
            with safe_open(split/directory/'lens.safetensors', framework='pt', device='cpu') as maps:
                for original in data['rows']:
                    row = copy.deepcopy(original)
                    layer, site = row['layer'], row['site']
                    key = f"{'mid' if site == 'attention-output' else 'post'}.{layer}"
                    if site == 'attention-output':
                        x = (capture.get_tensor(f'mid.{layer}').float()-capture.get_tensor(f'pre.{layer}').float()).bfloat16()
                    else:
                        x = capture.get_tensor(f"{'mlp_output' if site == 'mlp-output' else 'post'}.{layer}")
                    if key in maps.keys():
                        matrix = maps.get_tensor(key).cuda().float()
                    else:
                        assert key == 'post.23'
                        matrix = torch.eye(x.shape[-1], device='cuda') if name == 'same' else torch.zeros((x.shape[-1],x.shape[-1]),device='cuda')
                    cells = []
                    for start in range(0,len(x),32):
                        z = (x[start:start+32].cuda().float() @ matrix.T).to(head.weight.dtype)
                        logits = head(decoder.norm(z)).float()
                        scores, ids = logits.topk(8,dim=-1)
                        for ti, ts in zip(ids.tolist(),scores.tolist()):
                            cells.append([{'id':i,'text':tokenizer.decode([i]),'score':s} for i,s in zip(ti,ts)])
                    row['hover_readouts'] = cells
                    row['readouts'] = [c[:2] for c in cells]
                    rows.append(row)
                    print(name, site, layer, flush=True)
            data['lens_rows'][name] = rows
    data.pop('rows',None)
    (out/'readouts.json').write_text(json.dumps(data,ensure_ascii=False))
    write_html(data,out/'sky-lenses.html')

if __name__ == '__main__':
    main()

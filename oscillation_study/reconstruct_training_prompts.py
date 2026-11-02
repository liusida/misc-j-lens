# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets", "transformers"]
# ///
"""Reproduce the local v2 fitter's first-100-nonempty pile-10k selection."""
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"
import hashlib
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
source=Path('/home/liusida/research/llm-debugger/llmdbg/scripts/fit_lenses/v2/fit_lenses.py')
arrow=Path('/home/liusida/hf_cache/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow')
dataset=Dataset.from_file(str(arrow))
tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B',local_files_only=True)
rows=[]
for index,row in enumerate(dataset):
    text=str(row['text'])
    if not text.strip():
        continue
    ids=tokenizer(text,truncation=True,max_length=128)['input_ids']
    rows.append(dict(prompt_index=len(rows),dataset_row=index,text=text,input_ids=ids,token_count=len(ids),truncated_text=tokenizer.decode(ids),sha256=hashlib.sha256(text.encode()).hexdigest()))
    if len(rows)==100:
        break
cooking=json.loads((HERE/'random_comparison/summary.json').read_text())['prompts']['cooking']['prompt']
matches=[r['prompt_index'] for r in rows if cooking in r['text']]
prefix_matches=[r['prompt_index'] for r in rows if cooking[:80] in r['text']]
report=dict(dataset='NeelNanda/pile-10k',cached_arrow=str(arrow),dataset_fingerprint=dataset._fingerprint,fitting_source=str(source),fitting_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),recipe='First 100 nonempty train documents in dataset order; no shuffle; tokenizer truncation to 128',prompt_count=len(rows),sequence_tokens=sum(r['token_count'] for r in rows),valid_positions=sum(max(0,r['token_count']-5) for r in rows),cooking_exact_matches=matches,cooking_first_80_character_matches=prefix_matches,caveat='Reconstructed from currently available local fitter and cached dataset; original manifest does not pin dataset revision or fitter commit.',prompts=rows)
out=HERE/'training_prompt_audit'
out.mkdir(exist_ok=True)
(out/'reconstructed_100_prompts.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='prompts'},indent=2))
print('First three document prefixes:')
for r in rows[:3]: print(r['dataset_row'],repr(r['text'][:130]))

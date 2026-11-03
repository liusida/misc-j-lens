# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "huggingface-hub", "safetensors", "matplotlib"]
# ///
"""Verify saved paired averages, refresh plots and canonical self-only output."""
import csv
import json
from pathlib import Path
import torch
from fit_paired_self_normal_25 import plot
from fit_exact_self_lens_25 import plot_metrics

ROOT=Path(__file__).resolve().parents[2]
paired=ROOT/'artifacts/runs/j/legacy/paired_self_normal_25'
selfdir=ROOT/'artifacts/runs/j/legacy/exact_self_lens_25'
metrics=json.loads((paired/'metrics.json').read_text())
assert [r['n_prompts'] for r in metrics]==list(range(1,26))
lenses={name:torch.load(paired/f'{name}_lens.pt',map_location='cpu',weights_only=True) for name in ['self','normal']}
for name,directory,pattern in [('self',selfdir,'prompt_{:02d}_J.pt'),('normal',paired,'normal_prompt_{:02d}_J.pt')]:
    total=torch.zeros_like(lenses[name]['J'])
    for i in range(1,26):total+=torch.load(directory/pattern.format(i),map_location='cpu',weights_only=True)
    difference=(total/25-lenses[name]['J']).abs().max().item()
    print(name,'saved-average max discrepancy:',difference)
    assert difference<1e-5
torch.save(dict(J=lenses['normal']['J']-lenses['self']['J'],n_prompts=25,
    description='Normal minus same-position Jacobian; future contribution, with BF16 roundoff'),paired/'future_difference.pt')
plot(metrics,paired)
torch.save(lenses['self'],selfdir/'lens.pt')
torch.save(lenses['self'],selfdir/'running_fit.pt')
canonical=[dict(n_prompts=r['n_prompts'],cosine_to_v2=r['self_cosine_to_v2'],relative_update=r['self_lim'],
    frobenius_norm=None,elapsed_seconds=r['elapsed_seconds']) for r in metrics]
(selfdir/'metrics.json').write_text(json.dumps(canonical,indent=2)+'\n')
with (selfdir/'metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(canonical[0]));writer.writeheader();writer.writerows(canonical)
plot_metrics(canonical,selfdir)
readouts=json.loads((paired/'sport_readouts.json').read_text())
canonical_readouts=[dict(position=r['position'],token=r['token'],exact_self_only=r['self'],v2_reference=r['v2']) for r in readouts]
(selfdir/'sport_readouts.json').write_text(json.dumps(canonical_readouts,indent=2,ensure_ascii=False)+'\n')
print('n self_to_v2 normal_to_v2 self_vs_normal self_lim normal_lim')
for r in metrics:
    if r['n_prompts'] in [1,5,10,15,20,25]:
        print(*(r[k] for k in ['n_prompts','self_cosine_to_v2','normal_cosine_to_v2','self_normal_cosine','self_lim','normal_lim']))
for r in readouts:
    if r['position'] in [16,19,25]:
        print('Position',r['position'],repr(r['token']))
        for name in ['self','normal','v2']:print(name,[i['token'] for i in r[name][:6]])

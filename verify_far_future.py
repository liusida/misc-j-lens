# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.7"]
# ///
"""Check saved lag>=3 matrices and report comparisons with prior lenses."""
import csv
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
out=ROOT/'j_lens_output/far_future_lens_25'
paired=ROOT/'j_lens_output/paired_self_normal_25'
partitions=[(3,0,1),(3,1,1),(3,2,1),(2,0,-1),(2,1,-1)]
checkpoint=torch.load(out/'lens.pt',map_location='cpu',weights_only=True)
assert checkpoint['n_prompts']==25
total=torch.zeros_like(checkpoint['J']);errors=[]
for p in range(1,26):
    near=torch.zeros_like(total)
    for width,offset,sign in partitions:
        block=torch.load(out/f'prompt_{p:02d}_width_{width}_offset_{offset}.pt',map_location='cpu',weights_only=True)
        assert block.shape==total.shape and torch.isfinite(block).all()
        near+=sign*block
    normal=torch.load(paired/f'normal_prompt_{p:02d}_J.pt',map_location='cpu',weights_only=True)
    far=torch.load(out/f'prompt_{p:02d}_J.pt',map_location='cpu',weights_only=True)
    error=float((far-(normal-near)).abs().max());assert error<1e-5
    errors.append(error);total+=far
mean_error=float((total/25-checkpoint['J']).abs().max());assert mean_error<1e-5
paths=dict(self=paired/'self_lens.pt',normal=paired/'normal_lens.pt',future=ROOT/'j_lens_output/future_only_lens_25/lens.pt',far=out/'lens.pt')
lenses={label:torch.load(path,map_location='cpu',weights_only=True)['J'] for label,path in paths.items()}
cosines={a:{b:float(torch.nn.functional.cosine_similarity(A.flatten(),B.flatten(),dim=0)) for b,B in lenses.items()} for a,A in lenses.items()}
with (out/'lens_cosines.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['lens',*lenses])
    for label in lenses:writer.writerow([label,*[cosines[label][b] for b in lenses]])
metrics=json.loads((out/'metrics.json').read_text());assert [r['n_prompts'] for r in metrics]==list(range(1,26))
summary=dict(max_per_prompt_reconstruction_error=max(errors),mean_reconstruction_error=mean_error,
    matrix_cosines=cosines,final_metrics=metrics[-1])
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
rows=json.loads((out/'sport_readouts.json').read_text())
for row in rows:
    if row['position'] in [16,19,25]:print('Sport prompt',row['position'],repr(row['token']),[r['token'] for r in row['far']])

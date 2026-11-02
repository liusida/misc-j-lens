# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Export the existing cosine matrix with every pair labeled as PNG."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(__file__).resolve().parent/'j_lens_output/paris_soccer_pair_cosines'
matrix=np.load(folder/'cosines.npy')
with (folder/'pairs.csv').open() as handle:
    rows=list(csv.DictReader(handle))
# Preserve the saved target-major matrix ordering; change tick visibility only.
ticks=[i for i,row in enumerate(rows) if int(row['target_position'])-int(row['source_position']) in (0,1)]
labels=[rows[i]['label'] for i in ticks]
fig,ax=plt.subplots(figsize=(48,48))
im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r',interpolation='nearest')
ax.set_xticks(ticks,labels,rotation=90,fontsize=5)
ax.set_yticks(ticks,labels,fontsize=5)
ax.tick_params(length=1,width=.2,pad=1)
ax.set_xlabel('Source L12 → target L23; * = same position; no * = future position')
ax.set_ylabel('Source L12 → target L23; * = same position; no * = future position')
ax.set_title('Paris: individual soccer Jacobian directions\nAll 1,378 pairs; original target-major order; ticks show same-position and next-position pairs only')
fig.colorbar(im,ax=ax,label='Cosine similarity',shrink=.8)
output=folder/'heatmap_same_and_next_ticks.png'
fig.savefig(output,dpi=250,bbox_inches='tight')
print(output)

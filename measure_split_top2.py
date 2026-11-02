# /// script
# dependencies = ["matplotlib"]
# ///
"""Compare exact token-ID top-2 matches for residual-post readouts."""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent/'j_lens_output/split_sky_comparison'
data = json.loads((ROOT/'readouts.json').read_text())
tokens = data['tokens']
results = []
for method, rows in data['lens_rows'].items():
    current = following = total = 0
    for row in rows:
        if row['site'] != 'residual-post':
            continue
        # Use identical positions for both metrics; the final position has no
        # observed next token. Include prompt and generated text, all 24 layers.
        for position in range(len(tokens)-1):
            ids = {item['id'] for item in row['readouts'][position][:2]}
            current += tokens[position]['id'] in ids
            following += tokens[position+1]['id'] in ids
            total += 1
    results.append(dict(lens=data['lens_labels'][method],current_top2=current,
                        next_top2=following,total=total))
with (ROOT/'top2_counts.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(results[0]))
    writer.writeheader();writer.writerows(results)
fig, ax = plt.subplots(figsize=(8,5),constrained_layout=True)
x = [0, 1]
keys = ['current_top2', 'next_top2']
for offset, r, color in zip([-.25, 0, .25], results, ['#3979b9', '#e38c36', '#479568']):
    heights = [r[key] for key in keys]
    bars = ax.bar([v+offset for v in x],heights,width=.23,label=r['lens'],color=color)
    ax.bar_label(bars,labels=[f"{r[key]:,}\n({r[key]/r['total']:.1%})" for key in keys],padding=4,fontsize=10)
ax.set_xticks(x,['Current', 'Next'])
ax.set_ylabel('Token–layer pairs with target in top 2')
ax.set_title(f"Current vs next token in top 2\n24 residual-post layers × 338 positions = {results[0]['total']:,} pairs per lens")
ax.legend();ax.set_ylim(0,max(r[k] for r in results for k in ['current_top2','next_top2'])*1.27)
ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'top2_comparison.png',dpi=180)
print(json.dumps(results,indent=2))
print(ROOT/'top2_comparison.png')

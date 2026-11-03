# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "matplotlib", "numpy"]
# ///
"""Replot saved responses, coloring random directions by mean magnitude rank."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK JP", "Droid Sans Fallback"]
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import torch

folder = Path(__file__).resolve().parents[2] / "artifacts/runs/oscillation/random_comparison"
random = torch.load(folder/"vectors.pt", map_location="cpu", weights_only=True)
top = torch.load(folder/"top1_responses.pt", map_location="cpu", weights_only=True)
top10 = torch.load(folder/"top10_responses.pt", map_location="cpu", weights_only=True) if (folder/"top10_responses.pt").exists() else None
token_cmap = LinearSegmentedColormap.from_list("red_to_pink", ["#d00000", "#f9b9cf"])
transported = torch.load(folder/"transported_responses.pt", map_location="cpu", weights_only=True) if (folder/"transported_responses.pt").exists() else None
metadata = json.loads((folder/"summary.json").read_text())
cmap = LinearSegmentedColormap.from_list("blue_to_yellow", ["#1546b3", "#21a7c4", "#f1cf22"])
fig, axes = plt.subplots(4, 1, figsize=(14, 15), layout="constrained")
ranks = []
for ax, (name, info) in zip(axes, metadata["prompts"].items()):
    norms = random["responses"][name].norm(dim=-1).numpy()
    means = norms[:,1:].mean(axis=1)
    order = np.argsort(-means)
    count = len(order)
    magnitude = top[name]["response"].norm(dim=1).numpy()
    lag = np.arange(1, norms.shape[1])
    mean_x = int(lag[-1]) + 4
    random_mean_norm = random["responses"][name][:,1:].mean(dim=1).norm(dim=-1).numpy()
    with (folder/f"{name}.csv").open() as f:
        labels = {int(r["lag"]): f"{r['target_token']!r}\n{r['target_position']}" for r in csv.DictReader(f)}
    single, sax = plt.subplots(figsize=(18, 5.5), layout="constrained")
    for rank, index in reversed(list(enumerate(order))):
        color = cmap(rank/(count-1))
        for target in [ax, sax]:
            target.semilogy(lag, norms[index,1:], color=color, alpha=0.75, lw=0.9)
            target.scatter(mean_x, random_mean_norm[index], color=color, alpha=0.75, s=18)
        ranks.append(dict(prompt=name,direction=int(index),rank=rank+1,mean_future_response_norm=float(means[index])))
    for target in [ax, sax]:
        target.semilogy(lag, np.median(norms,axis=0)[1:], color="black",lw=1.8,label="Random-50 median")
        target.scatter(mean_x, np.median(random_mean_norm), color="black", s=35)
        if top10 is not None:
            for entry in reversed(top10[name]):
                target.semilogy(lag, entry["response"].norm(dim=1).numpy()[1:], color=token_cmap((entry["rank"]-1)/9), lw=1.7, label=f"#{entry['rank']} {entry['token']!r}")
                target.scatter(mean_x, float(entry["response"][1:].mean(dim=0).norm()), color=token_cmap((entry["rank"]-1)/9), s=35)
        else:
            target.semilogy(lag, magnitude[1:], color="#d62728",lw=2,marker=".",label=f"Top-1: {top[name]['token']!r}")
        if transported is not None:
            target.semilogy(lag, transported[name]["response"].norm(dim=1).numpy()[1:], color="#8a2be2", lw=2, marker=".", label="Full transport: unit Jᵀ(Jh)")
            target.scatter(mean_x, float(transported[name]["response"][1:].mean(dim=0).norm()), color="#8a2be2", s=45)
        target.set(title=f"{name}: L12 position 17 ({info['source_token']!r})",xlabel="Future target lag",ylabel="Response norm (log scale)")
        target.grid(alpha=0.2); target.legend(ncol=3, fontsize=8)
        target.axvline(mean_x-2,color="gray",ls=":",lw=0.8)
        target.set_xticks(list(target.get_xticks())+[mean_x])
        ticks=target.get_xticks()
        target.set_xticks(ticks,["Mean\nresponse" if t==mean_x else f"{t:g}" for t in ticks])
        target.set_xlim(0,mean_x+1.5)
    sax.set_xticks(list(lag)+[mean_x],[labels[int(k)] for k in lag]+["Mean\nresponse"],rotation=65,ha="right",fontsize=8)
    sax.set_xlabel("Target token and absolute position")
    mapper = plt.cm.ScalarMappable(norm=Normalize(1,count), cmap=cmap)
    bar = single.colorbar(mapper,ax=sax,pad=0.01)
    bar.set_label("Mean magnitude rank: 1 = highest, 50 = lowest")
    for extension in ["png","pdf"]:
        suffix = "top10" if top10 is not None else "top1"
        single.savefig(folder/f"{name}_{suffix}_ranked.{extension}",dpi=170)
    plt.close(single)
mapper = plt.cm.ScalarMappable(norm=Normalize(1,50), cmap=cmap)
bar = fig.colorbar(mapper,ax=list(axes),pad=0.01,shrink=0.65)
bar.set_label("Rank by mean future response magnitude (within each prompt): 1 = highest, 50 = lowest")
for extension in ["png","pdf"]:
    suffix = "top10" if top10 is not None else "top1"
    fig.savefig(folder/f"{suffix}_vs_random50_ranked.{extension}",dpi=180)
with (folder/"random_magnitude_ranks.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(ranks[0])); writer.writeheader(); writer.writerows(ranks)
print(folder/f"{suffix}_vs_random50_ranked.png")

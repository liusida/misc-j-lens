# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "matplotlib", "numpy"]
# ///
"""Compare response directions to the full-transport response at the same lag."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import torch
import torch.nn.functional as F

plt.rcParams["font.family"]=["DejaVu Sans","Noto Sans CJK JP","Droid Sans Fallback"]
folder=Path(__file__).resolve().parent/"random_comparison"
random=torch.load(folder/"vectors.pt",map_location="cpu",weights_only=True)
top=torch.load(folder/"top10_responses.pt",map_location="cpu",weights_only=True)
full=torch.load(folder/"transported_responses.pt",map_location="cpu",weights_only=True)
metadata=json.loads((folder/"summary.json").read_text())
random_cmap=LinearSegmentedColormap.from_list("blue_yellow",["#1546b3","#21a7c4","#f1cf22"])
token_cmap=LinearSegmentedColormap.from_list("red_pink",["#d00000","#f9b9cf"])
fig,axes=plt.subplots(4,1,figsize=(15,15),layout="constrained")
rows=[]
for ax,(name,info) in zip(axes,metadata["prompts"].items()):
    reference=full[name]["response"]
    responses=random["responses"][name]
    cosines=F.cosine_similarity(responses,reference[None],dim=-1).numpy()
    means=responses.norm(dim=-1)[:,1:].mean(dim=1).numpy()
    order=np.argsort(-means)
    lag=np.arange(1,len(reference))
    mean_x=int(lag[-1])+4
    reference_mean=reference[1:].mean(dim=0)
    random_mean_cos=F.cosine_similarity(responses[:,1:].mean(dim=1),reference_mean[None],dim=-1).numpy()
    single,sax=plt.subplots(figsize=(18,5.5),layout="constrained")
    for rank,index in reversed(list(enumerate(order))):
        for target in [ax,sax]:
            target.plot(lag,cosines[index,1:],color=random_cmap(rank/(len(order)-1)),alpha=0.6,lw=0.8)
            target.scatter(mean_x,random_mean_cos[index],color=random_cmap(rank/(len(order)-1)),alpha=0.65,s=18)
        for k in lag:
            rows.append(dict(prompt=name,method="random",index=int(index),lag=int(k),cosine=float(cosines[index,k])))
    for target in [ax,sax]:
        target.plot(lag,np.median(cosines[:,1:],axis=0),color="black",lw=1.8,label="Random-50 median")
        target.scatter(mean_x,np.median(random_mean_cos),color="black",s=35)
    for entry in reversed(top[name]):
        cosine=F.cosine_similarity(entry["response"],reference,dim=-1).numpy()
        for target in [ax,sax]:
            target.plot(lag,cosine[1:],color=token_cmap((entry["rank"]-1)/9),lw=1.7,label=f"#{entry['rank']} {entry['token']!r}")
            target.scatter(mean_x,float(F.cosine_similarity(entry["response"][1:].mean(dim=0),reference_mean,dim=0)),color=token_cmap((entry["rank"]-1)/9),s=35)
        for k in lag:
            rows.append(dict(prompt=name,method=entry["token"],index=entry["rank"],lag=int(k),cosine=float(cosine[k])))
    for target in [ax,sax]:
        target.axhline(1,color="#8a2be2",lw=1.7,ls="--",label="Full-transport reference (self = 1)")
        target.axhline(0,color="gray",lw=0.6)
        target.set(title=f"{name}: cosine to full-transport response at each target; source L12:17 {info['source_token']!r}",xlabel="Future target lag",ylabel="Cosine similarity",ylim=(-1.05,1.05))
        target.grid(alpha=0.2);target.legend(ncol=3,fontsize=8)
        target.scatter(mean_x,1,color="#8a2be2",s=45)
        target.axvline(mean_x-2,color="gray",ls=":",lw=0.8)
        ticks=list(target.get_xticks())+[mean_x]
        target.set_xticks(ticks,["Mean\nresponse" if t==mean_x else f"{t:g}" for t in ticks])
        target.set_xlim(0,mean_x+1.5)
    with (folder/f"{name}.csv").open() as handle:
        labels={int(r["lag"]):f"{r['target_token']!r}\n{r['target_position']}" for r in csv.DictReader(handle)}
    sax.set_xticks(list(lag)+[mean_x],[labels[int(k)] for k in lag]+["Mean\nresponse"],rotation=65,ha="right",fontsize=8)
    sax.set_xlabel("Target token and absolute position")
    single.savefig(folder/f"{name}_response_cosines.png",dpi=170)
    plt.close(single)
bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1,50),cmap=random_cmap),ax=list(axes),pad=0.01,shrink=0.65)
bar.set_label("Random-direction rank by mean response norm: blue = highest, yellow = lowest")
for extension in ["png","pdf"]:
    fig.savefig(folder/f"response_cosines_to_full_transport.{extension}",dpi=180)
with (folder/"response_cosines.csv").open("w",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(folder/"response_cosines_to_full_transport.png")

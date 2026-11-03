# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch", "transformers>=5", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Measure first five reconstructed fitting inputs; render magnitude and cosine."""
from run import *
from matplotlib.colors import LinearSegmentedColormap, Normalize
import torch.nn.functional as F

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=5)
    parser.add_argument('--measure-only',action='store_true')
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    folder=Path(__file__).resolve().parents[2]/"artifacts/runs/oscillation/training_five"
    folder.mkdir(exist_ok=True)
    audit=json.loads((folder.parent/"training_prompt_audit/reconstructed_100_prompts.json").read_text())
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B",local_files_only=True)
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,"Qwen/Qwen3.5-2B",dtype=torch.float32,local_files_only=True)
    model=from_hf(hf,tokenizer)
    path=hf_hub_download("sida/llmdbg","workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors",repo_type="dataset",revision="980e34bc0f0460c437bb4fa0238a58b9b50158d3",local_files_only=True)
    J=load_file(path)["post.12"].cuda().float()
    generator=torch.Generator().manual_seed(42)
    random=F.normalize(torch.randn(50,model.d_model,generator=generator),dim=1).cuda()
    summaries=[]
    for prompt in audit["prompts"][:args.count]:
        index=prompt["prompt_index"]
        checkpoint=folder/f"prompt_{index}.pt"
        if checkpoint.exists():
            print(f"Reusing {checkpoint}",flush=True)
            continue
        ids=torch.tensor([prompt["input_ids"]],device="cuda")
        with ActivationRecorder(model.layers,at=[12]) as rec:
            model.forward(ids)
            h=rec.activations[12][0,17].clone()
        logits=model.unembed((J@h)[None])[0].float()
        top_ids=logits.topk(10).indices
        token_v=F.normalize((J.T@((1+model._final_norm.weight.float())*model._lm_head.weight[top_ids]).T).T,dim=1)
        full=F.normalize(J.T@(J@h),dim=0)
        directions=torch.cat([random,token_v,full[None]],dim=0)
        responses=[];errors=[]
        for start in range(0,len(directions),args.batch_size):
            batch=directions[start:start+args.batch_size]
            def evaluate(delta):
                def hook(module,inputs,output):
                    x=output if torch.is_tensor(output) else output[0]
                    y=x.clone();y[:,17]+=delta*batch
                    return y if torch.is_tensor(output) else (y,*output[1:])
                handle=model.layers[12].register_forward_hook(hook)
                try:
                    with ActivationRecorder(model.layers,at=[model.n_layers-1]) as rec:
                        model.forward(ids.expand(len(batch),-1))
                        return rec.activations[model.n_layers-1].clone()
                finally:
                    handle.remove()
            estimates=[]
            for epsilon in [0.01,0.005]:
                delta=epsilon*float(h.norm())
                estimates.append((evaluate(delta)-evaluate(-delta))/(2*delta))
            error=(estimates[0]-estimates[1]).flatten(1).norm(dim=1)/estimates[1].flatten(1).norm(dim=1)
            responses.append(estimates[-1][:,17:].cpu());errors.extend(error.cpu().tolist())
            print(f"Prompt {index}: {min(start+args.batch_size,61)}/61 directions",flush=True)
        result=dict(prompt=prompt,source_position=17,source_token=tokenizer.decode([prompt['input_ids'][17]]),tokens=[tokenizer.decode([q]) for q in prompt['input_ids']],top_tokens=[tokenizer.decode([q]) for q in top_ids.tolist()],top_ids=top_ids.cpu(),directions=directions.cpu(),responses=torch.cat(responses),epsilon_errors=errors)
        torch.save(result,checkpoint)
    if args.measure_only:
        return
    plt.rcParams['font.family']=['DejaVu Sans','Noto Sans CJK JP','Droid Sans Fallback']
    rcmap=LinearSegmentedColormap.from_list('blue_yellow',['#1546b3','#21a7c4','#f1cf22'])
    tcmap=LinearSegmentedColormap.from_list('red_pink',['#d00000','#f9b9cf'])
    figures={metric:plt.subplots(args.count,1,figsize=(17,4*args.count),layout='constrained') for metric in ['magnitude','cosine']}
    rows=[]
    for index in range(args.count):
        result=torch.load(folder/f'prompt_{index}.pt',weights_only=True)
        response=result['responses']; reference=response[-1]
        norms=response.norm(dim=-1).numpy()
        mean=response[:,1:].mean(dim=1)
        mean_norm=mean.norm(dim=-1).numpy()
        cos=F.cosine_similarity(response,reference[None],dim=-1).numpy()
        mean_cos=F.cosine_similarity(mean,mean[-1][None],dim=-1).numpy()
        lag=np.arange(1,response.shape[1]);mean_x=int(lag[-1])+5
        order=np.argsort(-norms[:50,1:].mean(axis=1))
        for metric,(fig,axes) in figures.items():
            ax=axes[index];values=norms if metric=='magnitude' else cos; averages=mean_norm if metric=='magnitude' else mean_cos
            for rank,k in reversed(list(enumerate(order))):
                color=rcmap(rank/49)
                ax.plot(lag,values[k,1:],color=color,alpha=0.6,lw=0.8)
                ax.scatter(mean_x,averages[k],color=color,s=16,alpha=0.7)
            ax.plot(lag,np.median(values[:50,1:],axis=0),color='black',lw=1.7,label='Random-50 median')
            ax.scatter(mean_x,np.median(averages[:50]),color='black',s=30)
            for k in reversed(range(10)):
                color=tcmap(k/9)
                ax.plot(lag,values[50+k,1:],color=color,lw=1.5,label=f"#{k+1} {result['top_tokens'][k]!r}")
                ax.scatter(mean_x,averages[50+k],color=color,s=30)
            ax.plot(lag,values[-1,1:],color='#8a2be2',lw=2,label='Full transport: unit Jᵀ(Jh)')
            ax.scatter(mean_x,averages[-1],color='#8a2be2',s=40)
            ax.axvline(mean_x-2.5,color='gray',ls=':',lw=0.8)
            ax.set(title=f"Fitting prompt {index} (dataset row {result['prompt']['dataset_row']}): L12 position 17 {result['source_token']!r}",xlabel='Future target lag',ylabel='Response norm (log)' if metric=='magnitude' else 'Cosine to full-transport response')
            if metric=='magnitude':ax.set_yscale('log')
            else:ax.set_ylim(-1.05,1.05);ax.axhline(0,color='gray',lw=0.6)
            ticks=list(range(0,int(lag[-1])+1,10))+[mean_x]
            ax.set_xticks(ticks,[str(t) if t!=mean_x else 'Mean\nresponse' for t in ticks]);ax.set_xlim(0,mean_x+2)
            ax.legend(ncol=4,fontsize=8);ax.grid(alpha=0.2)
        for k in range(61):
            label=f'random_{k}' if k<50 else result['top_tokens'][k-50] if k<60 else 'full_transport'
            for t in lag:
                rows.append(dict(prompt=index,direction=label,lag=int(t),target_position=int(t+17),target_token=result['tokens'][t+17],magnitude=float(norms[k,t]),cosine=float(cos[k,t])))
            rows.append(dict(prompt=index,direction=label,lag='mean',target_position='',target_token='',magnitude=float(mean_norm[k]),cosine=float(mean_cos[k])))
        summaries.append(dict(prompt_index=index,source_token=result['source_token'],token_count=len(result['tokens']),top_tokens=result['top_tokens'],max_epsilon_error=max(result['epsilon_errors'])))
    for metric,(fig,axes) in figures.items():
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1,50),cmap=rcmap),ax=list(axes),pad=0.01,shrink=0.6)
        bar.set_label('Random-direction mean magnitude rank: blue highest → yellow lowest')
        for extension in ['png','pdf']:fig.savefig(folder/f'{metric}.{extension}',dpi=170)
    with (folder/'measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (folder/'summary.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()

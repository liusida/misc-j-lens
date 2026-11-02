# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch", "transformers>=5", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Measure the top ten readout directions with the existing offline protocol."""
from run import *

@torch.no_grad()
def main():
    folder=Path(__file__).resolve().parent/"random_comparison"
    metadata=json.loads((folder/"summary.json").read_text())
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B",local_files_only=True)
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,"Qwen/Qwen3.5-2B",dtype=torch.float32,local_files_only=True)
    model=from_hf(hf,tokenizer)
    path=hf_hub_download("sida/llmdbg","workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors",repo_type="dataset",revision="980e34bc0f0460c437bb4fa0238a58b9b50158d3",local_files_only=True)
    J=load_file(path)["post.12"].cuda().float()
    gain=1+model._final_norm.weight.float()
    saved, checks={}, {}
    for name,info in metadata["prompts"].items():
        ids=model.encode(info["prompt"],max_length=512)
        with ActivationRecorder(model.layers,at=[12]) as rec:
            model.forward(ids)
            h=rec.activations[12][0,17].clone()
        logits=model.unembed((J@h)[None])[0].float()
        top_ids=logits.topk(10).indices.tolist()
        entries=[]
        for rank,q in enumerate(top_ids,1):
            v=J.T@(gain*model._lm_head.weight[q]); v/=v.norm()
            def evaluate(delta):
                def hook(module,inputs,output):
                    x=output if torch.is_tensor(output) else output[0]
                    y=x.clone(); y[:,17]+=delta*v
                    return y if torch.is_tensor(output) else (y,*output[1:])
                handle=model.layers[12].register_forward_hook(hook)
                try:
                    with ActivationRecorder(model.layers,at=[model.n_layers-1]) as rec:
                        model.forward(ids)
                        return rec.activations[model.n_layers-1][0].clone()
                finally:
                    handle.remove()
            estimates=[]
            for fraction in [metadata["epsilon"],metadata["epsilon"]/2]:
                delta=fraction*float(h.norm())
                estimates.append((evaluate(delta)-evaluate(-delta))/(2*delta))
            error=float((estimates[0]-estimates[1]).norm()/estimates[1].norm())
            entry=dict(rank=rank,token_id=q,token=tokenizer.decode([q]),logit=float(logits[q]),epsilon_relative_change=error,direction=v.cpu(),response=estimates[-1][17:].cpu())
            entries.append(entry)
            print(name,rank,repr(entry["token"]),f"epsilon error={error:.5f}",flush=True)
        saved[name]=entries
        checks[name]=[{k:v for k,v in e.items() if k not in ["direction","response"]} for e in entries]
    torch.save(saved,folder/"top10_responses.pt")
    (folder/"top10_summary.json").write_text(json.dumps(checks,indent=2)+"\n")

if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch", "transformers>=5", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Measure the unit hidden-state direction in all four cached experiments."""
from run import *

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=["hidden", "transported"], default="hidden")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[2] / "artifacts/runs/oscillation/random_comparison"
    metadata = json.loads((folder/"summary.json").read_text())
    random = torch.load(folder/"vectors.pt",map_location="cpu",weights_only=True)
    top = torch.load(folder/"top1_responses.pt",map_location="cpu",weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B",local_files_only=True)
    hf = from_pretrained_to_cuda(AutoModelForCausalLM,"Qwen/Qwen3.5-2B",dtype=torch.float32,local_files_only=True)
    model = from_hf(hf,tokenizer)
    if args.direction == "transported":
        path = hf_hub_download("sida/llmdbg", "workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors", repo_type="dataset", revision="980e34bc0f0460c437bb4fa0238a58b9b50158d3", local_files_only=True)
        J = load_file(path)["post.12"].cuda().float()
    saved, stats = {}, {}
    for name,info in metadata["prompts"].items():
        ids = model.encode(info["prompt"],max_length=512)
        with ActivationRecorder(model.layers,at=[12]) as rec:
            model.forward(ids)
            h = rec.activations[12][0,17].clone()
        raw_direction = J.T @ (J @ h) if args.direction == "transported" else h
        v = raw_direction/raw_direction.norm()
        def evaluate(delta):
            def hook(module,inputs,output):
                x = output if torch.is_tensor(output) else output[0]
                y = x.clone(); y[:,17] += delta*v
                return y if torch.is_tensor(output) else (y,*output[1:])
            handle = model.layers[12].register_forward_hook(hook)
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
        response=estimates[-1][17:]
        magnitudes=response.norm(dim=1).cpu().numpy()[1:]
        random_norm=random["responses"][name].norm(dim=-1).numpy()[:,1:]
        top_norm=top[name]["response"].norm(dim=1).numpy()[1:]
        stats[name]=dict(relative_epsilon_change=float((estimates[0]-estimates[1]).norm()/estimates[1].norm()),fraction_above_random_median=float(np.mean(magnitudes>np.median(random_norm,axis=0))),fraction_above_all_random=float(np.mean(magnitudes>random_norm.max(axis=0))),fraction_above_top_token=float(np.mean(magnitudes>top_norm)),median_ratio_to_random_median=float(np.median(magnitudes/np.median(random_norm,axis=0))),cosine_with_top_token_direction=float(v.cpu()@top[name]["direction"]))
        saved[name]=dict(direction=v.cpu(),response=response.cpu())
        print(name,json.dumps(stats[name]),flush=True)
    prefix = args.direction
    torch.save(saved,folder/f"{prefix}_responses.pt")
    (folder/f"{prefix}_comparison.json").write_text(json.dumps(stats,indent=2)+"\n")

if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch", "transformers>=5", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Overlay each prompt's top-1 lens direction on saved random-50 responses."""
from run import *

@torch.no_grad()
def main():
    folder = Path(__file__).resolve().parent / "random_comparison"
    metadata = json.loads((folder/"summary.json").read_text())
    saved = torch.load(folder/"vectors.pt", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, "Qwen/Qwen3.5-2B", dtype=torch.float32, local_files_only=True)
    model = from_hf(hf, tokenizer)
    path = hf_hub_download("sida/llmdbg", "workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors", repo_type="dataset", revision="980e34bc0f0460c437bb4fa0238a58b9b50158d3", local_files_only=True)
    J = load_file(path)["post.12"].cuda().float()
    figure, axes = plt.subplots(4, 1, figsize=(13, 15), layout="constrained")
    stats, response_data = {}, {}
    for ax, (name, info) in zip(axes, metadata["prompts"].items()):
        ids = model.encode(info["prompt"], max_length=512)
        with ActivationRecorder(model.layers, at=[12]) as rec:
            model.forward(ids)
            h = rec.activations[12][0, 17].clone()
        logits = model.unembed((J@h)[None])[0].float()
        q = int(logits.argmax())
        word = tokenizer.decode([q])
        v = J.T @ ((1+model._final_norm.weight.float())*model._lm_head.weight[q])
        v /= v.norm()
        def evaluate(delta):
            def hook(module, inputs, output):
                x = output if torch.is_tensor(output) else output[0]
                y = x.clone(); y[:,17] += delta*v
                return y if torch.is_tensor(output) else (y,*output[1:])
            handle = model.layers[12].register_forward_hook(hook)
            try:
                with ActivationRecorder(model.layers, at=[model.n_layers-1]) as rec:
                    model.forward(ids)
                    return rec.activations[model.n_layers-1][0].clone()
            finally:
                handle.remove()
        estimates = []
        for fraction in [metadata["epsilon"], metadata["epsilon"]/2]:
            delta = fraction*float(h.norm())
            estimates.append((evaluate(delta)-evaluate(-delta))/(2*delta))
        response = estimates[-1][17:]
        magnitude = response.norm(dim=1).cpu().numpy()
        random_norms = saved["responses"][name].norm(dim=-1).numpy()
        median = np.median(random_norms,axis=0)
        lag = np.arange(len(magnitude))
        for curve in random_norms:
            ax.semilogy(lag[1:],curve[1:],color="#2878B5",alpha=0.15)
        ax.semilogy(lag[1:],median[1:],color="black",lw=2,label="Random-50 median")
        ax.semilogy(lag[1:],magnitude[1:],color="#d62728",lw=2,marker=".",label=f"Top-1 direction: {word!r}")
        ax.set(title=f"{name}: L12 source position 17, {info['source_token']!r}",xlabel="Future target lag",ylabel="Response norm (log scale)")
        ax.legend(); ax.grid(alpha=0.2)
        row = dict(top_token=word, fraction_future_positions_above_random_median=float(np.mean(magnitude[1:]>median[1:])), median_future_ratio_to_random_median=float(np.median(magnitude[1:]/median[1:])), epsilon_relative_change=float((estimates[0]-estimates[1]).norm()/estimates[1].norm()))
        stats[name] = row
        response_data[name] = dict(direction=v.cpu(), response=response.cpu(), token=word)
        print(name,json.dumps(row),flush=True)
        # Separate token-labelled figure for this prompt.
        tokens=[tokenizer.decode([int(x)]) for x in ids[0]]
        single, sax = plt.subplots(figsize=(17,5.5),layout="constrained")
        for curve in random_norms:
            sax.semilogy(lag[1:],curve[1:],color="#2878B5",alpha=0.15)
        sax.semilogy(lag[1:],median[1:],color="black",label="Random-50 median",lw=2)
        sax.semilogy(lag[1:],magnitude[1:],color="#d62728",label=f"Top-1: {word!r}",marker=".",lw=2)
        sax.set_xticks(lag[1:],[f"{tokens[t]!r}\n{t}" for t in range(18,len(tokens))],rotation=65,ha="right",fontsize=8)
        sax.set(title=f"{name}: top-1 direction versus 50 random unit directions",xlabel="Target token and position",ylabel="Response norm (log scale)")
        sax.legend(); sax.grid(alpha=0.2)
        single.savefig(folder/f"{name}_top1_overlay.png",dpi=170)
        plt.close(single)
    for extension in ["png","pdf"]:
        figure.savefig(folder/f"top1_vs_random50.{extension}",dpi=180)
    (folder/"top1_comparison.json").write_text(json.dumps(stats,indent=2)+"\n")
    torch.save(response_data,folder/"top1_responses.pt")

if __name__ == "__main__":
    main()

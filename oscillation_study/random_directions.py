# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch", "transformers>=5", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Offline FP32 CUDA comparison of 50 shared isotropic random directions."""
from run import *
from other_prompts import PROMPTS

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    folder = Path(__file__).resolve().parent / "random_comparison"
    folder.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, "Qwen/Qwen3.5-2B", dtype=torch.float32, local_files_only=True)
    model = from_hf(hf, tokenizer)
    revision = "980e34bc0f0460c437bb4fa0238a58b9b50158d3"
    path = hf_hub_download("sida/llmdbg", "workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors", repo_type="dataset", revision=revision, local_files_only=True)
    J = load_file(path)["post.12"].cuda().float()
    generator = torch.Generator().manual_seed(args.seed)
    directions = torch.randn(args.count, model.d_model, generator=generator)
    directions = torch.nn.functional.normalize(directions, dim=1).cuda()
    passages = {"paris": PROMPT, **PROMPTS}
    all_stats, saved = {}, {}
    overview, overview_axes = plt.subplots(4, 1, figsize=(13, 15), layout="constrained")
    for overview_ax, (name, prompt) in zip(overview_axes, passages.items()):
        ids = model.encode(prompt, max_length=512)
        tokens = [tokenizer.decode([int(x)]) for x in ids[0]]
        position, layer, last = 17, 12, model.n_layers-1
        with ActivationRecorder(model.layers, at=[layer]) as rec:
            model.forward(ids)
            hnorm = float(rec.activations[layer][0, position].norm())
        responses, stats, rows = [], [], []
        for index, v in enumerate(directions):
            def evaluate(delta):
                def hook(module, inputs, output):
                    x = output if torch.is_tensor(output) else output[0]
                    y = x.clone()
                    y[:, position] += delta*v
                    return y if torch.is_tensor(output) else (y, *output[1:])
                handle = model.layers[layer].register_forward_hook(hook)
                try:
                    with ActivationRecorder(model.layers, at=[last]) as rec:
                        model.forward(ids)
                        return rec.activations[last][0].clone()
                finally:
                    handle.remove()
            estimates = []
            for fraction in [args.epsilon, args.epsilon/2]:
                delta = fraction*hnorm
                estimates.append((evaluate(delta)-evaluate(-delta))/(2*delta))
            r = estimates[-1][position:]
            error = float((estimates[0]-estimates[1]).norm()/estimates[1].norm())
            magnitude = r.norm(dim=1).cpu().numpy()
            projection = (r @ torch.nn.functional.normalize(J@v, dim=0)).cpu().numpy()
            lag = np.arange(len(magnitude))
            slope, intercept = np.polyfit(lag[1:], np.log(magnitude[1:]), 1)
            predicted = intercept+slope*lag[1:]
            log_y = np.log(magnitude[1:])
            r2 = 1-np.sum((log_y-predicted)**2)/np.sum((log_y-log_y.mean())**2)
            stats.append(dict(direction=index, relative_epsilon_change=error, slope=float(slope), log_r_squared=float(r2), decay_length=float(-1/slope) if slope<0 else None))
            for k in lag:
                rows.append(dict(direction=index, lag=int(k), target_position=int(k+position), target_token=tokens[k+position], response_norm=float(magnitude[k]), signed_projection=float(projection[k])))
            responses.append(r.cpu())
            if (index+1)%10 == 0:
                print(f"{name}: {index+1}/{args.count}, epsilon error={error:.4g}", flush=True)
        norms = torch.stack(responses).norm(dim=-1).numpy()
        projections = np.array([[row["signed_projection"] for row in rows if row["direction"]==i] for i in range(args.count)])
        fig, axes = plt.subplots(3, 1, figsize=(17, 12), layout="constrained")
        for ax, values, start in [(axes[0], projections, 0), (axes[1], projections, 1), (axes[2], norms, 1)]:
            for curve in values:
                ax.plot(lag[start:], curve[start:], color="#2878B5", alpha=0.18, lw=0.8)
            ax.plot(lag[start:], np.median(values, axis=0)[start:], color="black", lw=2, label="Median of 50 directions")
            ax.set_xticks(lag, [f"{tokens[t]!r}\n{t}" for t in range(position,len(tokens))], rotation=65, ha="right", fontsize=8)
            ax.set_xlabel("Target token and absolute position")
            ax.legend(); ax.grid(alpha=0.2)
        axes[0].set(title=f"{name}: 50 random directions, source L12:{position} {tokens[position]!r}", ylabel="Signed projection onto normalized Jv")
        axes[1].set(title="Future positions only", ylabel="Signed projection onto normalized Jv")
        axes[0].axhline(0, color="gray", lw=0.6)
        axes[1].axhline(0, color="gray", lw=0.6)
        axes[2].set_yscale("log")
        axes[2].set(title="Future response magnitude (log scale)", ylabel="Response norm")
        median = np.median(norms, axis=0)
        slope, intercept = np.polyfit(lag[1:], np.log(median[1:]), 1)
        axes[2].plot(lag[1:], np.exp(intercept+slope*lag[1:]), "--", color="orange", label="Exponential fit to median")
        axes[2].legend()
        for extension in ["png", "pdf"]:
            fig.savefig(folder/f"{name}.{extension}", dpi=160)
        plt.close(fig)
        for curve in norms:
            overview_ax.semilogy(lag[1:], curve[1:], color="#2878B5", alpha=0.15)
        overview_ax.semilogy(lag[1:], median[1:], color="black", label="Median")
        overview_ax.semilogy(lag[1:], np.exp(intercept+slope*lag[1:]), "--", color="orange", label="Exponential fit to median")
        overview_ax.set(title=f"{name}: source token {tokens[position]!r}", xlabel="Future target lag", ylabel="Response magnitude (log)")
        overview_ax.legend(); overview_ax.grid(alpha=0.2)
        with (folder/f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        all_stats[name] = dict(prompt=prompt, source_token=tokens[position], results=stats,
                              r_squared_median=float(np.median([s["log_r_squared"] for s in stats])),
                              r_squared_range=[float(min(s["log_r_squared"] for s in stats)),float(max(s["log_r_squared"] for s in stats))],
                              max_epsilon_error=max(s["relative_epsilon_change"] for s in stats))
        saved[name] = torch.stack(responses)
        print(name, all_stats[name]["r_squared_median"], flush=True)
    overview.savefig(folder/"all_prompts.png", dpi=180)
    plt.close(overview)
    torch.save(dict(directions=directions.cpu(), responses=saved),folder/"vectors.pt")
    (folder/"summary.json").write_text(json.dumps(dict(seed=args.seed,count=args.count,epsilon=args.epsilon,dtype="float32",prompts=all_stats),indent=2)+"\n")

if __name__ == "__main__":
    main()

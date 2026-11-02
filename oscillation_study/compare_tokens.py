# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch", "transformers>=5", "huggingface-hub", "safetensors", "matplotlib", "numpy"]
# ///
"""Compare downstream decay for multiple token directions, offline on CUDA."""
from run import *

@torch.no_grad()
def compare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", default=[" centuries", " banana", " quantum", " guitar", " legal", " swimming"])
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "comparison")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    name = "Qwen/Qwen3.5-2B"
    revision = "980e34bc0f0460c437bb4fa0238a58b9b50158d3"
    path = hf_hub_download("sida/llmdbg", "workspace-lenses/v2/Qwen--Qwen3.5-2B/j/lens.safetensors", repo_type="dataset", revision=revision, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, name, dtype=torch.float32, local_files_only=True)
    model = from_hf(hf, tokenizer)
    ids = model.encode(args.prompt, max_length=512)
    tokens = [tokenizer.decode([int(x)]) for x in ids[0]]
    layer, position, last = 12, 17, model.n_layers-1
    with ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(ids)
        h = rec.activations[layer][0, position].clone()
    J = load_file(path)["post.12"].cuda().float()
    gain = 1 + model._final_norm.weight.float()
    fig, axes = plt.subplots(3, 1, figsize=(17, 12), layout="constrained")
    rows, statistics, raw = [], [], {}
    for word in args.tokens:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) != 1:
            raise ValueError(f"Not one token: {word!r}")
        v = J.T @ (gain * model._lm_head.weight[word_ids[0]])
        v /= v.norm()
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
        for epsilon in [args.epsilon, args.epsilon/2]:
            delta = epsilon*float(h.norm())
            estimates.append((evaluate(delta)-evaluate(-delta))/(2*delta))
        response = estimates[-1][position:]
        relative_error = float((estimates[0]-estimates[1]).norm()/estimates[1].norm())
        ref = torch.nn.functional.normalize(J@v, dim=0)
        signed = (response@ref).cpu().numpy()
        magnitude = response.norm(dim=1).cpu().numpy()
        lag = np.arange(len(magnitude))
        # Fit only future positions; self-position residual path is qualitatively different.
        slope, intercept = np.polyfit(lag[1:], np.log(magnitude[1:]), 1)
        prediction_log = intercept+slope*lag[1:]
        residual = np.log(magnitude[1:])-prediction_log
        r2 = 1-np.sum(residual**2)/np.sum((np.log(magnitude[1:])-np.log(magnitude[1:]).mean())**2)
        stat = dict(token=word, epsilon_relative_change=relative_error, log_magnitude_slope=float(slope), log_space_r_squared=float(r2), e_folding_tokens=float(-1/slope) if slope<0 else None)
        statistics.append(stat)
        print(json.dumps(stat), flush=True)
        axes[0].plot(lag, signed, marker=".", label=repr(word))
        axes[1].plot(lag[1:], signed[1:], marker=".", label=repr(word))
        line, = axes[2].semilogy(lag[1:], magnitude[1:], marker=".", label=repr(word))
        axes[2].semilogy(lag[1:], np.exp(prediction_log), "--", color=line.get_color(), alpha=0.6)
        for k in lag:
            rows.append(dict(token_direction=word, target_position=int(k+position), target_token=tokens[k+position], lag=int(k), projection=float(signed[k]), response_norm=float(magnitude[k])))
        raw[word] = response.cpu()
    labels = [f"{tokens[t]!r}\n{t}" for t in range(position, len(tokens))]
    for ax in axes:
        ax.legend(ncol=3)
        ax.grid(alpha=0.2)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=8)
        ax.set_xlabel("Target token and absolute position (source = L12, position 17)")
    axes[0].set(title="Six token directions: signed response, including source position", ylabel="Projection onto Jv / ‖Jv‖")
    axes[1].set(title="Future positions only: signed response", ylabel="Projection onto Jv / ‖Jv‖")
    axes[2].set(title="Future response magnitude, log scale; dashed = exponential fits", ylabel="‖K[t′,17]v‖ (log scale)")
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[1].axhline(0, color="gray", lw=0.6)
    for extension in ["png", "pdf"]:
        fig.savefig(output/f"token_comparison.{extension}", dpi=170)
    with (output/"responses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/"decay_summary.json").write_text(json.dumps(dict(dtype="float32", epsilon=args.epsilon, lens_revision=revision, prompt=args.prompt, source_position=position, source_token=tokens[position], results=statistics), indent=2)+"\n")
    torch.save(raw, output/"response_vectors.pt")

if __name__ == "__main__":
    compare()

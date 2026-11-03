# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Repeat the offline CUDA direction comparison on three fixed passages."""
import json
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[1] / "artifacts/runs/oscillation/other_prompts"
PROMPTS = {
    "forest": "Forests regulate climate, protect watersheds, and provide habitats for countless species. A mature forest contains layers ranging from the shaded floor to the canopy, with fungi, insects, birds, and large mammals participating in interconnected cycles of nutrients and energy. Rainfall and seasonal changes influence how these communities develop over time.",
    "software": "A computer program is a sequence of instructions expressed in a language that a machine can execute. Large systems are usually divided into modules with clear interfaces, allowing programmers to test each part, locate errors, and change behavior without rewriting everything. Good documentation explains the assumptions and makes future maintenance easier for the whole team.",
    "cooking": "To prepare a simple vegetable soup, first wash the carrots, potatoes, onions, and celery. Cut the vegetables into small pieces and heat some oil in a large pot. Cook the onions until soft, then add the remaining vegetables and enough water to cover them. Simmer gently until everything is tender, season with salt and pepper, and serve warm with bread.",
}

def main():
    summaries = {}
    for name, prompt in PROMPTS.items():
        folder = OUTPUT / name
        print(f"Running {name}", flush=True)
        subprocess.run(["uv", "run", str(HERE / "compare_tokens.py"), "--prompt", prompt, "--output", str(folder)], check=True)
        summaries[name] = json.loads((folder / "decay_summary.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), layout="constrained")
    import csv
    for ax, (name, summary) in zip(axes, summaries.items()):
        with (OUTPUT/name/"responses.csv").open() as f:
            rows = list(csv.DictReader(f))
        for stat in summary["results"]:
            selected = [r for r in rows if r["token_direction"] == stat["token"] and int(r["lag"])>0]
            x = np.array([int(r["lag"]) for r in selected])
            y = np.array([float(r["response_norm"]) for r in selected])
            line, = ax.semilogy(x, y, label=f"{stat['token'].strip()} (R²={stat['log_space_r_squared']:.2f})")
            slope, intercept = np.polyfit(x, np.log(y), 1)
            ax.semilogy(x, np.exp(intercept+slope*x), "--", color=line.get_color(), alpha=0.55)
        ax.set(title=f"{name}: source L12 position 17, {summary['source_token']!r}", xlabel="Future target lag", ylabel="Response magnitude (log scale)")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(OUTPUT/"decay_comparison.png", dpi=180)
    (OUTPUT/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")

if __name__ == "__main__":
    main()

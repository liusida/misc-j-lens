# Downstream oscillation experiment

Run from the project root:

```bash
uv run experiments/oscillation/run.py
```

Uses CUDA and the locally cached `sida/llmdbg` v2 Qwen3.5-2B J-Lens, pinned
to revision `980e34bc0f0460c437bb4fa0238a58b9b50158d3`. Hugging Face
offline mode is enforced. Default query: block-output layer 12, position 17.

Two unit input directions are tested: the original hidden state h, and the
top decoded token's direction J-transpose times its final readout vector.
The first method uses the continuous hidden state without choosing a word;
J h is its expected average output direction. J h itself is in final-layer
coordinates and is therefore used as an output reference, not injected at L12.

For each direction v, symmetric perturbations estimate K[t',t]v at every
current and future target position. We run epsilon and epsilon/2 to expose
finite-difference sensitivity; BF16 rounding and nonlinear effects can matter.
Try `--epsilon 0.05` and `--epsilon 0.2` to assess stability. This measures
local sensitivity, not the origin or full causal meaning of the activation.

Outputs are stored under `artifacts/runs/oscillation/output/` (with a local
compatibility symlink): a plot, per-position CSV,
raw response vectors, pairwise cosine matrices, frequency curves, and metadata.
Cosine near +1 means effects point together; -1 means they point oppositely.
Near-zero response vectors have unreliable cosine values. The frequency plot
uses the norm of the complex weighted mean of response vectors. A peak alone
does not establish periodicity: decay, short windows and irregular context can
also produce spectral structure. Repeat over prompts for stronger evidence.

`--token ' centuries'` overrides automatic top-token selection.
`--prompt`, `--position`, `--layer`, and `--output` customize the experiment.
The word projection in the CSV is before RMS normalization; it is not a
derivative of the final token probability.

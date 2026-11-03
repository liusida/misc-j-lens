# J-Lens research workspace

Experiments and reusable tooling for Jacobian Lens (J-Lens), relevance-based
R-Lens, temporal target splits, activation decoding, and causal interventions
on Qwen models.

## Layout

- `src/jlens_study/`: reusable fitting and path helpers;
- `scripts/fit/`: supported fitting entry points;
- `scripts/decode/`: supported decoding and comparison entry points;
- `experiments/`: self-contained studies grouped by research question;
- `artifacts/`: generated lenses, checkpoints, reports, and comparisons;
- `docs/`: protocols, research notes, and blog material;
- `references/`: papers, reading list, and the separate upstream checkout;
- `debugger_examples/`: `.llmdbg.py` examples.

Large generated tensors and run outputs under `artifacts/` are ignored. The
small JSON provenance records for the current R-Lens fit are intentionally
tracked.

## Current workflows

Fit matched Normal, Same-token, and Future-only lenses:

```bash
uv run scripts/fit/fit_split_lenses.py --method j
uv run scripts/fit/fit_split_lenses.py --method r --exclude-mid --prompts 87
```

Generate the six-way J-Lens/R-Lens decoding comparison:

```bash
uv run scripts/decode/compare_j_r_split_lenses.py
```

The resulting page is written to
`artifacts/comparisons/j_r_split_l12/comparison.html`.

See [the split-fitting protocol](docs/split-lens-fitting.md) and the READMEs
inside individual experiment directories for methodological details.

## Artifact location

`src/jlens_study/paths.py` defines canonical paths. Set
`JLENS_ARTIFACT_ROOT=/another/path` to keep large artifacts outside this
checkout. Experiment scripts use the canonical `artifacts/` and `references/`
locations directly; no compatibility links are required.

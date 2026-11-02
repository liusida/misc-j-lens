# Single-Jacobian replication

Question: does one unrelated same-position Jacobian reliably improve agreement
with the trained J-Lens over the ordinary logit lens?

- Qwen3.5-2B; source layer outputs 0, 3, 6, 9, 12, 15, 18, 21; final output 23.
- Ten fixed source passages; three positions per passage sampled with seed
  20261031, excluding positions before 16 and the final position.
- Ten distinct evaluation passages, shared across all matrices and layers.
- Each of 240 Jacobians is evaluated independently. No Jacobians are averaged.
- Readouts use the model's final normalization and unembedding, with the cached
  pinned trained lens's matching `post.L` matrix as reference.
- CSVs record every trained top token's rank. Summary reports reciprocal rank,
  median rank and top-10/100/1000 coverage, also excluding the first 16 positions.
  Reciprocal-rank and coverage summaries weight evaluation passages equally.
- BF16 ties use competition ranks. Agreement with the trained lens is the
  criterion, not independent semantic ground truth.

Run from the project root:

```bash
uv run single_jacobian_study/run.py
```

The queued launch waits on the existing fitter's `run.lock`. It starts when
that lock is released (including if the fitter exits with an error). Individual
matrices and completed CSVs are reused on restart. Outputs stay in `results/`.

## Next-position replication

```bash
uv run single_jacobian_study/run.py --target-lag 1
uv run single_jacobian_study/plot_target_lags.py
```

This changes only the differentiated target position from `t` to `t+1`:
`d h_final[t+1] / d h_layer[t]`. The supplied next token remains fixed during
differentiation. Seeds, source positions, prompts, trained reference and metrics
match the original experiment. Outputs and checkpoints are separate in
`results_next/`; the original results are preserved. Matched plots show both
variants and the logit lens, including the positions ≥16 view.

Hypothesis: next-position Jacobians may help earlier layers more than
same-position Jacobians. This is a prediction to test, not an assumed outcome.

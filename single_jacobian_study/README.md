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

## One-token (sentence-initial) Jacobians

Question: if the entire prompt is a single token—the same object as a
sentence’s first position—does that same-position Jacobian still beat the
logit lens under the metrics above?

```bash
export HF_HOME=/cmlscratch/feijhan/.cache/huggingface
uv run single_jacobian_study/run_one_token.py
uv run single_jacobian_study/next_token_accuracy_one_token.py
uv run single_jacobian_study/plot_one_token_blog.py
```

- Thirty distinct length-1 source prompts: the first token of each original
  source passage plus twenty additional sentence-initial tokens.
- Differentiate only \(J=\partial h_{23,0}/\partial h_{\ell,0}\). No averaging.
- Same ten evaluation passages, trained-lens reference, competition ranks,
  passage-equal MRR / top-\(k\) coverage, and next-token accuracy plots as the
  main study. Outputs live in `results_one_token/` and
  `next_token_accuracy_one_token/`; blog figures go to `../blog-post/`.

## Matched-token first / middle / last

Same measured token id \(T\); only left context changes (causal same-position
\(J\) ignores tokens after \(T\)).

```bash
export HF_HOME=/cmlscratch/feijhan/.cache/huggingface
# special-token filler (default <|endoftext|>)
uv run single_jacobian_study/run_position_compare.py --filler special
uv run single_jacobian_study/plot_position_compare.py --results results_position_special --tag special

# natural-text prefixes instead of specials
uv run single_jacobian_study/run_position_compare.py --filler text
uv run single_jacobian_study/plot_position_compare.py --results results_position_text --tag text
```

Roles: `first` prefix 0; `middle` prefix 31 (+ suffix 31 specials); `last`
prefix 63. Override filler with `--special-token '<|im_start|>'`.

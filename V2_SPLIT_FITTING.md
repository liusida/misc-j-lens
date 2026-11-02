# V2 same-token and future-only fitting

Run from this project root:

```bash
uv run fit_v2_split_lenses.py
```

The run fits all 48 v2 sites: `embed`, `mid.0`–`mid.23`, and
`post.0`–`post.22`. The target is `post.23`. It uses the first 100 nonempty
training documents of `NeelNanda/pile-10k`, in dataset order, truncated to 128
tokens; positions 0–3 and the last position are excluded. The prompt selection
matches v2's 12,203 sequence tokens and 11,703 valid positions.

- Model/tokenizer revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Dataset revision: `127bfedcd5047750df5ccf3a12979a47bfa0bafa`.
- CUDA, BF16 model, native SDPA forward, dimension batch 16, FP32 accumulation.
- All 2048 output coordinates are differentiated; no random projections.

For each prompt, average over valid source positions; at each source sum over
the selected valid targets. Finally average the per-prompt matrices equally.
Normal selects `target >= source`, Same-token selects `target == source`, and
Future-only selects `target > source`. Future-only is computed as the matched
FP32 Normal mean minus the Same-token mean. It is not renormalized by pair count.

`reference_v2_fitter.py` is a verbatim local copy of the original llmdbg v2
fitter. Both passes call its `prompt_jacobians` unchanged. For the Same-token
pass only, a context manager suppresses cross-token derivatives in causal token
mixers. The diagonal block of a product of causal Jacobians is the product of
their diagonal blocks, so this retains the exact mathematical same-token
derivatives. All native forward values, including SDPA, remain unchanged.
The equivalent recurrent FP32 adjoint changes ordinary BF16 rounding versus
the native chunk backward; validation records this numerical difference.

Validate without fitting:

```bash
uv run fit_v2_split_lenses.py --validate-only
uv run fit_v2_split_lenses.py --validate-fp32
```

Validation compares six sites at three positions against isolated-target
ordinary autodiff, checks identical forward values, and checks zero cross-token
gradients. The FP32 check tests derivative equivalence separately from BF16
rounding. Fitting always uses BF16.

Outputs are under `j_lens_output/v2_split_lenses/`:

- `same_token/lens.safetensors` and `future_only/lens.safetensors`: requested
  v2-format lenses, with manifests and the same site keys as v2.
- `normal/lens.safetensors`: matched Normal control.
- `fit.checkpoint.pt`: FP32 accumulated sums, completed count, metrics, and
  configuration. Restarting the same command resumes automatically.
- `pending_*.pt`: completed half of a prompt, reused after interruption; removed
  only after that prompt's combined checkpoint is saved.
- `status.json`, `metrics.json`: progress and per-site similarity/update metrics.
- `config.json`, `prompts.json`, `validation*.json`: protocol and checks.

Published lens files contain the current running mean. Their manifests say
`complete: false` until all 100 prompts are finished. Subtraction is performed
before FP16 export, so separately rounded artifacts need not sum bit-for-bit.

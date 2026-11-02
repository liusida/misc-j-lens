# One-prompt J-Lens demonstration

This repository contains a deliberately small, inspectable reproduction of one
training sample for one layer of the Jacobian lens (J-Lens), using
`Qwen/Qwen3.5-2B`.

## Run

The script is a PEP 723 `uv` script, so `uv` creates an isolated environment
from the dependency block at the top of the file:

```bash
uv run j_lens_one_prompt.py
```

The default prompt is a sufficiently long, pretraining-like paragraph rather
than a chat template. It must leave positions after the official default
16-token attention-sink exclusion:

> Paris is the capital and largest city of France. It stands on the Seine ...

Useful options:

```bash
uv run j_lens_one_prompt.py --layer 12 --vjp-batch-size 8
uv run j_lens_one_prompt.py --prompt "The quick brown fox jumps over the lazy"
```

Lower `--vjp-batch-size` if the batched backward pass runs out of memory. It
changes only performance, not the result.

## Exactly what is averaged

The script imports Anthropic's adapter, position mask, and activation recorder
directly from the cloned official repository in `anthropic-jacobian-lens/`.
Like that implementation, layer numbers refer to **transformer block outputs**,
and the target is the last block output before the final RMSNorm.

Let `h_layer[t, j]` be coordinate `j` at source token position `t`, and let
`h_target[t_prime, i]` be output coordinate `i` at a target-layer position.
The script first records the raw tensor

```text
G[i, t, j] = d sum_{valid t_prime} h_target[t_prime, i] / d h_layer[t, j]
```

Because the model is causal, terms with `t_prime < t` are zero. Thus the sum
contains the current and future positions allowed by the paper. This is the
raw, before-position-average object in the paper's reproduction pseudocode.

Following the official defaults, positions 0–15 are excluded as attention
sinks, and the final position is excluded because it has no next-token target.
The full `G` tensor retains every source position so you can see the excluded
rows too. The script then performs the two averaging stages:

```text
J_prompt[i, j] = mean over valid source position t of G[i, t, j]
J_layer[i, j]  = mean over prompts of J_prompt[prompt, i, j]
```

There is one prompt here, so the second mean is an identity. A real lens uses a
large prompt corpus (the paper's default is 1,000 sequences of 128 tokens).
This run is educational, not a statistically useful trained lens.

Note that the paper's published pseudocode **sums** target positions and
**averages** source positions. The mathematical expectation notation can make
this implementation detail easy to miss.

## Output data

`j_lens_output/one_prompt_layer.pt` is a `torch.save` dictionary containing:

- `raw_grads_before_position_mean`: full `G`, shaped `[D, T, D]`, stored as
  bfloat16;
- `j_prompt_after_position_mean`: `[D, D]`, averaged in float32;
- `j_layer_after_prompt_mean`: `[D, D]` (identical for this one-prompt run);
- source and final residual-stream activations, token IDs, decoded tokens, and
  the top J-Lens readout tokens for the final prompt position.

`j_lens_output/summary.json` records shapes, formulas, the readout, and a
numerical reconstruction check. Load and inspect a particular raw datum with:

```bash
uv run inspect_j_lens_data.py --output-dim 100 --input-dim 200
```

That prints every token-position value contributing to one scalar entry and
then prints both its manual mean and the stored matrix entry. Equivalently, in
Python:

```python
import torch

data = torch.load("j_lens_output/one_prompt_layer.pt", weights_only=False)
G = data["raw_grads_before_position_mean"]
J = data["j_prompt_after_position_mean"]
print(G[:, 0, :])       # all raw derivatives from source token position 0
print(G[100, :, 200])   # output dim 100 vs every source position, input dim 200
print(J[100, 200])      # mean of the preceding values
valid = data["valid_positions"]
print(G[100, valid, 200].float().mean())
```

## Relationship to a usable J-Lens

For a real J-Lens, repeat the per-prompt computation over a representative
corpus and average the resulting `[D, D]` matrices elementwise. The lens applied
to an activation is then approximately

```text
softmax(W_U @ final_norm(J_layer @ h_layer))
```

The script performs this final readout too, but a one-prompt Jacobian will be
strongly specific to its single context.

References are listed in `reference.txt`; the implementation follows the
reproduction pseudocode in *Verbalizable Representations Form a Global
Workspace in Language Models*.

The official reference code is cloned at `anthropic-jacobian-lens/` from
<https://github.com/anthropics/jacobian-lens>. In particular, compare
`jlens/fitting.py`, `jlens/hooks.py`, and `jlens/hf.py` with this demonstration.

# J-Lens: supplied text vs model-generated continuations

**Question:** Does fitting J-Lens on the model’s own generated continuations better explain its behavior during answering than fitting on supplied text?

Final hidden states contain next-token prediction information. Jacobians capture how activation perturbations change these states and predictions. Computing Jacobians on generated text still holds its tokens fixed; it does not capture a perturbation changing sampled tokens and branching into a different continuation.

**Test:** Use matched starting prompts and token budgets. Fit one lens on supplied continuations and another on model-generated continuations. Keep layer, masking, and averaging settings identical.

**Compare:** On held-out generated answers, measure readout stability, agreement between lenses, and prediction of small intervention effects. Inspect examples such as `h₁₂,₁₇ → " centuries"`.

**Hypothesis:** Generated continuations may better match answering contexts; supplied text may provide broader coverage. Neither is assumed better in advance.

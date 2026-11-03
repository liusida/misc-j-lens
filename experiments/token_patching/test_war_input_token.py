# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0"]
# ///
"""Compare input-token replacement with baseline and saved hidden-state patch."""
import json
import sys
from pathlib import Path
import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "references/upstream/anthropic-jacobian-lens"))
from jlens import from_hf


@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    hf = from_pretrained_to_cuda(AutoModelForCausalLM, "Qwen/Qwen3.5-2B", dtype=torch.bfloat16)
    model = from_hf(hf, tokenizer)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    ids = model.encode("Paris is the capital and largest city of France. It stands on the Seine and has been")
    war_ids = tokenizer.encode(" war", add_special_tokens=False)
    assert len(war_ids) == 1 and ids.shape[1] == 18
    word_id = war_ids[0]
    altered = ids.clone()
    altered[0, 17] = word_id

    def next_logits(tokens):
        output = model.forward(tokens)
        return model._lm_head(output.last_hidden_state[:, -1]).float()[0]

    runs = []
    for label, prefix in [("baseline", ids), ("input_token_replacement", altered)]:
        tokens = prefix.clone()
        generated, probabilities = [], []
        for step in range(60):
            logits = next_logits(tokens)
            probabilities.append(float(logits.softmax(-1)[word_id]))
            next_id = int(logits.argmax())
            generated.append(next_id)
            tokens = torch.cat([tokens, tokens.new_tensor([[next_id]])], dim=1)
            if next_id == tokenizer.eos_token_id:
                break
        row = dict(condition=label, prefix=tokenizer.decode(prefix[0]), generated_ids=generated,
                   generated_tokens=[tokenizer.decode([i]) for i in generated],
                   generated_text=tokenizer.decode(generated), war_probability_by_step=probabilities,
                   generated_war_count=generated.count(word_id))
        runs.append(row)
        print(label, row['prefix'], '\n', row['generated_text'], flush=True)

    # Same preceding continuation for both conditions at every compared step.
    baseline_ids = runs[0]['generated_ids']
    controlled = []
    for label, prefix in [("baseline", ids), ("input_token_replacement", altered)]:
        tail = prefix.new_tensor([baseline_ids])
        tokens = torch.cat([prefix, tail], dim=1)
        hidden = model.forward(tokens).last_hidden_state
        logits = model._lm_head(hidden[:, ids.shape[1]-1:ids.shape[1]-1+len(baseline_ids)]).float()[0]
        controlled.append(dict(condition=label, war_probability_by_step=logits.softmax(-1)[:, word_id].cpu().tolist()))
    result = dict(position=17, original_token=" been", replacement_token=" war", runs=runs,
                  baseline_continuation_control=controlled)
    out = ROOT / 'artifacts/runs/j/legacy/war_input_token_results.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print('Saved', out)
    print('step baseline_free replaced_free baseline_control replaced_control')
    for step in [1,2,3,4,5,6,10,20,40,60]:
        i=step-1
        print(step, *(r['war_probability_by_step'][i] for r in runs),
              *(r['war_probability_by_step'][i] for r in controlled))


if __name__ == '__main__':
    main()

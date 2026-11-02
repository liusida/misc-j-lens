# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0"]
# ///
"""Generate an answer and compare three L12 lenses in a standalone HTML table."""
import argparse
import html
import json
import sys
from pathlib import Path
import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'anthropic-jacobian-lens'))
from jlens import from_hf
from jlens.hooks import ActivationRecorder

TEMPLATE=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>L12 · Three-lens comparison · Computer science history</title>
<style>
:root {font-family:system-ui,sans-serif;color:#202936;background:#f3f5f8}body{margin:0;padding:24px}
main{max-width:1800px;margin:auto}h1{font-size:26px;margin:0 0 8px}p{line-height:1.5}.subtitle{color:#596576}
.card{background:white;border:1px solid #dce2ea;border-radius:10px;padding:18px;margin:16px 0}
.answer{white-space:pre-wrap;line-height:1.65}.controls{position:sticky;top:0;z-index:4;background:#f3f5f8;padding:12px 0;display:flex;gap:20px;flex-wrap:wrap;align-items:center}
input[type=search]{min-width:290px;padding:8px;border:1px solid #bcc7d4;border-radius:5px}select{padding:6px}label{font-size:14px}
table{width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;background:white;border:1px solid #dce2ea}
th{position:sticky;top:65px;z-index:3;background:#e9edf3;text-align:left;padding:12px;border-bottom:2px solid #b9c5d3}
th.pos{width:55px}th.phase{width:100px}th.token{width:165px}.normal-head{background:#fcebed}.self-head{background:#e8f0fb}.future-head{background:#f2eafb}
td{vertical-align:top;padding:10px 12px;border-bottom:1px solid #dce2ea;border-right:1px solid #eef1f5;font-size:13px;overflow-wrap:anywhere}
tr.generated td:first-child{border-left:4px solid #21936b}tr.prompt td:first-child{border-left:4px solid #8996a6}
code{font-family:ui-monospace,monospace;white-space:pre-wrap}.input-token{font-size:14px;font-weight:600;display:block;background:#f2f4f7;padding:7px;border-radius:4px}
.token-id{font-size:11px;color:#687586;margin-top:8px}.prediction{display:grid;grid-template-columns:22px minmax(50px,1fr) 66px;gap:6px;padding:4px 0;border-bottom:1px solid #f0f2f5;align-items:start}
.rank,.prob{color:#647184;font-size:11px}.prob{text-align:right;font-variant-numeric:tabular-nums}.first code{font-weight:700}.common{background:#fff3cc;border-radius:3px}
.count{color:#596576;font-size:13px}.small{font-size:12px;color:#647184}details{margin:12px 0}summary{cursor:pointer}a{color:#2166ac}
@media(max-width:900px){body{padding:10px}.table-wrap{overflow-x:auto}table{min-width:1100px}.controls{position:relative}th{top:0}}
</style></head><body><main>
<h1>L12 · Normal / same-position / future-only</h1>
<div class="subtitle">Qwen3.5-2B · post-block 12 → post-block 23 · matched 25-prompt lenses</div>
<section class="card"><strong>User prompt</strong><p>__PROMPT__</p>
<details open><summary>Model's unmodified greedy answer</summary><div class="answer">__ANSWER__</div></details>
<p class="small">All three lenses decode exactly the same L12 activations. They do not change generation. Each row is the state after reading the listed token.</p>
<details><summary>Definitions and numerical details</summary>
<p><b>Normal:</b> average over sources of the sum of same-position and future-target Jacobians.<br>
<b>Same-position:</b> average of same-position Jacobians only.<br>
<b>Future-only:</b> normal minus same-position, preserving the same source-position normalization.</p>
<p>Each transported activation is decoded with the model's actual final RMSNorm and unembedding. Percentages are lens softmax probabilities, not the model's actual next-token probabilities. Hover over a prediction for its logit and token ID.</p>
<p>Training: 25 saved pile-10k prompts, maximum 128 tokens, skip first 4 and last position; BF16 model, FP32 matrix accumulation. Same-position gradients use the validated causal diagonal method; future subtraction inherits its BF16 numerical differences. Input token decoding can show replacement characters for individual byte fragments.</p>
</details></section>
<div class="controls"><label>Rows <select id="phase"><option value="all">All positions</option><option value="prompt">Prompt only</option><option value="generated">Generated answer only</option></select></label>
<label>Top <select id="top"><option>3</option><option>5</option><option selected>10</option></select></label>
<input id="search" type="search" placeholder="Filter by input or decoded token…"><span id="count" class="count"></span></div>
<p class="small">Whitespace is displayed as ␠ (space), ↵ (newline), ⇥ (tab). Yellow marks vocabulary tokens appearing in all three displayed lists.</p>
<div class="table-wrap"><table><thead><tr><th class="pos">Pos</th><th class="phase">Part</th><th class="token">Input token</th><th class="normal-head">Normal</th><th class="self-head">Same-position only</th><th class="future-head">Future-only</th></tr></thead><tbody id="rows"></tbody></table></div>
</main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const visible=s=>s.replace(/ /g,'␠').replace(/\n/g,'↵').replace(/\t/g,'⇥').replace(/\r/g,'␍');
function render(){const n=Number(document.getElementById('top').value),phase=document.getElementById('phase').value,q=document.getElementById('search').value.toLowerCase();
 const rows=data.rows.filter(r=>(phase==='all'||r.phase===phase)&&(!q||[r.token,...['normal','self','future'].flatMap(k=>r[k].map(v=>v.token))].join(' ').toLowerCase().includes(q)));
 document.getElementById('count').textContent=`${rows.length} / ${data.rows.length} positions`;
 document.getElementById('rows').innerHTML=rows.map(r=>{const sets=['normal','self','future'].map(k=>new Set(r[k].slice(0,n).map(v=>v.token_id)));
 const common=id=>sets.every(s=>s.has(id));
 const cells=['normal','self','future'].map(k=>'<td>'+r[k].slice(0,n).map((v,i)=>`<div class="prediction ${i===0?'first':''}" title="token ID ${v.token_id}; logit ${v.logit.toFixed(5)}"><span class="rank">${i+1}</span><code class="${common(v.token_id)?'common':''}">${escape(visible(v.token))}</code><span class="prob">${(100*v.probability).toPrecision(3)}%</span></div>`).join('')+'</td>').join('');
 return `<tr class="${r.phase}" id="position-${r.position}"><td>${r.position}</td><td>${r.phase==='generated'?'Answer':'Prompt'}</td><td><code class="input-token">${escape(visible(r.token))}</code><div class="token-id">ID ${r.token_id}</div></td>${cells}</tr>`;}).join('');}
['phase','top','search'].forEach(id=>document.getElementById(id).addEventListener('input',render));render();
</script></body></html>'''


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prompt',default='Briefly introduce the history of computer science.')
    parser.add_argument('--max-new-tokens',type=int,default=240)
    parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--continue-generation',action='store_true')
    args=parser.parse_args()
    if args.render_only:
        out=ROOT/'j_lens_output/computer_science_three_lenses'
        data=json.loads((out/'readouts.json').read_text())
        safe_json=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
        page=TEMPLATE.replace('__PROMPT__',html.escape(data['user_prompt'])).replace('__ANSWER__',html.escape(data['generated_answer'])).replace('__DATA__',safe_json)
        (out/'comparison.html').write_text(page)
        print('Rendered',out/'comparison.html')
        return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out=ROOT/'j_lens_output/computer_science_three_lenses';out.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
    hf=from_pretrained_to_cuda(AutoModelForCausalLM,'Qwen/Qwen3.5-2B',dtype=torch.bfloat16,attn_implementation='eager')
    model=from_hf(hf,tokenizer)
    paired=ROOT/'j_lens_output/paired_self_normal_25'
    paths=dict(normal=paired/'normal_lens.pt',self=paired/'self_lens.pt',future=ROOT/'j_lens_output/future_only_lens_25/lens.pt')
    lenses={name:torch.load(path,map_location='cuda',weights_only=True)['J'].float() for name,path in paths.items()}
    chat=('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
          '<|im_start|>user\n'+args.prompt+'<|im_end|>\n'
          '<|im_start|>assistant\n<think>\n\n</think>')
    prefix=tokenizer(chat,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
    ids=prefix.clone();generated=[]
    if args.continue_generation:
        previous=json.loads((out/'readouts.json').read_text())
        if previous['chat_prompt']!=chat:raise ValueError('Saved prompt differs')
        previous_ids=[r['token_id'] for r in previous['rows']]
        if previous_ids[:prefix.shape[1]]!=prefix[0].tolist():raise ValueError('Saved tokenization differs')
        generated=previous_ids[prefix.shape[1]:]
        ids=prefix.new_tensor([previous_ids])
    stops={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}
    print('Generating unmodified answer...',flush=True)
    for step in range(len(generated),args.max_new_tokens):
        output=model.forward(ids)
        token=int(model._lm_head(output.last_hidden_state[:,-1]).float().argmax(-1))
        generated.append(token);ids=torch.cat([ids,ids.new_tensor([[token]])],dim=1)
        if token in stops:break
        if (step+1)%40==0:print(f'Generated {step+1} tokens',flush=True)
    answer=tokenizer.decode(generated,skip_special_tokens=True)
    print('Answer:',answer,flush=True)
    with ActivationRecorder(model.layers,at=[12]) as recorder:
        model.forward(ids);hidden=recorder.activations[12][0].float()
    rows=[dict(position=t,token_id=int(i),token=tokenizer.decode([int(i)]),phase='prompt' if t<prefix.shape[1] else 'generated') for t,i in enumerate(ids[0])]
    for name,J in lenses.items():
        for first in range(0,len(rows),8):
            logits=model.unembed(hidden[first:first+8]@J.T).float()
            probs=logits.softmax(-1)
            values,indices=logits.topk(10,dim=-1)
            for b in range(len(values)):
                rows[first+b][name]=[dict(token_id=int(i),token=tokenizer.decode([int(i)]),logit=float(v),probability=float(probs[b,i])) for v,i in zip(values[b],indices[b])]
        print('Decoded',name,flush=True)
    data=dict(model='Qwen/Qwen3.5-2B',layer=12,site='post',user_prompt=args.prompt,chat_prompt=chat,
        generated_answer=answer,generation='greedy, no intervention, use_cache=False',generation_truncated=generated[-1] not in stops,
        lens_paths={k:str(v) for k,v in paths.items()},rows=rows)
    json_text=json.dumps(data,ensure_ascii=False,indent=2)
    (out/'readouts.json').write_text(json_text+'\n')
    safe_json=json_text.replace('<','\\u003c')
    page=TEMPLATE.replace('__PROMPT__',html.escape(args.prompt)).replace('__ANSWER__',html.escape(answer)).replace('__DATA__',safe_json)
    path=out/'comparison.html';path.write_text(page)
    print('Saved',path,'rows',len(rows),'truncated',data['generation_truncated'],flush=True)


if __name__=='__main__':main()

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Render saved normal/future readouts with token positions as columns."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'artifacts/comparisons/j_split_l12_legacy'

TEMPLATE=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>L12 · Normal vs Future-only · Horizontal tokens</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#202936;font-family:system-ui,sans-serif}
header{padding:20px 24px 10px}h1{font-size:24px;margin:0 0 8px}.muted{color:#657185;font-size:13px}
.panel{margin:12px 24px;padding:16px;background:white;border:1px solid #dce2ea;border-radius:8px}
.answer{white-space:pre-wrap;line-height:1.6;max-height:230px;overflow:auto}summary{cursor:pointer}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding:12px 24px}select,input,button{font:inherit;padding:7px;border:1px solid #bdc8d5;border-radius:5px;background:white}
input[type=search]{width:290px}input[type=number]{width:80px}button{cursor:pointer}
.legend{padding:0 24px 10px;font-size:12px;color:#657185}.common{background:#fff0b7;border-radius:3px}
.viewport{margin:0 24px 24px;overflow-x:auto}
.token-block{margin-bottom:22px;border:1px solid #c6d0dc;background:white;width:max-content}
table{border-collapse:separate;border-spacing:0;table-layout:fixed}
th,td{border-right:1px solid #dce2ea;border-bottom:1px solid #dce2ea;vertical-align:top}
thead th{position:sticky;top:0;z-index:3;background:#edf1f6;padding:7px;width:145px;min-width:145px;text-align:left}
.corner{left:0;z-index:5;width:115px;min-width:115px}
tbody th{position:sticky;left:0;z-index:2;width:115px;min-width:115px;text-align:left;padding:12px 8px}
.normal{background:#fcebed}.future{background:#f2eafb}td{width:145px;min-width:145px;padding:6px}
.position{font-size:11px;color:#647184;margin-bottom:5px}.input{font-family:ui-monospace,monospace;font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere}
.prediction{display:grid;grid-template-columns:14px minmax(35px,1fr) 42px;gap:3px;padding:4px 0;border-bottom:1px solid #f0f2f5;font-size:11px}
.prediction code{font-family:ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.first code{font-weight:700}
.rank,.prob{color:#657185;font-size:10px}.prob{text-align:right;font-variant-numeric:tabular-nums}
.hide-percentages .prob{display:none}.hide-percentages .prediction{grid-template-columns:14px minmax(35px,1fr)}
.generated{border-top:4px solid #21936b}.prompt{border-top:4px solid #8996a6}.count{font-size:13px;color:#657185}
</style></head><body>
<header><h1>L12 · Normal vs Future-only</h1><div class="muted">Token positions run left → right. Two lenses decode the same activations. Qwen3.5-2B · post-block 12 · matched 25-prompt lenses.</div></header>
<div class="panel"><strong id="prompt"></strong><details><summary>Generated answer</summary><div class="answer" id="answer"></div></details></div>
<div class="controls"><label>Positions <select id="phase"><option value="all">All</option><option value="prompt">Prompt only</option><option value="generated">Answer only</option></select></label>
<label>Top <select id="top"><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option>8</option><option>9</option><option selected>10</option></select></label>
<label><input type="checkbox" id="percentages" checked> Show percentages</label>
<input id="search" type="search" placeholder="Filter by input or decoded token…">
<label>Go to position <input type="number" id="jump" min="0" value="0"></label><button id="go">Go</button><span id="count" class="count"></span></div>
<div class="legend">Positions wrap into blocks to fit the window; read left to right, then downward. <span class="common">Yellow</span> = decoded token appears in both displayed top-K lists at that position. ␠ = space; ↵ = newline; ⇥ = tab. Percentages are lens softmax probabilities, not actual next-token probabilities. Hover for logit and token ID.</div>
<div class="viewport" id="viewport"></div>
<div class="legend">Future-only = normal − same-position, with identical source averaging; inherits the recorded BF16 numerical differences. No interventions were applied to generation.</div>
<script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const visible=s=>s.replace(/ /g,'␠').replace(/\n/g,'↵').replace(/\t/g,'⇥').replace(/\r/g,'␍');
document.getElementById('prompt').textContent=data.user_prompt;document.getElementById('answer').textContent=data.generated_answer;
function render(){const n=Number(document.getElementById('top').value),phase=document.getElementById('phase').value,q=document.getElementById('search').value.toLowerCase();
 const rows=data.rows.filter(r=>(phase==='all'||r.phase===phase)&&(!q||[r.token,...['normal','future'].flatMap(k=>r[k].map(v=>v.token))].join(' ').toLowerCase().includes(q)));
 document.getElementById('count').textContent=`${rows.length} / ${data.rows.length} positions`;
 const viewport=document.getElementById('viewport');
 const columns=Math.max(1,Math.floor((viewport.clientWidth-117)/145));
 const blocks=[];
 for(let first=0;first<rows.length;first+=columns){const chunk=rows.slice(first,first+columns);
 const head='<tr><th class="corner">Lens ↓ / position →</th>'+chunk.map(r=>`<th class="${r.phase}" id="position-${r.position}"><div class="position">${r.position} · ${r.phase==='generated'?'Answer':'Prompt'} · ID ${r.token_id}</div><div class="input">${esc(visible(r.token))}</div></th>`).join('')+'</tr>';
 const body=['normal','future'].map(k=>`<tr><th class="${k}">${k==='normal'?'Normal':'Future-only'}</th>`+chunk.map(r=>{const other=new Set(r[k==='normal'?'future':'normal'].slice(0,n).map(v=>v.token_id));return '<td>'+r[k].slice(0,n).map((v,i)=>`<div class="prediction ${i===0?'first':''}" title="logit ${v.logit.toFixed(5)}; token ID ${v.token_id}"><span class="rank">${i+1}</span><code class="${other.has(v.token_id)?'common':''}">${esc(visible(v.token))}</code><span class="prob">${(100*v.probability).toPrecision(3)}%</span></div>`).join('')+'</td>';}).join('')+'</tr>').join('');
 blocks.push(`<div class="token-block"><table style="width:${115+145*chunk.length}px"><thead>${head}</thead><tbody>${body}</tbody></table></div>`);}
 viewport.innerHTML=blocks.join('');}
['phase','top','search'].forEach(id=>document.getElementById(id).addEventListener('input',render));
document.getElementById('percentages').addEventListener('change',()=>{document.getElementById('viewport').classList.toggle('hide-percentages',!document.getElementById('percentages').checked);});
document.getElementById('go').addEventListener('click',()=>{const cell=document.getElementById('position-'+Number(document.getElementById('jump').value));if(cell){cell.scrollIntoView({behavior:'smooth',block:'center',inline:'nearest'});}});
render();
if(typeof ResizeObserver!=='undefined'){let lastWidth=document.getElementById('viewport').clientWidth;new ResizeObserver(entries=>{const width=Math.floor(entries[0].contentRect.width);if(width!==lastWidth){lastWidth=width;render();}}).observe(document.getElementById('viewport'));}
</script></body></html>'''

data=json.loads((FOLDER/'readouts.json').read_text())
compact={k:data[k] for k in ['user_prompt','generated_answer']}
compact['rows']=[{k:r[k] for k in ['position','phase','token_id','token','normal','future']} for r in data['rows']]
payload=json.dumps(compact,ensure_ascii=False).replace('<','\\u003c')
output=FOLDER/'normal_future_horizontal.html'
output.write_text(TEMPLATE.replace('__DATA__',payload))
print(output)
print('Columns:',len(compact['rows']),'lens rows: 2')

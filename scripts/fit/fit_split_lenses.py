# /// script
# requires-python = ">=3.11"
# dependencies = ["gb10-load-llm", "torch>=2.7", "transformers>=5.0", "safetensors", "datasets"]
# ///
"""Matched v2 J/R Normal/Same-token fits; Future-only = Normal - Same-token.

All 48 v2 sites, all 2048 output dimensions, first 100 nonempty pile prompts.
Native v2 prompt_jacobians is called unchanged for both passes; only the
same-token pass removes cross-position derivative edges, using the causal
diagonal-block identity. Forward SDPA values are untouched. BF16 roundoff
differences in the equivalent diagonal adjoint are recorded by validation.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

import torch
from datasets import Dataset
from gb10_load_llm import load_model_to_cuda
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jlens_study import paths
from jlens_study.fitting import reference_v2 as v2
from jlens_study.fitting.relp_rules import install_qwen35_relp_rules
from jlens_study.fitting.same_token_gradients import same_token_sdpa_gradients

ROOT = REPO_ROOT
MODEL = 'Qwen/Qwen3.5-2B'
MODEL_REV = '15852e8c16360a2fea060d615a32b45270f8a8fc'
DATA_REV = '127bfedcd5047750df5ccf3a12979a47bfa0bafa'
REFERENCE_ROOT = Path('/home/liusida/hf_cache/hub/datasets--sida--llmdbg/snapshots/980e34bc0f0460c437bb4fa0238a58b9b50158d3/workspace-lenses/v2/Qwen--Qwen3.5-2B')


def atomic_torch(obj, path):
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def atomic_json(obj, path):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def validate(model, tokenizer, prompt, include_mid=True):
    embed, layers, _, architecture = v2.model_parts(model)
    ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)['input_ids'].cuda()
    positions = [4, 17, ids.shape[1]-2]
    keys = (
        ['embed', 'mid.0', 'post.0', 'mid.12', 'post.12', 'mid.23']
        if include_mid else ['embed', 'post.0', 'post.12']
    )
    direction = torch.sin(torch.arange(2048, device='cuda').float()/17).to(next(model.parameters()).dtype)
    expected = {}
    expected_future = {}
    with v2.record_sites(embed, layers, architecture) as rec:
        model(ids, use_cache=False)
        target = rec['post.23']
        forward = {k:v.detach().clone() for k,v in rec.items()}
        sources = [rec[k] for k in keys]
        for t in positions:
            cot = torch.zeros_like(target); cot[0,t] = direction
            gradients = torch.autograd.grad(target, sources, cot, retain_graph=True)
            expected[t] = [g[0,t].float().detach() for g in gradients]
            cot.zero_(); cot[0,t+1:ids.shape[1]-1] = direction
            gradients = torch.autograd.grad(target, sources, cot, retain_graph=True)
            expected_future[t] = [g[0,t].float().detach() for g in gradients]
        cot = torch.zeros_like(target); cot[:,4:ids.shape[1]-1] = direction
        gradients = torch.autograd.grad(target, sources, cot)
        normal = [g.detach().float() for g in gradients]
    checks = []
    with same_token_sdpa_gradients(), v2.record_sites(embed, layers, architecture) as rec:
        model(ids, use_cache=False)
        forward_error = max(float((v.detach()-forward[k]).abs().max()) for k,v in rec.items())
        target = rec['post.23']; sources = [rec[k] for k in keys]
        cot = torch.zeros_like(target); cot[:,4:ids.shape[1]-1] = direction
        gradients = torch.autograd.grad(target, sources, cot, retain_graph=True)
        for t in positions:
            for i,k in enumerate(keys):
                actual = gradients[i][0,t].float()
                ref = expected[t][i]
                future = normal[i][0,t]-actual
                ref_future = expected_future[t][i]
                checks.append(dict(site=k, position=t, relative_error=float((actual-ref).norm()/ref.norm().clamp_min(1e-12)), cosine=cosine(actual,ref),
                    future_relative_error=float((future-ref_future).norm()/ref_future.norm()) if ref_future.norm()>0 else None,
                    future_absolute_error=float((future-ref_future).norm()),
                    future_reference_norm=float(ref_future.norm())))
        cot.zero_(); cot[0,17] = direction
        gradients = torch.autograd.grad(target, sources, cot)
        off_position = max(float(torch.cat([g[:,:17].flatten(),g[:,18:].flatten()]).abs().max()) for g in gradients)
    result = dict(forward_max_error=forward_error, off_position_gradient_max=off_position, checks=checks,
        method='full-coordinate causal diagonal blocks, not stochastic; equivalent recurrent FP32 adjoint',
        numerical_note='BF16 rounding differs from native chunk/SDPA backward; mathematically identical diagonal derivatives')
    if forward_error != 0 or off_position != 0 or any(c['relative_error']>.08 or c['cosine']<.995 for c in checks):
        raise RuntimeError('Validation failed: '+json.dumps(result))
    print('VALIDATION', json.dumps(result), flush=True)
    return result


def publish(out, sums, completed, config, metrics):
    requested = config['fitting_recipe']['prompts']
    normal = {k:v/completed for k,v in sums['normal'].items()}
    same = {k:v/completed for k,v in sums['same_token'].items()}
    future = {k:normal[k]-same[k] for k in normal}
    for name, matrices in [('normal',normal), ('same_token',same), ('future_only',future)]:
        folder = out/name; folder.mkdir(exist_ok=True)
        path = folder/'lens.safetensors'
        temp = folder/'lens.safetensors.tmp'
        save_file({k:v.half().contiguous() for k,v in matrices.items()}, temp)
        temp.replace(path)
        manifest = dict(config, format_version=2, artifact=path.name, sha256=v2.sha256(path),
            method=config['method'], method_name=name+'_'+config['estimator'], provenance='v2_matched_token_target_split',
            prompts=completed, requested_prompts=requested, complete=completed==requested,
            dtype='float16', hidden_size=2048, sites=config['sites'], tensor_shapes={k:list(v.shape) for k,v in matrices.items()},
            created_at=datetime.now(timezone.utc).isoformat(),
            reduction='equal prompt mean of mean over valid sources of sum over selected valid targets',
            target_selection={'normal':'target >= source', 'same_token':'target == source', 'future_only':'target > source'}[name])
        atomic_json(manifest, folder/'manifest.json')
    # FP32 checkpoint retains exact split before export rounding.
    reconstruction = max(float((normal[k]-same[k]-future[k]).abs().max()) for k in normal)
    atomic_json(dict(completed=completed,requested=requested,reconstruction_max_error=reconstruction,latest=metrics[-1],
        complete=completed==requested), out/'status.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('j', 'r'), default='j')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--validate-fp32', action='store_true', help='Check the equivalent derivative algebra in FP32, then exit; fitting remains BF16')
    parser.add_argument('--dim-batch', type=int, default=16)
    parser.add_argument('--prompts', type=int, default=100, choices=range(1, 101))
    parser.add_argument('--exclude-mid', action='store_true', help='Fit embed and post.* sites only')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out = args.output or paths.split_lens_dir(args.method)
    out.mkdir(parents=True, exist_ok=True)
    reference_dir = REFERENCE_ROOT/args.method
    manifest = json.loads((reference_dir/'manifest.json').read_text())
    selected_sites = dict(manifest['sites'])
    if args.exclude_mid:
        selected_sites['mid'] = []
    arrow = Path('/home/liusida/hf_cache/datasets/NeelNanda___pile-10k/default/0.0.0')/DATA_REV/'pile-10k-train.arrow'
    dataset = Dataset.from_file(str(arrow))
    texts = []
    for row in dataset:
        if str(row['text']).strip(): texts.append(str(row['text']))
        if len(texts)==100: break
    assert len(texts)==100
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV, local_files_only=True)
    ids = [tokenizer(t,truncation=True,max_length=128)['input_ids'] for t in texts]
    assert sum(map(len,ids))==manifest['sequence_tokens_last_run']==12203
    assert sum(len(i)-5 for i in ids)==manifest['valid_positions_last_run']==11703
    prompts = [dict(index=i,text=t,input_ids=ts,sha256=hashlib.sha256(t.encode()).hexdigest()) for i,(t,ts) in enumerate(zip(texts[:args.prompts],ids[:args.prompts]))]
    atomic_json(prompts,out/'prompts.json')
    config = dict(method=args.method,estimator='ordinary_jacobian' if args.method=='j' else 'relp',model_id=MODEL,model_revision=MODEL_REV,tokenizer_revision=MODEL_REV,dataset='NeelNanda/pile-10k',dataset_revision=DATA_REV,
        dataset_recipe='first 100 nonempty train documents, dataset order, no shuffle',
        target=manifest['target'],sites=selected_sites,max_seq_len=128,skip_first=4,
        fitting_recipe=dict(prompts=args.prompts,max_seq_len=128,skip_first=4,dimension_batch=args.dim_batch,model_compute_dtype='bfloat16',accumulator_dtype='float32'),
        attention_implementation='sdpa',rules=manifest['rules'],
        reference_sha256=manifest['sha256'],reference_fitter_sha256=v2.sha256(ROOT/'src/jlens_study/fitting/reference_v2.py'),
        relp_rules_sha256=v2.sha256(ROOT/'src/jlens_study/fitting/relp_rules.py') if args.method=='r' else None,
        diagonal_adjoint_sha256=v2.sha256(ROOT/'src/jlens_study/fitting/same_token_gradients.py'),
        block_adjoint_sha256=v2.sha256(ROOT/'src/jlens_study/fitting/exact_block_gradients.py'),history_helper_sha256=v2.sha256(ROOT/'src/jlens_study/fitting/exact_self_gradients.py'),
        future_method='matched FP32 normal minus same-token, identical prompt and source averaging',
        numerical_note='Exact causal diagonal identity; native forward unchanged; equivalent adjoint has ordinary BF16 rounding differences')
    atomic_json(config,out/'config.json')
    print('Loading pinned model on CUDA',flush=True)
    model = load_model_to_cuda(AutoModelForCausalLM,MODEL,device='cuda',dtype=torch.bfloat16,touch='auto',low_cpu_mem_usage=True,
        attn_implementation='sdpa',revision=MODEL_REV,local_files_only=True)
    model.eval().requires_grad_(False)
    if args.method == 'r':
        parity_ids = tokenizer('R-Lens parity check.', return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            before = model(parity_ids).logits
        rules = install_qwen35_relp_rules(model)
        expected_counts = (2 * len(model.model.layers) + 1, len(model.model.layers))
        actual_counts = (rules['norm_modules'], rules['mlp_modules'])
        if actual_counts != expected_counts:
            raise RuntimeError(f'Unexpected RelP module counts: {actual_counts} != {expected_counts}')
        with torch.inference_mode():
            after = model(parity_ids).logits
        parity_error = float((before-after).abs().max())
        if not torch.allclose(before, after, atol=1e-5, rtol=1e-4):
            raise RuntimeError(f'RelP changed forward values by {parity_error:.3e}')
        config['rules'] = rules
        config['relp_forward_parity_max_error'] = parity_error
        atomic_json(config,out/'config.json')
    if args.validate_fp32:
        model.float()
        checks=validate(model,tokenizer,texts[0],include_mid=not args.exclude_mid)
        if any(c['relative_error']>1e-4 for c in checks['checks']):
            raise RuntimeError('FP32 algebra validation failed')
        atomic_json(checks,out/'validation_fp32.json')
        return
    checks=validate(model,tokenizer,texts[0],include_mid=not args.exclude_mid)
    atomic_json(checks,out/'validation.json')
    if args.validate_only: return
    checkpoint=out/'fit.checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['config']!=config: raise ValueError('Checkpoint config mismatch')
        sums=state['sums'];completed=state['completed'];metrics=state['metrics']
    else:
        sums={'normal':{},'same_token':{}};completed=0;metrics=[]
    with safe_open(reference_dir/'lens.safetensors',framework='pt',device='cpu') as f:
        reference={
            k:f.get_tensor(k).float() for k in f.keys()
            if not args.exclude_mid or not k.startswith('mid.')
        }
    for index in range(completed,args.prompts):
        started=time.monotonic()
        current={}
        for mode in ['normal','same_token']:
            pending=out/f'pending_{index+1:03d}_{mode}.pt'
            if pending.exists():
                matrices=torch.load(pending,map_location='cpu',weights_only=True)
                print('Reused pending',index+1,mode,flush=True)
            else:
                print(f'PROMPT {index+1}/{args.prompts} {mode} START',flush=True)
                with same_token_sdpa_gradients() if mode=='same_token' else nullcontext():
                    matrices,seq_len,n_valid=v2.prompt_jacobians(
                        model,tokenizer,prompt=texts[index],dim_batch=args.dim_batch,
                        max_seq_len=128,skip_first=4,include_mid=not args.exclude_mid,
                    )
                assert set(matrices)==set(reference)
                if any(not torch.isfinite(m).all() for m in matrices.values()): raise RuntimeError('Nonfinite matrix')
                atomic_torch(matrices,pending)
                print(f'PROMPT {index+1}/{args.prompts} {mode} SAVED',flush=True)
            current[mode]=matrices
        row=dict(n_prompts=index+1,seconds=time.monotonic()-started,sites={})
        for key in reference:
            previous={mode:sums[mode][key]/index if index else None for mode in sums}
            for mode in sums:
                if key not in sums[mode]: sums[mode][key]=torch.zeros_like(current[mode][key])
                sums[mode][key]+=current[mode][key]
            n=sums['normal'][key]/(index+1);s=sums['same_token'][key]/(index+1);f=n-s
            row['sites'][key]=dict(normal_cosine_to_v2=cosine(n,reference[key]),same_cosine_to_v2=cosine(s,reference[key]),future_cosine_to_v2=cosine(f,reference[key]),
                same_norm=float(s.norm()),future_norm=float(f.norm()),normal_norm=float(n.norm()),
                normal_lim=None if not index else float((n-previous['normal']).norm()/previous['normal'].norm()),
                same_lim=None if not index else float((s-previous['same_token']).norm()/previous['same_token'].norm()))
        completed=index+1;metrics.append(row)
        atomic_torch(dict(sums=sums,completed=completed,metrics=metrics,config=config),checkpoint)
        atomic_json(metrics,out/'metrics.json')
        publish(out,sums,completed,config,metrics)
        for mode in current: (out/f'pending_{index+1:03d}_{mode}.pt').unlink()
        print('COMPLETED',completed,f'/{args.prompts}',json.dumps(row['sites']['post.12']),f'seconds={row["seconds"]:.1f}',flush=True)
    print('DONE',out,flush=True)


if __name__=='__main__':
    main()

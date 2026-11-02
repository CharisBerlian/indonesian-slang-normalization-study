"""Join independently frozen SVM/BERT predictions and run the paired evaluation."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from evaluate_models import LABELS, MODEL_SEEDS, evaluate_paired
from infer_svm_final import ROOT, checked_hash

OUT = ROOT / 'results/final_evaluation'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def contained(base, relative):
    path=(base / relative).resolve()
    if base.resolve() not in path.parents: raise ValueError('Prediction path escapes artifact directory')
    return path


def load_svm(dataset, manifest, selection_sha):
    meta=read_json(OUT/'svm_inference.json')
    if meta['selection_sha256'] != selection_sha or meta['protocol_id'] != manifest['protocol_id']:
        raise ValueError('SVM inference selection mismatch')
    path=OUT/f'svm_{dataset}_predictions.csv'; checked_hash(path,meta['output_sha256'][dataset])
    selected={(m['representation'],m['condition']):m for m in manifest['svm']['models'] if m['dataset']==dataset}
    expected_count={'prdec':793,'smsa':500}[dataset]
    with path.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    if len(rows) != expected_count*8: raise ValueError('Unexpected SVM prediction count')
    gold={}; fixed={m:{'42':{t:{} for t in ('original','normalized')}} for m in ('word','character')}
    for row in rows:
        rep,trained,condition=row['representation'],row['trained_condition'],row['input_condition']
        if (rep,trained) not in selected or condition not in ('original','normalized'):
            raise ValueError('Invalid SVM prediction key')
        if row['checkpoint_sha256'] != selected[(rep,trained)]['checkpoint_sha256']:
            raise ValueError('SVM prediction checkpoint differs from selection')
        if row['changed'] not in ('true','false'): raise ValueError('Invalid changed flag')
        item={'id':row['id'],'group_id':row['group_id'],'label':row['label'],'changed':row['changed']=='true'}
        if row['id'] in gold and gold[row['id']] != item: raise ValueError('SVM gold metadata differs')
        gold[row['id']]=item
        values=fixed[rep]['42'][trained].setdefault(condition,{})
        if row['id'] in values: raise ValueError('Duplicate SVM prediction')
        values[row['id']]=row['prediction']
        score_names=('negative','positive') if dataset=='prdec' else ('negative','neutral','positive')
        if not all(np.isfinite(float(row[f'score_{x}'])) for x in score_names): raise ValueError('Invalid SVM score')
    if len(gold)!=expected_count: raise ValueError('Unexpected held-out ID count')
    return list(gold.values()),fixed,meta


def load_bert(directory, dataset, gold, manifest, selection_sha):
    run=read_json(directory/'results.json'); gate=read_json(directory/'checkpoint_manifest.json')
    em=read_json(directory/'eval_manifest.json')
    if run['status']!='complete' or len(run['results'])!=24 or run['selection_sha256']!=selection_sha:
        raise ValueError('BERT inference not complete or selection mismatch')
    if not gate['all_validation_predictions_match'] or gate['selection_sha256']!=selection_sha or gate['source_state_sha256']!=manifest['indobert']['state_sha256']:
        raise ValueError('BERT pre-test checkpoint validation gate mismatch')
    frozen_models={m['run_id']:m for m in manifest['indobert']['models']}
    checks={m['run_id']:m for m in gate['checkpoints']}
    if len(frozen_models)!=12 or set(checks)!=set(frozen_models): raise ValueError('BERT checkpoint gate incomplete')
    config_path=ROOT/'results/indobert_full/config.json'
    checked_hash(config_path,manifest['indobert']['config_file_sha256'])
    configured=read_json(config_path)['label_maps'][dataset]
    if em['label_maps'][dataset] != configured or set(configured) != set(LABELS[dataset]) or set(configured.values()) != set(range(len(LABELS[dataset]))):
        raise ValueError('BERT frozen label mapping mismatch')
    inverse={v:k for k,v in configured.items()}; references={r['id']:r for r in gold}
    for run_id, item in frozen_models.items():
        check=checks[run_id]
        if not check['validation_predictions_match'] or len(check['weights_sha256'])!=64:
            raise ValueError('BERT checkpoint validation or weight hash missing')
        if item['remote_checkpoint_sha256'] and item['remote_checkpoint_sha256'] != check['weights_sha256']:
            raise ValueError('BERT checkpoint hash differs from freeze')
    fixed={'indobert':{s:{t:{} for t in ('original','normalized')} for s in MODEL_SEEDS['indobert']}}
    seen=set(); input_changed={}; artifact_hashes={}
    all_combos={(m['run_id'], c) for m in frozen_models.values() for c in ('original','normalized')}
    for rec in run['results']:
        rid=rec['run_id']; cond=rec['eval_condition']; key=(rid,cond)
        if key not in all_combos or key in seen: raise ValueError('Duplicate or unexpected BERT output')
        seen.add(key); selected=frozen_models[rid]
        if (rec['dataset'],rec['train_condition'],rec['seed']) != (selected['dataset'],selected['condition'],selected['seed']):
            raise ValueError('BERT run metadata does not match frozen model')
        if rec['learning_rate'] != selected['selected_learning_rate'] or rec['best_epoch'] != selected['best_epoch'] or rec['primary_protocol_pair'] != (cond == selected['condition']):
            raise ValueError('BERT selection metadata mismatch')
        path=contained(directory,rec['prediction_path']); checked_hash(path,rec['prediction_sha256']); artifact_hashes[rec['prediction_path']]=rec['prediction_sha256']
        if rec['dataset'] != dataset: continue
        rows=read_json(path)
        if len(rows)!=len(gold) or rec['count']!=len(gold): raise ValueError('BERT prediction count mismatch')
        values={}
        for row in rows:
            ident=row['id']; ref=references.get(ident)
            if not ref or ident in values: raise ValueError('BERT ID mismatch or duplicate')
            if row['group_id']!=ref['group_id'] or row['label'] not in inverse or inverse[row['label']]!=ref['label']:
                raise ValueError('BERT gold label/group mismatch')
            if row['prediction'] not in inverse: raise ValueError('BERT prediction outside label set')
            logits=np.asarray(row['logits'],dtype=float)
            if logits.shape!=(len(configured),) or not np.isfinite(logits).all() or int(np.argmax(logits))!=row['prediction']:
                raise ValueError('BERT logits do not match prediction')
            flag=row['bert_input_changed']
            if not isinstance(flag,bool) or (flag and not ref['changed']): raise ValueError('Invalid BERT input-change flag')
            if ident in input_changed and input_changed[ident]!=flag: raise ValueError('BERT input-change flags differ')
            input_changed[ident]=flag; values[ident]=inverse[row['prediction']]
        fixed['indobert'][str(selected['seed'])][selected['condition']][cond]=values
    if seen!=all_combos: raise ValueError('Missing BERT outputs')
    return fixed, input_changed, artifact_hashes, run


def fmt_interval(i): return f'[{i["lower"]:+.4f}, {i["upper"]:+.4f}]'


def write_report(results,output):
    lines=['# Frozen held-out evaluation','',
           'The model-selection freeze preceded held-out inference. Primary pipeline effects compare separately trained and selected normalized and original models. Estimates are macro-F1 differences in raw units; 95% percentile intervals use the same 2,000 group-bootstrap draws across models, conditions and seeds within each dataset. These nominal intervals describe test-sample uncertainty conditional on the fitted models; they are not simultaneous intervals. Transformer sample standard deviation across the three seeds is reported separately.', '',
           'The ±0.01 practical margin is investigator-selected. A lower interval bound above +0.01 supports a practically meaningful gain for the tested setting; an interval containing zero does not establish equivalence.', '',
           '| Dataset | Model | Original macro-F1 | Normalized macro-F1 | Effect | 95% interval | Effect seed SD |', '|---|---|---:|---:|---:|---:|---:|']
    for ds,r in results['datasets'].items():
        for model,m in r['models'].items():
            sd=m['effect_across_seeds']['sample_sd']; sd='—' if sd is None else f'{sd:.4f}'
            lines.append(f'| {ds} | {model} | {m["macro_f1_across_seeds"]["original"]["mean"]:.4f} | {m["macro_f1_across_seeds"]["normalized"]["mean"]:.4f} | {m["effect_across_seeds"]["mean"]:+.4f} | {fmt_interval(m["mean_effect_interval"])} | {sd} |')
    lines+=['','| Dataset | Direct contrast | Estimate | 95% interval |','|---|---|---:|---:|']
    for ds,r in results['datasets'].items():
        for name,c in r['direct_contrasts'].items(): lines.append(f'| {ds} | {name} | {c["estimate"]:+.4f} | {fmt_interval(c["interval"])} |')
    lines+=['','Per-class F1, support and changes are shown per seed in `results/final_evaluation/paired_results.json`, together with confusion matrices and changed/unchanged subgroup scores. The table below averages seed-specific per-class F1 for IndoBERT; counts are repeated observations of the same held-out examples, not independent seed samples.', '',
            '| Dataset | Model | Class | Support | Original F1 | Normalized F1 | Difference |', '|---|---|---|---:|---:|---:|---:|']
    for ds,r in results['datasets'].items():
        for name,m in r['models'].items():
            seeds=m['seeds']
            for label in r['labels']:
                o=np.mean([s['pipeline_metrics']['original']['per_class'][label]['f1'] for s in seeds.values()]); n=np.mean([s['pipeline_metrics']['normalized']['per_class'][label]['f1'] for s in seeds.values()]); support=next(iter(seeds.values()))['pipeline_metrics']['original']['per_class'][label]['support']
                lines.append(f'| {ds} | {name} | {label} | {support} | {o:.4f} | {n:.4f} | {n-o:+.4f} |')
    lines+=['','Fixed-model original-to-normalized input transitions below use the original-trained model. IndoBERT values are mean counts across the three seeds; per-seed results and both trained conditions are retained in the JSON. A correct-to-incorrect transition is an introduced error.', '',
            '| Dataset | Model | Text-changed / total | Incorrect → correct | Correct → incorrect |', '|---|---|---:|---:|---:|']
    for ds,r in results['datasets'].items():
        for name,m in r['models'].items():
            ts=[s['fixed_model']['original']['transitions']['all'] for s in m['seeds'].values()]; t=next(iter(m['seeds'].values()))['fixed_model']['original']['transitions']['changed']['n']
            lines.append(f'| {ds} | {name} | {t}/{r["n_rows"]} | {np.mean([x["incorrect_to_correct"] for x in ts]):.2f} | {np.mean([x["correct_to_incorrect"] for x in ts]):.2f} |')
    lines+=['','Changed subsets can be small, especially for SmSA; subgroup metrics are descriptive and do not establish equivalence. These outcomes apply only to the studied datasets, models and dictionary policy. No simultaneous coverage guarantee or powered equivalence claim is made.', '',
            'Provenance limitation: the frozen selection recorded transformer checkpoint identifiers and metadata but did not contain pre-freeze weight hashes. The provider-version source was subsequently hashed before held-out inference and each checkpoint reproduced its saved validation predictions; this does not establish a pre-freeze cryptographic binding of those weight bytes. The SVM selection did bind checkpoint hashes.', '',
            'All 20 selected model checkpoints produced both input conditions. Paired predictions and model/inference provenance are local; see `results/final_evaluation/paired_results.json`, the SVM inference metadata and the independently supplied IndoBERT inference artifacts. Dedicated normalization-only runtime was not measured here.', '']
    output.write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--indobert-dir',type=Path,required=True); args=parser.parse_args()
    t0=time.perf_counter(); selection_path=ROOT/'results/final_selection.json'; selection_sha=(ROOT/'results/final_selection.sha256').read_text(encoding='utf-8').split()[0]
    checked_hash(selection_path,selection_sha); manifest=read_json(selection_path)
    results={'scope':'Frozen held-out evaluation','selection_sha256':selection_sha,'protocol_id':manifest['protocol_id'],
             'evaluator_sha256':hashlib.sha256((ROOT/'src/evaluate_models.py').read_bytes()).hexdigest(),
             'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'datasets':{}}
    for dataset in ('prdec','smsa'):
        gold,svm,svm_meta=load_svm(dataset,manifest,selection_sha)
        bert,bert_changed,artifact_hashes,run=load_bert(args.indobert_dir,dataset,gold,manifest,selection_sha)
        fixed={**svm,**bert}
        pipeline={model:{seed:{condition:fixed[model][seed][condition][condition] for condition in ('original','normalized')} for seed in seeds} for model,seeds in MODEL_SEEDS.items()}
        evaluation=evaluate_paired(dataset,gold,pipeline,fixed)
        evaluation['bert_input_changed']=sum(bert_changed.values())
        results['datasets'][dataset]=evaluation
    results['inputs']={'svm_output_sha256':svm_meta['output_sha256'],'bert_prediction_sha256':artifact_hashes,
                       'bert_results_sha256':hashlib.sha256((args.indobert_dir/'results.json').read_bytes()).hexdigest(),
                       'bert_checkpoint_manifest_sha256':hashlib.sha256((args.indobert_dir/'checkpoint_manifest.json').read_bytes()).hexdigest()}
    results['elapsed_seconds']=time.perf_counter()-t0
    output=OUT/'paired_results.json'; output.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    write_report(results,ROOT/'reports/final_evaluation.md')
    print(f'Paired evaluation complete in {results["elapsed_seconds"]:.2f} seconds.',flush=True)


if __name__=='__main__': main()

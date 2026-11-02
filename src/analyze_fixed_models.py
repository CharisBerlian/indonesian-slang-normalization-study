"""Descriptive four-cell analysis of frozen held-out predictions.

Does not train, tune, create predictions, or overwrite the primary evaluation.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from evaluate_models import (LABELS, MODEL_SEEDS, _align, _gold, _interval, _seed_summary,
                             bootstrap_f1, classification_metrics, group_draw_weights,
                             prediction_transitions)
from infer_svm_final import ROOT, checked_hash
from run_final_evaluation import OUT, load_bert, load_svm, read_json

OUTPUT = OUT/'fixed_model_diagnostics.json'
REPORT = ROOT/'reports/fixed_model_diagnostics.md'
CELLS = {'O/O': ('original', 'original'), 'O/N': ('original', 'normalized'),
         'N/O': ('normalized', 'original'), 'N/N': ('normalized', 'normalized')}
DIRECTIONS = {'O_to_N': ('O/O', 'O/N'), 'N_to_O': ('N/N', 'N/O')}


def evaluate_fixed(dataset, gold, fixed, expected_draw_sha256=None):
    ids,y,groups,changed=_gold(gold,dataset); labels=LABELS[dataset]
    names, row_groups, weights=group_draw_weights(groups)
    draw_sha=hashlib.sha256(json.dumps(names.tolist(),ensure_ascii=False).encode('utf-8')+weights.astype('<i4').tobytes()).hexdigest()
    if expected_draw_sha256 is not None and draw_sha != expected_draw_sha256:
        raise ValueError('Group bootstrap draws differ from the original primary evaluation')
    result={'labels':list(labels),'n_rows':len(ids),'n_groups':len(names),'bootstrap_draws_sha256':draw_sha,
            'bootstrap_replicates':len(weights),'bootstrap_seed':42,'models':{}}
    for model,seeds in MODEL_SEEDS.items():
        if set(fixed[model])!=set(seeds): raise ValueError('Unexpected seed set')
        seed_results={}; seed_draws={name:[] for name in DIRECTIONS}
        unique={name:{'corrected':set(),'introduced':set(),'prediction_changed':set()} for name in DIRECTIONS}
        for seed in seeds:
            cells={name:_align(fixed[model][seed][train][condition],ids,labels)
                   for name,(train,condition) in CELLS.items()}
            metrics={name:classification_metrics(y,pred,labels) for name,pred in cells.items()}
            cell_draw={name:bootstrap_f1(y,pred,labels,row_groups,weights) for name,pred in cells.items()}
            current={'cells':metrics,'matched_pipeline_effect':metrics['N/N']['macro_f1']-metrics['O/O']['macro_f1'],
                     'directions':{}}
            for name,(before,after) in DIRECTIONS.items():
                old,new=cells[before],cells[after]
                transitions=prediction_transitions(y,old,new,changed,labels)
                before_ok=old==y; after_ok=new==y
                for flag,key in ((~before_ok & after_ok,'corrected'),(before_ok & ~after_ok,'introduced'),(old!=new,'prediction_changed')):
                    unique[name][key].update(np.asarray(ids)[flag].tolist())
                diff=metrics[after]['macro_f1']-metrics[before]['macro_f1']
                draws=cell_draw[after]-cell_draw[before]; seed_draws[name].append(draws)
                current['directions'][name]={'subtraction':f'{after} - {before}',
                    'macro_f1_difference':diff,'nominal_conditional_interval':_interval(draws),
                    'transitions':transitions}
            seed_results[seed]=current
        aggregated={'seeds':seed_results,'cells':{},'directions':{},
                    'matched_pipeline_effect':_seed_summary([seed_results[s]['matched_pipeline_effect'] for s in seeds])}
        for cell in CELLS:
            cell_values=[seed_results[s]['cells'][cell] for s in seeds]
            aggregated['cells'][cell]={'macro_f1':_seed_summary([x['macro_f1'] for x in cell_values]),
                'per_class':{label:{'support':cell_values[0]['per_class'][label]['support'],
                            'f1':_seed_summary([x['per_class'][label]['f1'] for x in cell_values])} for label in labels}}
        for name,(before,after) in DIRECTIONS.items():
            counts=[seed_results[s]['directions'][name]['transitions']['all'] for s in seeds]
            values=[seed_results[s]['directions'][name]['macro_f1_difference'] for s in seeds]
            aggregated['directions'][name]={'subtraction':f'{after} - {before}',
                'difference_across_seeds':_seed_summary(values),
                'nominal_conditional_interval_of_seed_mean':_interval(np.mean(seed_draws[name],axis=0)),
                'transition_count_seed_means':{key:float(np.mean([item[key] for item in counts]))
                    for key in ('incorrect_to_correct','correct_to_incorrect','correct_to_correct','incorrect_to_incorrect','prediction_changed')},
                'unique_reviews_any_seed':{key:len(value) for key,value in unique[name].items()}}
        result['models'][model]=aggregated
    return result


def fmt(v): return f'{v:+.4f}'
def interval(i): return f'[{i["lower"]:+.5f}, {i["upper"]:+.5f}]'


def write_report(result):
    lines=['# Frozen four-cell and fixed-model diagnostics','',
           'O/O, O/N, N/O and N/N denote training preprocessing / evaluation input. These analyses reuse the frozen predictions; no models were trained or tuned. Matched-pipeline effect is N/N - O/O. The two fixed-model directions are O/N - O/O and N/O - N/N, respectively. These quantities describe different comparisons and do not identify a causal decomposition of preprocessing, training or separately selected hyperparameters.', '',
           'Fixed-model intervals were added after inspection of test results and are post hoc, exploratory, descriptive intervals. They use the same 2,000 group-bootstrap draws (seed 42) as the primary evaluation. Each draw averages the three paired IndoBERT seed effects. Intervals are nominal 95% percentile intervals conditional on the fitted models, not simultaneous intervals; the additional comparisons have multiplicity and do not change the primary effects or their prespecified +0.01 decision rule. Intervals containing zero do not establish equivalence.', '',
           '| Dataset | Model | O/O | O/N | N/O | N/N | Matched N/N - O/O |', '|---|---|---:|---:|---:|---:|---:|']
    for ds,d in result['datasets'].items():
        for model,m in d['models'].items():
            vals=' | '.join(f'{m["cells"][c]["macro_f1"]["mean"]:.4f}' for c in CELLS)
            lines.append(f'| {ds} | {model} | {vals} | {fmt(m["matched_pipeline_effect"]["mean"])} |')
    lines+=['','| Dataset | Model | Fixed-model change | Mean difference | Nominal 95% interval | Seed SD |', '|---|---|---|---:|---:|---:|']
    for ds,d in result['datasets'].items():
        for model,m in d['models'].items():
            for direction in DIRECTIONS:
                x=m['directions'][direction]; sd=x['difference_across_seeds']['sample_sd']; sd='—' if sd is None else f'{sd:.4f}'
                lines.append(f'| {ds} | {model} | {x["subtraction"]} | {x["difference_across_seeds"]["mean"]:+.5f} | {interval(x["nominal_conditional_interval_of_seed_mean"])} | {sd} |')
    lines+=['','IndoBERT per-seed matched-pipeline scores are computed on the same reviews; the sample standard deviation across these three paired effects is training-seed variability, not a larger independent test sample.', '',
            '| Dataset | Seed | O/O | N/N | Paired N/N - O/O |', '|---|---:|---:|---:|---:|']
    for ds,d in result['datasets'].items():
        for seed,x in d['models']['indobert']['seeds'].items():
            lines.append(f'| {ds} | {seed} | {x["cells"]["O/O"]["macro_f1"]:.4f} | {x["cells"]["N/N"]["macro_f1"]:.4f} | {fmt(x["matched_pipeline_effect"])} |')
    lines+=['','Class F1 below compares matched pipelines; IndoBERT F1 values are arithmetic means across the same three seeds. Class supports count unique held-out reviews once. Full four-cell per-class metrics and per-seed details are in the JSON.', '',
            '| Dataset | Model | Class | Support | O/O F1 | N/N F1 | N/N - O/O |', '|---|---|---|---:|---:|---:|---:|']
    for ds,d in result['datasets'].items():
        for model,m in d['models'].items():
            for label in d['labels']:
                a=m['cells']['O/O']['per_class'][label]; b=m['cells']['N/N']['per_class'][label]
                lines.append(f'| {ds} | {model} | {label} | {a["support"]} | {a["f1"]["mean"]:.4f} | {b["f1"]["mean"]:.4f} | {fmt(b["f1"]["mean"]-a["f1"]["mean"])} |')
    lines+=['','Transition counts below use the indicated direction under a fixed trained model. “Corrected” means incorrect→correct and “introduced” means correct→incorrect. For IndoBERT, averages are seed-specific counts, not counts of unique reviews. Unique counts are unions across seeds within the indicated category; the same review may appear in both categories across different seeds. For SVMs, the single-seed means and unique counts coincide.', '',
            '| Dataset | Model | Direction | Corrected: mean / unique | Introduced: mean / unique | Prediction changed: mean / unique |', '|---|---|---|---:|---:|---:|']
    for ds,d in result['datasets'].items():
        for model,m in d['models'].items():
            for direction in DIRECTIONS:
                x=m['directions'][direction]; a=x['transition_count_seed_means']; u=x['unique_reviews_any_seed']
                lines.append(f'| {ds} | {model} | {x["subtraction"]} | {a["incorrect_to_correct"]:.2f} / {u["corrected"]} | {a["correct_to_incorrect"]:.2f} / {u["introduced"]} | {a["prediction_changed"]:.2f} / {u["prediction_changed"]} |')
    lines+=['','The SmSA IndoBERT matched-pipeline difference and its original-trained fixed-model input difference are separate observations. Few fixed-model prediction changes do not show how much of the pipeline difference was caused by training, substitution, or hyperparameter selection.', '',
            'The original practical margin was recorded as an investigator-selected one-percentage-point reporting threshold; the protocol records no further prior rationale. It is not an externally validated standard. The original criterion remains unchanged; these supplementary analyses add no sensitivity thresholds.', '',
            'Reproduce with `.venv-svm/Scripts/python.exe src/analyze_fixed_models.py`; inputs and script hashes are stored in `results/final_evaluation/fixed_model_diagnostics.json`. Individual-seed effects, all four per-class matrices and detailed transitions are retained there.', '']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def main():
    path=ROOT/'results/final_selection.json'; selection_hash=(ROOT/'results/final_selection.sha256').read_text(encoding='utf-8').split()[0]
    checked_hash(path,selection_hash); manifest=read_json(path); original_path=OUT/'paired_results.json'; original=read_json(original_path)
    if original['selection_sha256'] != selection_hash: raise ValueError('Primary evaluation selection mismatch')
    result={'scope':'Supplementary descriptive analysis of frozen held-out predictions; fixed-model intervals post hoc and exploratory',
            'selection_sha256':selection_hash,'primary_results_sha256':hashlib.sha256(original_path.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'inputs':original['inputs'],'datasets':{}}
    for dataset in ('prdec','smsa'):
        gold,svm,_=load_svm(dataset,manifest,selection_hash)
        bert,_,_,_=load_bert(ROOT/'results/indobert_test',dataset,gold,manifest,selection_hash)
        diagnostic=evaluate_fixed(dataset,gold,{**svm,**bert},original['datasets'][dataset]['bootstrap']['draws_sha256'])
        for model,m in diagnostic['models'].items():
            saved=original['datasets'][dataset]['models'][model]
            for seed,x in m['seeds'].items():
                if abs(x['matched_pipeline_effect']-saved['seeds'][seed]['effect'])>1e-12:
                    raise ValueError('Matched-pipeline result differs from original')
                for cell in ('O/O','N/N'):
                    condition='original' if cell=='O/O' else 'normalized'
                    if abs(x['cells'][cell]['macro_f1']-saved['seeds'][seed]['pipeline_metrics'][condition]['macro_f1'])>1e-12:
                        raise ValueError('Matched score differs from original')
        result['datasets'][dataset]=diagnostic
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_report(result)
    print('Descriptive four-cell diagnostics complete.')


if __name__=='__main__': main()

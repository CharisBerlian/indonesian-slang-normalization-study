"""Final frozen SVM inference. --validate-only never opens test examples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from normalization import cleanup, normalize
from train_svm_pilot import ROOT, digest, verify_lock
from train_svm_search import COUNTS, selected_rows

EXPECTED_TEST = {'prdec': 793, 'smsa': 500}
OUTPUT = ROOT / 'results/final_evaluation'


def checked_hash(path, expected):
    actual = digest(path.read_bytes())
    if actual != expected:
        raise ValueError(f'Hash mismatch: {path}')
    return actual


def verify_selection():
    path = ROOT / 'results/final_selection.json'
    expected = (ROOT / 'results/final_selection.sha256').read_text(encoding='utf-8').split()[0]
    selection_hash = checked_hash(path, expected)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1 or not manifest['no_test_predictions_at_freeze']:
        raise ValueError('Unrecognized or non-pretest selection freeze')
    if verify_lock() != manifest['protocol_id']:
        raise ValueError('Protocol differs from selection')
    checked_hash(ROOT / 'data/protocol_lock.json', manifest['protocol_lock_sha256'])
    checked_hash(ROOT / 'configs/protocol.json', manifest['protocol_config_sha256'])
    checked_hash(ROOT / manifest['svm']['summary_path'], manifest['svm']['summary_sha256'])
    for name, h in manifest['svm']['source_and_config_sha256'].items():
        checked_hash(ROOT / name, h)
    entries = manifest['svm']['models']
    expected_keys = {(ds, rep, cond) for ds in EXPECTED_TEST for rep in ('word','character') for cond in ('original','normalized')}
    keys = [(e['dataset'],e['representation'],e['condition']) for e in entries]
    if len(entries) != 8 or set(keys) != expected_keys:
        raise ValueError('Selected SVM entries are incomplete or duplicated')
    summary = json.loads((ROOT / manifest['svm']['summary_path']).read_text(encoding='utf-8'))
    models = {}; validation = {}; selected = {}
    for dataset in EXPECTED_TEST:
        train, valid, input_hash = selected_rows(dataset)
        selected[dataset] = (train, valid, input_hash)
    for entry in entries:
        ds, rep, cond = entry['dataset'],entry['representation'],entry['condition']
        name = f'{rep}_{cond}'; path = ROOT / entry['checkpoint_path']
        train, valid, input_hash = selected[ds]
        if input_hash != entry['selected_input_prefix_sha256']:
            raise ValueError('Selected input hash mismatch')
        checked_hash(path, entry['checkpoint_sha256'])
        checked_hash(ROOT / entry['validation_predictions_path'], entry['validation_predictions_sha256'])
        recorded = summary['datasets'][ds]['models'][name]
        if recorded['selected_C'] != entry['selected_C'] or recorded['selected_checkpoint_sha256'] != entry['checkpoint_sha256']:
            raise ValueError('Manifest/summary model mismatch')
        if ds not in validation:
            with (ROOT / entry['validation_predictions_path']).open(encoding='utf-8',newline='') as f:
                validation[ds] = list(csv.DictReader(f))
            if len(validation[ds]) != len(valid) or any(any(a[k] != b[k] for k in ('id','group_id','label')) for a,b in zip(validation[ds], valid)):
                raise ValueError('Saved validation records not aligned')
        model = joblib.load(path)
        if model['svm'].C != entry['selected_C']:
            raise ValueError('Loaded model C mismatch')
        original = model.predict([r[cond] for r in valid])
        target = np.asarray([r[f'{name}_C{entry["selected_C"]}'] for r in validation[ds]])
        if not np.array_equal(original, target):
            raise ValueError('Selected validation predictions differ')
        other = 'normalized' if cond == 'original' else 'original'
        alternate = model.predict([r[other] for r in valid])
        target_alt = np.asarray([r[f'{name}_selected_input_{other}'] for r in validation[ds]])
        if not np.array_equal(alternate, target_alt):
            raise ValueError('Switched validation predictions differ')
        models[(ds,rep,cond)] = (entry,model)
    return selection_hash, manifest, selected, models


def load_heldout(dataset, train, valid):
    """Read held-out records only after selection and validation checks complete."""
    skip = sum(COUNTS[dataset].values()); n = EXPECTED_TEST[dataset]
    assignments = {}
    with (ROOT / f'data/splits/{dataset}.csv').open(encoding='utf-8',newline='') as f:
        reader = csv.DictReader(f)
        for _ in range(skip): next(reader)
        for _ in range(n):
            row = next(reader)
            if row['split'] != 'test' or row['id'] in assignments: raise ValueError('Held-out split mismatch')
            assignments[row['id']] = row
        if next(reader, None) is not None: raise ValueError('Unexpected extra split rows')
    mapping = json.loads((ROOT / 'data/processed/normalization_mapping.json').read_text(encoding='utf-8'))
    records=[]; selected_hash=hashlib.sha256()
    with (ROOT / f'data/processed/{dataset}.jsonl').open('rb') as f:
        for _ in range(skip):
            if not f.readline(): raise ValueError('Missing prefix row')
        for _ in range(n):
            line=f.readline()
            if not line: raise ValueError('Missing held-out record')
            selected_hash.update(line); row=json.loads(line)
            assignment=assignments.pop(row['id'],None)
            if not assignment or any(row[k] != assignment[k] for k in ('split','group_id','label')):
                raise ValueError('Held-out metadata mismatch')
            actual=normalize(row['original'],mapping)
            if actual != (row['original'],row['normalized'],row['replacements']): raise ValueError('Held-out normalization mismatch')
            records.append(row)
        if f.readline(): raise ValueError('Unexpected extra processed rows')
    if assignments: raise ValueError('Missing held-out IDs')
    if {r['group_id'] for r in records} & {r['group_id'] for r in train+valid}:
        raise ValueError('Held-out group crosses training or validation')
    lookup={r['id']:r for r in records}; found=set()
    if dataset == 'prdec':
        with (ROOT/'data/raw/prdec.csv').open(encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.DictReader(f),1):
                key=f'prdec:{i}'
                if key in lookup:
                    target=lookup[key]
                    if cleanup(row['Customer Review']) != target['original'] or row['Sentiment'].strip().lower() != target['label']:
                        raise ValueError('Held-out raw source mismatch')
                    found.add(key)
    else:
        with (ROOT/'data/raw/smsa_test.tsv').open(encoding='utf-8') as f:
            for i,line in enumerate(f,1):
                key=f'smsa:test:{i}'
                if key in lookup:
                    text,label=line.rstrip('\r\n').rsplit('\t',1); target=lookup[key]
                    if cleanup(text) != target['original'] or label.strip().lower() != target['label']:
                        raise ValueError('Held-out raw source mismatch')
                    found.add(key)
    if found != set(lookup): raise ValueError('Held-out raw IDs missing')
    return records, selected_hash.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(); started=time.perf_counter()
    selection_hash, manifest, selected, models=verify_selection()
    print('Selection, hashes, and both validation input predictions verified for all 8 SVM models.',flush=True)
    if args.validate_only: return
    output_rows={}; input_hashes={}; times={}
    for ds in EXPECTED_TEST:
        train,valid,_=selected[ds]; rows,test_hash=load_heldout(ds,train,valid); input_hashes[ds]=test_hash
        outputs=[]
        for rep in ('word','character'):
            for trained in ('original','normalized'):
                entry,model=models[(ds,rep,trained)]; classes=model['svm'].classes_.tolist()
                for condition in ('original','normalized'):
                    t0=time.perf_counter(); texts=[r[condition] for r in rows]
                    pred=model.predict(texts); scores=model.decision_function(texts)
                    times[f'{ds}_{rep}_{trained}_{condition}']=time.perf_counter()-t0
                    if scores.ndim == 1:
                        scores=np.stack((-scores,scores),axis=1)
                    if scores.shape != (len(rows),len(classes)) or not np.isfinite(scores).all(): raise ValueError('Invalid decision scores')
                    for r,p,score in zip(rows,pred,scores):
                        by_class=dict(zip(classes, map(float,score)))
                        outputs.append({'id':r['id'],'group_id':r['group_id'],'label':r['label'],
                                        'changed':str(r['original'] != r['normalized']).lower(),
                                        'representation':rep,'trained_condition':trained,'input_condition':condition,
                                        'prediction':p,'score_negative':by_class.get('negative',''),
                                        'score_neutral':by_class.get('neutral',''),'score_positive':by_class.get('positive',''),
                                        'checkpoint_sha256':entry['checkpoint_sha256']})
                first=outputs[-2*len(rows):-len(rows)]; second=outputs[-len(rows):]
                if any(a['changed']=='false' and a['prediction'] != b['prediction'] for a,b in zip(first,second)):
                    raise ValueError('Unchanged text yielded changed prediction')
        output_rows[ds]=outputs
    OUTPUT.mkdir(parents=True,exist_ok=True)
    file_hashes={}
    for ds,rows in output_rows.items():
        path=OUTPUT/f'svm_{ds}_predictions.csv'
        with path.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        file_hashes[ds]=digest(path.read_bytes())
    meta={'scope':'Frozen held-out SVM predictions; no tuning or retraining',
          'selection_sha256':selection_hash,'protocol_id':manifest['protocol_id'],
          'source_sha256':digest(Path(__file__).read_bytes()),'heldout_processed_sha256':input_hashes,
          'output_sha256':file_hashes,'decision_scores':'Raw LinearSVC decision margins, aligned to class labels; binary negative margin is negated positive margin, not calibrated probabilities.',
          'timing_seconds':times,'total_seconds':time.perf_counter()-started,'python':sys.version}
    (OUTPUT/'svm_inference.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print('Frozen SVM inference complete.',flush=True)


if __name__ == '__main__': main()

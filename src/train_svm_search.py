"""Full two-dataset SVM grid using only frozen train/validation examples."""
from collections import Counter
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from normalization import normalize
from train_svm_pilot import ROOT, digest, make_pipeline, parameter_value, vector_diagnostic, verify_lock

OUTPUT = ROOT / 'results/svm_search'
COUNTS = {'prdec': {'train': 3702, 'valid': 793}, 'smsa': {'train': 10919, 'valid': 1260}}


def selected_rows(dataset, root=ROOT):
    """Read exactly the pre-established train/valid prefixes, never test rows."""
    counts = COUNTS[dataset]
    total = sum(counts.values())
    assignments = {}
    with (root / f'data/splits/{dataset}.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i in range(total):
            row = next(reader)
            expected = 'train' if i < counts['train'] else 'valid'
            if row['split'] != expected or row['id'] in assignments:
                raise ValueError(f'{dataset}: split prefix mismatch or duplicate ID')
            assignments[row['id']] = row
    mapping = json.loads((root / 'data/processed/normalization_mapping.json').read_text(encoding='utf-8'))
    records = []
    selected_hasher = hashlib.sha256()
    with (root / f'data/processed/{dataset}.jsonl').open('rb') as f:
        for i in range(total):
            line = f.readline()
            if not line:
                raise ValueError(f'{dataset}: missing processed row')
            selected_hasher.update(line)
            row = json.loads(line)
            expected = 'train' if i < counts['train'] else 'valid'
            split = assignments.pop(row['id'], None)
            if not split or row['split'] != expected or any(row[k] != split[k] for k in ('split', 'group_id', 'label')):
                raise ValueError(f'{dataset}: processed prefix does not match locked split metadata')
            original, normalized, replacements = normalize(row['original'], mapping)
            if (original, normalized, replacements) != (row['original'], row['normalized'], row['replacements']):
                raise ValueError(f'{dataset}: processed normalization mismatch')
            records.append(row)
    if assignments:
        raise ValueError(f'{dataset}: selected rows are incomplete')
    train, valid = records[:counts['train']], records[counts['train']:]
    if {r['group_id'] for r in train} & {r['group_id'] for r in valid}:
        raise ValueError(f'{dataset}: group crosses train and validation')
    return train, valid, selected_hasher.hexdigest()


def select_candidate(candidates):
    """Highest macro-F1, then the smaller C on an exact score tie."""
    return max(candidates, key=lambda item: (item['validation_macro_f1'], -item['C']))


def main():
    started = time.perf_counter()
    protocol = verify_lock()
    cfg = json.loads((ROOT / 'configs/protocol.json').read_text(encoding='utf-8'))
    if cfg['svm']['C'] != [0.1, 1, 10] or cfg['svm']['seed'] != 42:
        raise ValueError('Unexpected SVM grid or seed')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_paths = ['src/train_svm_search.py', 'src/train_svm_pilot.py', 'src/normalization.py',
                    'configs/protocol.json', 'data/protocol_lock.json', 'requirements-pilot-svm-lock.txt']
    results = {'scope': 'Validation model selection only; no test examples or predictions',
               'protocol_id': protocol,
               'source_and_config_sha256': {p: digest((ROOT / p).read_bytes()) for p in source_paths},
               'environment': {'python': sys.version, 'platform': platform.platform(),
                               'scikit_learn': sklearn.__version__, 'numpy': np.__version__,
                               'scipy': scipy.__version__, 'joblib': joblib.__version__,
                               'dependency_freeze': (ROOT / 'requirements-pilot-svm-lock.txt').read_text(encoding='utf-8-sig').splitlines()},
               'selection': 'maximum validation macro-F1; exact ties choose smaller C',
               'datasets': {}}
    report_lines = ['# SVM validation grid', '',
                    'Validation model selection only. These are not test results; no test examples were read for feature construction or prediction. Protocol-locked files, including test source files, were read as bytes only for hash verification.', '',
                    f'Protocol ID: `{protocol}`. Word and character TF-IDF are fitted on training only, separately per condition. C ∈ {0.1, 1, 10}; choose maximum validation macro-F1, with exact ties resolved by smaller C. Seed 42, fixed protocol settings; LinearSVC dual=True (explicit solver choice carried over from the pilot).', '',
                    'The processed JSONL files are not protocol-locked: selected train/validation prefix hashes are recorded in the summary. Each selected record is checked against frozen split ID, group, and label and against the frozen normalizer. PRDECT-ID original text is not independently rechecked against raw in this run because its raw source intersperses test rows; the earlier validation pilot did check selected originals.', '',
                    '| Dataset | Representation | Condition | C=0.1 F1 | C=1 F1 | C=10 F1 | Selected C |',
                    '|---|---|---|---:|---:|---:|---:|']
    diagnostics_lines = []
    for dataset in ('prdec', 'smsa'):
        train, valid, input_hash = selected_rows(dataset)
        ytrain = [r['label'] for r in train]
        yvalid = [r['label'] for r in valid]
        data = {'selected_counts': COUNTS[dataset], 'selected_processed_prefix_sha256': input_hash,
                'models': {}}
        results['datasets'][dataset] = data
        prediction_rows = [{'id': r['id'], 'group_id': r['group_id'], 'label': r['label']} for r in valid]
        for kind in ('word', 'character'):
            for condition in ('original', 'normalized'):
                name = f'{kind}_{condition}'
                candidates = []
                best_model = None
                best_pred = None
                for c in cfg['svm']['C']:
                    print(f'Fitting {dataset} {name} C={c}', flush=True)
                    model = make_pipeline(kind, cfg)
                    model.set_params(svm__C=c)
                    t0 = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit([r[condition] for r in train], ytrain)
                    fit_seconds = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    pred = model.predict([r[condition] for r in valid])
                    predict_seconds = time.perf_counter() - t0
                    key = f'{name}_C{c}'
                    for row, value in zip(prediction_rows, pred):
                        row[key] = value
                    candidate = {'C': c, 'validation_macro_f1': f1_score(yvalid, pred, average='macro', zero_division=0),
                                 'validation_accuracy': accuracy_score(yvalid, pred),
                                 'validation_report': classification_report(yvalid, pred, output_dict=True, zero_division=0),
                                 'fit_seconds': fit_seconds, 'validation_predict_seconds': predict_seconds,
                                 'svm_iterations': int(model['svm'].n_iter_),
                                 'warnings': [str(w.message) for w in caught],
                                 'vocabulary_size': len(model['tfidf'].vocabulary_)}
                    candidates.append(candidate)
                    if select_candidate(candidates) is candidate:
                        best_model, best_pred = model, pred
                    print(f'{dataset} {name} C={c}: validation macro-F1 {candidate["validation_macro_f1"]:.6f}, fit {fit_seconds:.2f}s', flush=True)
                chosen = select_candidate(candidates)
                path = OUTPUT / f'{dataset}_{name}.joblib'
                joblib.dump(best_model, path, compress=3)
                restored = joblib.load(path)
                if not np.array_equal(best_pred, restored.predict([r[condition] for r in valid])):
                    raise ValueError('Reloaded model predictions differ')
                alt = 'normalized' if condition == 'original' else 'original'
                alternative = restored.predict([r[alt] for r in valid])
                for row, value in zip(prediction_rows, alternative):
                    row[f'{name}_selected_input_{alt}'] = value
                result = {'grid': candidates, 'selected_C': chosen['C'],
                          'selected_checkpoint_sha256': digest(path.read_bytes()),
                          'selected_parameters': {'tfidf': {k: parameter_value(v) for k, v in best_model['tfidf'].get_params().items()},
                                                  'svm': best_model['svm'].get_params()},
                          'fixed_model_switched_input_prediction_changes': int(np.count_nonzero(best_pred != alternative)),
                          'vector_diagnostic_train': vector_diagnostic(best_model['tfidf'], train),
                          'vector_diagnostic_valid': vector_diagnostic(best_model['tfidf'], valid)}
                data['models'][name] = result
                report_lines.append(f'| {dataset} | {kind} | {condition} | ' + ' | '.join(f'{x["validation_macro_f1"]:.6f}' for x in candidates) + f' | {chosen["C"]} |')
                a, b = result['vector_diagnostic_train'], result['vector_diagnostic_valid']
                diagnostics_lines.append(f'| {dataset} | {name} | {a["changed_text_with_different_tfidf_vector"]}/{a["text_changed"]} | {b["changed_text_with_different_tfidf_vector"]}/{b["text_changed"]} | {result["fixed_model_switched_input_prediction_changes"]} |')
        with (OUTPUT / f'{dataset}_validation_predictions.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    results['total_seconds'] = time.perf_counter() - started
    results['total_fit_seconds'] = sum(item['fit_seconds'] for d in results['datasets'].values() for model in d['models'].values() for item in model['grid'])
    (OUTPUT / 'summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report_lines.extend(['', f'Total runtime: {results["total_seconds"]:.2f} seconds; summed fit time: {results["total_fit_seconds"]:.2f} seconds.', '',
                         'Fixed-vectorizer survival counts show changed TF-IDF vectors among text-changed examples; they measure whether any change survives filtering, not individual-replacement attribution. Input-switch prediction changes hold the selected model fixed and are not retrained-pipeline effects.', '',
                         '| Dataset | Selected model | Train changed vectors / changed texts | Valid changed vectors / changed texts | Valid prediction changes on input switch |', '|---|---|---:|---:|---:|', *diagnostics_lines, '',
                         'The summary includes per-class metrics, full grid warnings, settings, input and code hashes, environment, and timings. Paired validation predictions for every C and both input forms under each selected model are saved alongside eight selected model checkpoints; each checkpoint was reloaded and prediction-checked. Use the pinned `requirements-pilot-svm-lock.txt` environment and put `src` on the import path when loading trusted models.', ''])
    (ROOT / 'reports/svm_search.md').write_text('\n'.join(report_lines), encoding='utf-8')
    print('Full validation search complete', flush=True)


if __name__ == '__main__':
    main()

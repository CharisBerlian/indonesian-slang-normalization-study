"""Frozen PRDECT-ID validation-only pilot; never select or predict test examples."""
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from normalization import cleanup, normalize, word_tokens

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/svm_pilot'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_lock(root=ROOT):
    lock = json.loads((root / 'data/protocol_lock.json').read_text(encoding='utf-8'))
    hashes = lock['files']
    for relative, expected in hashes.items():
        actual = digest((root / relative).read_bytes())
        if actual != expected:
            raise ValueError(f'Protocol lock mismatch: {relative}')
    computed = digest(json.dumps(hashes, sort_keys=True).encode())
    if computed != lock['protocol_id']:
        raise ValueError('Protocol fingerprint mismatch')
    return lock['protocol_id']


def load_selected(root=ROOT):
    selected = {}
    with (root / 'data/splits/prdec.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] in ('train', 'valid'):
                if row['id'] in selected:
                    raise ValueError('Duplicate split ID')
                selected[row['id']] = row
    counts = Counter(row['split'] for row in selected.values())
    if counts != {'train': 3702, 'valid': 793}:
        raise ValueError(f'Unexpected selected split counts: {counts}')
    # Read only the sorted training/validation prefix; stop before any test JSONL row.
    records = {}
    with (root / 'data/processed/prdec.jsonl').open(encoding='utf-8') as f:
        for _ in range(len(selected)):
            row = json.loads(next(f))
            ident = row['id']
            if ident not in selected or ident in records:
                raise ValueError('Processed prefix differs from frozen selected IDs')
            expected = selected[ident]
            if any(row[k] != expected[k] for k in ('split', 'group_id', 'label')):
                raise ValueError('Processed split metadata mismatch')
            records[ident] = row
    if records.keys() != selected.keys():
        raise ValueError('Selected IDs do not match processed records')
    mapping = json.loads((root / 'data/processed/normalization_mapping.json').read_text(encoding='utf-8'))
    # Only access source text/labels for selected row numbers. Skip all other source rows.
    remaining = set(records)
    with (root / 'data/raw/prdec.csv').open(encoding='utf-8-sig', newline='') as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            ident = f'prdec:{i}'
            if ident not in records:
                continue
            item = records[ident]
            if cleanup(row['Customer Review']) != item['original'] or row['Sentiment'].strip().lower() != item['label']:
                raise ValueError('Processed selected row differs from locked raw source')
            original, normalized, replacements = normalize(item['original'], mapping)
            if (original, normalized, replacements) != (item['original'], item['normalized'], item['replacements']):
                raise ValueError('Processed normalization mismatch')
            remaining.remove(ident)
            if not remaining:
                break
    if remaining:
        raise ValueError('Selected source rows missing')
    train = [r for r in records.values() if r['split'] == 'train']
    valid = [r for r in records.values() if r['split'] == 'valid']
    if {r['group_id'] for r in train} & {r['group_id'] for r in valid}:
        raise ValueError('Training and validation groups overlap')
    return train, valid


def make_pipeline(kind, cfg):
    tf = cfg['tfidf']
    args = dict(min_df=tf['min_df'], sublinear_tf=tf['sublinear_tf'],
                smooth_idf=tf['smooth_idf'], norm=tf['norm'], use_idf=True,
                dtype=np.float64)
    if kind == 'word':
        args.update(analyzer='word', tokenizer=word_tokens, token_pattern=None,
                    lowercase=False, ngram_range=tuple(tf['word_ngram_range']),
                    max_features=tf['word_max_features'])
    elif kind == 'character':
        # Match the audit tokenizer's casefold policy before character n-grams.
        args.update(analyzer='char', preprocessor=str.casefold, lowercase=False,
                    ngram_range=tuple(tf['character_ngram_range']),
                    max_features=tf['character_max_features'])
    else:
        raise ValueError('Unknown representation')
    svm = cfg['svm']
    return Pipeline([('tfidf', TfidfVectorizer(**args)),
                     ('svm', LinearSVC(C=1, random_state=svm['seed'], loss=svm['loss'],
                                       penalty=svm['penalty'], class_weight=svm['class_weight'],
                                       tol=svm['tol'], max_iter=svm['max_iter'], dual=True))])


def vector_diagnostic(vectorizer, rows):
    original = vectorizer.transform([r['original'] for r in rows])
    normalized = vectorizer.transform([r['normalized'] for r in rows])
    difference = original - normalized
    difference.eliminate_zeros()
    changed_vectors = np.asarray(difference.getnnz(axis=1)).ravel() > 0
    changed_text = np.array([r['original'] != r['normalized'] for r in rows])
    return {'rows': len(rows), 'text_changed': int(changed_text.sum()),
            'changed_text_with_different_tfidf_vector': int((changed_text & changed_vectors).sum()),
            'changed_text_with_identical_tfidf_vector': int((changed_text & ~changed_vectors).sum()),
            'all_rows_with_different_tfidf_vector': int(changed_vectors.sum()),
            'original_zero_vectors': int((original.getnnz(axis=1) == 0).sum()),
            'normalized_zero_vectors': int((normalized.getnnz(axis=1) == 0).sum())}


def parameter_value(value):
    if callable(value):
        return f'{getattr(value, "__module__", None) or "builtins"}.{value.__qualname__}'
    return value


def main():
    started = time.perf_counter()
    protocol_id = verify_lock()
    cfg = json.loads((ROOT / 'configs/protocol.json').read_text(encoding='utf-8'))
    train, valid = load_selected()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    y_train, y_valid = [r['label'] for r in train], [r['label'] for r in valid]
    results = {'scope': 'Preliminary untuned validation only; no test predictions',
               'protocol_id': protocol_id, 'dataset': 'PRDECT-ID',
               'selected_counts': {'train': len(train), 'valid': len(valid)},
               'environment': {'python': sys.version, 'platform': platform.platform(),
                               'scikit_learn': sklearn.__version__, 'numpy': np.__version__,
                               'scipy': scipy.__version__, 'joblib': joblib.__version__},
               'models': {}}
    predictions = [{'id': r['id'], 'group_id': r['group_id'], 'label': r['label']} for r in valid]
    for kind in ('word', 'character'):
        for condition in ('original', 'normalized'):
            name = f'{kind}_{condition}'
            print(f'Fitting {name}', flush=True)
            model = make_pipeline(kind, cfg)
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit([r[condition] for r in train], y_train)
            fit_sec = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = model.predict([r[condition] for r in valid])
            predict_sec = time.perf_counter() - t0
            for row, prediction in zip(predictions, pred):
                row[name] = prediction
            path = OUTPUT / f'{name}.joblib'
            joblib.dump(model, path, compress=3)
            restored = joblib.load(path)
            if not np.array_equal(pred, restored.predict([r[condition] for r in valid])):
                raise ValueError('Reloaded checkpoint predictions differ')
            other = 'normalized' if condition == 'original' else 'original'
            alternative = model.predict([r[other] for r in valid])
            entry = {'fit_seconds': fit_sec, 'validation_predict_seconds': predict_sec,
                     'vocabulary_size': len(model['tfidf'].vocabulary_),
                     'svm_iterations': int(model['svm'].n_iter_),
                     'warnings': [str(w.message) for w in caught],
                     'validation_accuracy': accuracy_score(y_valid, pred),
                     'validation_macro_f1': f1_score(y_valid, pred, average='macro', zero_division=0),
                     'validation_report': classification_report(y_valid, pred, output_dict=True, zero_division=0),
                     'fixed_model_other_input_prediction_changes': int(np.count_nonzero(pred != alternative)),
                     'vector_diagnostic_train': vector_diagnostic(model['tfidf'], train),
                     'vector_diagnostic_valid': vector_diagnostic(model['tfidf'], valid),
                     'checkpoint_sha256': digest(path.read_bytes()),
                     'parameters': {'tfidf': {k: parameter_value(v)
                                              for k, v in model['tfidf'].get_params().items()},
                                    'svm': model['svm'].get_params()}}
            results['models'][name] = entry
            print(f'{name}: macro-F1={entry["validation_macro_f1"]:.4f}; fit={fit_sec:.1f}s', flush=True)
    results['total_seconds'] = time.perf_counter() - started
    with (OUTPUT / 'validation_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (OUTPUT / 'summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# PRDECT-ID SVM pilot: validation only', '',
             'Preliminary, untuned C=1 pilot. These are validation results, not paper test results. No test examples were selected for feature construction or prediction. Locked source files are hashed and the source CSV is scanned to locate selected rows; nonselected rows are skipped before their text or labels are accessed.', '',
             f'Protocol ID: `{protocol_id}`. Frozen training: {len(train):,}; validation: {len(valid):,}. Training-only vocabularies; fixed paired rows; seed 42.', '',
             'Word: audit Unicode tokenization with casefold, 1–2 grams. Character: casefold, 3–5 contiguous character grams. Both: min_df=2, sublinear TF, smooth IDF, L2 normalization; max features 50,000 word / 100,000 character. LinearSVC uses C=1, squared hinge, L2 penalty, dual=True, tolerance 0.0001 and 20,000 maximum iterations. The dual solver is an explicit pilot choice; it was not fixed in the existing protocol.', '',
             '| Model | Validation macro-F1 | Accuracy | Vocabulary | Fit seconds | Iterations |', '|---|---:|---:|---:|---:|---:|']
    for name, r in results['models'].items():
        lines.append(f'| {name} | {r["validation_macro_f1"]:.4f} | {r["validation_accuracy"]:.4f} | {r["vocabulary_size"]:,} | {r["fit_seconds"]:.2f} | {r["svm_iterations"]} |')
    lines += ['', 'Vector survival is measured by transforming each original/normalized pair through the same fitted vectorizer and comparing the resulting TF-IDF vectors. Counts are among text-changed rows, so this measures whether any vector-level change survives all vocabulary filtering; it does not attribute survival to individual replacements.', '',
              '| Fixed vectorizer | Train changed vectors / changed texts | Validation changed vectors / changed texts | Validation changed predictions when input switched |', '|---|---:|---:|---:|']
    for name, r in results['models'].items():
        a, b = r['vector_diagnostic_train'], r['vector_diagnostic_valid']
        lines.append(f'| {name} | {a["changed_text_with_different_tfidf_vector"]} / {a["text_changed"]} | {b["changed_text_with_different_tfidf_vector"]} / {b["text_changed"]} | {r["fixed_model_other_input_prediction_changes"]} |')
    lines += ['', 'The switch-input diagnostic holds each trained model and its vocabulary fixed. It differs from the comparison between separately trained pipelines and is not a causal estimate. No hyperparameter selection or statistical inference was performed.', '',
              'Models, row-aligned validation predictions, per-class metrics, warnings, timings, parameters and environment are in `results/svm_pilot/`. Each saved model was reloaded and checked against all validation predictions. Install `requirements-pilot-svm-lock.txt`, then run `python src/train_svm_pilot.py` from this project. Load only trusted joblib files using the same environment; add `src` to Python’s import path so the frozen tokenizer is available.', '']
    (ROOT / 'reports/svm_pilot.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Pilot complete', flush=True)


if __name__ == '__main__':
    main()

"""Paired metrics on caller-supplied predictions; this module loads no datasets or models.

Prediction maps are keyed by record ID so misalignment is rejected.  Full use
requires three paired IndoBERT seeds and one seed each for both SVM families.
"""
from collections import Counter
import hashlib
import json

import numpy as np

LABELS = {'prdec': ('negative', 'positive'), 'smsa': ('negative', 'neutral', 'positive')}
MODEL_SEEDS = {'word': ('42',), 'character': ('42',), 'indobert': ('13', '37', '73')}
CONDITIONS = ('original', 'normalized')
REPLICATES = 2000
BOOTSTRAP_SEED = 42
MARGIN = 0.01


def _gold(gold, dataset):
    if dataset not in LABELS or not gold:
        raise ValueError('Unknown dataset or empty gold records')
    ids, labels, groups, changed = [], [], [], []
    for row in gold:
        ident, label, group = row['id'], row['label'], row['group_id']
        flag = row['changed']
        if not isinstance(ident, str) or not ident or not isinstance(group, str) or not group:
            raise ValueError('IDs and group IDs must be nonempty strings')
        if label not in LABELS[dataset] or not isinstance(flag, (bool, np.bool_)):
            raise ValueError('Gold label or changed flag is invalid')
        ids.append(ident); labels.append(label); groups.append(group); changed.append(bool(flag))
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate gold IDs')
    return ids, np.asarray(labels), np.asarray(groups), np.asarray(changed)


def _align(mapping, ids, label_set):
    if not isinstance(mapping, dict) or set(mapping) != set(ids):
        raise ValueError('Prediction IDs must match gold IDs exactly')
    values = [mapping[i] for i in ids]
    if any(value not in label_set for value in values):
        raise ValueError('Prediction outside the frozen label set')
    return np.asarray(values)


def classification_metrics(y, pred, labels):
    """Accuracy, fixed-label macro/per-class metrics, and true-by-predicted matrix."""
    if len(y) != len(pred) or not len(y):
        raise ValueError('Predictions and nonempty gold must have equal lengths')
    if not set(y).issubset(labels) or not set(pred).issubset(labels):
        raise ValueError('Label outside fixed label set')
    matrix = [[int(np.count_nonzero((y == a) & (pred == b))) for b in labels] for a in labels]
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]; actual = sum(matrix[i]); predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * tp / (actual + predicted) if actual + predicted else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': actual}
    return {'accuracy': float(np.mean(y == pred)),
            'macro_f1': float(np.mean([v['f1'] for v in per_class.values()])),
            'labels': list(labels), 'confusion_matrix': matrix, 'per_class': per_class}


def group_draw_weights(groups, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    """Deterministic bootstrap group multiplicities; columns follow sorted group IDs."""
    if replicates <= 0 or not len(groups):
        raise ValueError('Bootstrap needs positive replicate and group counts')
    names, row_groups = np.unique(groups, return_inverse=True)
    n = len(names)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(replicates, n))
    weights = np.zeros((replicates, n), dtype=np.int32)
    np.add.at(weights, (np.arange(replicates)[:, None], draws), 1)
    return names, row_groups, weights


def bootstrap_f1(y, pred, labels, row_groups, weights):
    """Weighted group confusion counts for each bootstrap draw, fixed label set."""
    scores = []
    for label in labels:
        actual = y == label; predicted = pred == label
        tp = np.bincount(row_groups, weights=(actual & predicted), minlength=weights.shape[1])
        denom = np.bincount(row_groups, weights=actual.astype(np.int8) + predicted.astype(np.int8), minlength=weights.shape[1])
        num = 2 * (weights @ tp); den = weights @ denom
        scores.append(np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0))
    return np.mean(scores, axis=0)


def _interval(draws):
    low, high = np.percentile(draws, [2.5, 97.5], method='linear')
    return {'lower': float(low), 'upper': float(high), 'level': 0.95,
            'method': 'percentile; nominal; conditional on fitted models'}


def _seed_summary(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'sample_sd': float(a.std(ddof=1)) if len(a) > 1 else None}


def subset_metrics(y, pred, changed, labels):
    """Descriptive changed/unchanged results, retaining the full label set."""
    result = {}
    for name, mask in (('changed', changed), ('unchanged', ~changed)):
        result[name] = {'n': int(mask.sum()),
                        'class_support': {label: int(np.count_nonzero(mask & (y == label))) for label in labels},
                        'metrics': classification_metrics(y[mask], pred[mask], labels) if mask.any() else None}
    return result


def prediction_transitions(y, original_input, normalized_input, changed, labels):
    """Compare two inputs under ONE fixed model, including changed/unchanged subsets."""
    if len(y) != len(original_input) or len(y) != len(normalized_input) or len(y) != len(changed):
        raise ValueError('Transition arrays differ in length')
    if np.any((~changed) & (original_input != normalized_input)):
        raise ValueError('Unchanged input produced different predictions under a fixed model')
    def counts(mask):
        old_correct = original_input == y; new_correct = normalized_input == y
        result = {'n': int(mask.sum()), 'class_support': {label: int(np.count_nonzero(mask & (y == label))) for label in labels},
                  'incorrect_to_correct': int(np.count_nonzero(mask & ~old_correct & new_correct)),
                  'correct_to_incorrect': int(np.count_nonzero(mask & old_correct & ~new_correct)),
                  'correct_to_correct': int(np.count_nonzero(mask & old_correct & new_correct)),
                  'incorrect_to_incorrect': int(np.count_nonzero(mask & ~old_correct & ~new_correct)),
                  'prediction_changed': int(np.count_nonzero(mask & (original_input != normalized_input)))}
        result['per_class'] = {}
        for label in labels:
            subset = mask & (y == label)
            result['per_class'][label] = {'support': int(subset.sum()),
                'incorrect_to_correct': int(np.count_nonzero(subset & ~old_correct & new_correct)),
                'correct_to_incorrect': int(np.count_nonzero(subset & old_correct & ~new_correct))}
        return result
    return {'all': counts(np.ones(len(y), dtype=bool)), 'changed': counts(changed), 'unchanged': counts(~changed)}


def evaluate_paired(dataset, gold, predictions, fixed_model_predictions=None,
                    replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    """Evaluate provided aligned predictions without accessing source data.

    gold: rows with id, group_id, label, changed (bool).
    predictions[model][seed][condition][id] = predicted label; model/seed keys
    must exactly match MODEL_SEEDS, with condition original or normalized.
    fixed_model_predictions, when present, has the same outer keys but adds
    [trained_condition][input_condition][id]. Its matching-input predictions
    must equal the pipeline predictions. The same group draws are shared across
    all models, preprocessing conditions and seeds in this dataset.
    """
    ids, y, groups, changed = _gold(gold, dataset)
    labels = LABELS[dataset]
    if set(predictions) != set(MODEL_SEEDS):
        raise ValueError('All three model families must be present')
    if fixed_model_predictions is not None and set(fixed_model_predictions) != set(MODEL_SEEDS):
        raise ValueError('Fixed-model predictions must include all model families')
    names, row_groups, weights = group_draw_weights(groups, replicates, seed)
    # Hash group names and the actual weights, so the paired draw stream is auditable.
    draw_hash = hashlib.sha256(json.dumps(names.tolist(), ensure_ascii=False).encode('utf-8') + weights.astype('<i4').tobytes()).hexdigest()
    result = {'dataset': dataset, 'n_rows': len(ids), 'n_groups': len(names),
              'labels': list(labels), 'bootstrap': {'replicates': replicates, 'seed': seed,
              'group_order': 'sorted group IDs', 'draws_sha256': draw_hash,
              'sampling': 'groups with replacement; all member rows repeated',
              'interval': '95% nominal percentile, conditional on trained models'},
              'practical_margin': MARGIN, 'models': {}}
    draw_effects = {}; point_effects = {}
    for model, seeds in MODEL_SEEDS.items():
        if set(predictions[model]) != set(seeds):
            raise ValueError(f'Unexpected seed set for {model}')
        if fixed_model_predictions is not None and set(fixed_model_predictions[model]) != set(seeds):
            raise ValueError(f'Unexpected fixed-model seed set for {model}')
        entries = {}; deltas = []; draws_by_seed = []
        for model_seed in seeds:
            inputs = predictions[model][model_seed]
            if set(inputs) != set(CONDITIONS):
                raise ValueError('Both preprocessing conditions are required')
            aligned = {c: _align(inputs[c], ids, labels) for c in CONDITIONS}
            metrics = {c: classification_metrics(y, aligned[c], labels) for c in CONDITIONS}
            effect = metrics['normalized']['macro_f1'] - metrics['original']['macro_f1']
            draw = bootstrap_f1(y, aligned['normalized'], labels, row_groups, weights) - bootstrap_f1(y, aligned['original'], labels, row_groups, weights)
            entry = {'pipeline_metrics': metrics,
                     'pipeline_subset_metrics': {c: subset_metrics(y, aligned[c], changed, labels) for c in CONDITIONS},
                     'effect': effect, 'effect_interval': _interval(draw)}
            if fixed_model_predictions is not None:
                fixed = fixed_model_predictions[model][model_seed]
                if set(fixed) != set(CONDITIONS):
                    raise ValueError('Both fixed trained conditions are required')
                entry['fixed_model'] = {}
                for trained in CONDITIONS:
                    if set(fixed[trained]) != set(CONDITIONS):
                        raise ValueError('Both fixed-model input conditions are required')
                    alt = {c: _align(fixed[trained][c], ids, labels) for c in CONDITIONS}
                    if not np.array_equal(alt[trained], aligned[trained]):
                        raise ValueError('Fixed-model matching input differs from pipeline predictions')
                    entry['fixed_model'][trained] = {'input_metrics': {c: classification_metrics(y, alt[c], labels) for c in CONDITIONS},
                                                     'input_subset_metrics': {c: subset_metrics(y, alt[c], changed, labels) for c in CONDITIONS},
                                                     'transitions': prediction_transitions(y, alt['original'], alt['normalized'], changed, labels)}
            entries[model_seed] = entry; deltas.append(effect); draws_by_seed.append(draw)
        averaged_draw = np.mean(draws_by_seed, axis=0)
        draw_effects[model] = averaged_draw; point_effects[model] = float(np.mean(deltas))
        scores = {c: _seed_summary([entries[s]['pipeline_metrics'][c]['macro_f1'] for s in seeds]) for c in CONDITIONS}
        result['models'][model] = {'seeds': entries, 'macro_f1_across_seeds': scores,
                                   'effect_across_seeds': _seed_summary(deltas),
                                   'mean_effect_interval': _interval(averaged_draw)}
    result['direct_contrasts'] = {}
    for other in ('character', 'indobert'):
        key = f'word_minus_{other}'
        result['direct_contrasts'][key] = {'estimate': point_effects['word'] - point_effects[other],
                                            'interval': _interval(draw_effects['word'] - draw_effects[other])}
    return result

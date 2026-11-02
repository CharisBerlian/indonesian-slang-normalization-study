from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate_models import (LABELS, MODEL_SEEDS, bootstrap_f1, classification_metrics,
                             evaluate_paired, group_draw_weights, prediction_transitions)


def fixture():
    rows = [{'id':'a','group_id':'g1','label':'positive','changed':True},
            {'id':'b','group_id':'g1','label':'negative','changed':True},
            {'id':'c','group_id':'g2','label':'positive','changed':False},
            {'id':'d','group_id':'g3','label':'negative','changed':True}]
    original = {'a':'negative','b':'negative','c':'positive','d':'positive'}
    normalized = {'a':'positive','b':'positive','c':'positive','d':'negative'}
    predictions = {model: {seed: {'original': dict(original), 'normalized': dict(normalized)}
                            for seed in seeds} for model, seeds in MODEL_SEEDS.items()}
    fixed = {model: {seed: {train: {'original': dict(original), 'normalized': dict(normalized)}
                            for train in ('original','normalized')}
                     for seed in seeds} for model, seeds in MODEL_SEEDS.items()}
    return rows, predictions, fixed


class EvaluationTests(unittest.TestCase):
    def test_fixed_label_f1_and_absent_class(self):
        y = np.array(['negative','positive','positive']); p = np.array(['negative','negative','positive'])
        result = classification_metrics(y, p, LABELS['smsa'])
        self.assertAlmostEqual(result['macro_f1'], f1_score(y, p, labels=list(LABELS['smsa']), average='macro', zero_division=0))
        self.assertEqual(result['per_class']['neutral']['f1'], 0)
        self.assertEqual(result['confusion_matrix'], [[1,0,0],[0,0,0],[1,0,1]])

    def test_group_resampling_matches_expanded_rows(self):
        gold, predictions, _ = fixture(); labels=LABELS['prdec']
        y = np.array([r['label'] for r in gold]); groups=np.array([r['group_id'] for r in gold])
        p = np.array([predictions['word']['42']['original'][r['id']] for r in gold])
        _, row_groups, weights = group_draw_weights(groups, replicates=41, seed=42)
        actual = bootstrap_f1(y, p, labels, row_groups, weights)
        expected=[]
        for draw in weights:
            ix = np.repeat(np.arange(len(y)), draw[row_groups])
            expected.append(f1_score(y[ix], p[ix], labels=list(labels), average='macro', zero_division=0))
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(weights, group_draw_weights(groups, replicates=41, seed=42)[2])

    def test_paired_effect_contrasts_and_fixed_transitions(self):
        gold, predictions, fixed = fixture()
        result=evaluate_paired('prdec',gold,predictions,fixed,replicates=41)
        self.assertGreater(result['models']['word']['effect_across_seeds']['mean'], 0)
        self.assertEqual(result['models']['indobert']['effect_across_seeds']['sample_sd'], 0)
        for contrast in result['direct_contrasts'].values():
            self.assertEqual(contrast['estimate'], 0)
            self.assertEqual(contrast['interval']['lower'], 0)
            self.assertEqual(contrast['interval']['upper'], 0)
        t=result['models']['word']['seeds']['42']['fixed_model']['original']['transitions']
        subset=result['models']['word']['seeds']['42']['pipeline_subset_metrics']['original']
        self.assertEqual(subset['changed']['n'], 3)
        self.assertEqual(subset['unchanged']['class_support']['positive'], 1)
        self.assertEqual([t['all'][k] for k in ('incorrect_to_correct','correct_to_incorrect','correct_to_correct','incorrect_to_incorrect')], [2,1,1,0])
        self.assertEqual(t['unchanged']['n'], 1)
        self.assertEqual(t['unchanged']['prediction_changed'], 0)

    def test_rejects_misaligned_ids_and_unchanged_prediction_change(self):
        gold, predictions, fixed = fixture()
        bad=deepcopy(predictions); bad['character']['42']['original']['unknown']=bad['character']['42']['original'].pop('a')
        with self.assertRaisesRegex(ValueError, 'Prediction IDs'):
            evaluate_paired('prdec',gold,bad,replicates=4)
        fixed['word']['42']['original']['normalized']['c']='negative'
        with self.assertRaisesRegex(ValueError, 'Unchanged input'):
            evaluate_paired('prdec',gold,predictions,fixed,replicates=4)

    def test_zero_effect_and_sample_seed_variability(self):
        gold, predictions, _=fixture()
        for model in predictions:
            for seed in predictions[model]:
                predictions[model][seed]['normalized']=dict(predictions[model][seed]['original'])
        result=evaluate_paired('prdec',gold,predictions,replicates=50)
        for model in MODEL_SEEDS:
            self.assertEqual(result['models'][model]['mean_effect_interval']['lower'],0)
            self.assertEqual(result['models'][model]['mean_effect_interval']['upper'],0)


if __name__ == '__main__': unittest.main()

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_fixed_models import evaluate_fixed
from test_evaluate_models import fixture


class FixedDiagnosticTests(unittest.TestCase):
    def test_directions_and_original_estimand(self):
        gold,_,fixed=fixture()
        result=evaluate_fixed('prdec',gold,fixed)
        word=result['models']['word']; seed=word['seeds']['42']
        oo=seed['cells']['O/O']['macro_f1']; on=seed['cells']['O/N']['macro_f1']
        no=seed['cells']['N/O']['macro_f1']; nn=seed['cells']['N/N']['macro_f1']
        self.assertAlmostEqual(seed['matched_pipeline_effect'],nn-oo)
        self.assertAlmostEqual(seed['directions']['O_to_N']['macro_f1_difference'],on-oo)
        self.assertAlmostEqual(seed['directions']['N_to_O']['macro_f1_difference'],no-nn)
        forward=seed['directions']['O_to_N']['transitions']['all']
        reverse=seed['directions']['N_to_O']['transitions']['all']
        self.assertEqual(forward['incorrect_to_correct'],reverse['correct_to_incorrect'])
        self.assertEqual(forward['correct_to_incorrect'],reverse['incorrect_to_correct'])
        self.assertAlmostEqual(word['directions']['O_to_N']['nominal_conditional_interval_of_seed_mean']['lower'],
                               -word['directions']['N_to_O']['nominal_conditional_interval_of_seed_mean']['upper'])

    def test_seed_mean_counts_are_not_unique_review_counts(self):
        gold,_,fixed=fixture()
        # Correct review a in two seeds, review d in the third; each seed fixes one.
        for seed in ('13','37'):
            fixed['indobert'][seed]['original']['normalized']['d']='positive'
        fixed['indobert']['73']['original']['normalized']['a']='negative'
        result=evaluate_fixed('prdec',gold,fixed)
        direction=result['models']['indobert']['directions']['O_to_N']
        self.assertEqual(direction['transition_count_seed_means']['incorrect_to_correct'],1)
        self.assertEqual(direction['unique_reviews_any_seed']['corrected'],2)

    def test_rejects_bootstrap_stream_mismatch(self):
        gold,_,fixed=fixture()
        with self.assertRaisesRegex(ValueError,'bootstrap draws differ'):
            evaluate_fixed('prdec',gold,fixed,expected_draw_sha256='wrong')


if __name__=='__main__': unittest.main()

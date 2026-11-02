import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import train_svm_search as search


class SearchTests(unittest.TestCase):
    def test_highest_score_and_smaller_c_on_exact_tie(self):
        candidates = [{'C': 10, 'validation_macro_f1': .8}, {'C': .1, 'validation_macro_f1': .8},
                      {'C': 1, 'validation_macro_f1': .79}]
        self.assertEqual(search.select_candidate(candidates)['C'], .1)
        candidates[-1]['validation_macro_f1'] = .81
        self.assertEqual(search.select_candidate(candidates)['C'], 1)

    def test_loader_stops_before_test_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/splits').mkdir(parents=True)
            (root / 'data/processed').mkdir(parents=True)
            (root / 'data/processed/normalization_mapping.json').write_text('{}', encoding='utf-8')
            rows = [dict(id='x:1', group_id='g1', label='positive', split='train', original='baik', normalized='baik', replacements=[]),
                    dict(id='x:2', group_id='g2', label='negative', split='valid', original='buruk', normalized='buruk', replacements=[])]
            with (root / 'data/splits/example.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'group_id', 'label', 'split'])
                writer.writeheader()
                writer.writerows([{k: row[k] for k in writer.fieldnames} for row in rows])
                f.write('test,sentinel,sentinel,test\n')
            with (root / 'data/processed/example.jsonl').open('w', encoding='utf-8', newline='\n') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
                f.write('THIS IS NOT JSON AND MUST NOT BE PARSED\n')
            search.COUNTS['example'] = {'train': 1, 'valid': 1}
            try:
                train, valid, digest = search.selected_rows('example', root)
            finally:
                del search.COUNTS['example']
            self.assertEqual([train[0]['id'], valid[0]['id']], ['x:1', 'x:2'])
            self.assertEqual(len(digest), 64)


if __name__ == '__main__':
    unittest.main()

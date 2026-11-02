import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train_svm_pilot import digest, make_pipeline, vector_diagnostic, verify_lock


class PilotTests(unittest.TestCase):
    def test_lock_detects_changed_file_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'frozen').write_bytes(b'original')
            hashes = {'frozen': digest(b'original')}
            lock = {'files': hashes, 'protocol_id': digest(json.dumps(hashes, sort_keys=True).encode())}
            (root / 'data/protocol_lock.json').write_text(json.dumps(lock), encoding='utf-8')
            self.assertEqual(verify_lock(root), lock['protocol_id'])
            (root / 'frozen').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Protocol lock mismatch'):
                verify_lock(root)

    def test_training_vocabulary_excludes_validation_only_words_and_reload(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/protocol.json').read_text(encoding='utf-8'))
        train = ['baik baik bagus', 'baik bagus', 'buruk jelek', 'buruk jelek']
        labels = ['positive', 'positive', 'negative', 'negative']
        validation = ['baikk unikvalidasi', 'buruk unikvalidasi']
        model = make_pipeline('word', cfg).fit(train, labels)
        self.assertNotIn('unikvalidasi', model['tfidf'].vocabulary_)
        before = model.predict(validation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(model, path)
            after = joblib.load(path).predict(validation)
        np.testing.assert_array_equal(before, after)
        self.assertNotIn('unikvalidasi', model['tfidf'].vocabulary_)

    def test_vector_survival_uses_fixed_vocabulary(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/protocol.json').read_text(encoding='utf-8'))
        model = make_pipeline('word', cfg).fit(['baik bagus', 'baik bagus', 'buruk jelek', 'buruk jelek'],
                                                ['positive', 'positive', 'negative', 'negative'])
        rows = [{'original': 'baik tidakdikenal', 'normalized': 'baik takdikenal'},
                {'original': 'baik', 'normalized': 'buruk'},
                {'original': 'baik', 'normalized': 'baik'}]
        diagnostic = vector_diagnostic(model['tfidf'], rows)
        self.assertEqual(diagnostic['text_changed'], 2)
        self.assertEqual(diagnostic['changed_text_with_different_tfidf_vector'], 1)
        self.assertEqual(diagnostic['changed_text_with_identical_tfidf_vector'], 1)


if __name__ == '__main__':
    unittest.main()

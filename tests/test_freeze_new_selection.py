import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from copy import deepcopy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from freeze_new_selection import best_epoch, check_history_metrics, pretest_guard, select_tuning, require_score, validate_training_provenance
from selection_io import read_verified_selection
from weight_integrity import evaluator_mode, enforce_frozen_weight_hash, HISTORICAL_SELECTION_SHA256


class FreezeGuardTests(unittest.TestCase):
    def test_guard_refuses_any_prior_selection_or_heldout_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'results').mkdir()
            pretest_guard(root)
            for relative in ('final_selection.json','final_selection.sha256','final_evaluation','indobert_test'):
                path=root/'results'/relative
                if relative in ('final_evaluation','indobert_test'): path.mkdir()
                else: path.write_text('existing',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Refusing freeze'): pretest_guard(root)
                if path.is_dir(): path.rmdir()
                else: path.unlink()

    def test_selection_bytes_require_exact_companion_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=root/'selection.json'; sha=root/'selection.sha256'
            manifest.write_bytes(b'{"condition":"validation"}\n')
            digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
            sha.write_text(digest+'  selection.json\n',encoding='utf-8')
            self.assertEqual(read_verified_selection(manifest,sha)[1],digest)
            manifest.write_bytes(b'{"condition":"different"}\n')
            with self.assertRaisesRegex(ValueError,'mismatch'): read_verified_selection(manifest,sha)
            sha.write_text('not-a-valid-hash',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Malformed'): read_verified_selection(manifest,sha)

    def test_frozen_validation_tie_rules(self):
        history=[{'epoch':i,'metrics':{'macro_f1':.7 if i not in (2,4) else .8}} for i in range(1,6)]
        self.assertEqual(best_epoch(history)['epoch'],2)
        records=[{'learning_rate':2e-5,'best_epoch':4,'best_macro_f1':.8},
                 {'learning_rate':3e-5,'best_epoch':2,'best_macro_f1':.8},
                 {'learning_rate':5e-5,'best_epoch':2,'best_macro_f1':.8}]
        self.assertEqual(select_tuning(records)['learning_rate'],3e-5)
        records.pop()
        with self.assertRaisesRegex(ValueError,'three frozen'): select_tuning(records)

    def test_nonfinite_metrics_and_frozen_weight_gate(self):
        for actual, expected in ((1.0,float('nan')),(float('nan'),1.0)):
            with self.assertRaisesRegex(ValueError,'score mismatch'): require_score(actual,expected)
        digest='a'*64; enforce_frozen_weight_hash(digest,digest)
        with self.assertRaisesRegex(ValueError,'mismatch'): enforce_frozen_weight_hash(digest,'b'*64)
        with self.assertRaisesRegex(ValueError,'valid SHA'): enforce_frozen_weight_hash(None,digest)
        bound={'indobert':{'models':[{'remote_checkpoint_sha256':digest} for _ in range(12)]}}
        self.assertEqual(evaluator_mode(bound,'c'*64),'prefreeze_hash_bound')
        unbound={'indobert':{'models':[{'remote_checkpoint_sha256':None} for _ in range(12)]}}
        self.assertEqual(evaluator_mode(unbound,HISTORICAL_SELECTION_SHA256),'historical_unbound')
        with self.assertRaisesRegex(ValueError,'historical freeze'): evaluator_mode(unbound,'c'*64)
        strict=(Path(__file__).resolve().parents[1]/'src/evaluate_indobert_test_frozen.py').read_text(encoding='utf-8')
        self.assertLess(strict.index('enforce_frozen_weight_hash('),strict.index('model=AutoModelForSequenceClassification'))
        self.assertLess(strict.index('enforce_frozen_weight_hash('),strict.index("test=unpack(TEST_B85"))

    def test_training_provenance_rejects_changed_settings(self):
        labels={'prdec':{'negative':0,'positive':1},'smsa':{'negative':0,'neutral':1,'positive':2}}
        protocol={'indobert':{'epochs':5,'effective_batch_size':16,'weight_decay':.01,'warmup_fraction':.1,'gradient_clip':1.0}}
        resources={'indobert_revision':'revision'}
        config={'model':'indobenchmark/indobert-base-p1','revision':'revision','epochs':5,'batch_size':16,'max_length':128,
                'weight_decay':.01,'warmup_fraction':.1,'warmup_rounding':'ceil','clip':1.0,'exclude_decay':['bias','LayerNorm.weight'],
                'label_maps':labels,'tokenizer_sha256':'tok','trainer_sha256':'trainer','payload_sha256':'payload',
                'image_digest':'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'}
        env={'python':'3.13.15','torch':'2.11.0+cu128','transformers':'5.16.1','numpy':'2.1.3','cuda':'12.8','cudnn':91900,'device':'Tesla T4'}
        build={'trainer_sha256':'trainer','payload_sha256':'payload','protocol_id':'pid','tokenizer_sha256':'tok',
               'label_maps':labels,'contains_test_records':False,'contains_raw_text':False}
        args=(config,env,protocol,resources,build,'pid','tok','trainer','payload')
        self.assertEqual(validate_training_provenance(*args),labels)
        for position,key,value in ((0,'epochs',4),(1,'cuda','wrong'),(4,'payload_sha256','wrong'),(4,'contains_test_records',True)):
            changed=list(deepcopy(args));changed[position][key]=value
            with self.assertRaises(ValueError): validate_training_provenance(*changed)

    def test_epoch_metrics_validate_confusion_counts(self):
        valid=[{'label':'negative'},{'label':'positive'}]
        history=[{'metrics':{'accuracy':1.0,'macro_f1':1.0,'per_class':{
            '0':{'support':1,'tp':1,'fp':0,'fn':0,'f1':1.0},
            '1':{'support':1,'tp':1,'fp':0,'fn':0,'f1':1.0}}}}]
        check_history_metrics(history,valid,{'negative':0,'positive':1})
        history[0]['metrics']['per_class']['0']['support']=2
        with self.assertRaisesRegex(ValueError,'class counts'): check_history_metrics(history,valid,{'negative':0,'positive':1})


if __name__=='__main__': unittest.main()

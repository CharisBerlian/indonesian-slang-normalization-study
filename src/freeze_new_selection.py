"""Freeze a NEW run's validated selections before test inference; never overwrite a freeze."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from train_svm_pilot import ROOT, verify_lock
from train_svm_search import selected_rows, select_candidate


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def checked(path, expected):
    if file_sha(path)!=expected: raise ValueError(f'Hash mismatch: {path}')


def pretest_guard(root):
    for path in (root/'results/final_selection.json',root/'results/final_selection.sha256',
                 root/'results/final_evaluation',root/'results/indobert_test'):
        if path.exists(): raise ValueError(f'Refusing freeze: existing selection or held-out artifact {path}')


def best_epoch(history):
    if len(history)!=5 or {e['epoch'] for e in history}!={1,2,3,4,5}:
        raise ValueError('Expected five distinct epoch results')
    if not all(np.isfinite(e['metrics']['macro_f1']) for e in history): raise ValueError('Nonfinite validation metric')
    return max(history,key=lambda e:(e['metrics']['macro_f1'],-e['epoch']))


def select_tuning(records):
    if len(records)!=3 or {x['learning_rate'] for x in records}!={2e-5,3e-5,5e-5}:
        raise ValueError('Expected exactly the three frozen learning rates')
    return max(records,key=lambda r:(r['best_macro_f1'],-r['best_epoch'],-r['learning_rate']))


def require_score(actual,expected):
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual-expected)>1e-12:
        raise ValueError('Validation score mismatch')


def check_history_metrics(history, valid, label_map):
    """Recompute all recorded epoch metrics from their class confusion counts."""
    supports={str(v):sum(r['label']==k for r in valid) for k,v in label_map.items()}
    for epoch in history:
        metric=epoch['metrics']; per=metric['per_class']
        if set(per)!=set(supports): raise ValueError('Epoch label set mismatch')
        scores=[]; total_tp=0; total_fp=0; total_fn=0
        for label,n in supports.items():
            x=per[label]; tp,fp,fn=x['tp'],x['fp'],x['fn']
            if any(not isinstance(z,int) or z<0 for z in (tp,fp,fn)) or x['support']!=n or tp+fn!=n:
                raise ValueError('Invalid epoch class counts')
            value=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
            require_score(value,x['f1']); scores.append(value); total_tp+=tp; total_fp+=fp; total_fn+=fn
        if total_fp!=total_fn: raise ValueError('Epoch confusion totals mismatch')
        require_score(sum(scores)/len(scores),metric['macro_f1'])
        require_score(total_tp/len(valid),metric['accuracy'])


def checked_training_payload(root, selected_rows_by_dataset, labels):
    """Rebuild canonical train/valid token IDs in memory and hash them only."""
    from tokenizers import Tokenizer, __version__ as tokenizers_version
    if tokenizers_version!='0.22.1': raise ValueError('Unexpected tokenizer version')
    tokenizer=Tokenizer.from_file(str(root/'data/processed/tokenizer.json'))
    tokenizer.enable_truncation(max_length=128)
    payload={}
    for ds in ('prdec','smsa'):
        train,valid,_=selected_rows_by_dataset[ds]; data=[]
        for row in train+valid:
            item={'id':row['id'],'split':row['split'],'label':labels[ds][row['label']]}
            for condition in ('original','normalized'):
                encoded=tokenizer.encode(row[condition],add_special_tokens=True)
                if len(encoded.ids)>128 or any(v!=1 for v in encoded.attention_mask) or any(encoded.type_ids):
                    raise ValueError('Unexpected tokenizer encoding')
                item[condition]=encoded.ids
            data.append(item)
        payload[ds]=data
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


def validate_training_provenance(config, env, protocol, resources, build, protocol_id,
                                 tokenizer_sha, trainer_sha, payload_sha):
    label_maps={'prdec':{'negative':0,'positive':1},'smsa':{'negative':0,'neutral':1,'positive':2}}
    settings={'model':'indobenchmark/indobert-base-p1','revision':resources['indobert_revision'],
              'epochs':protocol['indobert']['epochs'],'batch_size':protocol['indobert']['effective_batch_size'],
              'max_length':128,'weight_decay':protocol['indobert']['weight_decay'],
              'warmup_fraction':protocol['indobert']['warmup_fraction'],'warmup_rounding':'ceil',
              'clip':protocol['indobert']['gradient_clip'],'exclude_decay':['bias','LayerNorm.weight'],
              'label_maps':label_maps,
              'image_digest':'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'}
    if any(config.get(k)!=v for k,v in settings.items()): raise ValueError('Transformer training settings differ from protocol')
    expected_env={'python':'3.13.15','torch':'2.11.0+cu128','transformers':'5.16.1',
                  'numpy':'2.1.3','cuda':'12.8','cudnn':91900,'device':'Tesla T4'}
    if any(env.get(k)!=v for k,v in expected_env.items()): raise ValueError('Transformer training environment differs from recorded configuration')
    if config['tokenizer_sha256']!=tokenizer_sha: raise ValueError('Transformer tokenizer mismatch')
    if (config['trainer_sha256']!=trainer_sha or build['trainer_sha256']!=trainer_sha or
        config['payload_sha256']!=payload_sha or build['payload_sha256']!=payload_sha or
        build['protocol_id']!=protocol_id or build['tokenizer_sha256']!=tokenizer_sha or
        build['label_maps']!=label_maps or build['contains_test_records'] or build['contains_raw_text']):
        raise ValueError('Transformer source or training payload mismatch')
    return label_maps


def svm_selection(root, protocol_id, selected_rows_by_dataset):
    path=root/'results/svm_search/summary.json'; summary=json.loads(path.read_text(encoding='utf-8'))
    if summary['protocol_id']!=protocol_id: raise ValueError('SVM protocol mismatch')
    for relative,h in summary['source_and_config_sha256'].items(): checked(root/relative,h)
    entries=[]
    for ds in ('prdec','smsa'):
        _,valid,input_hash=selected_rows_by_dataset[ds]
        d=summary['datasets'][ds]
        if d['selected_processed_prefix_sha256']!=input_hash: raise ValueError('SVM training/validation prefix mismatch')
        pred_path=root/f'results/svm_search/{ds}_validation_predictions.csv'
        with pred_path.open(encoding='utf-8',newline='') as f: predictions=list(csv.DictReader(f))
        if len(predictions)!=len(valid) or any(any(a[k]!=b[k] for k in ('id','group_id','label')) for a,b in zip(predictions,valid)):
            raise ValueError('SVM validation IDs do not match frozen split')
        y=[r['label'] for r in valid]
        if set(d['models'])!={'word_original','word_normalized','character_original','character_normalized'}:
            raise ValueError('SVM conditions incomplete')
        for representation in ('word','character'):
            for condition in ('original','normalized'):
                name=f'{representation}_{condition}'; result=d['models'][name]; grid=result['grid']
                if len(grid)!=3 or {x['C'] for x in grid}!={.1,1,10}: raise ValueError('SVM grid mismatch')
                for item in grid:
                    pred=[r[f'{name}_C{item["C"]}'] for r in predictions]
                    require_score(f1_score(y,pred,average='macro',zero_division=0),item['validation_macro_f1'])
                chosen=select_candidate(grid)
                if chosen['C']!=result['selected_C']: raise ValueError('SVM selected C mismatch')
                model_path=root/f'results/svm_search/{ds}_{name}.joblib'; checked(model_path,result['selected_checkpoint_sha256'])
                model=joblib.load(model_path)
                if model['svm'].C!=chosen['C'] or model['svm'].random_state!=42: raise ValueError('SVM checkpoint setting mismatch')
                own=model.predict([r[condition] for r in valid]); saved=np.array([r[f'{name}_C{chosen["C"]}'] for r in predictions])
                if not np.array_equal(own,saved): raise ValueError('SVM validation prediction mismatch')
                other='normalized' if condition=='original' else 'original'
                alternate=model.predict([r[other] for r in valid]); saved_other=np.array([r[f'{name}_selected_input_{other}'] for r in predictions])
                if not np.array_equal(alternate,saved_other): raise ValueError('SVM alternate validation prediction mismatch')
                entries.append({'dataset':ds,'representation':representation,'condition':condition,'selected_C':chosen['C'],
                    'checkpoint_path':model_path.relative_to(root).as_posix(),'checkpoint_sha256':file_sha(model_path),
                    'selected_input_prefix_sha256':input_hash,
                    'validation_predictions_path':pred_path.relative_to(root).as_posix(),
                    'validation_predictions_sha256':file_sha(pred_path)})
    return {'summary_path':path.relative_to(root).as_posix(),'summary_sha256':file_sha(path),
            'source_and_config_sha256':summary['source_and_config_sha256'],'models':entries}


def bert_selection(root, checkpoint_root, protocol_id, selected_rows_by_dataset):
    base=root/'results/indobert_full'; config_path=base/'config.json'; state_path=base/'state.json'; env_path=base/'environment.json'
    config=json.loads(config_path.read_text(encoding='utf-8')); state=json.loads(state_path.read_text(encoding='utf-8'))
    env=json.loads(env_path.read_text(encoding='utf-8'))
    config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if state['status']!='complete' or state['config_sha256']!=config_sha or env['config_sha256']!=config_sha or config['protocol_id']!=protocol_id:
        raise ValueError('IndoBERT training state/config mismatch')
    lock=json.loads((root/'data/protocol_lock.json').read_text(encoding='utf-8'))
    protocol=json.loads((root/'configs/protocol.json').read_text(encoding='utf-8'))
    resources=json.loads((root/'configs/resources.json').read_text(encoding='utf-8'))
    build=json.loads((base/'build_manifest.json').read_text(encoding='utf-8'))
    label_maps={'prdec':{'negative':0,'positive':1},'smsa':{'negative':0,'neutral':1,'positive':2}}
    trainer_sha=hashlib.sha256((root/'src/train_indobert_full.py').read_text(encoding='utf-8').encode()).hexdigest()
    payload_sha=checked_training_payload(root,selected_rows_by_dataset,label_maps)
    validate_training_provenance(config,env,protocol,resources,build,protocol_id,
                                 lock['files']['data/processed/tokenizer.json'],trainer_sha,payload_sha)
    if config['tuning_seed']!=42 or config['tuning_lrs']!=[2e-5,3e-5,5e-5] or config['final_seeds']!=[13,37,73]:
        raise ValueError('Transformer search grid differs from protocol')
    if len(state['runs'])!=24: raise ValueError('Expected 24 transformer training runs')
    records=[]
    for rid,record in state['runs'].items():
        if record['run_id']!=rid or not record['completed'] or not record['best_checkpoint_predictions_match'] or not record['best_checkpoint_inference_finite']:
            raise ValueError('Transformer training run incomplete')
        train=selected_rows_by_dataset[record['dataset']][0]
        expected_steps=((len(train)+config['batch_size']-1)//config['batch_size'])*config['epochs']
        if record['steps']!=expected_steps: raise ValueError('Transformer training step count mismatch')
        runpath=base/'runs'/f'{rid}.json'
        if json.loads(runpath.read_text(encoding='utf-8'))!=record: raise ValueError('Run metadata differs from state')
        epoch=best_epoch(record['history'])
        if epoch['epoch']!=record['best_epoch']: raise ValueError('Transformer best-epoch choice mismatch')
        require_score(epoch['metrics']['macro_f1'],record['best_macro_f1'])
        valid=selected_rows_by_dataset[record['dataset']][1]; labels=config['label_maps'][record['dataset']]
        check_history_metrics(record['history'],valid,labels)
        predpath=base/'predictions'/f'{rid}.json'; preds=json.loads(predpath.read_text(encoding='utf-8'))
        if len(preds)!=len(valid) or any(p['id']!=r['id'] or p['label']!=labels[r['label']] for p,r in zip(preds,valid)):
            raise ValueError('Transformer validation rows do not match locked split')
        if any(p['prediction'] not in labels.values() for p in preds): raise ValueError('Transformer prediction label invalid')
        require_score(f1_score([p['label'] for p in preds],[p['prediction'] for p in preds],labels=list(labels.values()),average='macro',zero_division=0),record['best_macro_f1'])
        records.append(record)
    if set(state['selected'])!={f'{d}_{c}' for d in ('prdec','smsa') for c in ('original','normalized')}:
        raise ValueError('Transformer tuning selection set mismatch')
    selected_tuning={}; entries=[]
    for ds in ('prdec','smsa'):
        for condition in ('original','normalized'):
            relevant=[r for r in records if r['dataset']==ds and r['condition']==condition]
            tuning=[r for r in relevant if r['kind']=='tuning' and r['seed']==42]
            best=select_tuning(tuning); name=f'{ds}_{condition}'; expected=state['selected'][name]
            if any(expected[k]!=best[k] for k in ('run_id','learning_rate','best_epoch','best_macro_f1')):
                raise ValueError('Transformer selected tuning run mismatch')
            selected_tuning[name]=expected
            finals=[r for r in relevant if r['kind']=='final']
            if len(finals)!=3 or {r['seed'] for r in finals}!={13,37,73} or any(r['learning_rate']!=best['learning_rate'] for r in finals):
                raise ValueError('Transformer final seeds or learning rate mismatch')
            for record in sorted(finals,key=lambda r:r['seed']):
                rid=record['run_id']; weights=checkpoint_root/rid/'model.safetensors'; cfg=checkpoint_root/rid/'config.json'
                if not weights.is_file() or not cfg.is_file(): raise ValueError(f'Missing accessible checkpoint: {rid}')
                # Hash all model bytes while still pre-test; validation reproduction is gated again at inference.
                entries.append({'dataset':ds,'condition':condition,'seed':record['seed'],'selected_learning_rate':record['learning_rate'],
                    'best_epoch':record['best_epoch'],'run_id':rid,'remote_checkpoint_path':f'indobert_full/checkpoints/{rid}/model.safetensors',
                    'remote_checkpoint_bytes':weights.stat().st_size,'remote_checkpoint_sha256':file_sha(weights),
                    'model_config_sha256':file_sha(cfg),'validation_predictions_path':f'results/indobert_full/predictions/{rid}.json',
                    'validation_predictions_sha256':file_sha(base/'predictions'/f'{rid}.json'),
                    'run_record_sha256':file_sha(base/'runs'/f'{rid}.json')})
    return {'checkpoint_storage':'private supplied checkpoint directory, copied/mounted for private inference',
            'remote_weight_sha256_status':'All selected checkpoint hashes calculated before this freeze.',
            'state_sha256':file_sha(state_path),'config_sha256':config_sha,'config_file_sha256':file_sha(config_path),
            'environment_file_sha256':file_sha(env_path),'payload_sha256':config['payload_sha256'],
            'tokenizer_sha256':config['tokenizer_sha256'],'trainer_sha256':config['trainer_sha256'],
            'image_digest':config['image_digest'],'selected_tuning':selected_tuning,'models':entries}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root',type=Path,default=ROOT/'results/indobert_full/checkpoints',
                        help='Private local/mounted directory containing run_id/model.safetensors and config.json')
    args=parser.parse_args(); pretest_guard(ROOT)
    protocol=verify_lock(); subsets={d:selected_rows(d) for d in ('prdec','smsa')}
    svm=svm_selection(ROOT,protocol,subsets); bert=bert_selection(ROOT,args.checkpoint_root,protocol,subsets)
    pretest_guard(ROOT)
    result={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'New validation-only selection freeze, before test inference',
            'protocol_id':protocol,'protocol_lock_sha256':file_sha(ROOT/'data/protocol_lock.json'),
            'protocol_config_sha256':file_sha(ROOT/'configs/protocol.json'),
            'no_test_predictions_at_freeze':True,'svm':svm,'indobert':bert}
    blob=(json.dumps(result,indent=2)+'\n').encode('utf-8'); digest=hashlib.sha256(blob).hexdigest()
    # Exclusive create refuses to overwrite any existing freeze, even if files appear after guard checks.
    with (ROOT/'results/final_selection.json').open('xb') as f: f.write(blob)
    with (ROOT/'results/final_selection.sha256').open('x',encoding='utf-8',newline='\n') as f: f.write(digest+'  final_selection.json\n')
    print(f'Created new validation-only selection freeze: {digest}')


if __name__=='__main__': main()

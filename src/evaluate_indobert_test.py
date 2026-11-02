"""Private held-out inference for frozen final models; validates all models before test decoding."""
import base64,concurrent.futures,gzip,hashlib,json,math,os,platform,subprocess,sys,threading,time,urllib.request
from pathlib import Path
import numpy as np
import torch,transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
START=time.monotonic();OUT=Path('/kaggle/working/indobert_test');OUT.mkdir(parents=True,exist_ok=True)
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,sort_keys=True,indent=2),encoding='utf8');os.replace(tmp,path)
def event(name,**kwargs):
 x={'event':name,'seconds':round(time.monotonic()-START,2),**kwargs}
 with (OUT/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
 print('TEST_EVENT',json.dumps(x),flush=True)
assert platform.python_version()=='3.13.15'
assert torch.__version__=='2.11.0+cu128' and transformers.__version__=='5.16.1'
assert torch.cuda.is_available();torch.cuda.set_device(0);torch.set_num_threads(2)
assert 'T4' in torch.cuda.get_device_name(0)
selection_raw=base64.b85decode(SELECTION_B85)
assert sha_bytes(selection_raw)==EVAL_MANIFEST['selection_sha256']
selection=json.loads(selection_raw);assert selection['no_test_predictions_at_freeze']
models=selection['indobert']['models'];assert len(models)==12
assert len({m['run_id'] for m in models})==12
assert {(m['dataset'],m['condition'],m['seed']) for m in models}=={(d,c,s) for d in ('prdec','smsa') for c in ('original','normalized') for s in (13,37,73)}
assert all('_final_' in m['run_id'] and '_tuning_' not in m['run_id'] for m in models)
# Fetch the frozen version-1 outputs into private scratch storage. Signed URLs are
# injected only into the private submitted source and never logged or saved locally.
source=Path('/kaggle/temp/indobert_full');source.mkdir(parents=True,exist_ok=True)
small={'state.json':selection['indobert']['state_sha256'],'config.json':selection['indobert']['config_file_sha256'],'environment.json':selection['indobert']['environment_file_sha256']}
weights={}
for m in models:
 rid=m['run_id'];small[f'checkpoints/{rid}/config.json']=m['model_config_sha256']
 small[f'predictions/{rid}.json']=m['validation_predictions_sha256']
 small[f'runs/{rid}.json']=m['run_record_sha256']
 weights[f'checkpoints/{rid}/model.safetensors']=m['remote_checkpoint_bytes']
assert set(SOURCE_URLS)==set(small)|set(weights)
def get_small(rel,expected):
 target=source/rel;target.parent.mkdir(parents=True,exist_ok=True)
 try:
  with urllib.request.urlopen(SOURCE_URLS[rel],timeout=60) as response:
   assert response.status==200
   data=response.read(2*1024*1024+1);assert len(data)<=2*1024*1024
 except Exception:raise RuntimeError('Could not fetch required source file: '+rel) from None
 assert sha_bytes(data)==expected, 'source hash mismatch: '+rel
 target.write_bytes(data)
def get_weight(rel,length):
 target=source/rel;target.parent.mkdir(parents=True,exist_ok=True)
 chunk=8*1024*1024;lock=threading.Lock();counter=[0]
 with target.open('wb') as output:
  output.truncate(length)
  def fetch(start):
   end=min(start+chunk,length)-1
   for attempt in range(4):
    try:
     if time.monotonic()-START>3000:raise RuntimeError('soft time limit')
     request=urllib.request.Request(SOURCE_URLS[rel],headers={'Range':f'bytes={start}-{end}'})
     with urllib.request.urlopen(request,timeout=90) as response:
      assert response.status==206 and response.headers['Content-Range']==f'bytes {start}-{end}/{length}'
      data=response.read();assert len(data)==end-start+1
     with lock:output.seek(start);output.write(data);counter[0]+=len(data)
     return
    except Exception:
     if attempt==3:raise RuntimeError('Could not fetch weight range: '+rel) from None
     time.sleep(attempt+1)
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
   list(pool.map(fetch,range(0,length,chunk)))
 assert counter[0]==length and target.stat().st_size==length
 event('checkpoint_fetched',run_id=Path(rel).parent.name,bytes=length)
for rel,digest in small.items():get_small(rel,digest)
for rel,length in weights.items():get_weight(rel,length)
assert sha_file(source/'state.json')==selection['indobert']['state_sha256']
assert sha_file(source/'config.json')==selection['indobert']['config_file_sha256']
assert sha_file(source/'environment.json')==selection['indobert']['environment_file_sha256']
state=json.loads((source/'state.json').read_text());assert state['status']=='complete'
config=json.loads((source/'config.json').read_text());assert config['protocol_id']==selection['protocol_id']
assert config['payload_sha256']==selection['indobert']['payload_sha256']
# Verify every frozen source file before loading any model.
for m in models:
 rid=m['run_id']; p=source/'checkpoints'/rid
 assert m['remote_checkpoint_path']==f'indobert_full/checkpoints/{rid}/model.safetensors'
 assert (p/'model.safetensors').stat().st_size==m['remote_checkpoint_bytes']
 assert sha_file(p/'config.json')==m['model_config_sha256']
 assert sha_file(source/'predictions'/f'{rid}.json')==m['validation_predictions_sha256']
 assert sha_file(source/'runs'/f'{rid}.json')==m['run_record_sha256']
 rr=state['runs'][rid]
 assert rr['completed'] and rr['best_checkpoint_predictions_match'] and rr['best_epoch']==m['best_epoch'] and rr['learning_rate']==m['selected_learning_rate']
(OUT/'frozen_selection.json').write_bytes(selection_raw)
atomic(OUT/'eval_manifest.json',EVAL_MANIFEST)
atomic(OUT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':torch.cuda.get_device_name(0),'image_digest':selection['indobert']['image_digest'],'source_state_sha256':sha_file(source/'state.json')})
(OUT/'pip_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout,encoding='utf8')
def unpack(encoded,expected):
 raw=gzip.decompress(base64.b85decode(encoded));assert sha_bytes(raw)==expected
 return json.loads(raw)
validation=unpack(VALID_B85,EVAL_MANIFEST['valid_payload_sha256'])
for ds in ('prdec','smsa'):
 assert len(validation[ds])==EVAL_MANIFEST['counts'][ds]['valid']
 assert len({r['id'] for r in validation[ds]})==len(validation[ds])
def collate(items):
 width=max(len(x['ids']) for x in items)
 ids=torch.tensor([x['ids']+[0]*(width-len(x['ids'])) for x in items],dtype=torch.long,device='cuda:0')
 masks=torch.tensor([[1]*len(x['ids'])+[0]*(width-len(x['ids'])) for x in items],dtype=torch.long,device='cuda:0')
 return {'input_ids':ids,'attention_mask':masks,'token_type_ids':torch.zeros_like(ids)}
def infer(model,rows,condition):
 items=[{'ids':r[condition]} for r in rows];loader=DataLoader(items,batch_size=16,shuffle=False,collate_fn=collate)
 preds=[];logits=[]
 with torch.no_grad():
  for batch in loader:
   output=model(**batch).logits
   assert torch.isfinite(output).all().item()
   logits.extend(output.cpu().tolist());preds.extend(output.argmax(-1).cpu().tolist())
 return preds,logits
def metric(rows,pred,n):
 y=[r['label'] for r in rows];per={};fs=[]
 for label in range(n):
  tp=sum(a==label and b==label for a,b in zip(y,pred));fp=sum(a!=label and b==label for a,b in zip(y,pred));fn=sum(a==label and b!=label for a,b in zip(y,pred))
  score=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0;fs.append(score);per[str(label)]={'support':sum(a==label for a in y),'tp':tp,'fp':fp,'fn':fn,'f1':score}
 return {'accuracy':sum(a==b for a,b in zip(y,pred))/len(y),'macro_f1':sum(fs)/n,'per_class':per}
event('source_verified',models=len(models),selection_sha256=EVAL_MANIFEST['selection_sha256'])
# Absolute gate: all 12 checkpoints must match their frozen validation predictions.
verified=[]
for m in models:
 if time.monotonic()-START>3000:raise RuntimeError('Inference soft time limit reached before validation gate')
 t0=time.monotonic();rid=m['run_id'];cp=source/'checkpoints'/rid;digest=sha_file(cp/'model.safetensors')
 model=AutoModelForSequenceClassification.from_pretrained(cp,local_files_only=True).to('cuda:0').eval()
 rows=validation[m['dataset']];pred,_=infer(model,rows,m['condition'])
 expected=json.loads((source/'predictions'/f'{rid}.json').read_text())
 actual=[{'id':r['id'],'label':r['label'],'prediction':p} for r,p in zip(rows,pred)]
 assert actual==expected, f'Validation prediction mismatch for {rid}'
 verified.append({'run_id':rid,'weights_sha256':digest,'validation_count':len(rows),'validation_predictions_match':True,'seconds':time.monotonic()-t0})
 atomic(OUT/'checkpoint_validation_progress.json',verified);event('validation_reproduced',run_id=rid,weights_sha256=digest)
 del model;torch.cuda.empty_cache()
assert len(verified)==12
atomic(OUT/'checkpoint_manifest.json',{'selection_sha256':EVAL_MANIFEST['selection_sha256'],'source_state_sha256':selection['indobert']['state_sha256'],'checkpoints':verified,'all_validation_predictions_match':True})
event('validation_gate_passed',models=12)
# Test data are only decoded and processed after all 12 validation checks pass.
test=unpack(TEST_B85,EVAL_MANIFEST['test_payload_sha256'])
for ds in ('prdec','smsa'):
 assert len(test[ds])==EVAL_MANIFEST['counts'][ds]['test']
 assert len({r['id'] for r in test[ds]})==len(test[ds])
results=[]
for m in models:
 if time.monotonic()-START>3000:raise RuntimeError('Inference soft time limit reached')
 t0=time.monotonic();rid=m['run_id'];cp=source/'checkpoints'/rid
 torch.cuda.reset_peak_memory_stats(0)
 model=AutoModelForSequenceClassification.from_pretrained(cp,local_files_only=True).to('cuda:0').eval()
 for condition in ('original','normalized'):
  condition_start=time.monotonic();rows=test[m['dataset']];pred,logits=infer(model,rows,condition)
  payload=[{'id':r['id'],'group_id':r['group_id'],'label':r['label'],'prediction':p,'logits':l,'bert_input_changed':r['bert_input_changed']} for r,p,l in zip(rows,pred,logits)]
  dest=OUT/'predictions'/f'{rid}__eval_{condition}.json';atomic(dest,payload)
  scores=metric(rows,pred,len(EVAL_MANIFEST['label_maps'][m['dataset']]))
  rec={'run_id':rid,'dataset':m['dataset'],'train_condition':m['condition'],'eval_condition':condition,'seed':m['seed'],'learning_rate':m['selected_learning_rate'],'best_epoch':m['best_epoch'],'primary_protocol_pair':condition==m['condition'],'metrics':scores,'count':len(rows),'prediction_path':str(dest.relative_to(OUT)),'prediction_sha256':sha_file(dest),'seconds':time.monotonic()-condition_start}
  results.append(rec);atomic(OUT/'results.json',{'status':'running','results':results});event('test_condition_completed',run_id=rid,eval_condition=condition)
 event('model_completed',run_id=rid,seconds=time.monotonic()-t0,peak_gpu_bytes=torch.cuda.max_memory_allocated(0))
 del model;torch.cuda.empty_cache()
assert len(results)==24
atomic(OUT/'results.json',{'status':'complete','results':results,'total_seconds':time.monotonic()-START,'selection_sha256':EVAL_MANIFEST['selection_sha256']})
event('complete',conditions=24)

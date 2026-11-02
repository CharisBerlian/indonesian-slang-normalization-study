"""Private train/validation-only IndoBERT tuning and final-seed jobs."""
import base64, hashlib, json, lzma, math, os, platform, random, shutil, struct, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch, transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

START=time.monotonic()
SOFT_DEADLINE=10*3600  # Leave one hour before the 11-hour Kaggle session limit for output finalization.
OUT=Path('/kaggle/working/indobert_full')
RESUME=os.environ.get('INDOBERT_RESUME_DIR')
if RESUME:
    source=Path(RESUME)
    if OUT.exists(): raise RuntimeError('Resume destination must be empty')
    shutil.copytree(source,OUT)
OUT.mkdir(parents=True,exist_ok=True)

def atomic(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf8')
    os.replace(tmp,path)

def event(name,**kw):
    obj={'event':name,'elapsed_seconds':round(time.monotonic()-START,2),**kw}
    with (OUT/'events.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(obj)+'\n'); f.flush(); os.fsync(f.fileno())
    print('FULL_RUN_EVENT',json.dumps(obj),flush=True)

assert platform.python_version()=='3.13.15',platform.python_version()
assert torch.__version__=='2.11.0+cu128',torch.__version__
assert transformers.__version__=='5.16.1',transformers.__version__
assert torch.cuda.is_available()
torch.cuda.set_device(0); torch.set_num_threads(2)
assert 'T4' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0)
packed=lzma.decompress(base64.b85decode(PAYLOAD_B64))
assert hashlib.sha256(packed).hexdigest()==BUILD_MANIFEST['packed_sha256']
pos=0
def take(n):
    global pos
    part=packed[pos:pos+n]; assert len(part)==n; pos+=n; return part
def unpack(fmt): return struct.unpack(fmt,take(struct.calcsize(fmt)))
data={}
for ds in ('prdec','smsa'):
    count,=unpack('<I'); rows=[]
    for _ in range(count):
        ilen,=unpack('<B'); ident=take(ilen).decode('utf8')
        split,label,n,ops=unpack('<BBBB')
        original=list(unpack('<'+'H'*n)); edits=[]
        for _ in range(ops):
            a,b,k=unpack('<BBB'); edits.append((a,b,list(unpack('<'+'H'*k))))
        normalized=original.copy()
        for a,b,replacement in reversed(edits): normalized[a:b]=replacement
        rows.append({'id':ident,'split':('train' if split==0 else 'valid'),'label':label,'original':original,'normalized':normalized})
    data[ds]=rows
assert pos==len(packed)
raw=json.dumps(data,separators=(',',':')).encode()
assert hashlib.sha256(raw).hexdigest()==BUILD_MANIFEST['payload_sha256']
del raw,packed
assert set(data)=={'prdec','smsa'}
for ds in data:
    assert all(r['split'] in ('train','valid') for r in data[ds])
    assert len({r['id'] for r in data[ds]})==len(data[ds])
    for split in ('train','valid'):
        assert sum(r['split']==split for r in data[ds])==BUILD_MANIFEST['statistics'][ds][split]['count']
    assert all(0<len(r[c])<=128 for r in data[ds] for c in ('original','normalized'))
config={'protocol_id':BUILD_MANIFEST['protocol_id'],'tokenizer_sha256':BUILD_MANIFEST['tokenizer_sha256'],'payload_sha256':BUILD_MANIFEST['payload_sha256'],'trainer_sha256':BUILD_MANIFEST['trainer_sha256'],'model':'indobenchmark/indobert-base-p1','revision':'c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2','epochs':5,'batch_size':16,'max_length':128,'weight_decay':0.01,'exclude_decay':['bias','LayerNorm.weight'],'warmup_fraction':0.1,'warmup_rounding':'ceil','clip':1.0,'tuning_lrs':[2e-5,3e-5,5e-5],'tuning_seed':42,'final_seeds':[13,37,73],'selection':'highest validation macro-F1; earlier epoch then lower learning rate on ties','deadline_seconds':SOFT_DEADLINE,'session_timeout_seconds':39600,'image_digest':'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435','label_maps':BUILD_MANIFEST['label_maps']}
config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if (OUT/'config.json').exists():
    assert json.loads((OUT/'config.json').read_text())==config, 'resume config mismatch'
else: atomic(OUT/'config.json',config)
state_path=OUT/'state.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {'config_sha256':config_hash,'runs':{},'selected':{},'status':'running'}
assert state['config_sha256']==config_hash
state['status']='running'; atomic(state_path,state)
freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
(OUT/'pip_freeze.txt').write_text(freeze,encoding='utf8')
nvidia=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
(OUT/'nvidia_smi.txt').write_text(nvidia,encoding='utf8')
env={'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':torch.cuda.get_device_name(0),'gpu_count':torch.cuda.device_count(),'config_sha256':config_hash,'resume_source':bool(RESUME)}
atomic(OUT/'environment.json',env); atomic(OUT/'build_manifest.json',BUILD_MANIFEST)
event('dataset_loaded',counts=BUILD_MANIFEST['statistics'],environment=env)

class Deadline(Exception): pass

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def collate(items):
    width=max(len(x['ids']) for x in items)
    ids=torch.tensor([x['ids']+[0]*(width-len(x['ids'])) for x in items],dtype=torch.long,device='cuda:0')
    mask=torch.tensor([[1]*len(x['ids'])+[0]*(width-len(x['ids'])) for x in items],dtype=torch.long,device='cuda:0')
    labels=torch.tensor([x['label'] for x in items],dtype=torch.long,device='cuda:0')
    return {'input_ids':ids,'attention_mask':mask,'token_type_ids':torch.zeros_like(ids),'labels':labels}

def row_data(ds,split,condition):
    return [{'id':r['id'],'ids':r[condition],'label':r['label']} for r in data[ds] if r['split']==split]

def scores(y,p,n):
    by_class={}; fs=[]
    for label in range(n):
        tp=sum(a==label and b==label for a,b in zip(y,p)); fp=sum(a!=label and b==label for a,b in zip(y,p)); fn=sum(a==label and b!=label for a,b in zip(y,p))
        f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        by_class[str(label)]={'f1':f,'support':sum(a==label for a in y),'tp':tp,'fp':fp,'fn':fn}; fs.append(f)
    return {'macro_f1':sum(fs)/n,'accuracy':sum(a==b for a,b in zip(y,p))/len(y),'per_class':by_class}

def checkpoint(model,path):
    tmp=path.with_name(path.name+'.tmp')
    if tmp.exists(): shutil.rmtree(tmp)
    model.save_pretrained(tmp,safe_serialization=True)
    if path.exists(): shutil.rmtree(path)
    os.replace(tmp,path)

def run(ds,condition,kind,seed,lr):
    run_id=f'{ds}_{condition}_{kind}_seed{seed}_lr{lr:.0e}'
    prev=state['runs'].get(run_id,{})
    if prev.get('completed'): return run_id
    if time.monotonic()-START>=SOFT_DEADLINE: raise Deadline()
    path=OUT/'checkpoints'/run_id
    # Incomplete runs restart from their initial seed, never from a partially saved optimizer.
    if path.exists(): shutil.rmtree(path)
    seed_all(seed); torch.cuda.reset_peak_memory_stats(0); t0=time.monotonic()
    labels=BUILD_MANIFEST['label_maps'][ds]; inverse={v:k for k,v in labels.items()}
    model=AutoModelForSequenceClassification.from_pretrained(config['model'],revision=config['revision'],num_labels=len(labels),id2label=inverse,label2id=labels).to('cuda:0')
    before=model.bert.embeddings.word_embeddings.weight.detach().clone()
    yes=[]; no=[]
    for n,p in model.named_parameters():
        assert p.requires_grad
        (no if n.endswith('.bias') or 'LayerNorm.weight' in n else yes).append(p)
    optimizer=torch.optim.AdamW([{'params':yes,'weight_decay':0.01},{'params':no,'weight_decay':0.0}],lr=lr)
    train_rows=row_data(ds,'train',condition); valid_rows=row_data(ds,'valid',condition)
    train=DataLoader(train_rows,batch_size=16,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(seed))
    valid=DataLoader(valid_rows,batch_size=16,shuffle=False,collate_fn=collate)
    total=len(train)*5; warmup=math.ceil(total*.1)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:step/max(1,warmup) if step<warmup else max(0.0,(total-step)/max(1,total-warmup)))
    history=[]; best=-1.0; best_epoch=None; steps=0
    event('run_started',run_id=run_id,train=len(train_rows),valid=len(valid_rows),total_steps=total,warmup_steps=warmup)
    for epoch in range(1,6):
        model.train(); losses=[]
        for batch in train:
            if time.monotonic()-START>=SOFT_DEADLINE: raise Deadline()
            optimizer.zero_grad(set_to_none=True); loss=model(**batch).loss
            if not torch.isfinite(loss).item(): raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); losses.append(loss.item()); steps+=1
        model.eval(); y=[]; pred=[]; vl=[]
        with torch.no_grad():
            for batch in valid:
                out=model(**batch); vl.append(out.loss.item()); y.extend(batch['labels'].tolist()); pred.extend(out.logits.argmax(-1).tolist())
        metric=scores(y,pred,len(labels)); val=metric['macro_f1']
        row={'epoch':epoch,'train_loss':sum(losses)/len(losses),'valid_loss':sum(vl)/len(vl),'metrics':metric,'seconds_elapsed':time.monotonic()-t0}
        history.append(row); atomic(OUT/'runs'/f'{run_id}.json',{'run_id':run_id,'history':history,'completed':False,'best_epoch':best_epoch,'best_macro_f1':best})
        if val>best:
            best=val; best_epoch=epoch; checkpoint(model,path)
            atomic(OUT/'predictions'/f'{run_id}.json',[{'id':r['id'],'label':a,'prediction':b} for r,a,b in zip(valid_rows,y,pred)])
        atomic(OUT/'runs'/f'{run_id}.json',{'run_id':run_id,'history':history,'completed':False,'best_epoch':best_epoch,'best_macro_f1':best})
        event('epoch_completed',run_id=run_id,epoch=epoch,macro_f1=val,best_epoch=best_epoch)
    change=(model.bert.embeddings.word_embeddings.weight.detach()-before).abs().max().item()
    assert change>0 and math.isfinite(change)
    # Compare the reloaded best checkpoint with saved best-epoch predictions for every validation ID.
    del model; torch.cuda.empty_cache()
    reloaded=AutoModelForSequenceClassification.from_pretrained(path).to('cuda:0').eval()
    reload_y=[]; reload_p=[]
    with torch.no_grad():
        for batch in valid:
            out=reloaded(**batch)
            assert torch.isfinite(out.logits).all().item()
            reload_y.extend(batch['labels'].tolist()); reload_p.extend(out.logits.argmax(-1).tolist())
    saved=json.loads((OUT/'predictions'/f'{run_id}.json').read_text(encoding='utf8'))
    assert saved==[{'id':r['id'],'label':a,'prediction':b} for r,a,b in zip(valid_rows,reload_y,reload_p)], 'best checkpoint reload predictions mismatch'
    result={'run_id':run_id,'dataset':ds,'condition':condition,'kind':kind,'seed':seed,'learning_rate':lr,'history':history,'best_epoch':best_epoch,'best_macro_f1':best,'steps':steps,'embedding_max_abs_update':change,'best_checkpoint_inference_finite':True,'best_checkpoint_predictions_match':True,'peak_gpu_bytes':torch.cuda.max_memory_allocated(0),'seconds':time.monotonic()-t0,'completed':True,'checkpoint':str(path.relative_to(OUT))}
    atomic(OUT/'runs'/f'{run_id}.json',result); state['runs'][run_id]=result; atomic(state_path,state)
    event('run_completed',run_id=run_id,best_epoch=best_epoch,best_macro_f1=best,seconds=result['seconds'])
    del reloaded,optimizer,before; torch.cuda.empty_cache()
    return run_id

try:
    for ds in ('prdec','smsa'):
        for condition in ('original','normalized'):
            ids=[run(ds,condition,'tuning',42,lr) for lr in (2e-5,3e-5,5e-5)]
            best_id=min(ids,key=lambda i:(-state['runs'][i]['best_macro_f1'],state['runs'][i]['best_epoch'],state['runs'][i]['learning_rate']))
            key=ds+'_'+condition
            state['selected'][key]={'run_id':best_id,'learning_rate':state['runs'][best_id]['learning_rate'],'best_epoch':state['runs'][best_id]['best_epoch'],'best_macro_f1':state['runs'][best_id]['best_macro_f1']}
            atomic(state_path,state); event('tuning_selected',key=key,**state['selected'][key])
            for other in ids:
                if other!=best_id:
                    p=OUT/state['runs'][other]['checkpoint']
                    if p.exists(): shutil.rmtree(p)
    for ds in ('prdec','smsa'):
        for condition in ('original','normalized'):
            lr=state['selected'][ds+'_'+condition]['learning_rate']
            for seed in (13,37,73): run(ds,condition,'final',seed,lr)
    assert len([r for r in state['runs'].values() if r.get('completed')])==24
    state['status']='complete'; atomic(state_path,state); event('all_runs_completed',runs=24)
except Deadline:
    state['status']='deadline_stopped_resume_completed_runs'; atomic(state_path,state); event('deadline_stopped',completed=sum(r.get('completed',False) for r in state['runs'].values()))
except BaseException as exc:
    state['status']='failed'; state['failure']=repr(exc); atomic(state_path,state); event('failed',error=repr(exc)); traceback.print_exc(); raise

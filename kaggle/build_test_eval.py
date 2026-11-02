"""Build private test-inference package after the final selection is frozen."""
import argparse,base64,csv,gzip,hashlib,json,sys
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from selection_io import read_verified_selection
from weight_integrity import evaluator_mode
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--selection',type=Path,default=ROOT/'results/final_selection.json',help='Frozen pre-test manifest')
parser.add_argument('--selection-sha256',type=Path,default=ROOT/'results/final_selection.sha256',help='Companion SHA-256 file')
args=parser.parse_args()
selection_bytes,EXPECTED_SELECTION=read_verified_selection(args.selection,args.selection_sha256)
selection=json.loads(selection_bytes); assert selection['no_test_predictions_at_freeze']
mode=evaluator_mode(selection,EXPECTED_SELECTION)
lock=json.loads((ROOT/'data/protocol_lock.json').read_text())
assert selection['protocol_id']==lock['protocol_id']
for name in ('configs/protocol.json','configs/resources.json','data/processed/tokenizer.json','data/splits/prdec.csv','data/splits/smsa.csv','src/audit_data.py','src/normalization.py'):
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==lock['files'][name],name
tok=Tokenizer.from_file(str(ROOT/'data/processed/tokenizer.json'));tok.enable_truncation(max_length=128)
label_maps={'prdec':{'negative':0,'positive':1},'smsa':{'negative':0,'neutral':1,'positive':2}}
expected={'prdec':{'valid':793,'test':793},'smsa':{'valid':1260,'test':500}}
all_splits={'valid':{},'test':{}}
for ds in ('prdec','smsa'):
 with (ROOT/f'data/splits/{ds}.csv').open(encoding='utf8',newline='') as f: split={r['id']:r for r in csv.DictReader(f) if r['split'] in ('valid','test')}
 rows={'valid':[],'test':[]};seen=set()
 with (ROOT/f'data/processed/{ds}.jsonl').open(encoding='utf8') as f:
  for line in f:
   r=json.loads(line)
   if r['split'] not in rows: continue
   s=split[r['id']];assert all(str(r[k])==s[k] for k in ('id','split','group_id','label'))
   assert r['id'] not in seen;seen.add(r['id'])
   item={'id':r['id'],'group_id':r['group_id'],'label':label_maps[ds][r['label']]}
   for condition in ('original','normalized'):
    e=tok.encode(r[condition],add_special_tokens=True)
    assert len(e.ids)<=128 and all(x==1 for x in e.attention_mask) and all(x==0 for x in e.type_ids)
    item[condition]=e.ids
   item['bert_input_changed']=item['original']!=item['normalized']
   rows[r['split']].append(item)
 assert seen==set(split)
 for s in ('valid','test'):
  assert len(rows[s])==expected[ds][s]
  all_splits[s][ds]=rows[s]
def encode(data):
 raw=json.dumps(data,separators=(',',':')).encode();return base64.b85encode(gzip.compress(raw,mtime=0)).decode(),hashlib.sha256(raw).hexdigest()
v,vhash=encode(all_splits['valid']);t,thash=encode(all_splits['test'])
manifest={'selection_sha256':EXPECTED_SELECTION,'protocol_id':lock['protocol_id'],'tokenizer_sha256':lock['files']['data/processed/tokenizer.json'],'valid_payload_sha256':vhash,'test_payload_sha256':thash,'counts':expected,'label_maps':label_maps,'no_raw_text':True}
if mode=='prefreeze_hash_bound':
 code=(ROOT/'src/weight_integrity.py').read_text(encoding='utf8')+'\n'+(ROOT/'src/evaluate_indobert_test_frozen.py').read_text(encoding='utf8')
else:
 code=(ROOT/'src/evaluate_indobert_test.py').read_text(encoding='utf8')
manifest['weight_hash_mode']=mode
manifest['evaluator_sha256']=hashlib.sha256(code.encode()).hexdigest()
script='SELECTION_B85 = '+repr(base64.b85encode(selection_bytes).decode())+'\nVALID_B85 = '+repr(v)+'\nTEST_B85 = '+repr(t)+'\nEVAL_MANIFEST = '+repr(manifest)+'\n'+code
(ROOT/'kaggle/private_test_eval.py').write_text(script,encoding='utf8')
(ROOT/'kaggle/test_eval_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps({'script_chars':len(script),'manifest':manifest},indent=2))
assert len(script)<990000

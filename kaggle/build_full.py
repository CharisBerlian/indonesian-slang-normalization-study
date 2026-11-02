"""Build private, token-ID-only full-training script from frozen train/valid splits."""
import base64, csv, difflib, hashlib, json, lzma, struct
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
lock=json.loads((ROOT/'data/protocol_lock.json').read_text(encoding='utf8'))
for name in ['configs/protocol.json','configs/resources.json','data/processed/tokenizer.json','data/splits/prdec.csv','data/splits/smsa.csv','src/audit_data.py','src/normalization.py']:
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==lock['files'][name], name
assert __import__('tokenizers').__version__=='0.22.1'
tok=Tokenizer.from_file(str(ROOT/'data/processed/tokenizer.json')); tok.enable_truncation(max_length=128)
payload={}; stats={}
label_maps={'prdec':{'negative':0,'positive':1},'smsa':{'negative':0,'neutral':1,'positive':2}}
expected={'prdec':{'train':3702,'valid':793},'smsa':{'train':10919,'valid':1260}}
for ds in ('prdec','smsa'):
    with (ROOT/f'data/splits/{ds}.csv').open(encoding='utf8',newline='') as f:
        source={r['id']:r for r in csv.DictReader(f) if r['split'] in ('train','valid')}
    rows=[]; seen=set()
    with (ROOT/f'data/processed/{ds}.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['split'] not in ('train','valid'): continue
            s=source[r['id']]
            assert all(str(r[k])==s[k] for k in ('id','split','group_id','label'))
            assert r['id'] not in seen; seen.add(r['id'])
            obj={'id':r['id'],'split':r['split'],'label':label_maps[ds][r['label']]}
            for condition in ('original','normalized'):
                e=tok.encode(r[condition],add_special_tokens=True)
                assert len(e.ids)<=128 and all(x==1 for x in e.attention_mask) and all(x==0 for x in e.type_ids)
                obj[condition]=e.ids
            rows.append(obj)
    assert seen==set(source)
    assert {s:sum(r['split']==s for r in rows) for s in ('train','valid')}==expected[ds]
    payload[ds]=rows
    stats[ds]={s:{'count':sum(r['split']==s for r in rows),'labels':dict(Counter(str(r['label']) for r in rows if r['split']==s))} for s in ('train','valid')}
raw=json.dumps(payload,separators=(',',':')).encode()
manifest={'protocol_id':lock['protocol_id'],'payload_sha256':hashlib.sha256(raw).hexdigest(),'tokenizer_sha256':lock['files']['data/processed/tokenizer.json'],'statistics':stats,'label_maps':label_maps,'contains_test_records':False,'contains_raw_text':False}
# Compact original uint16 token IDs and edit patches for normalized tokens.
packed=bytearray()
for ds in ('prdec','smsa'):
    packed.extend(struct.pack('<I',len(payload[ds])))
    for r in payload[ds]:
        ident=r['id'].encode('utf8'); original=r['original']; normalized=r['normalized']
        edits=[(a,b,normalized[c:d]) for op,a,b,c,d in difflib.SequenceMatcher(None,original,normalized,autojunk=False).get_opcodes() if op!='equal']
        assert len(ident)<256 and len(original)<=128 and len(edits)<256
        packed.extend(struct.pack('<B',len(ident))); packed.extend(ident)
        packed.extend(struct.pack('<BBBB',0 if r['split']=='train' else 1,r['label'],len(original),len(edits)))
        packed.extend(struct.pack('<'+'H'*len(original),*original))
        for a,b,replacement in edits:
            assert 0<=a<=b<=128 and len(replacement)<=128
            packed.extend(struct.pack('<BBB',a,b,len(replacement)))
            packed.extend(struct.pack('<'+'H'*len(replacement),*replacement))
manifest['packed_sha256']=hashlib.sha256(packed).hexdigest()
encoded=base64.b85encode(lzma.compress(bytes(packed),preset=9)).decode()
code=(ROOT/'src/train_indobert_full.py').read_text(encoding='utf8')
manifest['trainer_sha256']=hashlib.sha256(code.encode()).hexdigest()
script='PAYLOAD_B64 = '+repr(encoded)+'\nBUILD_MANIFEST = '+repr(manifest)+'\n'+code
(ROOT/'kaggle/private_full.py').write_text(script,encoding='utf8')
(ROOT/'kaggle/full_payload_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps({'script_bytes':len(script),'payload_sha256':manifest['payload_sha256'],'statistics':stats},indent=2))

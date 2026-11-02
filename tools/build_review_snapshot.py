"""Create an allowlist-only local research review snapshot; no publication."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
VERSION='2.0.0'
ALLOWLIST=(
    'REPRODUCIBILITY.md','REVIEW_SNAPSHOT.md',
    'configs/protocol.json','configs/resources.json','data/manifest.json','data/protocol_lock.json',
    'requirements-audit-lock.txt','requirements-pilot-svm-lock.txt','requirements-figure-lock.txt','requirements-freeze.txt',
    'src/normalization.py','src/audit_data.py','src/fetch_resources.py','src/verify_frozen_protocol.py',
    'src/train_svm_pilot.py','src/train_svm_search.py','src/train_indobert_full.py',
    'src/evaluate_models.py','src/evaluate_indobert_test.py','src/evaluate_indobert_test_frozen.py',
    'src/selection_io.py','src/weight_integrity.py','src/freeze_new_selection.py',
    'src/infer_svm_final.py','src/run_final_evaluation.py','src/analyze_fixed_models.py','src/generate_study_facts.py',
    'kaggle/build_full.py','kaggle/build_test_eval.py',
    'tests/test_data_protocol.py','tests/test_svm_pilot.py','tests/test_svm_search.py',
    'tests/test_evaluate_models.py','tests/test_fixed_model_diagnostics.py','tests/test_freeze_new_selection.py',
    'paper/main.tex','paper/supplement.tex','paper/references.bib','paper/IEEEtran.cls',
    'paper/build.ps1','paper/build_supplement.ps1',
    'paper/sections/abstract.tex','paper/sections/introduction.tex','paper/sections/literature_review.tex',
    'paper/sections/methodology.tex','paper/sections/results.tex',
    'paper/figures/create_normalization_effects.py','paper/figures/create_workflow.py',
    'paper/figures/normalization_effects.pdf','paper/figures/normalization_effects.png',
    'paper/figures/proposed_workflow.pdf','paper/figures/proposed_workflow.png',
    'output/pdf/indonesian_slang_sentiment_full_study.pdf','output/pdf/indonesian_slang_sentiment_supplement.pdf',
    'reports/audit_summary.json','reports/final_evaluation.md','reports/fixed_model_diagnostics.md',
    'reports/study_facts.md','reports/resource_availability.md',
    'results/final_evaluation/paired_results.json','results/final_evaluation/fixed_model_diagnostics.json',
    'results/final_evaluation/study_facts.json','tools/build_review_snapshot.py',
)
SECRET_PATTERNS=(re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                 re.compile(rb'(?i)authorization:\s*bearer\s+[A-Za-z0-9_.\-]{20,}'),
                 re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
                 re.compile(rb'\bAKIA[0-9A-Z]{16}\b'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/review_snapshot_v2')
    args=parser.parse_args(); dest=args.output.resolve(); archive=dest.with_suffix('.zip'); digest_file=archive.with_suffix('.zip.sha256')
    if not dest.is_relative_to(ROOT.resolve()): raise ValueError('Snapshot must be inside the workspace')
    if dest.exists() or archive.exists() or digest_file.exists(): raise FileExistsError('Refusing to overwrite an existing snapshot or archive')
    if len(ALLOWLIST)!=len(set(ALLOWLIST)): raise ValueError('Duplicate allowlist path')
    payload={}
    for name in ALLOWLIST:
        path=ROOT/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()): raise ValueError(f'Missing or unsafe allowlist file: {name}')
        data=path.read_bytes()
        if path.suffix.lower() not in ('.pdf','.png') and any(p.search(data) for p in SECRET_PATTERNS):
            raise ValueError(f'Potential secret pattern in allowlisted file: {name}')
        payload[name]=data
    manifest={'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'Local aggregate-only review preparation; no public release or license grant',
              'exclusions':['raw or normalized reviews','dictionary data and mappings','row-level IDs, labels, splits, groups and predictions',
                            'token payloads and generated private scripts','credentials','model weights'],
              'scan_limit':'Allowlist and limited secret-pattern scan; not a legal, privacy, anonymization or permission determination.',
              'files':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in payload.items()]}
    dest.mkdir(parents=True)
    for name,data in payload.items():
        path=dest/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    manifest_bytes=(json.dumps(manifest,indent=2)+'\n').encode('utf-8')
    (dest/'MANIFEST.json').write_bytes(manifest_bytes)
    with archive.open('xb') as f:
        with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for name,data in payload.items():z.writestr(name,data)
            z.writestr('MANIFEST.json',manifest_bytes)
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    with digest_file.open('x',encoding='utf-8',newline='\n') as f:f.write(f'{sha}  {archive.name}\n')
    print(f'Created {dest} and {archive}; {len(payload)} allowlisted files, archive SHA-256 {sha}')


if __name__=='__main__':main()

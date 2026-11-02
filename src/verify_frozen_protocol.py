"""Verify every file and the joint fingerprint in the existing protocol lock."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    lock=json.loads((root/'data/protocol_lock.json').read_text(encoding='utf-8'))
    for name,expected in lock['files'].items():
        path=(root/name).resolve()
        if root.resolve() not in path.parents: raise ValueError('Locked path escapes workspace')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        if h.hexdigest()!=expected: raise ValueError(f'Protocol lock mismatch: {name}')
    actual=hashlib.sha256(json.dumps(lock['files'],sort_keys=True).encode('utf-8')).hexdigest()
    if actual!=lock['protocol_id']: raise ValueError('Joint protocol fingerprint mismatch')
    return {'protocol_id':actual,'locked_file_count':len(lock['files'])}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(verify(),sort_keys=True))


if __name__=='__main__': main()

"""Read a selection manifest only when its companion SHA matches exact bytes."""
import hashlib
import hmac
from pathlib import Path
import re


def read_verified_selection(selection_path, hash_path):
    data=Path(selection_path).read_bytes()
    fields=Path(hash_path).read_text(encoding='utf-8').split()
    if not fields or not re.fullmatch(r'[0-9a-f]{64}',fields[0]):
        raise ValueError('Malformed selection SHA-256')
    actual=hashlib.sha256(data).hexdigest()
    if not hmac.compare_digest(actual,fields[0]):
        raise ValueError('Selection SHA-256 mismatch')
    return data,actual

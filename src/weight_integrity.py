"""Fail-closed verification for pre-freeze transformer weight hashes."""
import hmac
import re

HISTORICAL_SELECTION_SHA256 = '160bc9442b8912dfdccd79d6e2788566a433a08fab87e28e6184f5cc55ed8b3d'


def evaluator_mode(selection, selection_sha256):
    models=selection['indobert']['models']
    if len(models)!=12: raise ValueError('Expected twelve selected transformer models')
    hashes=[m['remote_checkpoint_sha256'] for m in models]
    if all(h is None for h in hashes):
        if selection_sha256!=HISTORICAL_SELECTION_SHA256:
            raise ValueError('Only the exact historical freeze may omit weight hashes')
        return 'historical_unbound'
    if any(not isinstance(h,str) or not re.fullmatch(r'[0-9a-f]{64}',h) for h in hashes):
        raise ValueError('New selections must bind all transformer checkpoint hashes')
    return 'prefreeze_hash_bound'


def enforce_frozen_weight_hash(expected, actual):
    if not isinstance(expected,str) or not re.fullmatch(r'[0-9a-f]{64}',expected):
        raise ValueError('New frozen checkpoint must have a valid SHA-256')
    if not isinstance(actual,str) or not hmac.compare_digest(expected,actual):
        raise ValueError('Frozen checkpoint SHA-256 mismatch')

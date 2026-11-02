"""Read-only integrity and identity checks for an extracted review snapshot.

Uses only the standard library. Verifies bytes, not training or permissions.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, strict=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    expected = {'MANIFEST.json'}
    for item in manifest['files']:
        name = item['path']
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name or name in expected:
            raise ValueError(f'Unsafe or duplicate manifest path: {name}')
        expected.add(name)
        path = root.joinpath(*rel.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or unsafe file: {name}')
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'Integrity mismatch: {name}')
    if strict:
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
                  if p.is_file() and '.git' not in p.relative_to(root).parts}
        if actual != expected:
            raise ValueError(f'Unexpected files: {sorted(actual - expected)}')
    identity = json.loads((root / 'ARTIFACT_IDENTITY.json').read_text())
    resources = json.loads((root / 'configs/resources.json').read_text())
    lock = json.loads((root / 'data/protocol_lock.json').read_text())
    source_manifest = json.loads((root / 'data/manifest.json').read_text())
    if manifest['version'] != identity['package_version']:
        raise ValueError('Package version mismatch')
    for key, value in identity['resource_revisions'].items():
        if resources[key] != value or source_manifest[key] != value:
            raise ValueError(f'Resource revision mismatch: {key}')
    protocol_id = hashlib.sha256(json.dumps(lock['files'], sort_keys=True).encode()).hexdigest()
    if protocol_id != lock['protocol_id'] or protocol_id != identity['protocol_id']:
        raise ValueError('Protocol fingerprint mismatch')
    for name, sha in identity['aggregate_result_sha256'].items():
        if digest(root / name) != sha:
            raise ValueError(f'Aggregate identity mismatch: {name}')
    included_lock_files = []
    for name, sha in lock['files'].items():
        if name in expected:
            if digest(root / name) != sha:
                raise ValueError(f'Bundled protocol file mismatch: {name}')
            included_lock_files.append(name)
    return {'version': manifest['version'], 'verified_payload_files': len(manifest['files']),
            'strict_file_inventory': strict, 'protocol_id': protocol_id,
            'verified_bundled_protocol_files': included_lock_files,
            'omitted_protocol_files': len(lock['files']) - len(included_lock_files),
            'note': 'Omitted raw/processed/split files require separate retrieval and reconstruction.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--strict', action='store_true', help='Reject unlisted files except Git metadata; use before builds/tests.')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.strict), indent=2))

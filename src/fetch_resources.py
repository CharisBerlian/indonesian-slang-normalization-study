"""Fetch pinned public resource files, never model weights or scraped reviews."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs/resources.json").read_text())
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    old = {f["path"]: f for f in previous["files"]}
    entries = []
    for spec in config["files"]:
        path = raw / spec["path"]
        if path.exists():
            content = path.read_bytes()
        else:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "Indonesian-NLP-research-audit/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
        digest = hashlib.sha256(content).hexdigest()
        expected = spec.get("sha256") or old.get(spec["path"], {}).get("sha256")
        if expected and digest != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        if not path.exists():
            path.write_bytes(content)
        entry = dict(spec, sha256=digest, bytes=len(content))
        entry["retrieved_utc"] = old.get(spec["path"], {}).get("retrieved_utc", datetime.now(timezone.utc).isoformat())
        entries.append(entry)
        print(spec["path"],len(content),digest[:12],flush=True)
    manifest_path.write_text(json.dumps(dict(config, files=entries), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()

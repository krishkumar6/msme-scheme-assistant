"""Download and unpack the prebuilt search index (Chroma + query adapter) into the data dir.

Used by the Dockerfile so a deployment doesn't need the 280 MB of source PDFs or a 25-minute
ingest. The bundle is `tar -czf index-bundle.tar.gz -C data index adapter`, attached to a GitHub
release; its SHA-256 is checked before anything is unpacked.

    python scripts/fetch_index.py <url> <sha256> [data_dir]
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path


def fetch(url: str, sha256: str, data_dir: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        sys.exit(f"index bundle checksum mismatch: expected {sha256}, got {digest}")
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(data_dir, filter="data")  # "data" filter: no absolute paths, links or devices
    print(f"unpacked {len(data) / 1e6:.1f} MB index bundle into {data_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    fetch(sys.argv[1], sys.argv[2], Path(sys.argv[3] if len(sys.argv) > 3 else "data"))

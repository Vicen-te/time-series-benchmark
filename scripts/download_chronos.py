"""Pre-download the Chronos foundation model into ``data/models/``.

The first attempt uses ``huggingface_hub`` (respects HF cache, resumable).
If SSL verification fails we fall back to a direct ``curl`` per file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import certifi  # noqa: E402

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())

REPO_ID = "amazon/chronos-bolt-small"
TARGET_DIR = PROJECT_ROOT / "data" / "models" / "chronos-bolt-small"
FILES = ("config.json", "generation_config.json", "model.safetensors")


def _via_huggingface_hub() -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=REPO_ID, local_dir=TARGET_DIR)
        return Path(path)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"huggingface_hub failed ({type(exc).__name__}); falling back to curl.")
        return None


def _via_curl(files: Iterable[str]) -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for fname in files:
        url = f"https://huggingface.co/{REPO_ID}/resolve/main/{fname}"
        dest = TARGET_DIR / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip {fname} (already present)")
            continue
        print(f"  downloading {fname}")
        subprocess.run(
            ["curl", "-sSL", "--ssl-no-revoke", "-o", str(dest), url],
            check=True,
        )
    return TARGET_DIR


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    result = _via_huggingface_hub()
    if result is None:
        result = _via_curl(FILES)
    print(f"Chronos weights ready at: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Launch the MLflow UI against ./mlruns/ with the 3.x file-store opt-in baked in.

MLflow 3.x deprecates the file-based tracking store and refuses to start by
default. Our Python entrypoints (``train.configure_mlflow`` and
``render_mlflow_summary``) opt in via ``os.environ.setdefault``, but the
``mlflow ui`` CLI bypasses our code -- so the env var has to be set before
spawning it. This wrapper handles that automatically.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def main() -> int:
    if not MLRUNS_DIR.exists():
        print("No runs yet. Run `python scripts/run_benchmark.py` first.")
        return 1
    # ``--workers 1`` avoids the noisy ``WinError 10022`` traceback that
    # MLflow's default multi-worker uvicorn emits on Windows: the OS does
    # not support the cross-process socket sharing the supervisor relies on,
    # so one worker dies on startup and is silently respawned. A single
    # worker is more than enough for a local browse-runs UI.
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            MLRUNS_DIR.resolve().as_uri(),
            "--workers",
            "1",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

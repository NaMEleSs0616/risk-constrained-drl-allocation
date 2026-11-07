#!/usr/bin/env python
"""Write MANIFEST.json: provenance and SHA-256 hashes for a release.

Records the git commit, the Python and key package versions, the SHA-256 of the local
price file (which is not redistributed, so users can check that their download matches),
and the SHA-256 and size of every git-tracked file under the result and paper folders.

Usage:
    python scripts/make_manifest.py --out MANIFEST.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
TRACKED_DIRS = ("outputs", "outputs_real", "outputs_real_followup", "outputs_real_nextopen",
                "outputs_syn_nextopen", "paper", "config")
PACKAGES = ("numpy", "pandas", "scipy", "torch", "stable-baselines3", "gymnasium", "scikit-learn", "matplotlib")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="MANIFEST.json")
    p.add_argument("--data", default="data/etf_daily.csv")
    args = p.parse_args(argv)

    files = {}
    for rel in git("ls-files", *TRACKED_DIRS).splitlines():
        f = ROOT / rel
        if f.is_file():
            files[rel] = {"sha256": sha256(f), "bytes": f.stat().st_size}
    data = ROOT / args.data
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit_at_creation": git("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "data": {
            "path": args.data,
            "sha256": sha256(data) if data.exists() else None,
            "bytes": data.stat().st_size if data.exists() else None,
            "provenance": "data/etf_daily.json (Yahoo Finance via scripts/download_data.py; not redistributed)",
        },
        "files": files,
    }
    (ROOT / args.out).write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(files)} files; data sha256 {manifest['data']['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

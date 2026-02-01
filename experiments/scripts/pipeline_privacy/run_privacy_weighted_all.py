#!/usr/bin/env python3

"""
Convenience runner to execute all weighted privacy pipeline scripts sequentially.

Scripts covered:
- adult_ctgan_privacy_pipeline_weighted_5.py
- adult_tvae_privacy_pipeline_weighted.py
- adult_tvae_privacy_pipeline_weighted_5.py
- cardio_ctgan_privacy_pipeline_weighted.py
- cardio_ctgan_privacy_pipeline_weighted_5.py
- cardio_tvae_privacy_pipeline_weighted.py
- cardio_tvae_privacy_pipeline_weighted_5.py
- credit_ctgan_privacy_pipeline_weighted.py
- credit_ctgan_privacy_pipeline_weighted_5.py
- credit_tvae_privacy_pipeline_weighted.py
- credit_tvae_privacy_pipeline_weighted_5.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_scripts(scripts: list[Path]) -> int:
    """Run each script with python3; stop on the first failure."""
    for script in scripts:
        print(f"\n[run] {script}")
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"[fail] {script} exited with code {result.returncode}")
            return result.returncode
    return 0


def main() -> int:
    base = Path(__file__).resolve().parent
    scripts = [
        base / "adult_ctgan_privacy_pipeline_weighted_5.py",
        base / "adult_tvae_privacy_pipeline_weighted.py",
        base / "adult_tvae_privacy_pipeline_weighted_5.py",
        base / "cardio_ctgan_privacy_pipeline_weighted.py",
        base / "cardio_ctgan_privacy_pipeline_weighted_5.py",
        base / "cardio_tvae_privacy_pipeline_weighted.py",
        base / "cardio_tvae_privacy_pipeline_weighted_5.py",
        base / "credit_ctgan_privacy_pipeline_weighted.py",
        base / "credit_ctgan_privacy_pipeline_weighted_5.py",
        base / "credit_tvae_privacy_pipeline_weighted.py",
        base / "credit_tvae_privacy_pipeline_weighted_5.py",
    ]
    return run_scripts(scripts)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python scripts/check_environment.py

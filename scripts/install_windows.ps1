$ErrorActionPreference = "Stop"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python scripts/check_environment.py
if ($LASTEXITCODE -ne 0) { throw "Environment check failed. Read environment_report.json." }

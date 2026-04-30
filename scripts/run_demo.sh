#!/usr/bin/env bash
set -euo pipefail

cd /data/project
source /data/project/.venv/bin/activate
export PYTHONPATH=/data/project/src

python /data/project/scripts/run_live_demo.py

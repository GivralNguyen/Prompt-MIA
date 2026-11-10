#!/bin/bash
#SBATCH --job-name=pmia_t6
#SBATCH --output=logs/t6-%j.out
#SBATCH --time=0:30:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# Table 6 (configs/table6.json): the per-model outlier detection results of Table 1 (results/outlier/), tabulated through
# the package pipeline; nothing is trained or evaluated here.
set -euo pipefail
PY=${PY:-python}
cd "$SLURM_SUBMIT_DIR"
$PY - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, '.')
from promptmia.workflow import build_plan, execute, load_manifest
config = load_manifest(Path('configs/table6.json'))
plan = build_plan(config, config['artifacts'])
print('PLOT table6', flush=True)
execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), Path('.').resolve(), 'plot', 'cpu')
print('DONE', flush=True)
EOF

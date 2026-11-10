#!/bin/bash
#SBATCH --job-name=pmia_fig12_14
#SBATCH --output=logs/fig12_14-%A_%a.out
#SBATCH --time=1:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=12-14
# Figures 12-14 (configs/figure12.json, figure13.json, figure14.json; array index = figure number). Each task draws one
# figure through the package pipeline from the Figure 4 attacks on the base models; nothing is trained or attacked here.
set -euo pipefail
PY=${PY:-python}
cd "$SLURM_SUBMIT_DIR"
$PY - "$SLURM_ARRAY_TASK_ID" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, '.')
from promptmia.workflow import build_plan, execute, load_manifest
number = int(sys.argv[1])
config = load_manifest(Path(f'configs/figure{number}.json'))
plan = build_plan(config, config['artifacts'])
print('PLOT', f'figure{number}', flush=True)
execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), Path('.').resolve(), 'plot', 'cpu')
print('DONE', number, flush=True)
EOF

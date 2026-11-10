#!/bin/bash
#SBATCH --job-name=pmia_t4
#SBATCH --output=logs/t4-%A_%a.out
#SBATCH --time=2:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3%5
# Table 4 (configs/table4.json). Each array task handles one job through the package pipeline so it writes the proper
# completion stamp. Set STAGE: 'evaluate' (array 0-3, the key-drift audit on each dataset's ViT-B/32 base model in
# runs/base, which is reused and never retrained here) or 'plot' (array 0, after evaluation).
set -euo pipefail
export HF_HUB_OFFLINE=1
PY=${PY:-python}
STAGE=${STAGE:-evaluate}
cd "$SLURM_SUBMIT_DIR"
$PY - "$SLURM_ARRAY_TASK_ID" "$STAGE" <<'EOF'
import os, sys
from pathlib import Path
sys.path.insert(0, '.')
from promptmia.workflow import asset_environment, build_plan, execute, load_manifest
root = Path('.').resolve()
asset_environment(root)
os.environ['HF_HUB_OFFLINE'] = '1'
i, stage = int(sys.argv[1]), sys.argv[2]
config = load_manifest(Path('configs/table4.json'))
plan = build_plan(config, config['artifacts'])
if stage == 'evaluate':
    job = plan['evaluation'][i]
    print('EVALUATE', job['name'], flush=True)
    execute(dict(training=plan['training'], evaluation=[job], artifacts=[]), root, 'evaluate', 'cpu')
elif stage == 'plot':
    print('PLOT table4', flush=True)
    execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), root, 'plot', 'cpu')
else:
    raise SystemExit(f'unknown STAGE {stage}')
print('DONE', stage, i, flush=True)
EOF

#!/bin/bash
#SBATCH --job-name=pmia_table2
#SBATCH --output=logs/table2-%A_%a.out
#SBATCH --time=08:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-31%5
# Table 2 (configs/table2.json). Each array task handles one job through the package pipeline so it writes the proper
# completion stamp. Set STAGE: 'train' (array 0-31, training seeds 1 and 2; the seed 0 models of Figures 4 and 7 are
# reused and never retrained here) or 'plot' (array 0, after training).
set -euo pipefail
export HF_HUB_OFFLINE=1
PY=${PY:-python}
STAGE=${STAGE:-train}
cd "$SLURM_SUBMIT_DIR"
[ "$STAGE" != plot ] && nvidia-smi -L
$PY - "$SLURM_ARRAY_TASK_ID" "$STAGE" <<'EOF'
import os, sys
from pathlib import Path
sys.path.insert(0, '.')
from promptmia.workflow import asset_environment, build_plan, execute, load_manifest
root = Path('.').resolve()
asset_environment(root)
os.environ['HF_HUB_OFFLINE'] = '1'
i, stage = int(sys.argv[1]), sys.argv[2]
config = load_manifest(Path('configs/table2.json'))
plan = build_plan(config, config['artifacts'])
if stage == 'train':
    job = [j for j in plan['training'] if j['seed'] != 0][i]
    print('TRAIN', job['name'], flush=True)
    execute(dict(training=[job], evaluation=[], artifacts=[]), root, 'train', 'cuda')
elif stage == 'plot':
    print('PLOT table2', flush=True)
    execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), root, 'plot', 'cpu')
else:
    raise SystemExit(f'unknown STAGE {stage}')
print('DONE', stage, i, flush=True)
EOF

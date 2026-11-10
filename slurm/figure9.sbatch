#!/bin/bash
#SBATCH --job-name=pmia_fig9
#SBATCH --output=logs/fig9-%A_%a.out
#SBATCH --time=16:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-11%5
# Figure 9 (configs/figure9.json). Each array task handles one job through the package pipeline so it writes the proper
# completion stamp. Set STAGE: 'train' (array 0-11, the 12 prompt-dropout models; the ViT-B/32 base models in runs/base are
# reused and never retrained here), 'evaluate' (array 0-11, the 12 dropout attacks; the no-defense results of Figure 6
# are reused), or 'plot' (array 0, after evaluation).
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
config = load_manifest(Path('configs/figure9.json'))
plan = build_plan(config, config['artifacts'])
if stage == 'train':
    job = [j for j in plan['training'] if j['name'].startswith('dropout/')][i]
    print('TRAIN', job['name'], flush=True)
    execute(dict(training=[job], evaluation=[], artifacts=[]), root, 'train', 'cuda')
elif stage == 'evaluate':
    job = [j for j in plan['evaluation'] if j['name'].startswith('dropout/')][i]
    print('EVALUATE', job['name'], flush=True)
    execute(dict(training=plan['training'], evaluation=[job], artifacts=[]), root, 'evaluate', 'cuda')
elif stage == 'plot':
    print('PLOT figure9', flush=True)
    execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), root, 'plot', 'cpu')
else:
    raise SystemExit(f'unknown STAGE {stage}')
print('DONE', stage, i, flush=True)
EOF

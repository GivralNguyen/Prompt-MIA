#!/bin/bash
#SBATCH --job-name=pmia_fig17
#SBATCH --output=logs/fig17-%A_%a.out
#SBATCH --time=16:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-39%5
# Figure 17 (configs/figure17.json). Each array task handles one job through the package pipeline so it writes the proper
# completion stamp. Set STAGE: 'train' (array 0-39, the 40 retrained models: M and N ablations, the key lr 1e-4 training
# dynamics and the local epochs, longest first; the ViT-B/32 base models in runs/base are reused and never retrained here),
# 'evaluate' (array 0-99, the 100 attacks; the base models' attacks of Figure 4 are reused), or 'plot' (array 0, after
# evaluation).
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
config = load_manifest(Path('configs/figure17.json'))
plan = build_plan(config, config['artifacts'])
# Relative training time per dataset at the default 5 local epochs.
cost = dict(cifar10=1, cifar100=1, tinyimagenet=1.85, fourdataset=2.15)
if stage == 'train':
    jobs = [j for j in plan['training'] if not j['name'].startswith('base/')]
    job = sorted(jobs, key=lambda j: -cost[j['dataset']]*j['local_epochs'])[i]
    print('TRAIN', job['name'], flush=True)
    execute(dict(training=[job], evaluation=[], artifacts=[]), root, 'train', 'cuda')
elif stage == 'evaluate':
    job = [j for j in plan['evaluation'] if not j['name'].startswith('main/')][i]
    print('EVALUATE', job['name'], flush=True)
    execute(dict(training=plan['training'], evaluation=[job], artifacts=[]), root, 'evaluate', 'cuda')
elif stage == 'plot':
    print('PLOT figure17', flush=True)
    execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), root, 'plot', 'cpu')
else:
    raise SystemExit(f'unknown STAGE {stage}')
print('DONE', stage, i, flush=True)
EOF

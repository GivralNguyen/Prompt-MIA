#!/bin/bash
#SBATCH --job-name=pmia_f21
#SBATCH --output=logs/f21-%A_%a.out
#SBATCH --time=3:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-7%5
# Figure 21 (configs/figure21.json). Each array task handles one job through the package pipeline so it writes the proper
# completion stamp. Set STAGE: 'train' (array 0-7, ViT-B/32 with 160 and 320 clients on the four datasets, longest
# first; the 80-client models and attacks are those of Table 1), 'evaluate' (array 0-7, the attack on each new model),
# or 'plot' (array 0, after evaluation).
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
config = load_manifest(Path('configs/figure21.json'))
plan = build_plan(config, config['artifacts'])
cost = dict(cifar10=1, cifar100=1, tinyimagenet=1.85, fourdataset=2.15)
if stage == 'train':
    jobs = sorted((j for j in plan['training'] if j['name'].startswith('clients/')),
                  key=lambda j: -cost[j['dataset']]*80/j['clients'])
    job = jobs[i]
    print('TRAIN', job['name'], flush=True)
    execute(dict(training=[job], evaluation=[], artifacts=[]), root, 'train', 'cuda')
elif stage == 'evaluate':
    job = [j for j in plan['evaluation'] if j['name'].startswith('clients/')][i]
    print('EVALUATE', job['name'], flush=True)
    execute(dict(training=plan['training'], evaluation=[job], artifacts=[]), root, 'evaluate', 'cuda')
elif stage == 'plot':
    print('PLOT figure21', flush=True)
    execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), root, 'plot', 'cpu')
else:
    raise SystemExit(f'unknown STAGE {stage}')
print('DONE', stage, i, flush=True)
EOF

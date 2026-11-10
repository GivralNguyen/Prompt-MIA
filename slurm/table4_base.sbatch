#!/bin/bash
#SBATCH --job-name=pmia_t4base
#SBATCH --output=logs/t4base-%A_%a.out
#SBATCH --time=16:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3%5
# Train the four ViT-B/32 base models Table 4 needs (configs/table4.json training[i]): cifar10, cifar100,
# tinyimagenet, fourdataset. Each task trains one model through the package pipeline so it writes the proper
# completion stamp; afterwards `python scripts/table4.py` skips training and runs the drift evaluation + table.
# Assets are the shared caches wired via symlinks (models/torch, models/huggingface, data).
set -euo pipefail
export HF_HUB_OFFLINE=1
PY=${PY:-python}
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
$PY - "$SLURM_ARRAY_TASK_ID" <<'EOF'
import os, sys
from pathlib import Path
sys.path.insert(0, '.')
from promptmia.workflow import asset_environment, build_plan, execute, load_manifest
root = Path('.').resolve()
asset_environment(root)
os.environ['HF_HUB_OFFLINE'] = '1'
i = int(sys.argv[1])
config = load_manifest(Path('configs/table4.json'))
plan = build_plan(config, config['artifacts'])
job = plan['training'][i]
print('TRAIN', job['name'], flush=True)
execute(dict(training=[job], evaluation=[], artifacts=[]), root, 'train', 'cuda')
print('DONE', job['name'], flush=True)
EOF

#!/bin/bash
#SBATCH --job-name=pmia_f20
#SBATCH --output=logs/f20-%A_%a.out
#SBATCH --time=3:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2%5
# Figure 20 (configs/figure20.json). Each array task handles one job through the package pipeline so it writes the proper
# completion stamp. Set STAGE: 'train' (array 0-2: ViLT on UPMC Food-101 with image and text, ViLT text only, CLIP
# ViT-B/32 on Fashion-Product), 'evaluate' (array 0-2, the attack on each model's round-60 keys), or 'plot' (array 0,
# after evaluation). Needs data/food101, data/fashionmm_clip and the ViLT, BERT and CLIP weights in models/ (docs/data.md).
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
config = load_manifest(Path('configs/figure20.json'))
plan = build_plan(config, config['artifacts'])
if stage == 'train':
    job = plan['training'][i]
    print('TRAIN', job['name'], flush=True)
    execute(dict(training=[job], evaluation=[], artifacts=[]), root, 'train', 'cuda')
elif stage == 'evaluate':
    job = plan['evaluation'][i]
    print('EVALUATE', job['name'], flush=True)
    execute(dict(training=plan['training'], evaluation=[job], artifacts=[]), root, 'evaluate', 'cuda')
elif stage == 'plot':
    print('PLOT figure20', flush=True)
    execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), root, 'plot', 'cpu')
else:
    raise SystemExit(f'unknown STAGE {stage}')
print('DONE', stage, i, flush=True)
EOF

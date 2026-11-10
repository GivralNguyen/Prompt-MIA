#!/bin/bash
#SBATCH --job-name=pmia_fig15_16
#SBATCH --output=logs/fig15_16-%A_%a.out
#SBATCH --time=1:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=15-16
# Figures 15 and 16 (configs/figure15.json, figure16.json; array index = figure number) on the CIFAR-10 ViT-B/32 base
# model in runs/base, which is reused and never retrained here. Set STAGE: 'evaluate' (the outlier detectors and the
# t-SNE of the key pool, with and without PromptMIA keys) or 'plot' (after evaluation). scikit-learn is the 1.5.2
# pinned by requirements.txt.
set -euo pipefail
export HF_HUB_OFFLINE=1
export PYTHONPATH=$HOME/.cache/sklearn-1.5.2${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
STAGE=${STAGE:-evaluate}
cd "$SLURM_SUBMIT_DIR"
$PY -c "import sklearn; print('scikit-learn', sklearn.__version__); assert sklearn.__version__ == '1.5.2'"
$PY - "$SLURM_ARRAY_TASK_ID" "$STAGE" <<'EOF'
import os, sys
from pathlib import Path
sys.path.insert(0, '.')
from promptmia.workflow import asset_environment, build_plan, execute, load_manifest
root = Path('.').resolve()
asset_environment(root)
os.environ['HF_HUB_OFFLINE'] = '1'
number, stage = int(sys.argv[1]), sys.argv[2]
config = load_manifest(Path(f'configs/figure{number}.json'))
plan = build_plan(config, config['artifacts'])
if stage == 'evaluate':
    print('EVALUATE', [j['name'] for j in plan['evaluation']], flush=True)
    execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=[]), root, 'evaluate', 'cpu')
elif stage == 'plot':
    print('PLOT', f'figure{number}', flush=True)
    execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), root, 'plot', 'cpu')
else:
    raise SystemExit(f'unknown STAGE {stage}')
print('DONE', stage, number, flush=True)
EOF

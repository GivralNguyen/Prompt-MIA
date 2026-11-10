#!/bin/bash
#SBATCH --job-name=pmia_table1
#SBATCH --output=logs/table1-%A_%a.out
#SBATCH --time=08:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-8%5
# Table 1 (configs/table1.json) on 9 trained models in runs/base. Each array task handles one job through the package
# pipeline so it writes the proper completion stamp. Set STAGE: 'evaluate' (array 0-8) or 'plot' (array 0, after
# evaluation). No stage trains, so this cannot overwrite a model. scikit-learn is the 1.5.2 pinned by requirements.txt.
set -euo pipefail
export HF_HUB_OFFLINE=1
export PYTHONPATH=$HOME/.cache/sklearn-1.5.2${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
STAGE=${STAGE:-evaluate}
cd "$SLURM_SUBMIT_DIR"
[ "$STAGE" != plot ] && nvidia-smi -L
$PY -c "import sklearn; print('scikit-learn', sklearn.__version__); assert sklearn.__version__ == '1.5.2'"
$PY - "$SLURM_ARRAY_TASK_ID" "$STAGE" <<'EOF'
import os, sys
from pathlib import Path
sys.path.insert(0, '.')
from promptmia.workflow import asset_environment, build_plan, execute, load_manifest
root = Path('.').resolve()
asset_environment(root)
os.environ['HF_HUB_OFFLINE'] = '1'
i, stage = int(sys.argv[1]), sys.argv[2]
config = load_manifest(Path('configs/table1.json'))
plan = build_plan(config, config['artifacts'])
if stage == 'evaluate':
    job = plan['evaluation'][i]
    print('EVALUATE', job['name'], flush=True)
    execute(dict(training=plan['training'], evaluation=[job], artifacts=[]), root, 'evaluate', 'cuda')
elif stage == 'plot':
    print('PLOT table1', flush=True)
    execute(dict(training=plan['training'], evaluation=plan['evaluation'], artifacts=plan['artifacts']), root, 'plot', 'cpu')
else:
    raise SystemExit(f'unknown STAGE {stage}')
print('DONE', stage, i, flush=True)
EOF

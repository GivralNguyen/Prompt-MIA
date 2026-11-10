#!/bin/bash
#SBATCH --job-name=pmia_fig5
#SBATCH --output=logs/fig5-%j.out
#SBATCH --time=02:00:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Figure 5 (configs/figure5.json) on the 12 trained models in runs/base. Only the evaluate and plot stages run, so this
# cannot retrain or overwrite a model; a missing or changed model is an error. scikit-learn is the 1.5.2 pinned by
# requirements.txt.
set -euo pipefail
export HF_HUB_OFFLINE=1
export PYTHONPATH=$HOME/.cache/sklearn-1.5.2${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
$PY -c "import sklearn; print('scikit-learn', sklearn.__version__); assert sklearn.__version__ == '1.5.2'"
$PY scripts/fig5.py --stage evaluate --device cuda
$PY scripts/fig5.py --stage plot
echo DONE

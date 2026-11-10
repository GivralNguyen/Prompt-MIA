#!/bin/bash
#SBATCH --job-name=pmia_fig2_3
#SBATCH --output=logs/fig2_3-%j.out
#SBATCH --time=00:30:00
# Set up your cluster-specific SBATCH options here (e.g. --partition, --account, --qos).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Figures 2 and 3 (configs/figure2.json, configs/figure3.json) on the trained model runs/base/cifar10_vit_b32.
# Only the evaluate and plot stages run, so this cannot retrain or overwrite the model; a missing or changed model is
# an error. scikit-learn is the 1.5.2 pinned by requirements.txt. A GPU is required to encode the queries.
set -euo pipefail
export HF_HUB_OFFLINE=1
export PYTHONPATH=$HOME/.cache/sklearn-1.5.2${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
$PY -c "import sklearn; print('scikit-learn', sklearn.__version__); assert sklearn.__version__ == '1.5.2'"
for FIG in fig2 fig3; do
  echo "=== $FIG ==="
  $PY scripts/$FIG.py --stage evaluate --device cuda
  $PY scripts/$FIG.py --stage plot
done
echo DONE

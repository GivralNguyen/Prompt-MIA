"""Show package versions and whether this machine can run CUDA experiments."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from promptmia.workflow import asset_environment, doctor


if __name__ == "__main__":
    asset_environment(ROOT)
    raise SystemExit(doctor(ROOT))

"""Figure 19: Large-batch membership inference, averaged over three models."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from promptmia.experiment import run_output


def main():
    # This configuration contains this output's training, measurement, and plotting settings.
    run_output(ROOT / "configs" / "figure19.json", __doc__)


if __name__ == "__main__":
    main()

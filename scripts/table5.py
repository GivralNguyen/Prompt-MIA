"""Table 5: Attack success rate under extreme non-IID data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from promptmia.experiment import run_output


def main():
    # This configuration contains this output's training, measurement, and plotting settings.
    run_output(ROOT / "configs" / "table5.json", __doc__)


if __name__ == "__main__":
    main()

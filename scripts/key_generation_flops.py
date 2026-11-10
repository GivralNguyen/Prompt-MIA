"""Count key-generation arithmetic for the paper's computational-overhead paragraph."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from promptmia.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "key_generation_flops.json")
    args = parser.parse_args()
    from promptmia.evaluation import key_generation_flops
    # Four injected keys replace four of the twenty global keys.
    result = dict(pool_size=20, injected_keys=4, benign_keys=16, dimension=768,
                  flops=key_generation_flops(16, 4, 768))
    write_json(args.output, result)
    print(f"Adversarial key generation: {result['flops']:,} FLOPs")


if __name__ == "__main__":
    main()

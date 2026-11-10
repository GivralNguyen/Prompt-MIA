"""Command-line options shared by the separate figure and table programs."""

import argparse
from pathlib import Path

from .io import project_root
from .workflow import asset_environment, build_plan, choose_device, execute, load_manifest, print_plan


def run_output(config_path, description, argv=None):
    """Run the training, measurements, and plotting for one paper output."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path(config_path),
                        help="Configuration for this figure or table")
    parser.add_argument("--root", type=Path, default=project_root(),
                        help="Project directory containing data, models, and results")
    parser.add_argument("--stage", choices=("all", "train", "evaluate", "plot"), default="all",
                        help="Run every required step, or only the selected stage")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, or auto")
    parser.add_argument("--dry-run", action="store_true", help="Show the required jobs without running them")
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs in the selected stage")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    asset_environment(root)
    try:
        config = load_manifest(args.config)
        if len(config["artifacts"]) != 1:
            raise ValueError("Each figure or table configuration must describe exactly one output.")
        plan = build_plan(config, config["artifacts"])
        if args.dry_run:
            print_plan(plan)
            return
        needs_device = args.stage != "plot" and (plan["training"] or plan["evaluation"])
        device = choose_device(args.device) if needs_device else "cpu"
        execute(plan, root, args.stage, device, args.force)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")

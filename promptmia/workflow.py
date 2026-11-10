"""Shared job execution used by the individual figure and table scripts."""

import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from .io import fingerprint, project_root, read_json, relative_name, write_json


def load_manifest(path):
    manifest = read_json(path)
    for section in ("training", "evaluation", "artifacts"):
        entries = manifest.get(section, [])
        field = "id" if section == "artifacts" else "name"
        names = [entry[field] for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate names in {section}")
        for name in names:
            relative_name(name)
    return manifest


def build_plan(manifest, artifacts):
    training = {job["name"]: job for job in manifest["training"]}
    evaluation = {job["name"]: job for job in manifest["evaluation"]}
    train_names, eval_names, visiting = set(), [], set()

    def visit(source):
        if source.startswith("train:"):
            name = source[6:]
            if name not in training:
                raise ValueError(f"Missing training job: {name}")
            train_names.add(name)
            return
        if source in eval_names:
            return
        if source in visiting:
            raise ValueError(f"Cyclic evaluation dependency: {source}")
        if source not in evaluation:
            raise ValueError(f"Missing evaluation job: {source}")
        visiting.add(source)
        job = evaluation[source]
        if job.get("run"):
            visit("train:" + job["run"])
        for dependency in job.get("depends_on", []):
            visit(dependency)
        eval_names.append(source)
        visiting.remove(source)

    for artifact in artifacts:
        for source in artifact.get("sources", []):
            visit(source)
        for panel in artifact.get("panels", []):
            for source in panel.get("sources", []):
                visit(source)
            for curve in panel.get("curves", []):
                for source in curve.get("sources", []):
                    visit(source)
    return {
        "training": [job for name, job in training.items() if name in train_names],
        "evaluation": [evaluation[name] for name in eval_names],
        "artifacts": artifacts,
    }


def print_plan(plan):
    for section, label in (("training", "TRAIN"), ("evaluation", "EVALUATE"), ("artifacts", "OUTPUT")):
        for job in plan[section]:
            name = job.get("name", job.get("id"))
            description = job.get("title", job.get("kind", job.get("dataset", "")))
            print(f"{label:8s} {name:48s} {description}")
    print(f"\n{len(plan['training'])} training jobs, {len(plan['evaluation'])} evaluation jobs, "
          f"{len(plan['artifacts'])} paper outputs")


def choose_device(requested):
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build or select --device cpu.")
    return requested


def code_fingerprint(root, section):
    """A changed implementation invalidates completion records automatically."""
    import hashlib
    digest = hashlib.sha256()
    files = {
        "training": ("training", "fl", "model", "backbones", "attack", "data", "defenses", "vilt", "clip", "io"),
        "evaluation": ("evaluation", "attack", "bound", "diagnostics", "loss_mia", "outliers", "model", "backbones", "data", "defenses", "vilt", "clip", "io"),
        "artifacts": ("plotting", "io"),
    }[section]
    for path in sorted(root / "promptmia" / (name + ".py") for name in files):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def asset_environment(root):
    """Set cache locations before importing the model libraries."""
    os.environ["TORCH_HOME"] = str(root / "models" / "torch")
    os.environ["HF_HOME"] = str(root / "models" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(root / "models" / "huggingface" / "hub")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache" / "matplotlib"))


def input_files(job, section, root):
    """Record the measured inputs, so changing a result also redraws its plots."""
    paths = []

    def add_training(name, full=False):
        run = root / "runs" / name
        paths.extend(run / f for f in ("config.json", "metrics.json"))
        if full:
            paths.extend(run / f for f in ("keys.pt", "state.pt"))
            if (run / "mia_losses.pt").exists():
                paths.append(run / "mia_losses.pt")
            if (run / "config.json").exists():
                cache = read_json(run / "config.json").get("query_cache")
                if cache:
                    paths.append(root / cache)

    def add_source(source):
        if source.startswith("train:"):
            add_training(source[6:])
        else:
            paths.append(root / "results" / (source + ".json"))

    if section == "evaluation":
        if job.get("run"):
            add_training(job["run"], full=True)
        for source in job.get("depends_on", []):
            add_source(source)
    elif section == "artifacts":
        if job.get("static_asset"):
            paths.append(root / job["static_asset"])
        for source in job.get("sources", []):
            add_source(source)
        for panel in job.get("panels", []):
            for source in panel.get("sources", []):
                add_source(source)
            for curve in panel.get("curves", []):
                for source in curve.get("sources", []):
                    add_source(source)
    records = []
    for path in sorted(set(paths)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input {path}. Run the required training/evaluation stage first.")
        stat = path.stat()
        label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        records.append((label, stat.st_size, stat.st_mtime_ns))
    return records


def require_training(job, root):
    """Prevent an evaluation from silently using a different training setup."""
    folder = root / "runs" / job["name"]
    config = read_json(folder / "config.json")
    metrics = read_json(folder / "metrics.json")
    changed = [key for key, value in job.items() if config.get(key) != value]
    if changed:
        raise ValueError(f"Training configuration changed for {job['name']}: {', '.join(changed)}. Run --stage train first.")
    if not metrics.get("completed", False):
        raise ValueError(f"Training {job['name']} has not completed. Run --stage train first.")


def execute(plan, root, stage, device, force=False):
    asset_environment(root)
    training = {job["name"]: job for job in plan["training"]}
    journal = root / "logs" / "reproduction.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    allowed = {"train": {"training"}, "evaluate": {"evaluation"},
               "plot": {"artifacts"}, "all": {"training", "evaluation", "artifacts"}}[stage]

    for section in ("training", "evaluation", "artifacts"):
        if section not in allowed:
            continue
        implementation = code_fingerprint(root, section)
        for job in plan[section]:
            name = job.get("name", job.get("id"))
            stamp_path = root / "logs" / "completed" / section / (name + ".json")
            if section == "evaluation" and job.get("run"):
                require_training(training[job["run"]], root)
            inputs = input_files(job, section, root)
            signature = fingerprint({"job": job, "implementation": implementation, "inputs": inputs})
            if stamp_path.exists() and not force:
                stamp = read_json(stamp_path)
                outputs = stamp.get("outputs", [])
                if stamp.get("signature") == signature and outputs and all((root / p).exists() for p in outputs):
                    print(f"SKIP {name} (complete)", flush=True)
                    continue
            started = time.time()
            # An interrupted rerun must never leave an old success marker usable.
            stamp_path.unlink(missing_ok=True)
            print(f"RUN  {section}: {name}", flush=True)
            try:
                if section == "training":
                    if job.get("backbone") == "vilt":
                        from .vilt import train_vilt
                        train_vilt(job, root, device)
                    elif job.get("backbone") == "clip":
                        from .clip import train_clip
                        train_clip(job, root, device)
                    else:
                        from .training import train
                        train(job, root, device)
                    outputs = [f"runs/{name}/{f}" for f in ("config.json", "metrics.json", "keys.pt", "state.pt")]
                    if job.get("mia_target") is not None:
                        outputs.append(f"runs/{name}/mia_losses.pt")
                    saved_config = read_json(root / "runs" / name / "config.json")
                    outputs.append(saved_config["query_cache"])
                    outputs.extend(f"runs/{name}/ckpt_round{r}.pt" for r in job.get("save_rounds", []))
                elif section == "evaluation":
                    from .evaluation import evaluate_job
                    result = evaluate_job(job, root, device)
                    path = root / "results" / (name + ".json")
                    if not path.exists():
                        write_json(path, result)
                    outputs = [str(path.relative_to(root))]
                else:
                    from .plotting import plot_artifact
                    paths = plot_artifact(job, root)
                    outputs = []
                    for p in paths:
                        path = Path(p)
                        if path.is_absolute():
                            path = path.relative_to(root)
                        outputs.append(str(path))
                missing = [path for path in outputs if not (root / path).is_file()]
                if missing:
                    raise RuntimeError(f"Job did not create required outputs: {missing}")
                record = {"name": name, "stage": section, "signature": signature,
                          "seconds": time.time() - started, "outputs": outputs}
                write_json(stamp_path, record)
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(record, status="complete")) + "\n")
            except Exception as exc:
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"name": name, "stage": section, "status": "failed", "error": str(exc)}) + "\n")
                raise


def doctor(root):
    """Inspect the installation without downloading data or starting a job."""
    packages = ("torch", "torchvision", "timm", "numpy", "scipy", "scikit-learn", "matplotlib", "Pillow", "transformers", "pyarrow", "datasets")
    versions = {}
    for line in (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            package, version = line.strip().split("==", 1)
            versions[package] = version
    print(f"Project: {root}\nPython:  {sys.version.split()[0]}")
    missing = []
    for name in packages:
        try:
            version = importlib.metadata.version(name)
            expected = versions.get(name)
            matches = expected is None or version.split("+", 1)[0] == expected
            print(f"{name:16s} {version}" + (f" (expected {expected})" if not matches else ""))
            if not matches:
                missing.append(name)
        except importlib.metadata.PackageNotFoundError:
            print(f"{name:16s} MISSING")
            missing.append(name)
    for name in ("data", "models", "cache", "runs"):
        print(f"{name:16s} {'present' if (root / name).exists() else 'not created yet'}")
    if not missing:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
    else:
        print("Install the versions in requirements.txt before running experiments.")
    return 1 if missing else 0


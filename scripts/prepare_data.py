"""Download supported image datasets, import manifests, or check local data."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
FOLDERS = {"mnistm": ("MNISTM", 10), "cinic10": ("cinic10-py", 10), "mmafedb": ("MMAFEDB", 7)}


def read_manifest(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="") as stream:
            records = list(csv.DictReader(stream))
    elif path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        raise ValueError("Use a CSV with a header or a JSONL manifest.")
    if not records:
        raise ValueError("The manifest is empty.")
    return records


def import_images(args):
    folder, n_classes = FOLDERS[args.dataset]
    target = args.root / folder / args.split
    records = read_manifest(args.manifest)
    image_root = Path(args.image_root or args.manifest.parent)
    for index, record in enumerate(records):
        if not {"image", "label"}.issubset(record):
            raise ValueError("Every row needs image and label fields.")
        label = int(record["label"])
        if not 0 <= label < n_classes:
            raise ValueError(f"Expected labels 0 through {n_classes - 1}; found {label}.")
        source = image_root / record["image"]
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = target / str(label) / f"{index:08d}{source.suffix.lower()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"Imported {len(records)} images into {target}")


def import_food101(args):
    records = read_manifest(args.manifest)
    image_root = Path(args.image_root or args.manifest.parent)
    folder = args.root / "food101"
    out = []
    for index, record in enumerate(records):
        if not {"image", "text", "label"}.issubset(record):
            raise ValueError("Food101 rows need image, text, and label fields.")
        label = int(record["label"])
        if not 0 <= label < 101:
            raise ValueError("Food101 labels must be integers from 0 through 100.")
        source = image_root / record["image"]
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = Path("images") / args.split / f"{index:08d}{source.suffix.lower()}"
        destination = folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        out.append(dict(image=relative.as_posix(), text=record["text"], label=label))
    manifest = folder / f"{args.split}.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "text", "label"])
        writer.writeheader(); writer.writerows(out)
    print(f"Imported {len(out)} image/text pairs into {manifest}")


def download(args):
    import torchvision.datasets as datasets
    args.root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        if name in ("cifar10", "cifar100", "fashionmnist"):
            cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100,
                   "fashionmnist": datasets.FashionMNIST}[name]
            folder = args.root / "FashionMNIST" if name == "fashionmnist" else args.root
            for train in (True, False):
                cls(str(folder), train=train, download=True)
        elif name == "tinyimagenet":
            archive = args.root / "tiny-imagenet-200.zip"
            if not archive.exists():
                temporary = archive.with_suffix(".zip.part")
                urllib.request.urlretrieve(TINY_URL, temporary)
                temporary.replace(archive)
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    destination = (args.root / member.filename).resolve()
                    if not destination.is_relative_to(args.root.resolve()):
                        raise ValueError("Archive contains a path outside the data directory.")
                bundle.extractall(args.root)
        print(f"Ready: {name}")


def validate(args):
    import numpy as np
    from promptmia.data import load_dataset
    for name in args.datasets:
        if name == 'fashion':
            from promptmia.clip import PackedCache
            train = PackedCache('train', 'cpu', args.root)
            test = PackedCache('test', 'cpu', args.root)
            if train.classes != test.classes:
                raise ValueError('Fashion-Product class orders must match between splits.')
            result = dict(train=len(train), test=len(test), classes=train.classes)
        elif name == "food101":
            from promptmia.vilt import Food101Dataset
            train = Food101Dataset(args.root, "train", args.num_classes)
            test = Food101Dataset(args.root, "test", args.num_classes)
            train.sample(0); test.sample(0)
            result = dict(train=len(train), test=len(test), classes=args.num_classes)
        else:
            data = load_dataset(name, root=args.root)
            data["train"][0]; data["test"][0]
            result = dict(train=len(data["train"]), test=len(data["test"]), classes=data["n_classes"],
                          train_class_counts=np.bincount(data["train_labels"], minlength=data["n_classes"]).tolist())
            if name == "fourdataset":
                result["train_per_source"] = np.bincount(data["train_source"]).tolist()
                result["test_per_source"] = np.bincount(data["test_source"]).tolist()
        print(name, json.dumps(result))


def build_fashion(args):
    model_root = args.model_root.expanduser().resolve()
    os.environ['HF_HOME'] = str(model_root / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(model_root / 'huggingface' / 'hub')
    from promptmia.clip import build_fashion_cache
    folder = build_fashion_cache(args.root, model_root / 'clip-vit-base-patch32',
                                 device=args.device, batch_size=args.batch_size)
    print(f'Prepared Fashion-Product in {folder}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data", help="Data directory inside this project by default.")
    commands = parser.add_subparsers(dest="command", required=True)
    downloader = commands.add_parser("download", help="Download standard public image datasets.")
    downloader.add_argument("--datasets", nargs="+", choices=["cifar10", "cifar100", "fashionmnist", "tinyimagenet"], required=True)
    downloader.set_defaults(action=download)
    image = commands.add_parser("import-images", help="Copy a labeled image manifest into the data directory.")
    image.add_argument("--dataset", choices=list(FOLDERS), required=True)
    food = commands.add_parser("import-food101", help="Copy Food101 images and write a portable paired-text manifest.")
    for command in (image, food):
        command.add_argument("--split", choices=["train", "test"], required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--image-root", type=Path, help="Base for image paths; defaults to the manifest directory.")
    image.set_defaults(action=import_images); food.set_defaults(action=import_food101)
    check = commands.add_parser("validate", help="Check file layout and print the exact selected sample counts.")
    check.add_argument("--datasets", nargs="+", choices=["cifar10", "cifar100", "tinyimagenet", "fourdataset", "food101", "fashion"], required=True)
    check.add_argument("--num-classes", type=int, default=8, help="Number of Food101 class IDs to include.")
    check.set_defaults(action=validate)
    fashion = commands.add_parser('build-fashion', help='Download Fashion-Product and cache the fixed CLIP inputs.')
    fashion.add_argument('--model-root', type=Path, default=ROOT / 'models')
    fashion.add_argument('--device', default='cuda')
    fashion.add_argument('--batch-size', type=int, default=256)
    fashion.set_defaults(action=build_fashion)
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.action(args)


if __name__ == "__main__":
    main()

"""Image datasets, deterministic client partitions, and compact image caches."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PROMPTMIA_DATA", PROJECT_ROOT / "data"))
NORM_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
PRE = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor()])
PRE_AUG = T.Compose([T.Resize(256), T.PILToTensor()])


class Normalizer:
    """Normalize uint8 batches with the ImageNet mean and standard deviation."""

    def __init__(self, device):
        levels = TF.to_tensor(np.arange(256, dtype=np.uint8).reshape(256, 1, 1).repeat(3, axis=2))
        self.lut = TF.normalize(levels, *NORM_STATS).reshape(-1).to(device)
        self.channel = (torch.arange(3, device=device) * 256).view(1, 3, 1, 1)

    def __call__(self, images):
        return self.lut[images.long() + self.channel]


class ImageCache:
    """Store resized uint8 images; transfer and normalize only the requested batch.

    CPU storage reduces GPU memory use. Set cache_device=device to keep all images
    beside the model when the full dataset fits in GPU memory.
    """

    def __init__(self, dataset, device, batch_size=500, cache_device=None, workers=0):
        if len(dataset) == 0:
            raise ValueError("Cannot cache an empty dataset.")
        self.device = torch.device(device)
        storage = torch.device(cache_device or "cpu")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        images, labels = [], []
        for x, y in loader:
            if x.dtype != torch.uint8:
                raise TypeError("ImageCache expects uint8 images from PRE or PRE_AUG.")
            images.append(x.to(storage))
            labels.append(torch.as_tensor(y, dtype=torch.long, device=storage))
        self.images, self.labels = torch.cat(images), torch.cat(labels)
        self.norm = Normalizer(self.device)

    def __len__(self):
        return len(self.images)

    def get(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.images.device)
        x = self.images[indices].to(self.device)
        return self.norm(x), self.labels[indices].to(self.device)


@torch.no_grad()
def compute_queries(model, cache, batch_size=500):
    """Evaluate the frozen query encoder once for each cached image."""
    return torch.cat([model.query(cache.get(ids)[0])
                      for ids in torch.arange(len(cache)).split(batch_size)])


@torch.no_grad()
def dataset_queries(query_fn, dataset, device, batch_size=500):
    normalizer = Normalizer(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return torch.cat([query_fn(normalizer(x.to(device))) for x, _ in loader])


class Images(Dataset):
    """A PIL image accessor with stable integer labels and source indices."""

    def __init__(self, get_image, labels, transform=PRE, source_indices=None):
        self.get_image = get_image
        self.labels = np.asarray(labels, dtype=np.int64)
        self.targets = self.labels
        self.transform = transform
        self.source_indices = np.arange(len(self.labels)) if source_indices is None else np.asarray(source_indices)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.get_image(int(index)).convert("RGB")
        return self.transform(image), int(self.labels[index])


def _torchvision_split(dataset_class, root, train, transform):
    dataset = dataset_class(str(root), train=train, download=False)
    return Images(lambda i: dataset[i][0], dataset.targets, transform)


def load_cifar10(root=None):
    data = load_dataset("cifar10", root=root)
    return data["train"], data["test"], data["n_classes"]


def _tinyimagenet(root, train, transform):
    base = root / "tiny-imagenet-200"
    classes = sorted(p.name for p in (base / "train").iterdir() if p.is_dir())
    class_to_id = {name: i for i, name in enumerate(classes)}
    if train:
        records = [(p, class_to_id[name]) for name in classes
                   for p in sorted((base / "train" / name / "images").glob("*")) if p.is_file()]
    else:
        records = []
        for line in (base / "val" / "val_annotations.txt").read_text().splitlines():
            filename, label = line.split("\t")[:2]
            records.append((base / "val" / "images" / filename, class_to_id[label]))
    return Images(lambda i: Image.open(records[i][0]), [r[1] for r in records], transform)


def balanced_indices(labels, n_classes, size, seed=0):
    """Take the first size//classes rows per class, then shuffle their order."""
    labels = np.asarray(labels, dtype=np.int64)
    if size <= 0 or n_classes <= 0 or len(labels) == 0:
        raise ValueError("Class count, subset size, and input dataset must be positive.")
    if np.any((labels < 0) | (labels >= n_classes)):
        raise ValueError("Labels must be integers from zero through n_classes - 1.")
    result = np.concatenate([np.flatnonzero(labels == label)[:size // n_classes]
                             for label in range(n_classes)]).astype(np.int64)
    np.random.RandomState(seed).shuffle(result)
    return result


def _mnistm(root, split):
    tensor_path = root / "MNISTM" / "processed" / f"mnist_m_{split}.pt"
    if tensor_path.exists():
        images, labels = torch.load(tensor_path, map_location="cpu", weights_only=False)
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError(f"Expected four-dimensional image tensor in {tensor_path}.")
        if images.shape[1] == 3 and images.shape[-1] != 3:
            images = images.transpose(0, 2, 3, 1)
        if images.dtype != np.uint8:
            raise ValueError("MNIST-M image tensors must contain uint8 RGB values.")
        return lambda i: Image.fromarray(images[i]), np.asarray(labels, dtype=np.int64), None
    folder = root / "MNISTM" / split
    dataset = torchvision.datasets.ImageFolder(str(folder))
    return lambda i: Image.open(dataset.samples[i][0]), np.asarray(dataset.targets), dataset.classes


def _fourdataset(root, train, transform=PRE):
    split, size = ("train", 30000) if train else ("test", 2500)
    get, labels, classes = _mnistm(root, split)
    sources = [("MNISTM", get, labels, 10, classes)]
    fashion = torchvision.datasets.FashionMNIST(str(root / "FashionMNIST"), train=train, download=False)
    sources.append(("FashionMNIST", lambda i: fashion[i][0], np.asarray(fashion.targets), 10, fashion.classes))
    for name, count in (("cinic10-py", 10), ("MMAFEDB", 7)):
        dataset = torchvision.datasets.ImageFolder(str(root / name / split))
        if len(dataset.classes) != count:
            raise ValueError(f"{name}/{split} must contain {count} classes.")
        sources.append((name, lambda i, d=dataset: Image.open(d.samples[i][0]),
                        np.asarray(dataset.targets), count, dataset.classes))
    getters, labels, source_ids, original_ids, manifest = [], [], [], [], []
    offset = 0
    for source_id, (name, get, lab, count, classes) in enumerate(sources):
        quota_size = int(np.ceil(size / count)) * count
        selected = balanced_indices(lab, count, quota_size, seed=0)
        getters.extend((get, int(i)) for i in selected)
        labels.extend((lab[selected] + offset).tolist())
        source_ids.extend([source_id] * len(selected))
        original_ids.extend(selected.tolist())
        manifest.append(dict(source=name, classes=classes, label_offset=offset,
                             selected_indices=selected.tolist(), available_counts=np.bincount(lab, minlength=count).tolist(),
                             selected_counts=np.bincount(lab[selected], minlength=count).tolist()))
        offset += count
    manifest_path = root / "splits" / f"fourdataset_{split}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(split=split, seed=0, size=len(labels), sources=manifest), indent=2)
    if not manifest_path.exists() or manifest_path.read_text() != payload:
        manifest_path.write_text(payload)
    dataset = Images(lambda i: getters[i][0](getters[i][1]), labels, transform, original_ids)
    return dataset, np.asarray(source_ids, dtype=np.int64)


def load_dataset(name, root=None, transform=PRE):
    """Load a benchmark from an explicit data directory without downloading it."""
    root = Path(root or DATA_ROOT).expanduser().resolve()
    result = dict(train_source=None, test_source=None)
    try:
        if name in ("cifar10", "cifar100"):
            cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
            result.update(train=_torchvision_split(cls, root, True, transform),
                          test=_torchvision_split(cls, root, False, transform),
                          n_classes=10 if name == "cifar10" else 100)
        elif name == "tinyimagenet":
            result.update(train=_tinyimagenet(root, True, transform), test=_tinyimagenet(root, False, transform), n_classes=200)
        elif name == "fourdataset":
            (train, tr_source), (test, te_source) = _fourdataset(root, True, transform), _fourdataset(root, False, transform)
            result.update(train=train, test=test, n_classes=37, train_source=tr_source, test_source=te_source)
        else:
            raise ValueError(f"Unknown dataset: {name}")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Dataset {name!r} is incomplete in {root}. See docs/data.md. {error}") from error
    result["train_labels"] = result["train"].labels
    result["fingerprint"] = dataset_fingerprint(name, root, result)
    return result


def dataset_fingerprint(name, root, data):
    """Identify the selected rows and source file metadata for query-cache reuse."""
    digest = hashlib.sha256()
    digest.update(name.encode())
    for split in ("train", "test"):
        digest.update(np.asarray(data[split].labels, dtype=np.int64).tobytes())
        digest.update(np.asarray(data[split].source_indices, dtype=np.int64).tobytes())
        if data.get(f"{split}_source") is not None:
            digest.update(np.asarray(data[f"{split}_source"], dtype=np.int64).tobytes())
    folders = {"cifar10": ["cifar-10-batches-py"], "cifar100": ["cifar-100-python"],
               "tinyimagenet": ["tiny-imagenet-200"],
               "fourdataset": ["MNISTM", "FashionMNIST", "cinic10-py", "MMAFEDB"]}[name]
    for folder in folders:
        for file in sorted((root / folder).rglob("*")):
            if file.is_file():
                info = file.stat()
                digest.update(f"{file.relative_to(root)}:{info.st_size}:{info.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def dirichlet_split(labels, n_clients, alpha, seed, min_size=32, max_attempts=10000):
    """Partition every row once using class-wise Dirichlet proportions."""
    labels = np.asarray(labels)
    if n_clients <= 0 or not np.isfinite(alpha) or alpha <= 0 or min_size < 1:
        raise ValueError("n_clients, alpha, and min_size must be positive.")
    if len(labels) < n_clients * min_size:
        raise ValueError(f"Need at least {n_clients * min_size} rows for {n_clients} clients with min_size={min_size}.")
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        parts = [[] for _ in range(n_clients)]
        for label in np.unique(labels):
            rows = rng.permutation(np.flatnonzero(labels == label))
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(proportions) * len(rows)).astype(int)[:-1]
            for client, chunk in zip(parts, np.split(rows, cuts)):
                client.extend(chunk.tolist())
        if min(map(len, parts)) >= min_size:
            return [np.sort(np.asarray(part, dtype=np.int64)) for part in parts]
    raise RuntimeError(f"Could not construct the Dirichlet partition in {max_attempts} attempts; increase alpha or reduce min_size.")


def extreme_split(labels, n_clients, n_dominated, seed, dominated_ratio=0.99, alpha=1.0):
    """Split 99% of each class across its dominant clients.

    The remaining rows go to the other clients in Dirichlet(alpha) proportions.
    Class rows retain their dataset order, and every row is assigned once.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n_classes = len(classes)
    if n_clients < 2 or not 0 < n_dominated < n_classes or not 0 < dominated_ratio < 1 or alpha <= 0:
        raise ValueError("Invalid client count, dominant-class count, or dominant ratio.")
    if not np.array_equal(classes, np.arange(n_classes)):
        raise ValueError('Extreme partition requires contiguous class labels starting at zero.')
    if n_clients * n_dominated < n_classes:
        raise ValueError("There must be enough dominant-class slots to cover every class.")
    rng = np.random.RandomState(seed)
    order = np.tile(np.arange(n_clients), n_dominated).reshape(n_dominated, n_clients)
    for row in order:
        rng.shuffle(row)
    step = n_dominated * n_clients / n_classes
    cuts = np.round(np.arange(step, n_dominated * n_clients, step)).astype(int)
    owners = np.split(order.flatten(), cuts[:n_classes - 1])
    parts = [[] for _ in range(n_clients)]
    for label in classes:
        rows = np.flatnonzero(labels == label)
        cutoff = int(len(rows) * dominated_ratio)
        selected = owners[label]
        for client, chunk in zip(selected, np.array_split(rows[:cutoff], len(selected))):
            parts[client].append(chunk)
        residual = rows[cutoff:]
        proportions = rng.dirichlet([alpha] * (n_clients - len(selected)), 1).flatten()
        other = np.setdiff1d(np.arange(n_clients), selected, assume_unique=True)
        rng.shuffle(other)
        chunks = np.split(residual, (np.cumsum(proportions)[:-1] * len(residual)).astype(int))
        for client, chunk in zip(other, chunks):
            parts[client].append(chunk)
    if any(len(part) == 0 for part in parts):
        raise ValueError("Extreme split produced an empty client; use more data or fewer clients.")
    return [np.concatenate(part).astype(np.int64) for part in parts]


def client_split(data, n_clients, alpha, seed):
    """Split FourDataset within each source; split other datasets as one population."""
    if data.get("train_source") is None:
        return dirichlet_split(data["train_labels"], n_clients, alpha, seed)
    sources = np.unique(data["train_source"])
    if n_clients % len(sources):
        raise ValueError("FourDataset requires a client count divisible by four.")
    parts = []
    for source in sources:
        rows = np.flatnonzero(data["train_source"] == source)
        local = dirichlet_split(np.asarray(data["train_labels"])[rows], n_clients // len(sources), alpha, seed + int(source))
        parts.extend(rows[part] for part in local)
    return parts

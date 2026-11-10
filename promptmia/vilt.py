"""Frozen ViLT with a shared prompt pool for the image/text experiments."""

import csv
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

from promptmia.data import DATA_ROOT, dirichlet_split
from promptmia.model import L2PViT, PromptPool
from promptmia.io import relative_name, write_json


IMAGE_SIZE = 384
MAX_TEXT_LEN = 40
PRE = T.Compose([T.Resize(IMAGE_SIZE), T.CenterCrop(IMAGE_SIZE), T.ToTensor(),
                 T.Normalize([0.5] * 3, [0.5] * 3)])


class Food101Dataset:
    """Read image/text pairs from a portable manifest or an Arrow table.

    A CSV/JSONL row contains image, text, and an integer label. Image paths are
    relative to the manifest directory. Arrow tables contain image bytes, text,
    and label columns.
    """

    def __init__(self, root=None, split="train", num_classes=8):
        root = Path(root or DATA_ROOT) / "food101"
        self.root, self.table, self.records = root, None, None
        manifest = next((root / f"{split}.{suffix}" for suffix in ("csv", "jsonl")
                         if (root / f"{split}.{suffix}").exists()), None)
        if manifest is not None:
            if manifest.suffix == ".csv":
                with manifest.open(newline="") as stream:
                    self.records = list(csv.DictReader(stream))
            else:
                self.records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
            for row in self.records:
                if not {"image", "text", "label"}.issubset(row):
                    raise ValueError(f"Every row of {manifest} needs image, text, and label fields.")
                row["label"] = int(row["label"])
            labels = np.asarray([row["label"] for row in self.records], dtype=np.int64)
        else:
            import pyarrow as pa
            arrow = root / f"food101_{split}.arrow"
            if not arrow.exists():
                raise FileNotFoundError(f"Missing {root / (split + '.csv')} or {arrow}; see docs/data.md.")
            self.table = pa.ipc.RecordBatchFileReader(pa.memory_map(str(arrow), "r")).read_all()
            if not {"image", "text", "label"}.issubset(self.table.column_names):
                raise ValueError(f"Arrow table {arrow} needs image, text, and label columns.")
            labels = np.asarray(self.table["label"].to_pylist(), dtype=np.int64)
        if np.any(labels < 0):
            raise ValueError("Food101 labels must be nonnegative integers.")
        self.rows = np.flatnonzero(labels < num_classes)
        self.labels = labels[self.rows]
        if not len(self.rows) or set(self.labels.tolist()) != set(range(num_classes)):
            raise ValueError(f"Food101 {split} must contain every label from 0 through {num_classes - 1}.")

    def __len__(self):
        return len(self.rows)

    def sample(self, index, modality="multimodal"):
        row = int(self.rows[index])
        if self.records is None:
            text = self.table["text"][row].as_py()
            image = self.table["image"][row].as_py() if modality == "multimodal" else None
        else:
            record = self.records[row]
            text = record["text"]
            image = self.root / record["image"] if modality == "multimodal" else None
        if isinstance(text, list):
            text = text[0] if text else ""
        if not isinstance(text, str):
            raise ValueError("Food101 text fields must contain strings.")
        if image is None:
            # A fixed image keeps the sequence shape unchanged in text-only runs.
            pixels = torch.ones(3, IMAGE_SIZE, IMAGE_SIZE)
        else:
            stream = io.BytesIO(image) if isinstance(image, bytes) else image
            with Image.open(stream) as picture:
                pixels = PRE(picture.convert("RGB"))
        return pixels, text, int(self.labels[index])


class ViLTBackbone(nn.Module):
    """Pretrained text/image embeddings and transformer blocks, held fixed."""

    def __init__(self, checkpoint, max_text_len=MAX_TEXT_LEN):
        super().__init__()
        import timm
        from transformers import BertConfig
        from transformers.models.bert.modeling_bert import BertEmbeddings
        self.vit = timm.create_model("vit_base_patch32_384", pretrained=False, num_classes=0)
        self.text_embeddings = BertEmbeddings(BertConfig(vocab_size=30522, hidden_size=768,
                                                        max_position_embeddings=max_text_len))
        self.token_type_embeddings = nn.Embedding(2, 768)
        self.pooler = nn.Linear(768, 768)
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing ViLT weights: {checkpoint}. Run scripts/download_models.py --models vilt.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        part = lambda prefix: {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        self.vit.load_state_dict(part("transformer."), strict=True)
        missing, unexpected = self.text_embeddings.load_state_dict(part("text_embeddings."), strict=False)
        if missing or not set(unexpected).issubset({"position_ids", "token_type_ids"}):
            raise ValueError(f"Unexpected text embedding state: missing={missing}, unexpected={unexpected}")
        self.token_type_embeddings.load_state_dict(part("token_type_embeddings."), strict=True)
        self.pooler.load_state_dict(part("pooler.dense."), strict=True)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def embed(self, images, input_ids, text_mask):
        text = self.text_embeddings(input_ids=input_ids)
        text = text + self.token_type_embeddings(torch.zeros_like(input_ids))
        vision = self.vit
        image = vision.patch_embed(images) + vision.pos_embed[:, 1:]
        cls = (vision.cls_token + vision.pos_embed[:, :1]).expand(len(image), -1, -1)
        image = torch.cat([cls, image], dim=1)
        image = image + self.token_type_embeddings(torch.ones(image.shape[:2], dtype=torch.long, device=image.device))
        mask = torch.cat([text_mask, torch.ones(image.shape[:2], dtype=text_mask.dtype, device=image.device)], dim=1)
        return torch.cat([text, image], dim=1), mask

    def blocks(self, sequence, mask):
        attention_mask = mask.bool()[:, None, None, :]
        for block in self.vit.blocks:
            sequence = block(sequence, attn_mask=attention_mask)
        return self.vit.norm(sequence)


class EmbeddingCache:
    """Keep frozen input embeddings in CPU memory and move requested batches."""

    def __init__(self, sequence, mask, labels, device=None):
        self.images, self.mask, self.labels = sequence, mask, labels
        self.device = torch.device(device or sequence.device)

    def __len__(self):
        return len(self.images)

    def get(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.images.device)
        return (self.images[indices].to(self.device), self.mask[indices].to(self.device)), self.labels[indices].to(self.device)


@torch.no_grad()
def load_food101(backbone, modality, device, root=None, tokenizer_path=None,
                 num_classes=8, batch_size=128, cache_device="cpu"):
    """Embed both splits once, with deterministic image resizing and tokenization."""
    from transformers import BertTokenizer
    if modality not in ("multimodal", "text"):
        raise ValueError("modality must be multimodal or text.")
    if tokenizer_path is None:
        tokenizer_path = Path(__file__).resolve().parents[1] / "models" / "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    storage = torch.device(cache_device)
    caches = []
    for split in ("train", "test"):
        dataset = Food101Dataset(root, split, num_classes)
        sequences, masks = [], []
        for start in range(0, len(dataset), batch_size):
            rows = [dataset.sample(i, modality) for i in range(start, min(start + batch_size, len(dataset)))]
            encoded = tokenizer([row[1] for row in rows], padding="max_length", truncation=True,
                                max_length=MAX_TEXT_LEN, return_tensors="pt")
            sequence, mask = backbone.embed(torch.stack([row[0] for row in rows]).to(device),
                                            encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            sequences.append(sequence.to(storage)); masks.append(mask.to(storage))
        caches.append(EmbeddingCache(torch.cat(sequences), torch.cat(masks),
                                     torch.as_tensor(dataset.labels, dtype=torch.long, device=storage), device))
    return tuple(caches)


class ViLTL2P(nn.Module):
    """Insert selected prompts after the text CLS token and train a classifier."""

    def __init__(self, backbone, M=20, N=4, query="max", num_classes=8):
        super().__init__()
        if query not in ("max", "cls"):
            raise ValueError("query must be max or cls.")
        self.backbone, self.query_mode = backbone, query
        self.pool = PromptPool(M, N, 768)
        self.pooler = nn.Linear(768, 768)
        self.pooler.load_state_dict(backbone.pooler.state_dict())
        self.head = nn.Sequential(nn.Linear(768, 1536), nn.LayerNorm(1536), nn.GELU(), nn.Linear(1536, num_classes))
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    trainable_state = L2PViT.trainable_state
    load_trainable_state = L2PViT.load_trainable_state

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def query(self, x):
        sequence, mask = x
        return sequence.max(dim=1).values if self.query_mode == "max" else self.backbone.blocks(sequence, mask)[:, 0]

    def forward(self, x, q=None):
        sequence, mask = x
        indices, key_loss = self.pool.select(self.query(x) if q is None else q)
        prompts = self.pool.prompt[indices]
        sequence = torch.cat([sequence[:, :1], prompts, sequence[:, 1:]], dim=1)
        mask = torch.cat([mask[:, :1], torch.ones(prompts.shape[:2], dtype=mask.dtype, device=mask.device), mask[:, 1:]], dim=1)
        hidden = self.backbone.blocks(sequence, mask)
        logits = self.head(torch.tanh(self.pooler(hidden[:, 0])))
        return logits, key_loss, indices


def train_vilt(config, root, device):
    """Train one configured Food101 run and save the inputs needed for evaluation."""
    from promptmia.fl import run_fedavg
    from promptmia.training import _save
    root = Path(root).resolve()
    cfg = dict(rounds=60, clients=20, clients_per_round=10, local_epochs=5, batch_size=64,
               lr=1e-4, key_lr=1e-3, lam=0.5, alpha=0.5, M=20, N=4, seed=0,
               modality="multimodal", query="max", diverse_select=True, num_classes=8, cache_device="cpu")
    cfg.update(config)
    cfg["dataset"], cfg["backbone"] = "food101", "vilt"
    cfg.setdefault("name", f"food101_{cfg['modality']}_vilt")
    if cfg.get("defense", "none") != "none":
        raise ValueError("The Food101 experiment uses the undefended prompt pool.")
    if cfg["rounds"] < 1 or cfg["local_epochs"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("rounds, local_epochs, and batch_size must be positive.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Choose --device cpu or run on a CUDA host.")
    cfg["device"] = str(device)
    run = root / "runs" / relative_name(cfg["name"])
    run.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metrics = dict(test_acc={}, elapsed_s=0.0, completed=False)
    write_json(run / "config.json", cfg)
    write_json(run / "metrics.json", metrics)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else root / path

    models = resolve(cfg.get("model_root", "models"))
    os.environ['HF_HOME'] = str(models / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(models / 'huggingface' / 'hub')
    checkpoint = resolve(cfg.get("vilt_checkpoint", models / "vilt_200k_mlm_itm.ckpt"))
    tokenizer = resolve(cfg.get("tokenizer_path", models / "bert-base-uncased"))
    data_root = resolve(cfg.get("data_root", "data"))
    backbone = ViLTBackbone(checkpoint).to(device)
    train_cache, test_cache = load_food101(backbone, cfg["modality"], device,
        root=data_root, tokenizer_path=tokenizer, num_classes=cfg["num_classes"],
        batch_size=cfg.get("embedding_batch_size", 128), cache_device=cfg["cache_device"])
    model = ViLTL2P(backbone, cfg["M"], cfg["N"], cfg["query"], cfg["num_classes"]).to(device)
    with torch.no_grad():
        encode = lambda cache: torch.cat([model.query(cache.get(ids)[0])
                                         for ids in torch.arange(len(cache)).split(cfg.get("query_batch_size", 256))])
        train_q, test_q = encode(train_cache), encode(test_cache)
    cfg["query_cache"] = f"cache/queries/food101_{cfg['modality']}_vilt.pt"
    query_file = root / cfg["query_cache"]
    query_file.parent.mkdir(parents=True, exist_ok=True)
    _save(query_file, dict(train=train_q.cpu(), test=test_q.cpu(), train_labels=train_cache.labels.cpu(),
                          test_labels=test_cache.labels.cpu(), train_source=None, test_source=None))
    clients = dirichlet_split(train_cache.labels.cpu().numpy(), cfg["clients"], cfg["alpha"], cfg["seed"])
    write_json(run / "clients.json", [p.tolist() for p in clients])
    _save(run / 'clients.pt', [torch.as_tensor(client) for client in clients])
    cfg.update(n_classes=cfg["num_classes"], embedding_dim=768,
               train_samples=len(train_cache), test_samples=len(test_cache))
    write_json(run / "config.json", cfg)
    selected_rounds = cfg.get("save_rounds", [0, 1, 20, cfg["rounds"]])
    if isinstance(selected_rounds, str):
        selected_rounds = [int(r) for r in selected_rounds.split(",")]
    saved_rounds, keys, accuracies = set(selected_rounds) | {cfg['rounds']}, {}, {}

    def on_round(round_index, state, accuracy):
        keys[round_index] = state["pool.key"].detach().cpu().clone()
        accuracies[round_index] = accuracy
        _save(run / "keys.pt", keys)
        metrics.update(test_acc=accuracies, elapsed_s=time.time() - started)
        write_json(run / "metrics.json", metrics)
        if round_index in saved_rounds:
            _save(run / f"ckpt_round{round_index}.pt", {name: value.detach().cpu().clone() for name, value in state.items()})

    run_fedavg(model, train_cache, train_q, test_cache, test_q, clients, dict(cfg, defense=None), on_round)
    _save(run / "state.pt", {name: value.cpu() for name, value in model.trainable_state().items()})
    metrics.update(completed=True, elapsed_s=time.time() - started)
    write_json(run / "metrics.json", metrics)
    return run

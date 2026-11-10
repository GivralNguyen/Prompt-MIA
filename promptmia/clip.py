"""Frozen CLIP prompt tuning on Fashion-Product images and product titles."""

from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn

from promptmia.data import DATA_ROOT, dirichlet_split
from promptmia.model import L2PViT, PromptPool
from promptmia.io import relative_name, write_json


CLIP_NAME = 'openai/clip-vit-base-patch32'
FASHION_DATASET = 'ashraq/fashion-product-images-small'
MAX_TEXT_LEN = 24


class PackedCache:
    """Token IDs and frozen image/text embeddings, stored on CPU or GPU."""

    def __init__(self, split, device, root=None, cache_device='cpu'):
        path = Path(root or DATA_ROOT) / 'fashionmm_clip' / f'{split}.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}; run scripts/prepare_data.py build-fashion.')
        self.device = torch.device(device)
        storage = torch.device(cache_device)
        with np.load(path, allow_pickle=False) as data:
            required = {'ids', 'img', 'txt', 'y', 'classes'}
            if not required.issubset(data.files):
                raise ValueError(f'{path} needs arrays {sorted(required)}.')
            self.ids = torch.as_tensor(data['ids'].astype(np.int64), device=storage)
            self.img = torch.as_tensor(data['img'].copy(), dtype=torch.float32, device=storage)
            self.txt = torch.as_tensor(data['txt'].copy(), dtype=torch.float32, device=storage)
            self.labels = torch.as_tensor(data['y'].astype(np.int64), device=storage)
            self.classes = data['classes'].astype(str).tolist()
        self.images = self.ids
        if self.ids.ndim != 2 or self.img.shape != (len(self), 512) or self.txt.shape != self.img.shape:
            raise ValueError('Fashion-Product needs token rows and two 512-dimensional embeddings per sample.')
        if len(self.ids) != len(self) or not len(self) or not torch.isfinite(self.img).all() or not torch.isfinite(self.txt).all():
            raise ValueError('Fashion-Product arrays must be nonempty, aligned, and finite.')
        if (self.labels < 0).any() or (self.labels >= len(self.classes)).any():
            raise ValueError('Fashion-Product labels must index its ordered class names.')

    def __len__(self):
        return len(self.labels)

    def get(self, indices):
        if not isinstance(indices, slice):
            indices = torch.as_tensor(indices, dtype=torch.long, device=self.ids.device)
        return ((self.ids[indices].to(self.device), self.img[indices].to(self.device),
                 self.txt[indices].to(self.device)), self.labels[indices].to(self.device))


class CLIPL2P(nn.Module):
    """Insert selected prompts and an image token into CLIP's frozen text tower."""

    def __init__(self, n_classes, M=20, N=4, model_path=None, text_model=None):
        super().__init__()
        if text_model is None:
            from transformers import CLIPTextModelWithProjection
            model_path = Path(model_path or Path(__file__).resolve().parents[1] / 'models' / 'clip-vit-base-patch32')
            text_model = CLIPTextModelWithProjection.from_pretrained(str(model_path), local_files_only=True)
        self.clip_text = text_model
        self.clip_text.requires_grad_(False).eval()
        self.pool = PromptPool(M, N, self.clip_text.config.hidden_size)
        self.head = nn.Sequential(nn.Linear(self.clip_text.config.projection_dim, 512), nn.Dropout(.5),
                                  nn.Linear(512, n_classes))

    trainable_state = L2PViT.trainable_state
    load_trainable_state = L2PViT.load_trainable_state

    def train(self, mode=True):
        super().train(mode)
        self.clip_text.eval()
        return self

    @torch.no_grad()
    def query(self, x):
        _, image, text = x
        return image / (image.norm(dim=-1, keepdim=True) + 1e-6) + text / (text.norm(dim=-1, keepdim=True) + 1e-6)

    def encode(self, ids, image, prompts):
        from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
        tower = self.clip_text.text_model
        tokens = tower.embeddings.token_embedding(ids)
        # Scale the image token to the average text-token norm in this batch.
        image_token = (image / (image.norm(dim=-1, keepdim=True) + 1e-6)
                       * tokens.norm(dim=-1).mean().detach()).unsqueeze(1)
        sequence = torch.cat([tokens[:, :1], prompts, image_token, tokens[:, 1:]], dim=1)
        sequence = tower.embeddings(inputs_embeds=sequence)
        mask = _create_4d_causal_attention_mask(sequence.shape[:2], sequence.dtype, sequence.device)
        hidden = tower.final_layer_norm(tower.encoder(inputs_embeds=sequence, causal_attention_mask=mask)[0])
        end = ids.argmax(dim=-1) + prompts.size(1) + 1
        return self.clip_text.text_projection(hidden[torch.arange(len(hidden), device=hidden.device), end])

    def forward(self, x, q=None):
        ids, image, _ = x
        selected, key_loss = self.pool.select(self.query(x) if q is None else q)
        return self.head(self.encode(ids, image, self.pool.prompt[selected])), key_loss, selected


def train_clip(config, root, device):
    """Train one CLIP experiment and save its queries, global keys, and metrics."""
    from promptmia.fl import run_fedavg
    from promptmia.training import _save
    root = Path(root).resolve()
    cfg = dict(rounds=60, clients=80, clients_per_round=10, local_epochs=5, batch_size=64,
               lr=1e-4, key_lr=1e-3, lam=.5, alpha=.5, M=20, N=4, seed=0, cache_device='cpu')
    cfg.update(config)
    cfg.update(dataset='fashion', backbone='clip')
    cfg.setdefault('name', 'clip/fashion')
    if cfg.get('defense', 'none') != 'none':
        raise ValueError('The Fashion-Product experiment uses the undefended prompt pool.')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Choose --device cpu or run on a CUDA host.')
    if min(cfg['rounds'], cfg['local_epochs'], cfg['batch_size']) < 1:
        raise ValueError('rounds, local_epochs, and batch_size must be positive.')
    cfg['device'] = str(device)
    run = root / 'runs' / relative_name(cfg['name'])
    run.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metrics = dict(test_acc={}, elapsed_s=0., completed=False)
    write_json(run / 'config.json', cfg)
    write_json(run / 'metrics.json', metrics)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    models = resolve(cfg.get('model_root', 'models'))
    os.environ['HF_HOME'] = str(models / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(models / 'huggingface' / 'hub')
    data_root = resolve(cfg.get('data_root', 'data'))
    model_path = resolve(cfg.get('clip_model', models / 'clip-vit-base-patch32'))
    train_cache = PackedCache('train', device, data_root, cfg['cache_device'])
    test_cache = PackedCache('test', device, data_root, cfg['cache_device'])
    if train_cache.classes != test_cache.classes:
        raise ValueError('Fashion-Product train/test class orders must match.')
    if cfg.get('num_classes') is not None and len(train_cache.classes) != int(cfg['num_classes']):
        raise ValueError('The prepared Fashion-Product class count differs from num_classes.')
    model = CLIPL2P(len(train_cache.classes), cfg['M'], cfg['N'], model_path=model_path).to(device)
    train_q = model.query(train_cache.get(slice(None))[0])
    test_q = model.query(test_cache.get(slice(None))[0])
    cfg['query_cache'] = 'cache/queries/fashion_clip.pt'
    _save(root / cfg['query_cache'], dict(train=train_q.cpu(), test=test_q.cpu(),
          train_labels=train_cache.labels.cpu(), test_labels=test_cache.labels.cpu(),
          train_source=None, test_source=None))
    clients = dirichlet_split(train_cache.labels.cpu().numpy(), cfg['clients'], cfg['alpha'], cfg['seed'])
    _save(run / 'clients.pt', [torch.as_tensor(client) for client in clients])
    cfg.update(n_classes=len(train_cache.classes), embedding_dim=512,
               train_samples=len(train_cache), test_samples=len(test_cache))
    write_json(run / 'config.json', cfg)
    requested_rounds = cfg.get('save_rounds', [0, 1, 20, cfg['rounds']])
    if isinstance(requested_rounds, str):
        requested_rounds = [int(r) for r in requested_rounds.split(',')]
    saved_rounds, keys = set(requested_rounds) | {cfg['rounds']}, {}

    def on_round(round_index, state, accuracy):
        keys[round_index] = state['pool.key'].detach().cpu().clone()
        metrics['test_acc'][round_index] = accuracy
        metrics['elapsed_s'] = time.time() - started
        _save(run / 'keys.pt', keys)
        write_json(run / 'metrics.json', metrics)
        if round_index in saved_rounds:
            _save(run / f'ckpt_round{round_index}.pt', {name: value.detach().cpu().clone() for name, value in state.items()})

    run_fedavg(model, train_cache, train_q, test_cache, test_q, clients, dict(cfg, defense=None), on_round)
    _save(run / 'state.pt', {name: value.cpu() for name, value in model.trainable_state().items()})
    metrics.update(completed=True, elapsed_s=time.time() - started)
    write_json(run / 'metrics.json', metrics)
    return run


@torch.no_grad()
def build_fashion_cache(root, model_path, device='cuda', batch_size=256, max_per_class=3000,
                        n_classes=10, seed=0, test_fraction=.2):
    """Prepare the fixed Fashion-Product split and both frozen CLIP embeddings."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
    root, model_path = Path(root), Path(model_path)
    output = root / 'fashionmm_clip'
    output.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(FASHION_DATASET, split='train', cache_dir=str(root / 'downloads' / 'fashion-product'))
    label_column = 'subCategory' if 'subCategory' in raw.column_names else 'masterCategory'
    text_column = 'productDisplayName' if 'productDisplayName' in raw.column_names else 'articleType'
    labels = raw[label_column]
    classes = [label for label, _ in Counter(labels).most_common(n_classes)]
    class_ids = {label: index for index, label in enumerate(classes)}
    rng = np.random.RandomState(seed)
    chosen = []
    for label in classes:
        available = np.flatnonzero(np.asarray(labels) == label)
        chosen.append(rng.choice(available, max_per_class, replace=False) if len(available) > max_per_class else available)
    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    targets = np.asarray([class_ids[labels[int(i)]] for i in chosen], dtype=np.int64)
    text = [str(raw[int(i)][text_column] or classes[targets[j]]) for j, i in enumerate(chosen)]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    ids = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_TEXT_LEN,
                    return_tensors='np')['input_ids'].astype(np.int32)
    vision = CLIPVisionModelWithProjection.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    image_embeddings = np.empty((len(chosen), 512), dtype=np.float32)
    for start in range(0, len(chosen), batch_size):
        images = [raw[int(i)]['image'].convert('RGB') for i in chosen[start:start + batch_size]]
        pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
        image_embeddings[start:start + len(images)] = vision(pixel_values=pixels).image_embeds.cpu().numpy()
        print(f'Fashion-Product image embeddings: {start + len(images)}/{len(chosen)}', flush=True)
    del vision
    text_model = CLIPTextModelWithProjection.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    text_embeddings = np.empty_like(image_embeddings)
    for start in range(0, len(ids), batch_size):
        tokens = torch.as_tensor(ids[start:start + batch_size].astype(np.int64), device=device)
        text_embeddings[start:start + len(tokens)] = text_model(input_ids=tokens).text_embeds.cpu().numpy()
    n_test = int(len(chosen) * test_fraction)
    for split, selected in [('test', slice(0, n_test)), ('train', slice(n_test, None))]:
        np.savez_compressed(output / f'{split}.npz', ids=ids[selected], img=image_embeddings[selected],
                            txt=text_embeddings[selected], y=targets[selected], classes=np.asarray(classes, dtype=str),
                            source_indices=chosen[selected], pad_id=np.asarray(tokenizer.pad_token_id),
                            max_len=np.asarray(MAX_TEXT_LEN))
    write_json(output / 'preparation.json', dict(dataset=FASHION_DATASET, label_column=label_column,
               text_column=text_column, classes=classes, seed=seed, max_per_class=max_per_class,
               max_text_len=MAX_TEXT_LEN, test_fraction=test_fraction, train_samples=len(chosen)-n_test,
               test_samples=n_test, source_indices=chosen.tolist()))
    return output

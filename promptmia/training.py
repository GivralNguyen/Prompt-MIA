"""Train one configured federated experiment and save its measurements."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from promptmia.attack import promptmia_inject
from promptmia.defenses import Defense, input_noise_sigma
from promptmia.fl import run_fedavg, per_sample_losses, evaluate_noisy
from promptmia.model import L2PViT


DEFAULTS = dict(dataset='cifar10', backbone='vit_b32', seed=0, rounds=60,
                local_epochs=5, batch_size=64, clients=80, clients_per_round=10,
                lr=1e-4, key_lr=1e-3, M=20, N=4, lam=0.5, alpha=0.5,
                split='dirichlet', aggregator='fedavg', defense='none',
                dp_eps=None, prompt_dropout=0.0, inject_round=0, inject_target=0,
                mia_target=None, head='cls', cache_device='cpu', eval_batch_size=500,
                sparse_ratio=0.8, mixup_alpha=0.2, save_rounds=[0, 1, 20, 60])


def _json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def _save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def train(config, root, device='cuda'):
    """Run one experiment; all relative paths are resolved inside root.

    Checkpoints contain trainable parameters. The frozen pretrained backbone is
    reconstructed by its configured model name when an evaluation loads a run.
    """
    from promptmia.data import load_dataset, ImageCache, compute_queries, client_split, extreme_split, PRE_AUG

    root = Path(root).resolve()
    cfg = dict(DEFAULTS, **config)
    name = cfg.get('name', f"{cfg['dataset']}_{cfg['backbone']}_s{cfg['seed']}")
    run = root / 'runs' / name
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Choose --device cpu or run on a CUDA host.')
    if cfg['rounds'] < 1 or cfg['local_epochs'] < 1 or cfg['batch_size'] < 1:
        raise ValueError('rounds, local_epochs, and batch_size must be positive.')
    if (cfg['defense'] == 'noise') != (cfg['dp_eps'] is not None):
        raise ValueError('The noise defense requires dp_eps; other defenses must omit it.')
    if cfg['mia_target'] is not None and cfg['defense'] not in ('none', 'sparse'):
        raise ValueError('Loss-based attack traces use unmodified training inputs.')

    # Place downloaded weights inside the project so its asset location is explicit.
    model_root = Path(cfg.get('model_root', 'models'))
    if not model_root.is_absolute():
        model_root = root / model_root
    torch.hub.set_dir(str(model_root / 'torch'))
    os.environ['HF_HOME'] = str(model_root / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(model_root / 'huggingface' / 'hub')
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get('tf32', False))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get('tf32', False))
    torch.backends.cudnn.benchmark = False

    cfg.update(name=name, device=str(dev))
    started = time.time()
    metrics = dict(test_acc={}, elapsed_s=0.0, completed=False)
    _json(run / 'config.json', cfg)
    _json(run / 'metrics.json', metrics)
    data_root = Path(cfg.get('data_root', 'data'))
    if not data_root.is_absolute():
        data_root = root / data_root
    data = load_dataset(cfg['dataset'], root=data_root)
    cfg.update(n_classes=data['n_classes'], embedding_dim=768,
               train_samples=len(data['train']), test_samples=len(data['test']))
    train_cache = ImageCache(data['train'], dev, cache_device=cfg['cache_device'])
    test_cache = ImageCache(data['test'], dev, cache_device=cfg['cache_device'])
    model = L2PViT(data['n_classes'], cfg['M'], cfg['N'], cfg['head'], cfg['backbone']).to(dev)
    model.pool.dropout = cfg['prompt_dropout']

    cache_name = f"{cfg['dataset']}_{cfg['backbone']}.pt"
    query_path = root / 'cache' / 'queries' / cache_name
    cfg['query_cache'] = query_path.relative_to(root).as_posix()
    labels = np.asarray(data['train_labels'], dtype=np.int64)
    metadata = dict(version=1, dataset=cfg['dataset'], backbone=cfg['backbone'],
                    dataset_fingerprint=data['fingerprint'],
                    data_root=str(data_root.resolve()), train_samples=len(train_cache),
                    test_samples=len(test_cache), labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                    test_labels_sha256=hashlib.sha256(np.asarray(data['test'].labels, dtype=np.int64).tobytes()).hexdigest(),
                    selection_sha256=hashlib.sha256(np.asarray(data['train'].source_indices, dtype=np.int64).tobytes()).hexdigest(),
                    preprocessing='resize256-centercrop224-imagenet', query='frozen_cls')
    queries = None
    if query_path.exists() and not cfg.get('refresh_queries', False):
        stored = torch.load(query_path, map_location='cpu', weights_only=False)
        if stored.get('metadata') == metadata:
            queries = stored
    if queries is None:
        print(f"Computing frozen queries: {cfg['dataset']} / {cfg['backbone']}", flush=True)
        queries = dict(train=compute_queries(model, train_cache).cpu(),
                       test=compute_queries(model, test_cache).cpu(),
                       train_labels=torch.as_tensor(labels),
                       test_labels=torch.as_tensor(np.asarray(data['test'].labels)),
                       train_source=None if data.get('train_source') is None else torch.as_tensor(data['train_source']),
                       test_source=None if data.get('test_source') is None else torch.as_tensor(data['test_source']),
                       metadata=metadata)
        _save(query_path, queries)
    clean_train_q = queries['train'].to(dev)
    test_q = queries['test'].to(dev)
    train_q = clean_train_q if cfg['defense'] in ('none', 'sparse') else None
    if cfg['defense'] == 'augment':
        augmented = load_dataset(cfg['dataset'], root=data_root, transform=PRE_AUG)
        # Color jitter draws from the global random stream, and iterating a DataLoader advances it.
        stream = torch.get_rng_state()
        train_cache = ImageCache(augmented['train'], dev, cache_device=cfg['cache_device'])
        torch.set_rng_state(stream)

    if cfg['split'] == 'extreme':
        dominated = cfg.get('n_dominated') or max(1, round(0.1 * data['n_classes']))
        clients = extreme_split(labels, cfg['clients'], dominated, cfg['seed'])
    else:
        clients = client_split(data, cfg['clients'], cfg['alpha'], cfg['seed'])
    _save(run / 'clients.pt', [torch.as_tensor(c, dtype=torch.long) for c in clients])
    cfg['client_sizes'] = [len(c) for c in clients]
    noise_sigma = None
    if cfg['defense'] == 'noise':
        cfg['dp_delta'] = cfg.get('dp_delta') or 1.0 / len(train_cache)
        noise_sigma = input_noise_sigma(cfg['dp_eps'], cfg['dp_delta'])
        cfg['noise_sigma'] = noise_sigma
    _json(run / 'config.json', cfg)
    defense = None if cfg['defense'] == 'none' else Defense(
        cfg['defense'], cfg['sparse_ratio'], cfg['mixup_alpha'], cfg['seed'],
        device=dev, noise_sigma=noise_sigma)
    keys = {}
    save_rounds = cfg['save_rounds']
    if isinstance(save_rounds, str):
        save_rounds = [int(r) for r in save_rounds.split(',')]
    save_rounds = set(save_rounds) | {cfg['rounds']}
    mia = None
    if cfg['mia_target'] is not None:
        target = int(cfg['mia_target'])
        if not 0 <= target < len(clients):
            raise ValueError('mia_target must be a valid client index.')
        members = clients[target]
        tests = np.arange(len(test_cache))
        mia = dict(target=target, member_index=torch.as_tensor(members), local={}, glob={})

    def on_client(r, client, chosen, local_model):
        if mia is not None and mia['target'] in chosen:
            mia['local'].setdefault(r, {})[client] = (
                per_sample_losses(local_model, train_cache, clean_train_q, members, cfg['eval_batch_size']),
                per_sample_losses(local_model, test_cache, test_q, tests, cfg['eval_batch_size']))

    def on_round(r, state, accuracy):
        keys[r] = state['pool.key'].detach().cpu().clone()
        metrics['test_acc'][r] = accuracy
        metrics['elapsed_s'] = time.time() - started
        _save(run / 'keys.pt', keys)
        _json(run / 'metrics.json', metrics)
        saved_state = {n: v.detach().cpu().clone() for n, v in state.items()}
        if r in save_rounds:
            _save(run / f'ckpt_round{r}.pt', saved_state)
        if r == cfg['rounds']:
            _save(run / 'state.pt', saved_state)
        if mia is not None and r > 0:
            mia['glob'][r] = (
                per_sample_losses(model, train_cache, clean_train_q, members, cfg['eval_batch_size']),
                per_sample_losses(model, test_cache, test_q, tests, cfg['eval_batch_size']))
            _save(run / 'mia_losses.pt', mia)

    def before_round(r, state):
        if r != cfg['inject_round']:
            return state
        target = int(cfg['inject_target'])
        if not 0 <= target < len(clean_train_q):
            raise ValueError('inject_target must be a training-set index.')
        injected, slots, interval = promptmia_inject(state['pool.key'], clean_train_q[target], cfg['N'],
                                                    0.02, 0.05, torch.Generator().manual_seed(cfg['seed']))
        _json(run / 'injection.json', dict(round=r, target=target, slots=slots.tolist(), interval=list(interval)))
        return dict(state, **{'pool.key': injected})

    runtime = dict(cfg, defense=defense, aggregation=cfg['aggregator'], before_round=before_round)
    run_fedavg(model, train_cache, train_q, test_cache, test_q, clients, runtime, on_round,
               on_client if mia is not None else None)
    if noise_sigma is not None:
        noisy_defense = Defense('noise', seed=cfg['seed'] + 1, device=dev, noise_sigma=noise_sigma)
        metrics['test_acc_noisy'] = evaluate_noisy(model, test_cache, noisy_defense, cfg['eval_batch_size'])
    metrics.update(elapsed_s=time.time() - started, completed=True)
    _json(run / 'metrics.json', metrics)
    return run

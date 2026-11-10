"""Measure the attacks and diagnostics used by the paper's figures and tables."""

from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _run(job, root):
    run = root / 'runs' / job['run']
    config = json.loads((run / 'config.json').read_text())
    series = _load(run / 'keys.pt')
    series = {int(k): v for k, v in series.items()}
    round_number = int(job.get('round', max(series)))
    if round_number not in series:
        raise ValueError(f"Round {round_number} is absent from {run / 'keys.pt'}")
    cache = Path(config.get('query_cache', f"cache/queries/{config['dataset']}_{config['backbone']}.pt"))
    queries = _load(cache if cache.is_absolute() else root / cache)
    return run, config, series[round_number], queries, round_number


def _metadata(job, config, round_number):
    return dict(name=job['name'], kind=job['kind'], run=job.get('run'),
                dataset=config.get('dataset'), backbone=config.get('backbone'),
                round=round_number, config=config, settings=job)


def _indices(n, limit, seed):
    if limit is None or int(limit) >= n:
        return np.arange(n)
    if int(limit) <= 0:
        raise ValueError('n_targets must be positive')
    return np.random.default_rng(seed).choice(n, int(limit), replace=False)


def _clients(config, queries):
    from .data import client_split, extreme_split
    labels = torch.as_tensor(queries['train_labels']).numpy()
    if config.get('split') == 'extreme':
        return extreme_split(labels, config['clients'],
                             config.get('n_dominated') or max(1, round(0.1 * len(np.unique(labels)))),
                             config.get('seed', 0))
    source = queries.get('train_source')
    return client_split(dict(train_labels=labels,
                             train_source=None if source is None else np.asarray(source)),
                        config['clients'], config.get('alpha', 0.5), config.get('seed', 0))


def _plain_attack(job, root, device):
    from .attack import security_game
    run, config, keys, queries, rnd = _run(job, root)
    train_q, test_q = queries['train'].to(device), queries['test'].to(device)
    if job.get('population', 'global') == 'client' or job.get('client_population'):
        client = int(job.get('client', 0))
        losses = _load(run / 'mia_losses.pt') if (run / 'mia_losses.pt').exists() else {}
        indices = losses.get('member_index') if losses.get('target') == client else None
        if indices is None:
            indices = _clients(config, queries)[client]
        train_q = train_q[torch.as_tensor(indices, device=device)]
    member_idx = torch.as_tensor(_indices(len(train_q), job.get('n_targets'), 1234), device=device)
    test_q = test_q[torch.as_tensor(_indices(len(test_q), job.get('n_targets'), 5678), device=device)]
    keys = keys.to(device)
    n_selected = int(job.get('N', config.get('N', 4)))
    sizes = list(map(int, job.get('batch_sizes', [1, 4, 8, 16, 32, 64, 128, 256])))
    if min(sizes) < 1 or max(sizes) > len(train_q):
        raise ValueError(f'Batch sizes must lie in [1, {len(train_q)}] for {job["name"]}')
    result = _metadata(job, config, rnd)
    result.update(batch_sizes=sizes, member_targets=len(member_idx), nonmember_targets=len(test_q), methods={})
    for method in job.get('methods', ['promptmia', 'naive']):
        if method not in ('promptmia', 'naive'):
            raise ValueError(f'Unknown attack method: {method}')
        result['methods'][method] = {}
        for batch in sizes:
            repeats = [security_game(keys, train_q, test_q, n_selected, batch,
                                     float(job.get('delta_min', .02)), float(job.get('delta', .05)),
                                     seed=int(job.get('seed', 0)) + 10000 * r + batch,
                                     naive=method == 'naive', member_idx=member_idx,
                                     chunk=int(job.get('chunk', 250)))
                       for r in range(int(job.get('repeats', 5)))]
            result['methods'][method][str(batch)] = {m: [float(x[m]) for x in repeats]
                                                     for m in ('tpr', 'fpr', 'advantage', 'asr')}
            print(f"{job['name']} {method} B={batch}: ASR={np.mean([x['asr'] for x in repeats]):.5f}", flush=True)
    return result


def _load_vision_model(config, run, keys, device):
    from .model import L2PViT
    state = _load(run / 'state.pt')
    classes = config.get('num_classes', config.get('n_classes'))
    if classes is None:
        classes = state['head.2.weight' if config.get('head', 'cls') == 'cls' else 'head.weight'].size(0)
    model = L2PViT(int(classes), keys.size(0), config.get('N', 4),
                   head=config.get('head', 'cls'), backbone=config.get('backbone', 'vit_b32')).to(device)
    model.load_trainable_state(state)
    model.eval()
    return model


def _defended_attack(job, root, device):
    """Apply the trained client's defense to every sampled attack batch."""
    from .attack import inject_batch, robust_inject
    from .data import ImageCache, load_dataset, PRE, PRE_AUG
    from .defenses import Defense, input_noise_sigma
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Initialize once per experiment, including ColorJitter's global generator.
    torch.manual_seed(int(job.get('seed', 0)))
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(int(job.get('seed', 0)))
    run, config, keys, queries, rnd = _run(job, root)
    defense_name = job.get('defense', config.get('defense', 'none'))
    dropout = float(job.get('prompt_dropout', config.get('prompt_dropout', 0)))
    train_q, test_q = queries['train'].to(device), queries['test'].to(device)
    keys = keys.to(device)
    n_selected = int(job.get('N', config.get('N', 4)))
    target_rng = np.random.default_rng(1234)
    limit = int(job.get('n_targets', 1000))
    members = torch.as_tensor(target_rng.choice(len(train_q), min(limit, len(train_q)), replace=False), device=device)
    nonmembers = torch.as_tensor(target_rng.choice(len(test_q), min(limit, len(test_q)), replace=False), device=device)
    sizes = list(map(int, job.get('batch_sizes', [1, 4, 8, 16, 32, 64, 128, 256])))
    if min(sizes) < 1 or max(sizes) > len(train_q):
        raise ValueError('An attack batch must fit in the training population without replacement')
    eot = int(job.get('eot', 0))
    if eot and defense_name not in ('augment', 'noise'):
        raise ValueError('Expectation over transformations requires augmentation or input noise')
    required = int(job.get('min_selected', math.ceil(n_selected/2) if dropout else n_selected))
    if not 1 <= required <= n_selected:
        raise ValueError('min_selected must lie between one and N')
    model = cache = test_cache = None
    if defense_name != 'none':
        if rnd != int(config.get('rounds', 60)):
            raise ValueError('Defense evaluation needs the final trained model state')
        model = _load_vision_model(config, run, keys, device)
        data_root = Path(config.get('data_root', root / 'data'))
        if not data_root.is_absolute():
            data_root = root / data_root
        data = load_dataset(config['dataset'], root=data_root,
                            transform=PRE_AUG if defense_name == 'augment' else PRE)
        cache = ImageCache(data['train'], device)
        if eot:
            test_cache = ImageCache(data['test'], device)
        if len(cache) != len(train_q):
            raise ValueError('Image order and query cache must use the same prepared dataset')
    sigma = config.get('noise_sigma')
    if defense_name == 'noise' and sigma is None:
        sigma = input_noise_sigma(config['dp_eps'], config.get('dp_delta') or 1 / len(train_q))
    result = _metadata(job, config, rnd)
    result.update(batch_sizes=sizes, member_targets=len(members), nonmember_targets=len(nonmembers),
                  min_selected=required, methods={})
    sparse_ratio = float(job.get('sparse_ratio', config.get('sparse_ratio', .8)))
    max_images = int(job.get('images_per_chunk', 512))
    dropout_gen = torch.Generator(device=device).manual_seed(int(job.get('dropout_seed', 4321)))
    # EoT key calibration
    calibration = dict(requested=float(job.get('eot_coverage', .95)), targets=0, calibrated=0,
                       achieved_sum=0.0, achieved_min=1.0)

    def rate(target_indices, is_member, batch, seed, method):
        gen = torch.Generator().manual_seed(seed)
        sample_gen = torch.Generator(device=device).manual_seed(seed)
        defense = None if defense_name == 'none' else Defense(
            defense_name, sparse_ratio, float(config.get('mixup_alpha', .2)),
            seed=seed, device=device, noise_sigma=sigma)
        hits = 0
        chunk_size = max(1, max_images // batch) if defense_name != 'none' else int(job.get('chunk', 250))
        for chunk in target_indices.split(chunk_size):
            target = (train_q if is_member else test_q)[chunk]
            with torch.no_grad():
                if eot and method == 'promptmia':
                    source = cache if is_member else test_cache
                    repeated_ids = chunk.repeat_interleave(eot)
                    copies = []
                    for ids in repeated_ids.split(max_images):
                        images, _ = defense.inputs(source, ids)
                        copies.append(model.query(images))
                    transformed = torch.cat(copies).reshape(len(chunk), eot, -1)
                    pools, slots, _, calibrated, achieved = robust_inject(
                        keys, target, transformed, n_selected, float(job.get('delta_min', .02)),
                        float(job.get('delta', .05)), gen, coverage=float(job.get('eot_coverage', .95)),
                        grid=int(job.get('eot_grid', 41)))
                    calibration['targets'] += int(calibrated.numel())
                    calibration['calibrated'] += int(calibrated.sum())
                    calibration['achieved_sum'] += float(achieved.sum())
                    calibration['achieved_min'] = min(calibration['achieved_min'], float(achieved.min()))
                else:
                    pools, slots = inject_batch(keys, target, n_selected, float(job.get('delta_min', .02)),
                                                float(job.get('delta', .05)), gen, naive=method == 'naive')
                random_scores = torch.rand(len(chunk), len(train_q), device=device, generator=sample_gen)
                if is_member:
                    random_scores[torch.arange(len(chunk), device=device), chunk] = -1
                    ids = torch.cat([chunk[:, None], random_scores.topk(batch - 1, dim=1).indices], dim=1)
                else:
                    ids = random_scores.topk(batch, dim=1).indices
                if defense_name in ('none', 'sparse'):
                    q = train_q[ids]
                elif defense_name == 'mixup':
                    # Mixing stays inside each client's sampled batch.
                    q = torch.stack([model.query(defense.inputs(cache, row)[0]) for row in ids])
                else:
                    q = torch.cat([model.query(defense.inputs(cache, part)[0])
                                   for part in ids.flatten().split(max_images)]).reshape(len(chunk), batch, -1)
                similarity = F.normalize(q, dim=2) @ F.normalize(pools, dim=2).transpose(1, 2)
                if dropout:
                    unavailable = torch.rand(similarity.shape, generator=dropout_gen, device=device) < dropout
                    similarity.masked_fill_(unavailable, -2)
                selected = similarity.topk(n_selected, dim=2).indices
                updated = torch.zeros(pools.shape[:2], dtype=torch.bool, device=device)
                updated.scatter_(1, selected.flatten(1), True)
            if defense_name == 'sparse':
                for row_number, (row, chosen) in enumerate(zip(ids, selected)):
                    images, labels = cache.get(row)
                    gradient = torch.zeros_like(model.pool.prompt)
                    for start in range(0, len(row), max_images):
                        sl = chosen[start:start + max_images]
                        prompts = model.pool.prompt.detach()[sl].clone().requires_grad_(True)
                        hidden = model.encode(images[start:start + max_images], prompts)
                        if model.head_type == 'prompt':
                            logits = model.head(hidden[:, 1:1 + n_selected].mean(1))
                        elif model.backbone == 'vit_b32':
                            logits = model.head(model.vit.heads(hidden[:, 0]))
                        else:
                            logits = model.head(hidden[:, 0])
                        loss = F.cross_entropy(logits, labels[start:start + max_images], reduction='sum') / batch
                        grad = torch.autograd.grad(loss, prompts)[0]
                        gradient.index_add_(0, sl.flatten(), grad.flatten(0, 1))
                    keep = int(gradient.numel() * (1 - sparse_ratio))
                    if keep:
                        threshold = gradient.abs().flatten().topk(keep).values[-1]
                        updated[row_number] = ((gradient.abs() >= threshold) & (gradient != 0)).any(1)
                    else:
                        updated[row_number].zero_()
            # Dropout uses the paper's half-of-N decision; other games require all N.
            hits += int((updated.gather(1, slots).sum(1) >= required).sum())
        return hits / len(target_indices)

    for method in job.get('methods', ['promptmia']):
        if method not in ('promptmia', 'naive'):
            raise ValueError(f'Unknown attack method: {method}')
        result['methods'][method] = {}
        for batch in sizes:
            records = []
            for repeat in range(int(job.get('repeats', 5))):
                seed = int(job.get('seed', 0)) + 10000 * repeat + batch
                tpr = rate(members, True, batch, seed, method)
                fpr = rate(nonmembers, False, batch, seed + 1, method)
                records.append(dict(tpr=tpr, fpr=fpr, advantage=tpr-fpr, asr=(1+tpr-fpr)/2))
            result['methods'][method][str(batch)] = {m: [r[m] for r in records] for m in records[0]}
            print(f"{job['name']} {method} B={batch}: ASR={np.mean([r['asr'] for r in records]):.5f}", flush=True)
    if calibration['targets']:
        fraction = calibration['calibrated'] / calibration['targets']
        result['eot_calibration'] = dict(
            coverage_requested=calibration['requested'], targets=calibration['targets'],
            fraction_calibrated=fraction, mean_achieved_coverage=calibration['achieved_sum'] / calibration['targets'],
            min_achieved_coverage=calibration['achieved_min'])
        if fraction < 1:
            print(f"{job['name']} WARNING: EoT calibration reached {calibration['requested']:.0%} coverage for only "
                  f"{fraction:.1%} of targets (min achieved {calibration['achieved_min']:.1%})", flush=True)
    return result


def _loss(job, root, device):
    from .loss_mia import attacks
    run, config, _, _, rnd = _run(job, root)
    result = _metadata(job, config, rnd)
    matched_fpr = None
    if job.get('attack_source'):
        source = json.loads((root / 'results' / (job['attack_source']+'.json')).read_text())
        if source['run'] != job['run']:
            raise ValueError('Loss comparisons must use PromptMIA results from the same trained run')
        pm = source['methods']['promptmia'][str(int(job.get('batch_size', 64)))]
        pm = {m: float(np.mean(v)) for m, v in pm.items()}
        matched_fpr = pm['fpr']
    measured = attacks(_load(run / 'mia_losses.pt'), fpr=matched_fpr)
    result['methods'] = {name: measured[name] for name in ('blackbox_loss', 'loss_series', 'fedmia_i')}
    if matched_fpr is not None:
        result['methods']['promptmia'] = pm
        result['matched_fpr'] = matched_fpr
    return result


def _outlier(job, root, device):
    from .attack import inject_batch
    from .outliers import flags, METHODS
    _, config, keys, queries, rnd = _run(job, root)
    ids = _indices(len(queries['train']), job.get('n_targets', 100), int(job.get('seed', 42)))
    n_selected = int(job.get('N', config.get('N', 4)))
    pools, slots = inject_batch(keys, queries['train'][ids], n_selected,
                                float(job.get('delta_min', .02)), float(job.get('delta', .05)),
                                torch.Generator().manual_seed(int(job.get('seed', 42))))
    counts = {m: dict(tp=0, fp=0, fn=0) for m in METHODS}
    for pool, selected in zip(pools, slots):
        truth = np.zeros(len(pool), bool)
        truth[selected.numpy()] = True
        for name, predicted in flags(F.normalize(pool.double(), dim=1).numpy()).items():
            counts[name]['tp'] += int((predicted & truth).sum())
            counts[name]['fp'] += int((predicted & ~truth).sum())
            counts[name]['fn'] += int((~predicted & truth).sum())
    clean = flags(F.normalize(keys.double(), dim=1).numpy())
    rate = float(job.get('injection_rate', .1))
    if not 0 < rate <= 1:
        raise ValueError('injection_rate must lie in (0, 1]')
    methods = {}
    for name, c in counts.items():
        tp, fn = rate * c['tp'], rate * c['fn']
        fp_clean = int(clean[name].sum())
        fp = rate * c['fp'] + (1-rate) * len(ids) * fp_clean
        precision = tp / (tp+fp) if tp+fp else 0.
        recall = tp / (tp+fn) if tp+fn else 0.
        methods[name] = dict(precision=precision, recall=recall,
                             f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                             counts=dict(c, fp_clean=fp_clean), clean_flagged_fraction=fp_clean/len(keys))
    return dict(_metadata(job, config, rnd), methods=methods, targets=len(ids), injection_rate=rate)


def _bound(job, root, device):
    from .bound import bounds
    _, config, keys, queries, rnd = _run(job, root)
    n = min(int(job.get('n_targets', 100)), len(queries['test']))
    lb, empirical, empty = bounds(keys, queries['test'], n, int(job.get('N', config.get('N', 4))),
                                  float(job.get('delta_min', .02)), float(job.get('delta', .05)))
    log_mean = float((np.logaddexp.reduce(lb * math.log(10)) - math.log(len(lb))) / math.log(10))
    return dict(_metadata(job, config, rnd), log10_mean_bound=log_mean,
                log10_bounds=lb.tolist(), empirical_fpr=empirical.tolist(),
                empty_components=np.asarray(empty).tolist(), empirical_fpr_mean=float(empirical.mean()))


def _first_image_queries(job, root, config, device):
    """Encode the first training images in one batch, with TF32 at PyTorch's default (on for the patch convolution)."""
    from torch.utils.data import Subset
    from .data import ImageCache, load_dataset
    from .model import L2PViT
    if torch.device(device).type != 'cuda':
        raise ValueError(f"{job['name']}: encoding queries needs a CUDA device, since TF32 only exists there")
    count = max(int(job.get('n_queries', 200)), int(job.get('target', 0)) + 1)
    data_root = Path(config.get('data_root', root / 'data'))
    data = load_dataset(config['dataset'], root=data_root if data_root.is_absolute() else root / data_root)
    cache = ImageCache(Subset(data['train'], range(count)), device)
    model = L2PViT(data['n_classes'], config.get('M', 20), config.get('N', 4), head=config.get('head', 'cls'),
                   backbone=config.get('backbone', 'vit_b32')).to(device).eval()
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, True
    try:
        with torch.no_grad():
            return model.query(cache.get(torch.arange(count))[0]).cpu()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _embedding(job, root, device):
    from sklearn.manifold import TSNE
    from .attack import promptmia_inject, naive_inject
    _, config, keys, queries, rnd = _run(job, root)
    source = job.get('query_source', 'encoded')
    if source not in ('encoded', 'cache'):
        raise ValueError(f"{job['name']}: unknown query_source {source}")
    if source == 'encoded' and (job.get('include_queries', True) or job.get('variant', 'benign') != 'benign'):
        if job.get('query_split', 'train') != 'train' or job.get('query_sampling', 'first') != 'first':
            raise ValueError(f"{job['name']}: encoded queries are the first training images; use query_source cache")
        queries = dict(queries, train=_first_image_queries(job, root, config, device))
    variant = job.get('variant', 'benign')
    target_index = int(job.get('target', 0))
    selected = []
    if variant == 'promptmia':
        keys, selected, _ = promptmia_inject(keys.double(), queries['train'][target_index].double(),
                                            config.get('N', 4), float(job.get('delta_min', .02)),
                                            float(job.get('delta', .05)), torch.Generator().manual_seed(0))
        selected = selected.tolist()
    elif variant == 'naive':
        keys, selected = naive_inject(keys, queries['train'][target_index], config.get('N', 4))
        selected = selected.tolist()
    elif variant != 'benign':
        raise ValueError(f'Unknown embedding variant {variant}')
    q = queries[job.get('query_split', 'train')]
    n = min(int(job.get('n_queries', 200)), len(q)) if job.get('include_queries', True) else 0
    ids = np.arange(n) if n == 0 or job.get('query_sampling', 'first') == 'first' else torch.randperm(
        len(q), generator=torch.Generator().manual_seed(42))[:n].numpy()
    if job.get('detect_outliers'):
        features = F.normalize(keys.double(), dim=1).numpy()
    elif n == 0:
        features = F.normalize(keys.double(), dim=1).float().numpy()
    else:
        features = F.normalize(keys.float(), dim=1).numpy()
    if n:
        features = np.concatenate([features, F.normalize(q[ids].float(), dim=1).numpy()])
    if len(features) < 3:
        raise ValueError('At least three vectors are needed for the embedding plot')
    coordinates = TSNE(n_components=2, perplexity=min(10, len(features)-1), random_state=42,
                       init='pca', learning_rate='auto').fit_transform(features)
    result = dict(_metadata(job, config, rnd), coordinates=coordinates.tolist(),
                  n_keys=len(keys), n_queries=n, injected_slots=selected, variant=variant,
                  query_indices=ids.tolist())
    if job.get('detect_outliers'):
        from .outliers import flags
        result['flagged'] = {name: np.flatnonzero(mask).tolist()
                              for name, mask in flags(F.normalize(keys.double(), dim=1).numpy()).items()}
    return result


def key_generation_flops(benign_keys, injected_keys, dimension):
    """Count scalar arithmetic in Algorithm 2; random-number draws are excluded."""
    k, n, d = benign_keys, injected_keys, dimension
    normalize = 3*d
    maximum_similarity = k*normalize + normalize + k*(2*d-1) + (k-1)
    sample_similarities = 4+3*n
    one_key = normalize+normalize+(2*d-1)+d+d+normalize+d+3+d+d+2*d+d
    return int(maximum_similarity+sample_similarities+n*one_key)


def _sparsity(job, root, device):
    from .model import L2PViT
    rows = []
    for backbone in job.get('backbones', ['vit_b32', 'deit_b16', 'convit_base']):
        model = L2PViT(int(job.get('num_classes', 10)), int(job.get('M', 20)),
                       int(job.get('N', 4)), head='cls', backbone=backbone, pretrained=False)
        total = sum(p.numel() for p in model.parameters())
        modified = int(job.get('N', 4))*model.pool.key.size(1)
        rows.append(dict(model=backbone, total_params=total, modified_params=modified,
                         sparsity_percent=100*modified/total))
        del model
    return dict(name=job['name'], kind='sparsity', settings=job, rows=rows)


def _flops(job, root, device):
    n, m, d = int(job.get('N', 4)), int(job.get('M', 20)), int(job.get('dimension', 768))
    if not 0 < n < m or d < 2:
        raise ValueError('Require 0 < N < M and an embedding dimension of at least two')
    operations = key_generation_flops(m-n, n, d)
    return dict(name=job['name'], kind='flops', settings=job, flops=operations,
                M=m, N=n, dimension=d, fraction_of_vit_forward=operations/4.4e9)


def _drift(job, root, device):
    from .diagnostics import drift_statistics
    run, config, _, queries, rnd = _run(job, root)
    result = drift_statistics(
        _load(run / 'keys.pt'), queries['train'], n_targets=int(job.get('n_targets', 300)),
        N=int(job.get('N', config.get('N', 4))), delta_min=float(job.get('delta_min', .02)),
        delta=float(job.get('delta', .05)), seed=int(job.get('seed', 0)),
        thresholds=job.get('thresholds', [1, 2, 3]), injection_rates=job.get('injection_rates'),
        steady_round=int(job.get('steady_round', 5)))
    return dict(_metadata(job, config, rnd), **result)


def _distinctiveness(job, root, device):
    from .diagnostics import distinctiveness_statistics
    _, config, keys, queries, rnd = _run(job, root)
    result = distinctiveness_statistics(
        keys.to(device), queries['train'].to(device), queries['test'].to(device), _clients(config, queries),
        n_targets=int(job.get('n_targets', 2000)), batch_sizes=list(map(int, job.get('batch_sizes', [64, 256]))),
        n_batches=int(job.get('n_batches', 32)), N=int(job.get('N', config.get('N', 4))),
        delta_min=float(job.get('delta_min', .02)), delta=float(job.get('delta', .05)),
        seed=int(job.get('seed', 0)), chunk=int(job.get('chunk', 250)))
    return dict(_metadata(job, config, rnd), **result)


def evaluate_job(job, root, device='cpu'):
    """Run one measurement and save its full precision values under results/."""
    root = Path(root).resolve()
    kind = job['kind']
    if kind == 'attack':
        config = json.loads((root / 'runs' / job['run'] / 'config.json').read_text())
        defense = job.get('defense', config.get('defense', 'none'))
        dropout = job.get('prompt_dropout', config.get('prompt_dropout', 0))
        function = _defended_attack if job.get('defended') or job.get('attack_round') or defense != 'none' or dropout else _plain_attack
    else:
        functions = dict(loss=_loss, outlier=_outlier, bound=_bound, embedding=_embedding,
                         sparsity=_sparsity, flops=_flops, drift=_drift, distinctiveness=_distinctiveness)
        if kind not in functions:
            raise ValueError(f'Unknown evaluation kind: {kind}')
        function = functions[kind]
    result = function(job, root, str(device))
    _json(root / 'results' / (job['name'] + '.json'), result)
    return result

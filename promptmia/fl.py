"""Local prompt training and server aggregation for federated learning."""
import numpy as np
import torch
import torch.nn.functional as F


def local_train(model, cache, queries, idx, epochs, batch_size, lr, lam, gen, key_lr, defense=None):
    """Train one client and return the prompt rows selected during its updates."""
    model.train()
    used = torch.zeros(len(model.pool.key), dtype=torch.bool, device=model.pool.key.device)
    other = [p for n, p in model.named_parameters() if p.requires_grad and n != 'pool.key']
    optimizer = torch.optim.Adam([{'params': other}, {'params': [model.pool.key], 'lr': key_lr}],
                                 lr=lr, betas=(0.9, 0.98), eps=1e-6)
    idx = torch.as_tensor(idx, dtype=torch.long, device=cache.device)
    for _ in range(epochs):
        order = idx[torch.randperm(len(idx), generator=gen).to(idx.device)]
        for ids in order.split(batch_size):
            if defense is None:
                x, y = cache.get(ids)
                logits, key_loss, selected = model(x, None if queries is None else queries[ids])
                loss = F.cross_entropy(logits, y) + lam * key_loss
            else:
                x, y = defense.inputs(cache, ids)
                query = queries[ids] if defense.clean_inputs and queries is not None else None
                logits, key_loss, selected = model(x, query)
                loss = defense.loss(logits, y) + lam * key_loss
            used[selected.flatten()] = True
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if defense is not None:
                defense.after_backward(model)
            optimizer.step()
    return used


def fedavg(states, weights):
    """Average every trainable tensor, weighting clients by their sample counts."""
    weights = torch.as_tensor(weights, dtype=torch.float32)
    if not states or len(states) != len(weights) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('FedAvg requires nonempty states and valid client weights.')
    weights = weights / weights.sum()
    names = set(states[0])
    if any(set(s) != names for s in states):
        raise ValueError('All clients must return the same trainable tensors.')
    return {name: sum(w * s[name] for w, s in zip(weights.tolist(), states)) for name in states[0]}


@torch.no_grad()
def per_sample_losses(model, cache, queries, idx, batch_size=500):
    """Cross-entropy of individual examples, used by the loss-based attacks."""
    model.eval()
    idx = torch.as_tensor(idx, dtype=torch.long, device=cache.device)
    values = []
    for ids in idx.split(batch_size):
        x, y = cache.get(ids)
        logits, _, _ = model(x, None if queries is None else queries[ids])
        values.append(F.cross_entropy(logits, y, reduction='none').cpu())
    return torch.cat(values)


@torch.no_grad()
def evaluate(model, cache, queries, batch_size=500):
    """Accuracy on the complete test split."""
    model.eval()
    correct = 0
    for ids in torch.arange(len(cache), device=cache.device).split(batch_size):
        x, y = cache.get(ids)
        logits, _, _ = model(x, None if queries is None else queries[ids])
        correct += (logits.argmax(1) == y).sum().item()
    return correct / len(cache)


@torch.no_grad()
def evaluate_noisy(model, cache, defense, batch_size=500):
    """Accuracy when fresh input noise is also applied at test time."""
    model.eval()
    correct = 0
    for ids in torch.arange(len(cache), device=cache.device).split(batch_size):
        x, y = cache.get(ids)
        logits, _, _ = model(defense.perturb(x))
        correct += (logits.argmax(1) == y).sum().item()
    return correct / len(cache)


def run_fedavg(model, train_cache, train_q, test_cache, test_q, client_idx, cfg,
               on_round=None, on_client=None):
    """Run communication rounds with sample-weighted FedAvg.

    Each client starts from cloned global parameters. Callbacks receive completed
    local or global updates so measurements do not change the aggregation order.
    """
    if not 1 <= cfg['clients_per_round'] <= len(client_idx):
        raise ValueError('clients_per_round must fit within the federation.')
    if cfg.get('aggregator', 'fedavg') != 'fedavg':
        raise ValueError('The experiment uses FedAvg aggregation.')
    if any(not len(indices) for indices in client_idx):
        raise ValueError('Every training client must contain at least one sample.')
    rng = np.random.default_rng(cfg['seed'])
    gen = torch.Generator().manual_seed(cfg['seed'])
    state = model.trainable_state()
    if on_round:
        on_round(0, state, None)
    sizes = np.asarray([len(c) for c in client_idx])
    frequencies = {}
    for r in range(1, cfg['rounds'] + 1):
        chosen = np.sort(rng.choice(len(client_idx), cfg['clients_per_round'], replace=False))
        if cfg.get('before_round'):
            state = cfg['before_round'](r, state)
        local_states = []
        for c in chosen:
            model.load_trainable_state(state)
            if cfg.get('diverse_select'):
                if c not in frequencies or len(frequencies[c]) != len(model.pool.key):
                    frequencies[c] = torch.zeros(len(model.pool.key), device=model.pool.key.device)
                model.pool.freq = frequencies[c]
            local_train(model, train_cache, train_q, client_idx[c], cfg['local_epochs'],
                        cfg['batch_size'], cfg['lr'], cfg['lam'], gen, cfg['key_lr'], cfg.get('defense'))
            if on_client:
                on_client(r, int(c), chosen.tolist(), model)
            local_states.append(model.trainable_state())
        model.pool.freq = None
        state = fedavg(local_states, sizes[chosen])
        model.load_trainable_state(state)
        acc = evaluate(model, test_cache, test_q, cfg.get('eval_batch_size', 500))
        print(f'round {r:3d} | clients {chosen.tolist()} | test accuracy {acc:.4f} | pool {len(model.pool.key)}', flush=True)
        if on_round:
            on_round(r, state, acc)
    return state

"""Key-drift auditing and target-distinctiveness measurements."""

import numpy as np
import torch
import torch.nn.functional as F


def mad_flags(displacements, multiplier):
    """Flag values above median + c times the unscaled median absolute deviation."""
    values = np.asarray(displacements)
    median = np.median(values)
    return values > median + multiplier*np.median(np.abs(values-median))


def drift_statistics(key_series, train_queries, n_targets=300, N=4, delta_min=.02, delta=.05,
                     seed=0, thresholds=(1, 2, 3), injection_rates=None, steady_round=5):
    from .attack import inject_batch
    rates = injection_rates or {'1/10': .1, '1/60': 1/60}
    if any(not 0 < rate <= 1 for rate in rates.values()):
        raise ValueError('Injection rates must lie in (0, 1]')
    key_series = {int(r): k.detach().cpu() for r, k in key_series.items()}
    rounds = sorted(key_series)
    if any(b-a != 1 for a, b in zip(rounds, rounds[1:])):
        raise ValueError('Drift auditing requires keys from every consecutive training round')
    series = torch.stack([key_series[r] for r in rounds]).double()
    drift = (series[1:]-series[:-1]).norm(dim=2)
    steady = drift[torch.tensor(rounds[1:]) >= steady_round].numpy()
    if len(steady) == 0:
        raise ValueError('No key transitions remain after steady_round')
    final = key_series[rounds[-1]]
    previous = key_series[rounds[-2]].double()
    count = min(int(n_targets), len(train_queries))
    if count < 1:
        raise ValueError('At least one audit target is required')
    ids = np.random.default_rng(seed).choice(len(train_queries), count, replace=False)
    pools, slots = inject_batch(final, train_queries[ids].cpu().float(), N, delta_min, delta,
                                torch.Generator().manual_seed(seed), evade='norm')
    slots = slots.numpy()
    M = len(final)
    poisoned_drift = (pools.double() - previous.unsqueeze(0)).norm(dim=2).numpy()  # (count, M)
    clean_counts = {c: float(np.mean([mad_flags(d, c).sum() for d in steady])) for c in thresholds}  # mean flags per intact clean round
    clean_fraction = {str(c): clean_counts[c]/M for c in thresholds}
    injected_counts = {str(c): dict(tp=0, fp=0, fn=0) for c in thresholds}
    for target in range(count):
        values = poisoned_drift[target]
        is_injected = np.zeros(M, bool)
        is_injected[slots[target]] = True
        for c in thresholds:
            flagged = mad_flags(values, c)
            counts = injected_counts[str(c)]
            counts['tp'] += int(flagged[is_injected].sum())
            counts['fp'] += int(flagged[~is_injected].sum())
            counts['fn'] += int((~flagged[is_injected]).sum())
    pooled = {}
    for label, rate in rates.items():
        clean_rounds = count*(1-rate)/rate  # clean rounds per attacked round at the injection rate (Table 1 accounting)
        pooled[label] = {}
        for c in thresholds:
            clean_fp = clean_rounds*clean_counts[c]  # expected false positives the clean rounds contribute (no true positives)
            counts = injected_counts[str(c)]
            tp, fp, fn = counts['tp'], counts['fp']+clean_fp, counts['fn']
            precision = tp/(tp+fp) if tp+fp else 0.
            recall = tp/(tp+fn) if tp+fn else 0.
            f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
            pooled[label][str(c)] = dict(precision=precision, recall=recall, f1=f1,
                                        tp=tp, fp=fp, fn=fn, clean_fp=clean_fp, clean_rounds=clean_rounds)
    return dict(clean_flagged_fraction=clean_fraction, injection_counts=injected_counts,
                pooled=pooled, targets=count, thresholds=list(thresholds), injection_rates=rates)


def quintiles(scores, tpr, fpr):
    """Equal-count bins ordered from the most to the least distinctive targets."""
    if len(scores) < 5:
        raise ValueError('Quintiles need at least five target samples')
    result = []
    for selected in np.array_split(np.argsort(scores, kind='stable'), 5):
        t, f = float(np.asarray(tpr)[selected].mean()), float(np.asarray(fpr)[selected].mean())
        result.append(dict(tpr=t, fpr=f, asr=(1+t-f)/2, targets=len(selected)))
    return result


@torch.no_grad()
def distinctiveness_statistics(keys, train_queries, test_queries, clients, n_targets=2000,
                               batch_sizes=(64, 256), n_batches=32, N=4,
                               delta_min=.02, delta=.05, seed=0, chunk=250):
    from .attack import inject_batch, all_injected_selected
    if n_batches < 1 or chunk < 1:
        raise ValueError('n_batches and chunk must be positive')
    count = min(int(n_targets), len(test_queries))
    if count < 5:
        raise ValueError('Distinctiveness needs at least five test targets')
    rng = np.random.default_rng(seed)
    target_indices = rng.choice(len(test_queries), count, replace=False)
    targets = test_queries[torch.as_tensor(target_indices, device=keys.device)]
    scores = (F.normalize(targets, dim=1) @ F.normalize(keys, dim=1).t()).max(1).values
    pools, slots = inject_batch(keys, targets, N, delta_min, delta, torch.Generator().manual_seed(seed))
    result = dict(target_indices=target_indices.tolist(), s_max=scores.tolist(), games={}, quintiles={}, excluded_clients={})
    for batch_size in batch_sizes:
        eligible = [c for c in clients if len(c) >= batch_size]
        if not eligible:
            raise ValueError(f'No client has enough samples for a batch of {batch_size}')
        fp = torch.zeros(count, device=keys.device)
        tp = torch.zeros_like(fp)
        for start in range(0, count, chunk):
            ids = torch.arange(start, min(start+chunk, count), device=keys.device)
            for _ in range(n_batches):
                batch = np.stack([rng.choice(eligible[rng.integers(len(eligible))], batch_size, replace=False)
                                  for _ in range(len(ids))])
                batch = torch.as_tensor(batch, device=keys.device)
                fp[ids] += all_injected_selected(pools[ids], slots[ids], train_queries[batch]).float()
                member = torch.cat([targets[ids, None], train_queries[batch[:, :batch_size-1]]], dim=1)
                tp[ids] += all_injected_selected(pools[ids], slots[ids], member).float()
        tpr, fpr = (tp/n_batches).cpu().numpy(), (fp/n_batches).cpu().numpy()
        result['games'][str(batch_size)] = dict(tpr=tpr.tolist(), fpr=fpr.tolist())
        result['quintiles'][str(batch_size)] = quintiles(scores.cpu().numpy(), tpr, fpr)
        result['excluded_clients'][str(batch_size)] = len(clients)-len(eligible)
    return result

"""Loss-based attacks for Figure 5. Higher scores indicate membership.

Blackbox-Loss uses one selected round; Loss-Series averages negative losses;
FedMIA-I compares each target-client loss to the other selected clients' losses.
"""
import numpy as np
import scipy.stats
from sklearn import metrics


def roc_metrics(member, nonmember):
    y = np.r_[np.ones(len(member)), np.zeros(len(nonmember))]
    fpr, tpr, _ = metrics.roc_curve(y, np.r_[member, nonmember])
    adv = float((tpr - fpr).max())
    out = dict(auc=float(metrics.auc(fpr, tpr)), advantage=adv, asr=(1 + adv) / 2)
    for f in (0.1, 0.01, 0.001):
        out[f'tpr@fpr={f}'] = float(tpr[np.searchsorted(fpr, f, side='right') - 1])
    return out


def at_fpr(member, nonmember, fpr):
    """The operating point with the largest FPR not above `fpr` (threshold on the scores): TPR, FPR, Advantage, ASR."""
    y = np.r_[np.ones(len(member)), np.zeros(len(nonmember))]
    f, t, _ = metrics.roc_curve(y, np.r_[member, nonmember], drop_intermediate=False)
    i = np.searchsorted(f, fpr, side='right') - 1
    adv = float(t[i] - f[i])
    return dict(tpr=float(t[i]), fpr=float(f[i]), advantage=adv, asr=(1 + adv) / 2)


def attacks(rec, fpr=None):
    """Read a saved mia_losses.pt record and return metrics for each attack."""
    t = rec['target']
    rounds = sorted(rec['local'])
    if not rounds:
        raise ValueError(f'Client {t} has no recorded local-model losses.')
    tgt = [(-rec['local'][r][t][0].numpy(), -rec['local'][r][t][1].numpy()) for r in rounds]
    cdf_m, cdf_n = [], []
    for r, (m, n) in zip(rounds, tgt):
        shadow = [c for c in rec['local'][r] if c != t]
        if not shadow:
            raise ValueError('FedMIA-I requires at least two clients in each evaluated round.')
        sm = -np.stack([rec['local'][r][c][0].numpy() for c in shadow])
        sn = -np.stack([rec['local'][r][c][1].numpy() for c in shadow])
        cdf_m.append(scipy.stats.norm.cdf(m, sm.mean(0), np.sqrt(sm.var(0) + 1e-8)))
        cdf_n.append(scipy.stats.norm.cdf(n, sn.mean(0), np.sqrt(sn.var(0) + 1e-8)))
    g = sorted(rec['glob'])
    scores = dict(loss_series=(np.mean([m for m, _ in tgt], 0), np.mean([n for _, n in tgt], 0)),
                  fedmia_i=(np.mean(cdf_m, 0), np.mean(cdf_n, 0)),
                  blackbox_loss_global=(-rec['glob'][g[-1]][0].numpy(), -rec['glob'][g[-1]][1].numpy()),
                  loss_series_global=(-np.mean([rec['glob'][r][0].numpy() for r in g], 0),
                                      -np.mean([rec['glob'][r][1].numpy() for r in g], 0)))
    res = {}
    per_round = [roc_metrics(m, n) for m, n in tgt]
    best = int(np.argmax([x['advantage'] for x in per_round]))
    res['blackbox_loss'] = dict(per_round[best], round=rounds[best])
    if fpr is not None:
        per_round = [at_fpr(m, n, fpr) for m, n in tgt]
        best = int(np.argmax([x['advantage'] for x in per_round]))
        res['blackbox_loss']['at_fpr'] = dict(per_round[best], round=rounds[best])
    for name, (m, n) in scores.items():
        res[name] = roc_metrics(m, n)
        if fpr is not None:
            res[name]['at_fpr'] = at_fpr(m, n, fpr)
    res['fedmia_i']['rounds'] = rounds
    return res

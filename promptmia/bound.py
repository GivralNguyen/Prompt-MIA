"""Per-sample Gaussian race probabilities used in Figure 10."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from promptmia.attack import inject_batch

LOG10E = 1 / math.log(10)


def bounds(keys, Q, n_targets=100, N=4, delta_min=0.02, delta=0.05):
    keys, Q = keys.detach().double().cpu(), Q.detach().double().cpu()
    if not 1 <= n_targets <= len(Q) or len(Q) < 2:
        raise ValueError('Require at least two queries and 1 <= n_targets <= len(Q).')
    tidx = np.random.default_rng(42).choice(len(Q), n_targets, replace=False)
    pools, S = inject_batch(keys, Q[tidx], N, delta_min, delta, torch.Generator().manual_seed(42))
    Qn = F.normalize(Q, dim=1)
    log_b, emp, n_empty = [], [], []
    for t, pool, s in zip(tidx, pools, S):
        nm = Qn[torch.arange(len(Qn)) != t]                               # non-member queries, T excluded
        P = F.normalize(pool, dim=1)
        adv_mask = torch.zeros(len(P), dtype=torch.bool); adv_mask[s] = True
        A, B = P[adv_mask], P[~adv_mask]                                   # (N, D), (K, D)
        assign = (nm @ B.t()).argmax(1)
        cnt = torch.bincount(assign, minlength=len(B))
        sq = ((nm - B[assign]) ** 2).sum(1)
        sigma = torch.sqrt(torch.zeros(len(B), dtype=sq.dtype).index_add_(0, assign, sq) / cnt.clamp(min=1) / P.size(1))
        diff = A[:, None] - B[None]                                        # (N, K, D): k_aj - k_bl
        z = torch.einsum('jld,id->ijl', diff, B) / ((sigma[:, None, None] * diff.norm(dim=2)[None]).clamp(min=1e-15))
        per_cluster = torch.special.log_ndtr(z).amin(dim=(1, 2))           # log of min_{j,l} Phi(z_ijl), per i
        observed = cnt >= 1
        log_b.append(float(per_cluster[observed].max()) * LOG10E)
        n_empty.append(int((~observed).sum()))
        top = (nm @ P.t()).topk(N, dim=1).indices
        emp.append(float(adv_mask[top].all(1).double().mean()))
    return np.array(log_b), np.array(emp), np.array(n_empty)

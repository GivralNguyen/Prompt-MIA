"""Adversarial key construction and the batch membership game (Algorithms 1–2)."""
import torch
import torch.nn.functional as F


def _validate(keys, N):
    if keys.ndim != 2 or not 0 < N < keys.size(0):
        raise ValueError('Keys must have shape (M, D), with 0 < N < M.')
    if not torch.isfinite(keys).all() or (keys.norm(dim=-1) == 0).any():
        raise ValueError('Keys must be finite, nonzero vectors.')


def _directions(q, shape, gen):
    """Draw unit directions perpendicular to each unit query."""
    r = torch.rand(shape, generator=gen, dtype=q.dtype).to(q.device)
    r = F.normalize(r, dim=-1)
    o = r - (r * q).sum(-1, keepdim=True) * q
    # A coordinate axis provides a stable fallback for a parallel random vector.
    axis = F.one_hot(q.abs().argmin(-1), q.size(-1)).to(q.dtype)
    fallback = axis - (axis * q).sum(-1, keepdim=True) * q
    o = torch.where(o.norm(dim=-1, keepdim=True) < 1e-12, fallback, o)
    return F.normalize(o, dim=-1)


def gen_key_with_similarity(q, s, gen):
    """Algorithm 1: cosine similarity s, with the target query's norm."""
    if q.ndim != 1 or q.numel() < 2 or not torch.isfinite(q).all() or q.norm() == 0:
        raise ValueError('The target query must be a finite, nonzero vector of dimension at least 2.')
    if not -1 <= s <= 1:
        raise ValueError('Cosine similarity must lie in [-1, 1].')
    q_hat = F.normalize(q, dim=0)
    o = _directions(q_hat, q.shape, gen)
    return (s * q_hat + max(0.0, 1.0 - s * s) ** 0.5 * o) * q.norm()


def injection_slots(keys, q, N):
    """Replace the N keys nearest to the target query."""
    _validate(keys, N)
    return (F.normalize(keys, dim=1) @ F.normalize(q, dim=0)).topk(N).indices


def gen_adv_key_set(q, benign_keys, N, delta_min, delta, gen):
    """Sample Algorithm 2's similarities, capped at the cosine limit of one."""
    if delta_min < 0 or delta < 0:
        raise ValueError('Similarity margins must be nonnegative.')
    s_max = (F.normalize(benign_keys, dim=1) @ F.normalize(q, dim=0)).max().item()
    lo, hi = min(s_max + delta_min, 1.0), min(s_max + delta_min + delta, 1.0)
    s = lo + (hi - lo) * torch.rand(N, generator=gen, dtype=torch.float64)
    return torch.stack([gen_key_with_similarity(q, si.item(), gen) for si in s]), (lo, hi)


def promptmia_inject(keys, q, N, delta_min, delta, gen):
    """Return the modified pool, its injected indices, and the sampled interval."""
    S = injection_slots(keys, q, N)
    mask = torch.ones(keys.size(0), dtype=torch.bool, device=keys.device)
    mask[S] = False
    adv, interval = gen_adv_key_set(q.double(), keys[mask].double(), N, delta_min, delta, gen)
    out = keys.clone()
    out[S] = adv.to(keys.dtype)
    return out, S, interval


def naive_slots_for(keys, q, N, slots):
    if slots == 'last':
        return torch.arange(keys.size(0) - N, keys.size(0), device=keys.device)
    if slots == 'nearest':
        return injection_slots(keys, q, N)
    raise ValueError(f'Unknown injected slot rule: {slots}')


def naive_inject(keys, q, N, slots='last'):
    """Naive injection places copies of the target query in the chosen slots."""
    _validate(keys, N)
    S = naive_slots_for(keys, q, N, slots)
    out = keys.clone()
    out[S] = q.to(keys.dtype)
    return out, S


def inject_batch(keys, q, N, delta_min, delta, gen, naive=False, naive_slots='last', evade=None):
    """Construct one key pool per target: (M,D), (T,D) -> (T,M,D), (T,N)."""
    _validate(keys, N)
    if q.ndim != 2 or q.size(1) != keys.size(1) or (q.norm(dim=1) == 0).any():
        raise ValueError('Targets must be nonzero queries with the keys\' embedding dimension.')
    if delta_min < 0 or delta < 0:
        raise ValueError('Similarity margins must be nonnegative.')
    if evade not in (None, 'norm'):
        raise ValueError('The rescaling option is None or norm.')
    keys64, q64 = keys.double(), q.double()
    qn = F.normalize(q64, dim=1)
    sim = qn @ F.normalize(keys64, dim=1).t()
    S = sim.topk(N, dim=1).indices
    T, M, D = q.size(0), keys.size(0), keys.size(1)
    if naive:
        if naive_slots == 'last':
            S = torch.arange(M - N, M, device=q.device).expand(T, N)
        elif naive_slots != 'nearest':
            raise ValueError(f'Unknown injected slot rule: {naive_slots}')
        adv = q64[:, None, :].expand(T, N, D)
    else:
        s_max = sim.scatter(1, S, float('-inf')).max(dim=1).values
        # Cosine similarities cannot exceed one, including in margin sweeps.
        lo = (s_max + delta_min).clamp(max=1.0)
        hi = (s_max + delta_min + delta).clamp(max=1.0)
        u = torch.rand(T, N, generator=gen, dtype=torch.float64).to(q.device)
        s = lo[:, None] + (hi - lo)[:, None] * u
        o = _directions(qn[:, None, :], (T, N, D), gen)
        adv = (s[..., None] * qn[:, None] + (1 - s * s).clamp(min=0).sqrt()[..., None] * o)
        norm = keys64[S].norm(dim=2, keepdim=True) if evade == 'norm' else q64.norm(dim=1)[:, None, None]
        adv = adv * norm
    pools = keys64.expand(T, M, D).clone()
    pools.scatter_(1, S[..., None].expand(T, N, D), adv)
    return pools.to(keys.dtype), S


def all_injected_selected(pools, S, batch_q):
    """True when the union of a batch's selections contains every injected key."""
    N = S.size(1)
    sim = F.normalize(batch_q, dim=2) @ F.normalize(pools, dim=2).transpose(1, 2)
    selected = sim.topk(N, dim=2).indices.flatten(1)
    used = torch.zeros(pools.shape[:2], dtype=torch.bool, device=pools.device).scatter_(1, selected, True)
    return used.gather(1, S).all(dim=1)


@torch.no_grad()
def security_game(keys, train_q, test_q, N, batch_size, delta_min, delta, seed,
                  naive=False, naive_slots='last', member_idx=None, chunk=250, evade=None):
    """Evaluate member/nonmember decisions with independently sampled batches.

    A member batch includes its target and B-1 other training samples. A
    nonmember batch contains B training samples and excludes the test target.
    Fillers are sampled without replacement. TPR and FPR are both measured.
    """
    _validate(keys, N)
    n = len(train_q)
    if not 1 <= batch_size <= n or not len(test_q) or chunk < 1:
        raise ValueError('A batch must fit in the training population; test targets and chunk must be nonempty.')
    gen = torch.Generator().manual_seed(seed)
    g_dev = torch.Generator(device=train_q.device).manual_seed(seed)
    if member_idx is None:
        member_idx = torch.arange(n, device=train_q.device)
    member_idx = torch.as_tensor(member_idx, dtype=torch.long, device=train_q.device)
    if not len(member_idx):
        raise ValueError('At least one member target is required.')

    def rate(targets, member):
        hits = 0
        for start in range(0, len(targets), chunk):
            t = targets[start:start + chunk]
            pools, S = inject_batch(keys, t, N, delta_min, delta, gen, naive, naive_slots, evade)
            score = torch.rand(len(t), n, device=t.device, generator=g_dev)
            if member:
                score[torch.arange(len(t), device=t.device), member_idx[start:start + chunk]] = -1.0
                batch = torch.cat([t[:, None], train_q[score.topk(batch_size - 1, dim=1).indices]], dim=1)
            else:
                batch = train_q[score.topk(batch_size, dim=1).indices]
            hits += all_injected_selected(pools, S, batch).sum().item()
        return hits / len(targets)

    tpr = rate(train_q[member_idx], True)
    fpr = rate(test_q, False)
    return dict(tpr=tpr, fpr=fpr, advantage=tpr - fpr, asr=(1 + tpr - fpr) / 2)


@torch.no_grad()
def robust_inject(keys, t_clean, qs, N, delta_min, delta, gen, coverage=.95, grid=41):
    """Calibrate keys on K defended copies of every target (T,K,D)."""
    _validate(keys, N)
    if qs.ndim != 3 or qs.shape[0] != len(t_clean) or qs.shape[2] != keys.shape[1] or not qs.shape[1]:
        raise ValueError('Defended queries need shape (targets, copies, embedding dimension).')
    if not 0 < coverage <= 1 or grid < 2 or delta_min < 0 or delta < 0:
        raise ValueError('Require coverage in (0,1], at least two grid points, and nonnegative margins.')
    T, K, D = qs.shape
    device = qs.device
    normalized_keys = F.normalize(keys.double(), dim=1)
    normalized_queries = F.normalize(qs.double(), dim=2)
    center = F.normalize(normalized_queries.mean(1), dim=1)
    similarity = center @ normalized_keys.t()
    slots = similarity.topk(N, dim=1).indices
    best_benign = (normalized_queries @ normalized_keys.t()).scatter(
        2, slots[:, None, :].expand(-1, K, -1), float('-inf')).max(2).values
    lower = (similarity.scatter(1, slots, float('-inf')).max(1).values + delta_min).clamp(max=1.0)
    orthogonal = _directions(center[:, None], (T, N, D), gen)
    spread = torch.rand(T, N, generator=gen, dtype=torch.float64).to(device)
    center_similarity = (normalized_queries * center[:, None]).sum(-1)
    orthogonal_similarity = normalized_queries @ orthogonal.transpose(1, 2)
    def coverage_at(base):
        values = (base[:, None] + delta * spread).clamp(max=1.0)
        cosine = (values[:, None] * center_similarity[..., None] +
                  (1 - values * values).clamp(min=0).sqrt()[:, None] * orthogonal_similarity)
        return (cosine.min(2).values > best_benign).double().mean(1)
    chosen = torch.ones(T, dtype=torch.float64, device=device)      # fallback for targets no grid value can cover
    calibrated = torch.zeros(T, dtype=torch.bool, device=device)
    for fraction in torch.linspace(1, 0, grid, dtype=torch.float64).tolist():
        base = lower + (1 - lower) * fraction
        sufficient = coverage_at(base) >= coverage
        chosen = torch.where(sufficient, base, chosen)
        calibrated |= sufficient
    achieved = coverage_at(chosen)                                   # coverage actually attained at each chosen similarity
    values = (chosen[:, None] + delta * spread).clamp(max=1.0)
    adversarial = (values[..., None] * center[:, None] +
                   (1 - values * values).clamp(min=0).sqrt()[..., None] * orthogonal)
    adversarial *= t_clean.double().norm(dim=1)[:, None, None]
    pools = keys.double().expand(T, -1, D).clone()
    pools.scatter_(1, slots[..., None].expand(T, N, D), adversarial)
    return pools.to(keys.dtype), slots, chosen, calibrated, achieved

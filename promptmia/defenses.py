"""Client defenses: gradient sparsification, mixup, augmentation, and input noise."""
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

DEFENSES = ('none', 'sparse', 'mixup', 'augment', 'noise')


def input_noise_sigma(eps, delta, L=1.0):
    """Noise scale used for the input perturbation experiments."""
    if eps <= 0 or not 0 < delta < 1 or L <= 0:
        raise ValueError('Require epsilon > 0, 0 < delta < 1, and sensitivity > 0.')
    return math.sqrt(2 * math.log(1.25 / delta)) * L / eps


class Defense:
    def __init__(self, name, sparse_ratio=0.8, mixup_alpha=0.2, seed=0, device='cpu', noise_sigma=None):
        assert name in DEFENSES[1:], name
        assert name != 'noise' or noise_sigma is not None, 'noise needs noise_sigma'
        self.name, self.sparse_ratio, self.alpha, self.sigma = name, sparse_ratio, mixup_alpha, noise_sigma
        self.rng = np.random.default_rng(seed + 1)                     # mixup lambda
        self.gen = torch.Generator(device=device).manual_seed(seed + 1)
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.clean_inputs = name == 'sparse'                           # cached q(x) still valid

    def inputs(self, cache, ids):
        """(normalised images, labels) for a local step; mixup also keeps lam and perm for the loss."""
        if self.name == 'augment':
            stored = ids.to(cache.images.device)
            device = cache.device
            return self.augment(cache, cache.images[stored].to(device)), cache.labels[stored].to(device)
        x, y = cache.get(ids)
        if self.name == 'mixup':
            self.lam = float(self.rng.beta(self.alpha, self.alpha))
            self.perm = torch.randperm(x.size(0), device=x.device, generator=self.gen)
            x = self.lam * x + (1.0 - self.lam) * x[self.perm]
        elif self.name == 'noise':
            x = self.perturb(x)
        return x, y

    def perturb(self, x):
        """normalised images -> images plus fresh N(0, sigma^2) noise per pixel"""
        return x + self.sigma * torch.randn(x.shape, generator=self.gen, device=x.device, dtype=x.dtype)

    def loss(self, logits, y):
        if self.name == 'mixup':
            return self.lam * F.cross_entropy(logits, y) + (1.0 - self.lam) * F.cross_entropy(logits, y[self.perm])
        return F.cross_entropy(logits, y)

    def augment(self, cache, u8):
        """uint8 (B, 3, 256, 256) -> normalised (B, 3, 224, 224)."""
        B = u8.size(0)
        padded = F.pad(u8, (8, 8, 8, 8))                                       # zero padding before the random crop
        top = torch.randint(0, padded.size(2) - 224 + 1, (B,), generator=self.gen, device=u8.device).tolist()
        left = torch.randint(0, padded.size(3) - 224 + 1, (B,), generator=self.gen, device=u8.device).tolist()
        flip = (torch.rand(B, generator=self.gen, device=u8.device) < 0.5).tolist()
        out = []
        for i in range(B):
            img = padded[i, :, top[i]:top[i] + 224, left[i]:left[i] + 224]
            if flip[i]:
                img = img.flip(-1)
            out.append(self.jitter(img))                                       # uint8 in, uint8 out
        return cache.norm(torch.stack(out))

    @torch.no_grad()
    def after_backward(self, model):
        if self.name != 'sparse':
            return
        for p in model.parameters():
            if p.grad is None or p.grad.numel() <= 1:
                continue
            keep = int(p.grad.numel() * (1.0 - self.sparse_ratio))
            if keep <= 0:
                p.grad.zero_()
            elif keep < p.grad.numel():
                thr = p.grad.abs().reshape(-1).topk(keep).values[-1]
                p.grad.mul_((p.grad.abs() >= thr).to(p.grad.dtype))

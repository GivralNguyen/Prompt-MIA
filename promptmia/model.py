"""L2P-style prompt pool on a frozen ViT-B/32, DeiT-B/16 or ConViT-Base (paper Sec. 2, Eq. 1-2)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from promptmia.backbones import vit_cls_query


class PromptPool(nn.Module):
    """M (key, prompt) pairs; each input selects the N keys with the highest cosine similarity to q(x) (Eq. 1)."""

    def __init__(self, M=20, N=4, dim=768, patch=32):
        super().__init__()
        self.N = N
        if not 0 < N <= M:
            raise ValueError('Prompt selection size N must be between 1 and M.')
        val = math.sqrt(6.0 / (3 * patch * patch + dim))
        self.prompt = nn.Parameter(torch.empty(M, dim).uniform_(-val, val))
        self.key = nn.Parameter(torch.empty(M, dim).uniform_(-val, val))
        self.dropout = 0.0   # prompt dropout (Fig. 9): in training, each key is unavailable to each sample w.p. dropout
        # Optional selection frequencies are maintained separately for each client.
        self.freq = None

    def select(self, q):
        sim = F.normalize(q, dim=1) @ F.normalize(self.key, dim=1).t()  # (B, M) cosine similarity
        if self.training and self.dropout > 0:
            # a dropped key ranks below every available one (cosine >= -1); it is selected only if fewer than N keys remain
            sim = sim.masked_fill(torch.rand(sim.shape, device=sim.device) < self.dropout, -2.0)
        if self.training and self.freq is not None:
            h = 1.0 - self.freq / self.freq.max() if self.freq.sum() > 0 else torch.ones_like(self.freq)
            idx = (sim * h).topk(self.N, dim=1).indices
            self.freq.index_add_(0, idx.flatten(), torch.ones(idx.numel(), device=idx.device))
            top_sim = sim.gather(1, idx)
        else:
            top_sim, idx = sim.topk(self.N, dim=1)
        key_loss = (1.0 - top_sim).sum(dim=1).mean()                     # sum_i gamma(q(x), k_i), cosine distance
        return idx, key_loss


TIMM_NAMES = {'deit_b16': 'deit_base_patch16_224.fb_in1k', 'convit_base': 'convit_base.fb_in1k'}


class L2PViT(nn.Module):
    """Frozen backbone with the selected prompts inserted after the class token:
      vit_b32     : torchvision ViT-B/32; prompts after CLS, before the encoder.
      deit_b16    : timm DeiT-B/16; prompts after CLS (no positional embedding), before the blocks.
      convit_base : timm ConViT-Base; CLS and prompts join at the transition from GPSA to MHSA blocks.

    head='cls'    : default. ViT-B/32: CLS -> frozen ImageNet head -> Linear-Dropout-Linear (the head behind the
                    classifier); DeiT/ConViT: normalised CLS -> Linear-Dropout-Linear.
    head='prompt' : classifier on the mean of the N prompt-token hidden states.
    """

    def __init__(self, num_classes, M=20, N=4, head='cls', backbone='vit_b32', pretrained=True):
        super().__init__()
        self.backbone = backbone
        if backbone == 'vit_b32':
            self.vit = torchvision.models.vit_b_32(weights='IMAGENET1K_V1' if pretrained else None)
            self.vit.requires_grad_(False)
            dim, cls_in = self.vit.hidden_dim, 1000
        else:
            import timm
            self.net = timm.create_model(TIMM_NAMES[backbone], pretrained=pretrained, num_classes=0, global_pool='')
            self.net.requires_grad_(False)
            dim = cls_in = self.net.embed_dim
        self.pool = PromptPool(M, N, dim)
        self.head_type = head
        if head == 'prompt':
            self.head = nn.Linear(dim, num_classes)
        elif head == 'cls':
            self.head = nn.Sequential(nn.Linear(cls_in, 512), nn.Dropout(0.5), nn.Linear(512, num_classes))
        else:
            raise ValueError(head)

    def frozen(self):
        return self.vit if self.backbone == 'vit_b32' else self.net

    def train(self, mode=True):
        super().train(mode)
        self.frozen().eval()  # frozen backbone always in eval mode
        return self

    @torch.no_grad()
    def query(self, x):
        """q(x): CLS output of the frozen, prompt-free backbone."""
        if self.backbone == 'vit_b32':
            return vit_cls_query(self.vit, x)
        return self.net.forward_features(x)[:, 0]

    def encode(self, x, prompts):
        """Hidden states of the backbone with `prompts` (B, n, D) right after CLS; n = 0 gives the plain backbone."""
        if self.backbone == 'vit_b32':
            tok = self.vit._process_input(x)
            tok = torch.cat([self.vit.class_token.expand(tok.size(0), -1, -1), tok], dim=1)
            tok = tok + self.vit.encoder.pos_embedding
            h = torch.cat([tok[:, :1], prompts, tok[:, 1:]], dim=1)          # no positional embedding on prompts
            return self.vit.encoder.ln(self.vit.encoder.layers(h))
        net = self.net
        if self.backbone == 'deit_b16':
            h = net.norm_pre(net.patch_drop(net._pos_embed(net.patch_embed(x))))   # [CLS] + patches, with pos
            h = torch.cat([h[:, :1], prompts, h[:, 1:]], dim=1)
            return net.norm(net.blocks(h))
        h = net.patch_embed(x)                                                     # convit_base
        if net.use_pos_embed:
            h = h + net.pos_embed
        h = net.pos_drop(h)
        cls = net.cls_token.expand(h.size(0), -1, -1)
        for u, blk in enumerate(net.blocks):
            if u == net.local_up_to_layer:
                h = torch.cat([cls, prompts, h], dim=1)
            h = blk(h)
        return net.norm(h)

    def forward(self, x, q=None):
        """Returns (logits, key_loss, selected indices). Pass a precomputed q(x) to skip the query pass."""
        if q is None:
            q = self.query(x)
        idx, key_loss = self.pool.select(q)
        h = self.encode(x, self.pool.prompt[idx])                            # prompts (B, N, D)
        if self.head_type == 'prompt':
            logits = self.head(h[:, 1:1 + self.pool.N].mean(dim=1))
        elif self.backbone == 'vit_b32':
            logits = self.head(self.vit.heads(h[:, 0]))
        else:
            logits = self.head(h[:, 0])
        return logits, key_loss, idx

    def trainable_state(self):
        """Clones of every trainable tensor: the pool and the head (what a client sends to the server)."""
        return {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        params = dict(self.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                if params[n].shape != v.shape:
                    if n not in ('pool.key', 'pool.prompt'):
                        raise ValueError(f'Unexpected state shape for {n}: {v.shape}')
                    module, attr = n.rsplit('.', 1)
                    setattr(self.get_submodule(module), attr, nn.Parameter(v.detach().clone()))
                else:
                    params[n].copy_(v)

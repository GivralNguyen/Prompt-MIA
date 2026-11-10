"""Frozen backbones and their query function q(x) = CLS output of the prompt-free model (Sec. 2)."""
import torch
import torch.nn as nn
import torchvision

BACKBONES = ('vit_b32', 'deit_b16', 'convit_base')


def vit_cls_query(vit, x):
    """torchvision ViT: CLS token after the encoder's final LayerNorm."""
    tok = vit._process_input(x)
    tok = torch.cat([vit.class_token.expand(tok.size(0), -1, -1), tok], dim=1)
    return vit.encoder(tok)[:, 0]


class QueryModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == 'vit_b32':
            self.net = torchvision.models.vit_b_32(weights='IMAGENET1K_V1')
        else:
            import timm
            self.net = timm.create_model({'deit_b16': 'deit_base_patch16_224.fb_in1k',
                                          'convit_base': 'convit_base.fb_in1k'}[name], pretrained=True)
        self.net.requires_grad_(False).eval()

    @torch.no_grad()
    def query(self, x):
        if self.name == 'vit_b32':
            return vit_cls_query(self.net, x)
        return self.net.forward_features(x)[:, 0]  # timm: CLS after the final norm (ConViT adds CLS mid-network)

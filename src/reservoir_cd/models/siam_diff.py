"""Baseline B: torchgeo FC-Siam-diff, per-date 4-class segmentation.

FCSiamDiff encodes both dates with shared weights and decodes the *difference* of the encoder
features, producing one map. We ask it for ``T * n_classes`` channels and split them into per-date
class logits [B, T, C, H, W]; the change target is derived from the per-date argmaxes.
"""
import torch
from torch import nn


class SiamDiffSeg(nn.Module):
    def __init__(self, in_channels: int = 6, n_classes: int = 4, t: int = 2, encoder_weights=None):
        super().__init__()
        from torchgeo.models import FCSiamDiff
        self.t, self.c = t, n_classes
        self.net = FCSiamDiff(in_channels=in_channels, classes=t * n_classes, t=t, encoder_weights=encoder_weights)

    def forward(self, image: torch.Tensor, **_) -> torch.Tensor:
        """image [B, T, C, H, W] -> logits [B, T, n_classes, H, W]."""
        out = self.net(image)
        return out.reshape(out.shape[0], self.t, self.c, *out.shape[-2:])


def build_siam_diff(cfg: dict) -> nn.Module:
    return SiamDiffSeg(cfg.get("in_channels", 6), cfg.get("n_classes", 4), cfg.get("t", 2), cfg.get("encoder_weights"))

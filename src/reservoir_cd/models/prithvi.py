"""Baselines C / D (and the E4 variants): Prithvi-EO-2.0 via the TerraTorch encoder-decoder factory.

Backbone: ``prithvi_eo_v2_300_tl`` (temporal + location coords), T = 2 frames, 6 HLS bands, tokens
of both dates -> ``SelectIndices`` -> ``ReshapeTokensToImage(effective_time_dim=T)`` (time folded
into channels) -> ``LearnedInterpolateToPyramidal`` -> UPerNet. The head outputs ``T * n_classes``
channels that are reshaped to per-date logits [B, T, C, H, W]; the change map is derived from the
two per-date predictions (``change_from_logits``).

Adaptation mode (E4) is selected purely by config (``configs/train_{C,D}.yaml`` -> ``model.adaptation``):
  mode: frozen              backbone frozen (Baseline C)
  mode: lora, r: 4|8|16     LoRA on attention q/k/v/proj, alpha = ``alpha`` (default 2r) (Baseline D = r 8, alpha 16)
  mode: full                full fine-tuning of the backbone
The decoder / necks / head are always trained. TerraTorch is imported lazily.
"""
import torch
from torch import nn

HLS_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
LORA_TARGET_REGEX = r".*blocks\.\d+\.attn\.(qkv\.(q_linear|k_linear|v_linear)|proj)"


class PrithviSeg(nn.Module):
    def __init__(self, backbone: str = "prithvi_eo_v2_300_tl", pretrained: bool = False, t: int = 2,
                 n_classes: int = 4, img_size: int = 256, decoder_channels: int = 256,
                 adaptation: dict | None = None, out_indices: tuple[int, ...] = (5, 11, 17, 23),
                 num_layers_hint: int | None = None):
        super().__init__()
        from terratorch.models import EncoderDecoderFactory
        ad = dict(adaptation or {"mode": "frozen"})
        self.t, self.c, self.mode = t, n_classes, ad["mode"]
        peft = None
        if self.mode == "lora":
            r = int(ad.get("r", 8))
            peft = {"method": "LORA", "replace_qkv": "qkv", "peft_config_kwargs": {
                "r": r, "lora_alpha": int(ad.get("alpha", 2 * r)), "lora_dropout": float(ad.get("dropout", 0.05)),
                "target_modules": ad.get("target_modules", LORA_TARGET_REGEX), "bias": "none"}}
        elif self.mode not in ("frozen", "full"):
            raise ValueError(f"unknown adaptation mode {self.mode}")
        self.model = EncoderDecoderFactory().build_model(
            task="segmentation", backbone=backbone, decoder="UperNetDecoder",
            backbone_kwargs=dict(pretrained=pretrained, num_frames=t, bands=HLS_BANDS, img_size=img_size,
                                 coords_encoding=["time", "location"]),
            decoder_kwargs=dict(channels=decoder_channels),
            necks=[{"name": "SelectIndices", "indices": list(out_indices)},
                   {"name": "ReshapeTokensToImage", "effective_time_dim": t},
                   {"name": "LearnedInterpolateToPyramidal"}],
            num_classes=t * n_classes, head_kwargs=dict(dropout=0.1), rescale=True, peft_config=peft)
        if self.mode == "frozen":
            for p in self.model.encoder.parameters():
                p.requires_grad = False

    def forward(self, image: torch.Tensor, temporal_coords: torch.Tensor | None = None,
                location_coords: torch.Tensor | None = None, **_) -> torch.Tensor:
        """image [B, T, C, H, W] -> per-date logits [B, T, n_classes, H, W]."""
        x = image.permute(0, 2, 1, 3, 4).contiguous()  # TerraTorch wants [B, C, T, H, W]
        kw = {}
        if temporal_coords is not None:
            kw["temporal_coords"] = temporal_coords
        if location_coords is not None:
            kw["location_coords"] = location_coords
        out = self.model(x, **kw)
        out = out.output if hasattr(out, "output") else out
        return out.reshape(out.shape[0], self.t, self.c, *out.shape[-2:])

    def trainable_summary(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": tot, "trainable": tr, "trainable_pct": 100.0 * tr / tot}


def build_prithvi(cfg: dict) -> nn.Module:
    keys = ("backbone", "pretrained", "t", "n_classes", "img_size", "decoder_channels", "adaptation", "out_indices")
    kw = {k: cfg[k] for k in keys if k in cfg}
    if "out_indices" in kw:
        kw["out_indices"] = tuple(kw["out_indices"])
    return PrithviSeg(**kw)


def change_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-date logits [B, 2, C, H, W] -> water-change map [B, H, W]: 0 none, 1 loss, 2 gain (Water = class 0)."""
    pred = logits.argmax(2)
    w1, w2 = pred[:, 0] == 0, pred[:, 1] == 0
    ch = torch.zeros_like(pred[:, 0])
    ch[w1 & ~w2] = 1
    ch[~w1 & w2] = 2
    return ch

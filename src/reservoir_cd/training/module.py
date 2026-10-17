"""Lightning module + datamodule for baselines B / C / D (per-date 4-class segmentation)."""
import lightning as L
import torch
from torch.utils.data import DataLoader

from reservoir_cd.datasets.torch_dataset import PrithviReservoirDataset
from reservoir_cd.models.losses import IGNORE_INDEX, per_date_ce
from reservoir_cd.models.prithvi import build_prithvi, change_from_logits
from reservoir_cd.models.siam_diff import build_siam_diff

N_CLASSES = 4


def build_model(mcfg: dict) -> torch.nn.Module:
    name = mcfg["name"]
    if name == "siam_diff":
        return build_siam_diff(mcfg)
    if name == "prithvi":
        return build_prithvi(mcfg)
    raise ValueError(name)


class ChangeModule(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg
        self.model = build_model(cfg["model"])
        w = cfg.get("loss", {}).get("class_weights")
        self.register_buffer("class_weights", torch.tensor(w, dtype=torch.float32) if w else None)
        self.strict_loading = False  # checkpoints may omit frozen (pretrained) backbone weights
        self.trainable_only = cfg.get("trainer", {}).get("save_trainable_only", True)
        self._reset_stats()

    def on_save_checkpoint(self, checkpoint):
        """Drop frozen parameters (pretrained backbone / LoRA base weights): ~1.3 GB -> a few hundred MB.

        They are re-created from the pretrained weights when the model is rebuilt, so resume and
        inference stay exact. Buffers (BatchNorm statistics etc.) and all trainable weights are kept.
        """
        if not self.trainable_only:
            return
        frozen = {n for n, p in self.named_parameters() if not p.requires_grad}
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k not in frozen}

    def forward(self, batch):
        return self.model(batch["image"], temporal_coords=batch.get("temporal_coords"),
                          location_coords=batch.get("location_coords"))

    def training_step(self, batch, _):
        loss = per_date_ce(self(batch), batch["mask"], self.class_weights)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch["image"].shape[0])
        return loss

    def _reset_stats(self):
        self.conf = None
        self.chg = torch.zeros(3, 3, dtype=torch.long)  # [target, pred]

    def validation_step(self, batch, _):
        logits = self(batch)
        loss = per_date_ce(logits, batch["mask"], self.class_weights)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch["image"].shape[0])
        pred, tgt = logits.argmax(2), batch["mask"]
        ok = tgt != IGNORE_INDEX
        k = N_CLASSES
        c = torch.bincount((tgt[ok] * k + pred[ok]).flatten(), minlength=k * k).reshape(k, k).cpu()
        self.conf = c if self.conf is None else self.conf + c
        ct, ch = batch["change"], change_from_logits(logits)
        ok = ct != IGNORE_INDEX
        self.chg += torch.bincount((ct[ok] * 3 + ch[ok]).flatten(), minlength=9).reshape(3, 3).cpu()

    def on_validation_epoch_end(self):
        if self.conf is not None:
            tp = self.conf.diag().float()
            iou = tp / (self.conf.sum(0) + self.conf.sum(1) - tp).clamp(min=1)
            self.log("val_miou", iou.mean(), prog_bar=True)
            self.log("val_water_iou", iou[0])
            self.log("val_acc", tp.sum() / self.conf.sum().clamp(min=1))
            tpc = self.chg.diag().float()[1:]
            f1 = 2 * tpc / (self.chg.sum(0)[1:] + self.chg.sum(1)[1:]).clamp(min=1)
            self.log("val_change_f1", f1.mean(), prog_bar=True)
        self._reset_stats()

    def configure_optimizers(self):
        o = self.cfg["optim"]
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=o["lr"], weight_decay=o.get("weight_decay", 0.01))
        if o.get("cosine_epochs"):
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=o["cosine_epochs"])
            return {"optimizer": opt, "lr_scheduler": sch}
        return opt


class ChangeDataModule(L.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def _ds(self, split, augment=False):
        d = self.cfg["data"]
        col = f"loso_{d['loso_holdout']}" if d.get("loso_holdout") else "split"
        return PrithviReservoirDataset(
            d["chips_dir"], split=split, split_column=col, label_source=d.get("label_source", "consensus"),
            exclusion_radius_m=d.get("exclusion_radius_m", 150.0), ground_truth_dir=d.get("ground_truth_dir", "."),
            augment=augment)

    def _dl(self, split, shuffle, augment=False):
        d = self.cfg["data"]
        return DataLoader(self._ds(split, augment), batch_size=d["batch_size"], shuffle=shuffle,
                          num_workers=d.get("num_workers", 2), drop_last=shuffle,
                          persistent_workers=d.get("num_workers", 2) > 0)

    def train_dataloader(self):
        return self._dl("train", True, self.cfg["data"].get("augment", True))

    def val_dataloader(self):
        return self._dl("val", False)

    def test_dataloader(self):
        return self._dl("test", False)

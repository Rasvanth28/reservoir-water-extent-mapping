"""Train a baseline from a YAML config. Checkpoints every epoch; auto-resume from last.ckpt.

Usage (local smoke test: random-init weights, 2 steps):
  .venv/bin/python -m reservoir_cd.training.train --config configs/train_D.yaml \
      --set model.pretrained=false --limit-train-batches 2 --limit-val-batches 1 --max-epochs 1 \
      --output-dir /tmp/run
Colab: --output-dir /content/drive/MyDrive/reservoir_cd/runs/<name> (checkpoints survive disconnects;
rerunning the same command resumes from <output-dir>/checkpoints/last.ckpt).
"""
import argparse
import copy
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from reservoir_cd.training.module import ChangeDataModule, ChangeModule


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for s in sets or []:
        k, v = s.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(v)
    return cfg


def run(cfg: dict, output_dir: str, limit_train_batches=None, limit_val_batches=None, max_epochs=None,
        accelerator: str = "auto") -> Path:
    t = cfg["trainer"]
    L.seed_everything(cfg["seed"], workers=True)
    out = Path(output_dir)
    ck = ModelCheckpoint(dirpath=out / "checkpoints", filename="epoch{epoch:03d}-{val_loss:.4f}", monitor="val_loss",
                         save_top_k=1, save_last=True, every_n_epochs=1, auto_insert_metric_name=False)
    ck.CHECKPOINT_NAME_LAST = "last"
    callbacks = [ck, EarlyStopping("val_loss", patience=t.get("patience", 10), mode="min")]
    loggers = [CSVLogger(out, name="logs", version=""), TensorBoardLogger(out, name="tb", version="")]
    trainer = L.Trainer(
        max_epochs=max_epochs or t["max_epochs"], accelerator=accelerator, devices=1,
        precision=t.get("precision", "32-true"), callbacks=callbacks, logger=loggers,
        deterministic=False, log_every_n_steps=t.get("log_every_n_steps", 5),
        limit_train_batches=limit_train_batches or 1.0, limit_val_batches=limit_val_batches or 1.0,
        gradient_clip_val=t.get("gradient_clip_val"), accumulate_grad_batches=t.get("accumulate_grad_batches", 1),
        enable_progress_bar=True, default_root_dir=out)
    last = out / "checkpoints" / "last.ckpt"
    (out).mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    trainer.fit(ChangeModule(cfg), ChangeDataModule(cfg), ckpt_path=str(last) if last.exists() else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. model.adaptation.r=16")
    ap.add_argument("--limit-train-batches", type=int)
    ap.add_argument("--limit-val-batches", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--accelerator", default="auto")
    a = ap.parse_args()
    cfg = apply_overrides(yaml.safe_load(open(a.config)), a.set)
    run(cfg, a.output_dir or cfg["output_dir"], a.limit_train_batches, a.limit_val_batches, a.max_epochs, a.accelerator)


if __name__ == "__main__":
    main()

"""Pretrain the Mel-MAE feature encoder used by the drift kernel."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from .config import load_config, save_config
from .data import MelDataset, collate
from .models.mae import MelMAE
from .utils import EMA, count_params, infinite, lr_lambda, random_crop, save_checkpoint, seed_everything


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/mae.yaml")
    p.add_argument("--workdir", default="runs/mae")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.steps=1000")


def build_mae(cfg, num_classes: int = 0):
    """``model.type: 1d`` (mel bins as channels) or ``2d`` (ResNet over frequency x time)."""
    m = cfg.model
    n_cls = num_classes if m.get("speaker_head", True) else 0
    if m.get("type", "1d") == "2d":
        from .models.mae2d import MelMAE2d

        return MelMAE2d(n_mels=m.n_mels, base_channels=m.base_channels, layers=tuple(m.layers),
                        input_patch=tuple(m.input_patch), num_classes=n_cls, mask_patch=tuple(m.mask_patch))
    return MelMAE(n_mels=m.n_mels, base_channels=m.base_channels, layers=tuple(m.layers), kernel=m.kernel,
                  num_classes=n_cls, mask_patch=m.mask_patch)


def load_mae(path: str | Path, device="cpu"):
    """Load the EMA weights of a pretrained Mel-MAE (frozen, eval mode)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    from .config import Config

    model = build_mae(Config(ckpt["config"]), ckpt.get("num_classes", 0))
    model.load_state_dict(ckpt["ema"])
    return model.to(device).eval().requires_grad_(False)


def run(args) -> None:
    from torch.utils.tensorboard import SummaryWriter

    cfg = load_config(args.config, args.overrides)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    save_config(cfg, work / "config.yaml")
    seed_everything(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() and not cfg.train.get("cpu", False) else "cpu"
    tc = cfg.train

    ds = MelDataset(cfg.data.root, "train", min_quality=cfg.data.min_quality, min_frames=tc.crop_frames,
                    max_frames=cfg.data.max_frames, filters=cfg.data.get("filters"))
    loader = DataLoader(ds, batch_size=tc.batch_size, sampler=RandomSampler(ds, replacement=True, num_samples=10**9),
                        collate_fn=collate, num_workers=tc.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=tc.num_workers > 0)
    model = build_mae(cfg, ds.num_speakers).to(device)
    print(f"Mel-MAE: {count_params(model):.2f}M params, {len(ds)} training utterances")
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, betas=(0.9, 0.95), weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc.warmup, tc.steps, tc.schedule))
    ema = EMA(model, tc.ema_decay)
    writer = SummaryWriter(work / "tb")

    step, t0 = 0, time.time()
    ckpt_path = work / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed from step {step}")

    it = infinite(loader)
    while step < tc.steps:
        batch = next(it)
        mel, _ = random_crop(batch["mel"], batch["mel_len"], tc.crop_frames)
        mel, spk = mel.to(device, non_blocking=True), batch["spk"].to(device)
        cls_w = tc.cls_weight if step >= tc.steps - tc.cls_finetune_steps else 0.0
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss, metrics = model(mel, mask_ratio=tc.mask_ratio, labels=spk, cls_weight=cls_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        sched.step()
        ema.update(model)
        step += 1
        if step % tc.log_every == 0:
            metrics = {k: float(v) for k, v in metrics.items()}
            for k, v in metrics.items():
                writer.add_scalar(f"mae/{k}", v, step)
            writer.add_scalar("mae/grad_norm", float(gnorm), step)
            writer.add_scalar("mae/lr", sched.get_last_lr()[0], step)
            rate = tc.log_every / (time.time() - t0)
            t0 = time.time()
            print(f"step {step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + f" ({rate:.1f} it/s)",
                  flush=True)
        if step % tc.save_every == 0 or step == tc.steps:
            state = dict(model=model.state_dict(), ema=ema.model.state_dict(), opt=opt.state_dict(),
                         sched=sched.state_dict(), step=step, config=cfg.to_dict(), num_classes=ds.num_speakers)
            save_checkpoint(ckpt_path, **state)
    save_checkpoint(work / "mae_ema.pt", ema=ema.model.state_dict(), config=cfg.to_dict(), num_classes=ds.num_speakers,
                    step=step)
    print(f"done: {work / 'mae_ema.pt'}")

"""Probe the SLM adversary (``slm:`` config block) at the real batch, without writing checkpoints.

    python scripts/slm_probe.py --config configs/tts_v3_slm.yaml --steps 40 slm.crops_per_cond=2
    python scripts/slm_probe.py --config configs/tts_v3_slm.yaml --grad-norms --disc-steps 50 --batches 4

The model is built as ``drifting-tts train`` builds it (``train.init_from``, learned tau, compiled generator, EMA).
``--steps``: full training steps (generator, encoder and discriminator updates): peak CUDA memory and the step rate
after ``--warmup`` steps. ``--grad-norms``: at the initial model, gradient norms of the drift term (with the prior,
duration and pitch terms on the encoder), the LSGAN term and the WavLM feature-matching term, on the generator and
on the encoder, first with a fresh discriminator, then after ``--disc-steps`` discriminator-only steps. Prints one
JSON line per measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from drifting_tts.config import load_config
from drifting_tts.models.tts import DriftingTTS
from drifting_tts.slm import build_slm, slm_enabled
from drifting_tts.train import CropBank, build_loader, build_taus, grow_speakers, training_step
from drifting_tts.train_mae import load_mae
from drifting_tts.utils import EMA, infinite, lr_lambda, seed_everything


def build(cfg, device):
    tc = cfg.train
    seed_everything(tc.seed)
    ds, loader = build_loader(cfg)
    model = DriftingTTS(cfg.model, num_speakers=ds.num_speakers, n_mels=ds.dim).to(device)
    if model.pitch_enabled:
        st = json.loads((Path(cfg.data.root) / "stats.json").read_text())
        model.lf0_stats.copy_(torch.tensor([st["lf0_mean"], st["lf0_std"]]))
    mae = load_mae(cfg.mae.path, device)
    if cfg.drift.get("mode") != "kyutai":
        raise ValueError("the probe follows the v3 recipe: drift.mode kyutai")
    taus = build_taus(mae, cfg, device)
    if tc.get("init_from"):
        init = torch.load(tc.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(grow_speakers(init.get("ema", init.get("model")), model.state_dict()), strict=False)
        if init.get("taus") and set(init["taus"]) == set(taus.state_dict()):
            taus.load_state_dict(init["taus"])
    groups = [{"params": list(model.parameters())},
              {"params": list(taus.parameters()), "lr": cfg.drift.kyutai.get("tau_lr") or tc.lr, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=tc.lr, betas=(0.9, 0.95), weight_decay=tc.weight_decay,
                            fused=True if device == "cuda" else None)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc.warmup, tc.steps, tc.schedule))
    ema = EMA(model, tc.ema_decay)
    if tc.get("compile", False):
        model.generator.compile()
    mae_bank = CropBank(cfg.drift.uncond_bank, ds.dim, cfg.drift.crop_frames, device)
    slm = build_slm(cfg, ds.stats, ds.backend, device) if slm_enabled(cfg) else None
    return model, mae, taus, opt, sched, ema, mae_bank, slm, infinite(loader)


def sync(device) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def train_steps(args, cfg, device) -> dict:
    model, mae, taus, opt, sched, ema, bank, slm, it = build(cfg, device)
    tc = cfg.train
    model.train()
    cuda = device == "cuda"
    sync(device)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() if cuda else 0
    t0, metrics = None, []
    for step in range(args.steps):
        if step == args.warmup:
            sync(device)
            t0 = time.time()
        loss, m, _ = training_step(model, mae, next(it), bank, cfg, device, taus=taus, slm=slm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), tc.grad_clip)
        torch.nn.utils.clip_grad_norm_(model.generator.parameters(), tc.grad_clip)
        torch.nn.utils.clip_grad_norm_(taus.parameters(), tc.grad_clip)
        opt.step()
        sched.step()
        ema.update(model)
        metrics.append({k: float(v) for k, v in m.items()})
    sync(device)
    rate = (args.steps - args.warmup) / (time.time() - t0)
    n = min(10, len(metrics))
    last = {k: sum(d.get(k, 0.0) for d in metrics[-n:]) / n for k in metrics[-1]}
    gb = lambda b: round(b / 2**30, 2)  # noqa: E731
    return {"what": "steps", "slm": slm_enabled(cfg), "crops_per_cond": cfg.slm.crops_per_cond if slm else 0,
            "alpha": cfg.slm.get("alpha") if slm else None, "batch": tc.batch_size, "G": cfg.drift.gen_per_cond,
            "it_s": round(rate, 3), "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()) if cuda else None,
            "peak_reserved_gb": gb(torch.cuda.max_memory_reserved()) if cuda else None, "state_gb": gb(base),
            "last10": {k: round(v, 4) for k, v in last.items()}}


def grad_norms(model, mae, taus, bank, slm, it, cfg, device, n: int) -> list[dict]:
    gen = [p for p in model.generator.parameters() if p.requires_grad]
    enc = [p for p in model.parameters() if p.requires_grad and not any(p is q for q in gen)]
    out, seen = [], {}
    generate, step = model.generate, slm.step

    def spy_generate(*a, **kw):  # keep the generated crops (drift samples first) to measure gradients w.r.t. them
        seen["x"] = generate(*a, **kw)
        return seen["x"]

    def spy_step(fake, *a, **kw):
        seen["fake"] = fake
        return step(fake, *a, **kw)

    model.generate, slm.step = spy_generate, spy_step
    for _ in range(n):
        loss, m, info = training_step(model, mae, next(it), bank, cfg, device, taus=taus, slm=slm)
        t = info["slm"]
        rest = loss - slm.weight * t["adv"] - slm.fm_weight * t["fm"]  # drift (+ prior, duration, pitch on the encoder)
        row = {"d_real": float(m["slm_d_real"]), "d_fake": float(m["slm_d_fake"]), "adv": float(t["adv"]),
               "fm": float(t["fm"])}
        # gradient norms w.r.t. the generated crops: drift on the drift samples, the SLM terms on the judged crops
        row["x_drift"] = float(torch.autograd.grad(rest, seen["x"], retain_graph=True)[0].float().norm())
        for k in ("adv", "fm"):
            row[f"x_{k}"] = float(torch.autograd.grad(t[k], seen["fake"], retain_graph=True)[0].norm())
        for name, params in (("gen", gen), ("enc", enc)):
            grads = {}
            for k, term in (("drift", rest), ("adv", t["adv"]), ("fm", t["fm"])):
                g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                grads[k] = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten().float()
                                      for x, p in zip(g, params)])
            for k, v in grads.items():
                row[f"{name}_{k}"] = float(v.norm())
            for k in ("adv", "fm"):
                row[f"{name}_cos_drift_{k}"] = float(torch.nn.functional.cosine_similarity(grads["drift"], grads[k], 0))
        out.append(row)
        del loss, info, t, rest
        seen.clear()
    model.generate, slm.step = generate, step
    return out


def probe_grad_norms(args, cfg, device) -> None:
    model, mae, taus, _, _, _, bank, slm, it = build(cfg, device)
    model.train()
    slm.disc_warmup, slm.separate_terms, slm.grad_clip = 0, True, 0.0
    rows = grad_norms(model, mae, taus, bank, slm, it, cfg, device, args.batches)
    print(json.dumps({"what": "grad_norms", "disc_steps": slm.steps - args.batches, "rows": rows}), flush=True)
    d = []
    with torch.no_grad():  # discriminator-only steps: the generator is not updated
        for _ in range(args.disc_steps):
            _, m, _ = training_step(model, mae, next(it), bank, cfg, device, taus=taus, slm=slm)
            d.append((float(m["slm_disc"]), float(m["slm_d_real"]), float(m["slm_d_fake"])))
    print(json.dumps({"what": "disc_only", "steps": args.disc_steps, "disc_loss_first_last": [d[0], d[-1]]}),
          flush=True)
    rows = grad_norms(model, mae, taus, bank, slm, it, cfg, device, args.batches)
    print(json.dumps({"what": "grad_norms", "disc_steps": slm.steps - args.batches, "rows": rows}), flush=True)


def slm_only(args, cfg, device) -> dict:
    """The adversary alone at the real crop count: ``batch_size`` real and ``batch_size * crops_per_cond`` generated
    crops (here: noisy real mels), one discriminator update and one backward to the mels."""
    from drifting_tts.data import MelDataset
    from drifting_tts.slm import audio_segments

    L, B, c = cfg.drift.crop_frames, cfg.train.batch_size, cfg.slm.crops_per_cond
    ds = MelDataset(cfg.data.root, "val", min_frames=L, max_frames=10**9, with_audio=True)
    items = [ds[i % len(ds)] for i in range(B)]
    mel = torch.stack([it["mel"][:, :L] for it in items]).to(device)
    audio = audio_segments(torch.stack([it["audio"][: L * 256] for it in items]).to(device),
                           torch.zeros(B, dtype=torch.long, device=device), L)
    slm = build_slm(cfg, ds.stats, ds.backend, device)
    slm.disc_warmup = 0
    sync(device)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    for _ in range(args.steps):
        fake = mel.repeat_interleave(c, 0) + 0.3 * torch.randn(B * c, *mel.shape[1:], device=device)
        fake.requires_grad_(True)
        sync(device)
        t = time.time()
        loss, _, _ = slm.step(fake, mel, audio)
        loss.backward()
        sync(device)
        times.append(time.time() - t)
    ms = 1000 * sum(times[args.warmup:]) / max(1, len(times) - args.warmup)
    return {"what": "slm_only", "real": B, "fake": B * c, "dtype": cfg.slm.get("dtype"), "ms_per_step": round(ms, 1),
            "extra_peak_gb": round((torch.cuda.max_memory_allocated() - base) / 2**30, 2),
            "state_gb": round(base / 2**30, 2)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--config", default="configs/tts_v3_slm.yaml")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=15, help="steps (compilation) before timing")
    p.add_argument("--grad-norms", action="store_true")
    p.add_argument("--slm-only", action="store_true", help="time and memory of the adversary alone")
    p.add_argument("--disc-steps", type=int, default=50)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.grad_norms:
        probe_grad_norms(args, cfg, args.device)
    elif args.slm_only:
        print(json.dumps(slm_only(args, cfg, args.device)), flush=True)
    else:
        print(json.dumps(train_steps(args, cfg, args.device)), flush=True)


if __name__ == "__main__":
    main()

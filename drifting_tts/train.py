"""Train the one-step drifting TTS model.

One optimisation step (cf. Algorithm 1 of the paper and ``train_step`` of the official code):

1. encode text, align it to the target mel with MAS, compute prior / duration losses;
2. crop the same random window from the target mel and from the aligned condition;
3. draw ``G`` generator samples per condition (one forward pass, no iteration);
4. positives  = the target crop and ``P - 1`` slightly perturbed views of it;
   negatives  = the other generated samples of the same condition (self masked) and, for
   training-time CFG, ``N`` real crops of *other* utterances from an unconditional memory bank
   (drawn before the batch's own crops are added), weighted by ``(alpha - 1)(G - 1) / N`` where
   ``alpha`` is sampled per condition and fed to the generator;
5. the drifting loss is computed in the multi-scale feature space of the frozen Mel-MAE and
   averaged over feature maps (``drift.reduce: mean``, as in the configs) or summed (``sum``, the
   official code); ``drift.mode``: ``official`` fixed temperatures, ``kyutai`` learned temperature as in
   Kyutai's released code, ``learned`` their blog pseudo-code.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .alignment import sequence_mask
from .config import Config, load_config, save_config
from .data import BucketBatchSampler, MelDataset, collate
from .drift import feature_drift_loss
from .latents import VAE_BACKENDS
from .models.text_encoder import align, duration_loss, expand, prior_loss, token_pitch
from .models.tts import DriftingTTS
from .train_mae import load_mae
from .utils import EMA, count_params, infinite, lr_lambda, rng_state, save_checkpoint, seed_everything, set_rng_state

# official: fixed temperatures (paper); kyutai: Kyutai's released learned-tau recipe; learned: their blog pseudo-code
DRIFT_MODES = ("official", "kyutai", "learned")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/tts.yaml")
    p.add_argument("--workdir", default="runs/tts")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.steps=1000")


class CropBank:
    """Ring buffer of real mel crops: samples of the unconditional distribution ``p(x | ∅)``."""

    def __init__(self, size: int, n_mels: int, frames: int, device):
        self.data = torch.zeros(size, n_mels, frames, device=device, dtype=torch.bfloat16)
        self.ptr = self.count = 0

    def push(self, x: torch.Tensor) -> None:
        size, n = self.data.shape[0], x.shape[0]
        x = x[-size:]  # of more than ``size`` crops only the last ``size`` would survive
        start = (self.ptr + n - x.shape[0]) % size
        k = min(x.shape[0], size - start)  # slice up to the end of the buffer, the rest wraps around
        self.data[start: start + k] = x[:k]
        self.data[: x.shape[0] - k] = x[k:]
        self.ptr, self.count = (self.ptr + n) % size, min(self.count + n, size)

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.count, (n,), device=self.data.device)
        return self.data[idx].float()

    def state_dict(self) -> dict:
        return {"data": self.data, "ptr": self.ptr, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        if state["data"].shape != self.data.shape:
            print("CFG bank not restored: drift.uncond_bank or drift.crop_frames changed", flush=True)
            return
        self.data.copy_(state["data"])
        self.ptr, self.count = int(state["ptr"]), int(state["count"])


def sample_cfg(n: int, cfg_min: float, cfg_max: float, power: float, no_cfg_frac: float, device) -> torch.Tensor:
    """Guidance scales with density ``∝ alpha^-power`` on ``[cfg_min, cfg_max]`` (official ``neg_cfg_pw``)."""
    u = torch.rand(n, device=device)
    pw = 1.0 - power
    if abs(pw) < 1e-6:
        alpha = torch.exp(torch.log(torch.tensor(cfg_min)) + u * (torch.log(torch.tensor(cfg_max / cfg_min))))
    else:
        alpha = (cfg_min**pw + u * (cfg_max**pw - cfg_min**pw)) ** (1 / pw)
    return torch.where(torch.rand(n, device=device) < no_cfg_frac, torch.ones_like(alpha), alpha)


def crop_frames(x: torch.Tensor, starts: torch.Tensor, frames: int) -> torch.Tensor:
    idx = starts[:, None] + torch.arange(frames, device=x.device)[None]
    return torch.gather(x, 2, idx[:, None, :].expand(-1, x.shape[1], -1))


def pitch_enabled(cfg: Config) -> bool:
    return bool(cfg.model.get("pitch", {}).get("enabled", False))


def activation_kwargs(cfg: Config) -> dict:
    dc = cfg.drift
    return dict(patch_sizes=tuple(dc.patch_sizes), every_k_block=dc.every_k_block, mel_patch=dc.mel_patch,
                spectral_detail=bool(dc.get("spectral_detail", False)))


def split_features(feats: dict, B: int, S: int) -> dict:
    return {k: v.reshape(B, S, *v.shape[1:]) for k, v in feats.items()}


def build_loader(cfg: Config, split: str = "train") -> tuple[MelDataset, DataLoader]:
    d, t = cfg.data, cfg.train
    ds = MelDataset(d.root, split, min_quality=d.min_quality, min_frames=cfg.drift.crop_frames,
                    max_frames=d.max_frames, with_f0=pitch_enabled(cfg), filters=d.get("filters"))
    sampler = BucketBatchSampler([ds.frames(i) for i in range(len(ds))], max_frames=t.batch_frames,
                                 max_batch=t.batch_size, seed=t.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=t.num_workers,
                        pin_memory=True, persistent_workers=t.num_workers > 0)
    return ds, loader


def training_step(
    model: DriftingTTS, mae, batch: dict, bank: CropBank, cfg: Config, device, log_taus=None, taus=None
) -> tuple:
    """One step. ``log_taus``: ``drift.mode: learned``; ``taus`` (from :func:`build_taus`): ``drift.mode: kyutai``."""
    dc, lc = cfg.drift, cfg.loss
    text, text_len, y, y_len, spk = (batch[k].to(device, non_blocking=True)
                                     for k in ("text", "text_len", "mel", "mel_len", "spk"))
    B, L, G, P, N = y.shape[0], dc.crop_frames, dc.gen_per_cond, dc.pos_views, dc.uncond_per_cond
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=y.is_cuda)
    y_mask = sequence_mask(y_len, y.shape[-1])[:, None].float()

    # 1. text encoder, alignment, prior and duration losses
    with amp:
        h, mu, logw, x_mask = model.encoder(text, text_len, spk)
    h, mu, logw = h.float(), mu.float(), logw.float()
    attn, logw_target = align(mu, x_mask, y, y_mask)
    l_prior = prior_loss(expand(mu, attn), y, y_mask)
    l_dur = duration_loss(logw, logw_target, x_mask)
    l_pitch = torch.zeros((), device=device)
    if model.pitch_enabled:  # teacher-forced token pitch for the generator, predictor trained on it
        lf0_mean, lf0_std = model.lf0_stats  # 0-dim device tensors: no host sync
        pitch_target = token_pitch(batch["f0"].to(device, non_blocking=True), attn, lf0_mean, lf0_std)
        h, pitch_pred = model.pitch_condition(h, x_mask, spk, pitch_target)
        l_pitch = ((pitch_pred.float() - pitch_target) ** 2 * x_mask).sum() / x_mask.sum()

    # 2. aligned random crops of target and condition
    starts = (torch.rand(B, device=device) * (y_len - L + 1).float()).long()
    y_c = crop_frames(y, starts, L)
    cond = crop_frames(model.frame_condition(h, mu, attn), starts, L)
    if dc.get("detach_cond", False):
        cond = cond.detach()

    # 3. G one-step samples per condition, conditioned on a random CFG scale
    alpha = sample_cfg(B, dc.cfg_min, dc.cfg_max, dc.cfg_power, dc.no_cfg_frac, device)
    z = torch.randn(B * G, y.shape[1], L, device=device)
    cond_g, spk_g, alpha_g = cond.repeat_interleave(G, 0), spk.repeat_interleave(G, 0), alpha.repeat_interleave(G, 0)
    K = model.generator.num_steps
    j = int(torch.randint(0, K, (1,))) if K > 1 else 0
    with amp:
        g = model.generator
        noise_labels = torch.randint(0, g.noise_classes, (B * G, max(1, g.noise_coords)), device=device)
        if j > 0:  # on-policy rollout (DriftTTS): reach state x_j with the current generator, no gradient
            with torch.no_grad():
                z = model.rollout(z, cond_g.detach(), spk_g, alpha_g, j, noise_labels=noise_labels).float()
        x = model.generate(z, cond_g, spk_g, alpha_g, noise_labels=noise_labels, step=j)
    x = x.float()

    # 4. positives (target + perturbed views) and unconditional negatives: real crops of *other* utterances,
    #    drawn before this batch enters the bank so that no condition gets its own target as a negative
    pos = y_c[:, None].expand(B, P, -1, -1).clone()
    pos[:, 1:] += dc.view_noise * torch.randn_like(pos[:, 1:])
    unc = bank.sample(B * N) if bank.count else y_c.roll(1, 0).repeat_interleave(N, 0)  # empty bank: first step
    bank.push(y_c.detach())
    w_unc = ((alpha - 1) * (G - 1) / N)[:, None].expand(B, N)

    # 5. drift loss in Mel-MAE feature space (``feature_dtype: fp32`` runs the frozen MAE without bf16 autocast)
    act = activation_kwargs(cfg)
    feat_amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=y.is_cuda and dc.get("feature_dtype") != "fp32")
    with feat_amp:
        with torch.no_grad():
            f_pos = split_features(mae.get_activations(pos.flatten(0, 1), **act), B, P)
            f_unc = split_features(mae.get_activations(unc, **act), B, N)
        f_gen = split_features(mae.get_activations(x, **act), B, G)
    tau_arg = taus["tau"] if taus is not None and "tau" in taus else taus  # "tau": one global temperature (Kyutai)
    l_drift, info = feature_drift_loss(f_gen, f_pos, f_unc, weight_neg=w_unc, temperatures=tuple(dc.temperatures),
                                       key_weights=dc.get("key_weights"), affinity_floor=dc.affinity_floor,
                                       reduce=dc.get("reduce", "sum"), log_taus=log_taus, taus=tau_arg,
                                       max_locations=dc.get("max_locations"))

    xg = x.view(B, G, *x.shape[1:])
    l_recon = (xg - y_c[:, None]).abs().mean()
    loss = lc.drift * l_drift + lc.prior * l_prior + lc.duration * l_dur + lc.recon * l_recon
    loss = loss + lc.get("pitch", 0.1) * l_pitch

    with torch.no_grad():
        metrics = {
            "loss": loss.detach(), "drift": l_drift.detach(), "prior": l_prior.detach(), "duration": l_dur.detach(),
            "recon_l1": l_recon.detach(), "pitch": l_pitch.detach(),
            "centroid_mse": ((xg.mean(1) - y_c) ** 2).mean(),
            "sample_mse": ((xg - y_c[:, None]) ** 2).mean(),
            "across_sample_std": xg.std(1).mean(),
            "alpha": alpha.mean(), "rollout_step": float(j),
        }
        if log_taus is None and taus is None:
            for R in dc.temperatures:  # raw (pre-normalisation) drift norms: the real convergence signal
                vals = [v for k, v in info.items() if k.startswith(f"force_{R:g}/")]
                metrics[f"force_{R:g}"] = torch.stack(vals).mean()
        else:
            metrics["tau_loss"] = info["tau_loss"]
            metrics["drift"] = l_drift.detach() - info["tau_loss"]
            metrics["force"] = torch.stack([v for k, v in info.items() if k.startswith("force/")]).mean()
            metrics["p_data"] = torch.stack([v for k, v in info.items() if k.startswith("p_data/")]).mean()
            taus = torch.stack([v for k, v in info.items() if k.startswith("tau/")])
            metrics["tau_mean"], metrics["tau_min"], metrics["tau_max"] = taus.mean(), taus.min(), taus.max()
    return loss, metrics, info


def dataset_stats(ds: MelDataset) -> dict:
    """What inference needs to turn the model's normalised frames into audio (stored in the exported checkpoint)."""
    return {**ds.stats.to_dict(), "backend": ds.backend, "dim": ds.dim, "frame_rate": ds.frame_rate,
            "latent_repeat": ds.latent_repeat}


def feature_keys(mae, cfg: Config, device) -> list[str]:
    with torch.no_grad():
        mel = torch.zeros(1, mae.n_mels, cfg.drift.crop_frames, device=device)
        return list(mae.get_activations(mel, **activation_kwargs(cfg)))


def _scalar_params(keys, init: float, device) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(float(init), device=device)) for k in keys})


def build_log_taus(mae, cfg: Config, device) -> torch.nn.ParameterDict:
    """One learnable log-temperature per Mel-MAE feature map (``drift.mode: learned``)."""
    return _scalar_params(feature_keys(mae, cfg, device), math.log(cfg.drift.tau_init), device)


def build_taus(mae, cfg: Config, device) -> torch.nn.ParameterDict:
    """Raw learned kernel temperatures (``drift.mode: kyutai``): ``{"tau": t}``, one global as in Kyutai, or one
    per Mel-MAE feature map with ``drift.kyutai.per_feature_tau``. Initialised at ``drift.kyutai.tau_init``."""
    kc = cfg.drift.get("kyutai", {})
    keys = feature_keys(mae, cfg, device) if kc.get("per_feature_tau", False) else ["tau"]
    return _scalar_params(keys, kc.get("tau_init", 1.0), device)


@torch.no_grad()
def log_samples(model, ds: MelDataset, vocoder, writer, step: int, out_dir: Path, device, n: int = 4,
                temperature: float = 0.5) -> None:
    """Vocoded validation samples at CFG scales 1 and 2 (noise ``temperature``); restores the model's mode."""
    import soundfile as sf

    from .audio import SAMPLE_RATE

    was_training = model.training
    model.eval()
    items = [ds[i] for i in range(min(n, len(ds)))]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items):
        text = it["text"][None].to(device)
        spk = torch.tensor([it["spk"]], device=device)
        for cfg_scale in (1.0, 2.0):
            g = torch.Generator(device=device).manual_seed(i)
            mel, _ = model.synthesize(text, torch.tensor([text.shape[1]], device=device), spk,
                                      cfg_scale=cfg_scale, temperature=temperature, generator=g)
            wav = vocoder(ds.stats.denormalize(mel))[0].cpu()
            writer.add_audio(f"sample{i}/cfg{cfg_scale:g}", wav, step, sample_rate=SAMPLE_RATE)
            sf.write(out_dir / f"step{step}_utt{i}_cfg{cfg_scale:g}.wav", wav.numpy(), SAMPLE_RATE)
    model.train(was_training)


def run(args) -> None:
    from torch.utils.tensorboard import SummaryWriter  # training only: loading / synthesis need no tensorboard

    cfg = load_config(args.config, args.overrides)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    save_config(cfg, work / "config.yaml")
    tc = cfg.train
    seed_everything(tc.seed)
    device = "cuda" if torch.cuda.is_available() and not tc.get("cpu", False) else "cpu"

    ds, loader = build_loader(cfg)
    val_ds = MelDataset(cfg.data.root, "val", min_frames=1, max_frames=10**9, filters=cfg.data.get("filters"))
    model = DriftingTTS(cfg.model, num_speakers=ds.num_speakers, n_mels=ds.dim).to(device)
    if model.pitch_enabled:
        st = json.loads((Path(cfg.data.root) / "stats.json").read_text())
        model.lf0_stats.copy_(torch.tensor([st["lf0_mean"], st["lf0_std"]]))
    mae = load_mae(cfg.mae.path, device)
    print(f"DriftingTTS: encoder {count_params(model.encoder):.2f}M + generator {count_params(model.generator):.2f}M "
          f"params; Mel-MAE {count_params(mae):.2f}M (frozen); {len(ds)} training utterances", flush=True)

    groups = [{"params": list(model.parameters())}]
    mode = cfg.drift.get("mode", "official")
    if mode not in DRIFT_MODES:
        raise ValueError(f"drift.mode must be one of {DRIFT_MODES}, got {mode!r}")
    log_taus = taus = None
    if mode == "learned":
        log_taus = build_log_taus(mae, cfg, device)
        groups.append({"params": list(log_taus.parameters()), "lr": cfg.drift.tau_lr, "weight_decay": 0.0})
    elif mode == "kyutai":  # Kyutai: tau in the model's AdamW group (same LR and schedule), no weight decay
        taus = build_taus(mae, cfg, device)
        tau_lr = cfg.drift.get("kyutai", {}).get("tau_lr") or tc.lr
        groups.append({"params": list(taus.parameters()), "lr": tau_lr, "weight_decay": 0.0})
    temps = log_taus if log_taus is not None else taus
    opt = torch.optim.AdamW(groups, lr=tc.lr, betas=(0.9, 0.95), weight_decay=tc.weight_decay,
                            fused=True if device == "cuda" else None)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc.warmup, tc.steps, tc.schedule))
    ema = EMA(model, tc.ema_decay)  # a deep copy made before compiling: sampling and export stay eager
    if tc.get("compile", False):  # in place, so state_dict keys are unchanged; the training shapes are static
        model.generator.compile()
    bank = CropBank(cfg.drift.uncond_bank, ds.dim, cfg.drift.crop_frames, device)
    writer = SummaryWriter(work / "tb")
    step, rng = 0, None
    if (work / "last.pt").exists():
        ck = torch.load(work / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        if log_taus is not None and ck.get("log_taus"):
            log_taus.load_state_dict(ck["log_taus"])
        if taus is not None and ck.get("taus"):
            taus.load_state_dict(ck["taus"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, rng = ck["step"], ck.get("rng")
        loader.batch_sampler.epoch = ck.get("epoch", 0)
        if "bank" in ck:  # older checkpoints: the bank refills within uncond_bank / batch_size steps
            bank.load_state_dict(ck["bank"])
        print(f"resumed from step {step}", flush=True)
    # carry over what `calibrate-durations` stored (duration_scale, temperature), although measured on an earlier model
    calib = {}
    if (work / "model_ema.pt").exists():
        prev = torch.load(work / "model_ema.pt", map_location="cpu", weights_only=False)
        calib = {k: prev[k] for k in ("duration_scale", "duration_scales", "temperature") if prev.get(k) is not None}
    duration_scale = calib.get("duration_scale")
    if duration_scale is not None:
        print(f"model_ema.pt keeps duration_scale={duration_scale:.3f} from an earlier calibration; it goes stale as "
              "training continues", flush=True)

    if step == 0 and tc.get("init_from"):
        init = torch.load(tc.init_from, map_location="cpu", weights_only=False)
        state = grow_speakers(init.get("ema", init.get("model")), model.state_dict())
        missing, unexpected = model.load_state_dict(state, strict=False)
        ema.model.load_state_dict(model.state_dict())
        print(f"initialised from {tc.init_from} (missing: {len(missing)}, unexpected: {len(unexpected)})", flush=True)
        if taus is not None and init.get("taus") and set(init["taus"]) == set(taus.state_dict()):
            taus.load_state_dict(init["taus"])  # fine-tuning continues at the learned kernel temperature
            print(f"kernel temperature from {tc.init_from}: {temps_summary(taus)}", flush=True)

    vocoder = None
    if tc.sample_every > 0 and device == "cuda":
        if ds.backend in VAE_BACKENDS:
            from .latents.vocoder import LatentVocoder

            vocoder = LatentVocoder(ds.backend, device, repeat=ds.latent_repeat)
        else:
            from .vocoder import Vocoder

            vocoder = Vocoder(device, backend=ds.backend)

    model.train()
    if rng is not None:
        set_rng_state(rng)
    it = infinite(loader)
    t0, agg = time.time(), {}
    while step < tc.steps:
        batch = next(it)
        loss, metrics, _ = training_step(model, mae, batch, bank, cfg, device, log_taus, taus)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip encoder and generator separately so drift gradients cannot starve the prior / duration terms
        gnorm_enc = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), tc.grad_clip)
        gnorm = torch.nn.utils.clip_grad_norm_(model.generator.parameters(), tc.grad_clip)
        if temps is not None:
            torch.nn.utils.clip_grad_norm_(temps.parameters(), tc.grad_clip)
        opt.step()
        sched.step()
        ema.update(model)
        step += 1
        metrics["grad_norm"], metrics["grad_norm_enc"] = gnorm, gnorm_enc
        for k, v in metrics.items():  # summed on the device: no host sync until the next log
            agg[k] = agg.get(k, 0.0) + (v.detach() if torch.is_tensor(v) else v)
        if step % tc.log_every == 0:
            n = tc.log_every
            agg = {k: float(v) / n for k, v in agg.items()}
            rate = n / (time.time() - t0)
            for k, v in agg.items():
                writer.add_scalar(f"train/{k}", v, step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            keys = ["loss", "drift", "prior", "duration", "pitch", "centroid_mse", "across_sample_std", "force_0.05",
                    "force", "tau_mean", "p_data", "grad_norm", "grad_norm_enc"]
            print(f"step {step} " + " ".join(f"{k}={agg[k]:.4g}" for k in keys if k in agg)
                  + f" ({rate:.2f} it/s)", flush=True)
            agg, t0 = {}, time.time()
        if vocoder is not None and step % tc.sample_every == 0:
            log_samples(ema.model, val_ds, vocoder, writer, step, work / "samples", device,
                        temperature=tc.get("sample_temperature", 0.5))
        if step % tc.save_every == 0 or step == tc.steps:
            save_checkpoint(work / "last.pt", model=model.state_dict(), ema=ema.model.state_dict(),
                            opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                            log_taus=None if log_taus is None else log_taus.state_dict(),
                            taus=None if taus is None else taus.state_dict(),
                            epoch=loader.batch_sampler.epoch, bank=bank.state_dict(), rng=rng_state(),
                            config=cfg.to_dict(), num_speakers=ds.num_speakers)
            save_checkpoint(work / "model_ema.pt", ema=ema.model.state_dict(), config=cfg.to_dict(),
                            num_speakers=ds.num_speakers, step=step, taus=None if taus is None else taus.state_dict(),
                            n_mels=ds.dim, stats=dataset_stats(ds), **calib)
    stale = "" if duration_scale is None else " (stale duration_scale: re-run `drifting-tts calibrate-durations`)"
    print(f"done: {work / 'model_ema.pt'}{stale}", flush=True)


def temps_summary(taus: torch.nn.ParameterDict) -> str:
    values = [float(t) for t in taus.values()]
    return f"{values[0]:.4g}" if len(values) == 1 else f"{min(values):.4g}-{max(values):.4g} ({len(values)} maps)"


def grow_speakers(state: dict, own: dict) -> dict:
    """Fit a checkpoint's speaker tables to a model with more speakers (a new voice added for fine-tuning): the old
    rows are kept and every new row starts at their mean."""
    state = dict(state)
    for k, v in state.items():
        grown = k in own and own[k].shape[0] > v.shape[0] and own[k].shape[1:] == v.shape[1:]
        if k.endswith("spk.weight") and grown:
            state[k] = torch.cat([v, v.mean(0, keepdim=True).expand(own[k].shape[0] - v.shape[0], -1)])
            print(f"{k}: {v.shape[0]} -> {own[k].shape[0]} speakers", flush=True)
    return state


def load_tts(path: str | Path, device="cpu") -> tuple[DriftingTTS, Config, dict]:
    """Load an exported (EMA) model: returns the model, its config and the mel statistics."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(ck["config"])
    model = DriftingTTS(cfg.model, num_speakers=ck["num_speakers"], n_mels=ck.get("n_mels", 100))
    model.load_state_dict(ck["ema"])
    model.duration_scale = float(ck.get("duration_scale", 1.0))  # set by `drifting-tts calibrate-durations`
    model.duration_scales = {int(k): float(v) for k, v in ck.get("duration_scales", {}).items()}  # per voice
    model.temperature = ck.get("temperature")  # preferred noise temperature (`calibrate-durations --temperature`)
    stats = ck.get("stats")
    if stats is None:
        from .data import MelStats

        s = MelStats.load(cfg.data.root)
        stats = {"mean": s.mean, "std": s.std, "backend": "vocos"}
    return model.to(device).eval(), cfg, stats


"""Train ARDrift (``train-ar``, #42): an autoregressive acoustic model with a one-step drifting head.

One step:

1. the text encoder and monotonic alignment, prior and duration losses (as ``train``);
2. the frame-aligned condition and the target mel, padded to whole patches; every utterance is used whole;
3. the causal backbone, teacher-forced on the previous target patches plus noise of a random level per utterance
   (up to ``drift.input_noise``), gives one state per patch. With ``drift.self_history`` > 0, that share of the
   utterances has the model's own patches as its history instead: each drawn by the head (at
   ``drift.self_history_temperature``) after the recorded ones, in a first teacher-forced pass without gradients, as at
   inference after one step. The targets stay the recording's, so the model learns not to carry its own errors on
   (#73: the podcast voices' pauses are up to 5 dB louder than the recordings' teacher-forced, up to 18 dB
   free-running);
4. at ``drift.max_rows`` random valid patches (rows), the head draws ``drift.gen_per_cond`` samples from noise; the
   drifting loss (:func:`~drifting_tts.drift.kyutai_drift_loss`, one learned temperature) pulls them to the recorded
   patch and pushes them apart, in the kernel space of :func:`~drifting_tts.models.ar.ar_features` (the whole patch,
   each frame, each frame's harmonic detail and, with ``drift.energy_bands``, each frame's band energies).

``train.eval_every`` evaluates every EMA inside the loop, on the held-out objective and on synthesised held-out
sentences, and ``train.ema_decays`` keeps more than one EMA (#102, :mod:`drifting_tts.loop_eval`).

``model.ar.prompt`` with ``drift.prompt`` trains voices from a prompt (#95, Kyutai's Pocket TTS recipe): an utterance
is cut at a word boundary, the frames before the cut are history without a text condition, the words after it are
the text, and the loss is taken after the cut (:func:`cut_prompts`). For a share of them the speaker row is the
"prompted" one, so the voice has to come from the prompt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .alignment import sequence_mask
from .audio import HOP_LENGTH, SAMPLE_RATE
from .config import Config, load_config, save_config
from .data import MelDataset, collate
from .drift import feature_drift_loss
from .latents import VAE_BACKENDS
from .loop_eval import AudioEval, build_emas, heldout_metrics, load_emas, other_emas, record
from .models.ar import ARDriftTTS, ar_features, patchify
from .models.text_encoder import align, duration_loss, expand, prior_loss
from .text import BLANK_ID, PAD_ID, frontend
from .train import build_loader, dataset_stats, grow_speakers
from .utils import count_params, infinite, lr_lambda, rng_state, save_checkpoint, seed_everything, set_rng_state


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/ar_tts.yaml")
    p.add_argument("--workdir", default="runs/ar_tts")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.steps=1000")


@torch.no_grad()
def cut_prompts(model: ARDriftTTS, text: torch.Tensor, text_len: torch.Tensor, y: torch.Tensor, y_len: torch.Tensor,
                spk: torch.Tensor, pc, space: int, always: torch.Tensor | None = None) -> dict:
    """Prompts for a batch (#95): an utterance that is long enough is cut where a word starts, between
    ``min_seconds`` and ``max_seconds`` and at least ``min_target`` seconds before its end (the model's own
    alignment gives the frame). The frames before the cut, less up to ``max_trim`` seconds of their end, are the
    prompt; the tokens and frames from the cut on are the utterance that is trained. Kyutai's loader on why the cut
    is at a word: "Without a boundary to cut at, the prompt would be a window of the target itself and the model
    would learn to copy it instead of reading the text."

    Returns ``text`` / ``text_len`` / ``mel`` / ``mel_len`` of the part after the cut (the whole utterance where
    there is no cut), ``prompt`` ``[B, n_mels, P]`` with ``prompt_len`` (whole patches; 0: no prompt), and
    ``dropped`` ``[B]``: the prompted utterances whose speaker row is the prompted one (``id_dropout``). ``always``
    ``[B]``: utterances of voices without a row (a corpus streamed with ``speakers: prompt``), which are cut
    whenever they can be and never keep a row."""
    B, k, dev = y.shape[0], model.patch, y.device
    fps = SAMPLE_RATE / HOP_LENGTH
    lo, hi, rest, trim = (int(round(float(pc.get(name, default)) * fps)) for name, default in
                          (("min_seconds", 2.0), ("max_seconds", 6.0), ("min_target", 1.5), ("max_trim", 0.5)))
    y_mask = sequence_mask(y_len, y.shape[-1])[:, None].float()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=y.is_cuda):
        _, mu, _, x_mask = model.encoder(text, text_len, spk)
    frames = align(mu.float(), x_mask.float(), y, y_mask)[0].sum(-1)  # [B, N] frames per token
    start = (frames.cumsum(-1) - frames).long()  # the first frame of every token
    # a cut is the blank after a space: the next word starts there, and the rest is again blank, character, blank, ...
    after_space = torch.nn.functional.pad(text[:, :-1] == space, (1, 0)) & (text == BLANK_ID)
    inside = torch.arange(text.shape[1], device=dev)[None] < text_len[:, None]
    ok = after_space & inside & (start >= lo) & (start <= hi) & (start <= y_len[:, None] - rest)
    always = torch.zeros(B, dtype=torch.bool, device=dev) if always is None else always
    ok &= ((torch.rand(B, device=dev) < float(pc.get("prob", 0.85))) | always)[:, None]
    texts, mels, prompts = [], [], []
    for b in range(B):
        cand = ok[b].nonzero()[:, 0]
        j = int(cand[torch.randint(len(cand), (1,), device=dev)]) if len(cand) else 0
        f0 = int(start[b, j]) if len(cand) else 0
        p = max(f0 - int(torch.randint(trim + 1, (1,), device=dev)), 0) // k * k if len(cand) else 0
        texts.append(text[b, j: int(text_len[b])])
        mels.append(y[b, :, f0: int(y_len[b])].T)
        prompts.append(y[b, :, :p].T)
    prompt_len = torch.tensor([p.shape[0] for p in prompts], device=dev)
    dropped = (prompt_len > 0) & ((torch.rand(B, device=dev) < float(pc.get("id_dropout", 0.5))) | always)
    return {"text": pad_sequence(texts, batch_first=True, padding_value=PAD_ID),
            "text_len": torch.tensor([t.numel() for t in texts], device=dev),
            "mel": pad_sequence(mels, batch_first=True).transpose(1, 2),
            "mel_len": torch.tensor([m.shape[0] for m in mels], device=dev),
            "prompt": pad_sequence(prompts, batch_first=True).transpose(1, 2), "prompt_len": prompt_len,
            "dropped": dropped}


def prepend_prompts(cond: torch.Tensor, y: torch.Tensor, y_len: torch.Tensor, prompt: torch.Tensor,
                    prompt_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Every utterance's prompt in front of it: the mel ``[B, n_mels, P_b + T_b]`` (prompt, then target) and the
    condition, zero over the prompt (the backbone gets no text for it). Padded to the longest."""
    B, total = y.shape[0], int((prompt_len + y_len).max())
    full_c, full_y = cond.new_zeros(B, cond.shape[1], total), y.new_zeros(B, y.shape[1], total)
    for b in range(B):
        p, t = int(prompt_len[b]), int(y_len[b])
        full_y[b, :, :p], full_y[b, :, p: p + t] = prompt[b, :, :p], y[b, :, :t]
        full_c[b, :, p: p + t] = cond[b, :, :t]
    return full_c, full_y


def training_step(model: ARDriftTTS, batch: dict, cfg: Config, device, tau: torch.Tensor,
                  mel_std: float | None = None, detail: bool = False) -> tuple:
    """One training step; ``mel_std``: the scalar std of the mel statistics (for ``drift.energy_bands``).
    ``detail``: the metrics also hold every feature map's drift loss, raw field and data mass (``loss/<map>``,
    ``force/<map>``, ``p_data/<map>``), for the evaluation in the loop."""
    dc, lc = cfg.drift, cfg.loss
    text, text_len, y, y_len, spk = (batch[k].to(device, non_blocking=True)
                                     for k in ("text", "text_len", "mel", "mel_len", "spk"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=y.is_cuda)
    B = y.shape[0]
    unknown = spk < 0  # voices without a speaker row (a corpus streamed with `speakers: prompt`, #104)
    if unknown.any():
        if not model.prompted:
            raise ValueError("utterances without a speaker row need a model trained with prompts (model.ar.prompt)")
        spk = torch.where(unknown, torch.full_like(spk, model.prompt_row), spk)  # without a prompt: the null voice
    lead, spk_emb, spk_in = torch.zeros_like(y_len), None, spk  # prompt frames, the encoder's and the backbone's voice
    prompting = model.prompted and dc.get("prompt")
    if prompting:  # voices from a prompt (#95): the utterance's own beginning, cut off at a word
        cut = cut_prompts(model, text, text_len, y, y_len, spk, dc.prompt,
                          frontend(cfg.model.text.get("language")).SYMBOL_TO_ID[" "], unknown)
        text, text_len, y, y_len, lead = cut["text"], cut["text_len"], cut["mel"], cut["mel_len"], cut["prompt_len"]
        with amp:
            vec = model.prompt_vector(cut["prompt"], lead)
        spk_emb = torch.where(cut["dropped"][:, None], vec.float(), model.encoder.spk(spk))
        spk_in = torch.where(cut["dropped"], torch.full_like(spk, model.prompt_row), spk)
    y_mask = sequence_mask(y_len, y.shape[-1])[:, None].float()
    with amp:
        h, mu, logw, x_mask = model.encoder(text, text_len, spk, spk_emb)
    h, mu, logw = h.float(), mu.float(), logw.float()
    attn, logw_target = align(mu, x_mask, y, y_mask)
    l_prior = prior_loss(expand(mu, attn), y, y_mask)
    l_dur = duration_loss(logw, logw_target, x_mask)

    k, C = model.patch, model.n_mels
    cond = model.frame_condition(h, mu, attn)
    if prompting:
        cond, y = prepend_prompts(cond, y, y_len, cut["prompt"], lead)
    cond, y = model.pad_frames(cond), model.pad_frames(y)
    noise = torch.rand(B, device=device) * float(dc.get("input_noise", 0.0))
    p_drop = float(dc.get("cond_dropout", 0.0))  # the null condition of guidance
    drop = torch.rand(B, device=device) < p_drop if p_drop > 0 else None
    hist, share = y, float(dc.get("self_history", 0.0))
    if share > 0:  # the model's own one-step patches as the history of a share of the utterances
        with amp:
            own = model.gta(cond, y, spk_in, float(dc.get("self_history_temperature", 0.5))).float()
        swap = (torch.rand(B, device=device) < share)[:, None, None]
        swap = swap & (torch.arange(y.shape[-1], device=device)[None, None] >= lead[:, None, None])  # not a prompt
        hist = torch.where(swap, own, y)
    with amp:
        s = model.teacher_forced(cond, hist, spk_in, noise, drop)  # [B, S, dim]
    S = s.shape[1]
    pos = torch.arange(S, device=device)[None] * k
    valid = (pos >= lead[:, None]) & (pos < (lead + y_len)[:, None])  # steps that start inside the utterance
    idx = valid.flatten().nonzero()[:, 0]
    if idx.numel() > dc.max_rows:  # a random subset of the steps (rows) keeps the head's batch bounded
        idx = idx[torch.randperm(idx.numel(), device=device)[: dc.max_rows]]
    rows = s.flatten(0, 1)[idx]  # [R, dim]
    target = patchify(y, k).flatten(0, 1)[idx]  # [R, k * C]
    base = patchify(cond[:, :C], k).flatten(0, 1)[idx]  # the aligned prior mean: the head predicts a residual
    R, G = rows.shape[0], int(dc.gen_per_cond)
    z = torch.randn(R * G, model.head.noise_dim, device=device)
    with amp:
        res = model.head(rows.repeat_interleave(G, 0), z)
    x = base[:, None] + res.float().view(R, G, -1)  # [R, G, k * C]
    opts = {"detail": bool(dc.get("detail", True)), "energy_bands": dc.get("energy_bands")}
    if opts["energy_bands"]:
        if mel_std is None:
            raise ValueError("drift.energy_bands needs the scalar std of the mel statistics (log-mel data)")
        opts["energy_scale"] = 2.0 * float(mel_std)
    f_gen = ar_features(x, k, C, **opts)
    with torch.no_grad():
        f_pos = ar_features(target[:, None], k, C, **opts)
    l_drift, info = feature_drift_loss(f_gen, f_pos, None, taus=tau, reduce="mean",
                                       key_weights=dc.get("key_weights"))
    loss = lc.drift * l_drift + lc.prior * l_prior + lc.duration * l_dur
    with torch.no_grad():
        metrics = {"loss": loss.detach(), "drift": l_drift.detach() - info["tau_loss"], "tau_loss": info["tau_loss"],
                   "prior": l_prior.detach(), "duration": l_dur.detach(),
                   "centroid_mse": ((x.mean(1) - target) ** 2).mean(), "sample_std": x.std(1).mean(),
                   "rows": float(R), "tau": tau.detach().reshape(()),
                   "p_data": torch.stack([v for kk, v in info.items() if kk.startswith("p_data/")]).mean()}
        if prompting:
            metrics["prompted"] = (lead > 0).float().mean()
            metrics["id_dropped"] = cut["dropped"].float().mean()
            metrics["no_row"] = unknown.float().mean()
        if detail:
            metrics.update({kk: v for kk, v in info.items() if kk.startswith(("loss/", "force/", "p_data/"))})
    return loss, metrics


@torch.no_grad()
def log_samples(model: ARDriftTTS, ds: MelDataset, vocoder, out_dir: Path, step: int, device, n: int = 3,
                temperature: float = 0.5) -> None:
    import soundfile as sf

    was = model.training
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(ds))):
        it = ds[i]
        text = it["text"][None].to(device)
        g = torch.Generator(device=device).manual_seed(i)
        mel, _ = model.synthesize(text, torch.tensor([text.shape[1]], device=device),
                                  torch.tensor([it["spk"]], device=device), temperature=temperature, generator=g)
        wav = vocoder(ds.stats.denormalize(mel))[0].cpu()
        sf.write(out_dir / f"step{step}_utt{i}.wav", wav.numpy(), 24000)
    model.train(was)


def run(args) -> None:
    from torch.utils.tensorboard import SummaryWriter

    cfg = load_config(args.config, args.overrides)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    save_config(cfg, work / "config.yaml")
    tc = cfg.train
    seed_everything(tc.seed)
    device = "cuda" if torch.cuda.is_available() and not tc.get("cpu", False) else "cpu"

    if cfg.data.get("stream"):  # a corpus larger than the disk, read from its parquet shards (drifting_tts/stream.py)
        from .stream import stream_loader

        ds, loader = stream_loader(cfg)
        val_ds = ds
    else:
        ds, loader = build_loader(cfg)
        val_ds = MelDataset(cfg.data.root, "val", min_frames=1, max_frames=10**9)
    if ds.language != cfg.model.text.get("language", "tr"):
        raise ValueError(f"{cfg.data.root} is {ds.language!r} text (stats.json), the model's text frontend is "
                         f"{cfg.model.text.get('language', 'tr')!r}: set model.text.language")
    model = ARDriftTTS(cfg.model, num_speakers=ds.num_speakers, n_mels=ds.dim).to(device)
    print(f"ARDrift: text encoder {count_params(model.encoder):.2f}M, backbone "
          f"{count_params(model) - count_params(model.encoder) - count_params(model.head):.2f}M, head "
          f"{count_params(model.head):.2f}M; "
          + ("streamed training data" if cfg.data.get("stream") else f"{len(ds)} training utterances"), flush=True)
    tau = torch.nn.Parameter(torch.tensor(float(cfg.drift.get("tau_init", 1.0)), device=device))
    betas = tuple(tc.get("betas", (0.9, 0.999)))
    opt = torch.optim.AdamW([{"params": list(model.parameters())},
                             {"params": [tau], "weight_decay": 0.0}],
                            lr=tc.lr, betas=betas, weight_decay=tc.weight_decay, fused=device == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc.warmup, tc.steps, tc.schedule))
    emas = build_emas(model, tc)  # train.ema_decays: the first is model_ema.pt
    ema = next(iter(emas.values()))
    writer = SummaryWriter(work / "tb")
    step, rng = 0, None
    if (work / "last.pt").exists():
        ck = torch.load(work / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        load_emas(emas, ck)
        tau.data.fill_(float(ck["tau"]))
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, rng = ck["step"], ck.get("rng")
        if cfg.data.get("stream"):  # another shard order than before the restart (the stream has no position)
            loader.dataset.epoch = ck.get("step", 0)
        else:
            loader.batch_sampler.epoch = ck.get("epoch", 0)
        print(f"resumed from step {step}", flush=True)
    elif tc.get("init_from"):  # fine-tuning (e.g. a new voice): speaker tables grown, temperature carried over
        init = torch.load(tc.init_from, map_location="cpu", weights_only=False)
        state = grow_speakers(init.get("ema", init.get("model")), model.state_dict())
        missing, unexpected = model.load_state_dict(state, strict=False)
        for e in emas.values():
            e.model.load_state_dict(model.state_dict())
        if init.get("tau") is not None:
            tau.data.fill_(float(init["tau"]))
        print(f"initialised from {tc.init_from} (missing {len(missing)}, unexpected {len(unexpected)}, "
              f"tau {float(tau):.4g})", flush=True)

    vocoder = None
    sample_every = tc.get("sample_every", 0) if device == "cuda" else 0
    eval_every = int(tc.get("eval_every", 0))
    if eval_every and not len(val_ds):
        print(f"no 'val' utterances in {cfg.data.root}: no evaluation in the loop", flush=True)
        eval_every = 0
    if sample_every or eval_every:
        if ds.backend in VAE_BACKENDS:  # latent frames (extract-latents): the codec's decoder
            from .latents.vocoder import LatentVocoder

            vocoder = LatentVocoder(ds.backend, device, repeat=ds.latent_repeat)
        else:
            from .vocoder import load_vocoder

            vocoder = load_vocoder(tc.get("vocoder", "vocos-v2"), device)

    mel_std = None if ds.stats.per_channel else ds.stats.std  # drift.energy_bands: log-mel data only

    def evaluate(at: int) -> None:  # every EMA on the held-out objective and on synthesised sentences (#102)
        t = time.time()
        for n, (name, e) in enumerate(emas.items()):
            res = heldout_metrics(lambda b, m=e.model: training_step(m, b, cfg, device, tau, mel_std, detail=True)[1],
                                  heldout)
            res.update(audio(e.model))
            record(writer, work, at, name, n == 0, res)
        print(f"  evaluation at step {at}: {time.time() - t:.0f} s", flush=True)

    if eval_every:
        ec = tc.get("eval") or {}
        temperature = float(ec.get("temperature", tc.get("sample_temperature", 0.5)))

        def synthesize(m, item, g):
            text = item["text"][None].to(device)
            return m.synthesize(text, torch.tensor([text.shape[1]], device=device),
                                torch.tensor([item["spk"]], device=device), temperature=temperature, generator=g)[0]

        audio = AudioEval(val_ds, vocoder, device, synthesize, num=ec.get("num", 24), asr=ec.get("asr", "large-v3"),
                          mos=ec.get("mos", "utmosv2"), f0=ec.get("f0", True), asr_device=ec.get("asr_device"))
        size = int(ec.get("batch_size", 8))
        heldout = [collate([val_ds[i] for i in range(s, min(s + size, len(val_ds)))])
                   for s in range(0, min(int(ec.get("batches", 8)) * size, len(val_ds)), size)]
        if ec.get("at_start", False) and step < tc.steps:  # the point a continued or fine-tuned run starts from
            evaluate(step)
    model.train()
    if rng is not None:
        set_rng_state(rng)
    it = infinite(loader)
    t0, agg = time.time(), {}
    while step < tc.steps:
        loss, metrics = training_step(model, next(it), cfg, device, tau, mel_std)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm_enc = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), tc.grad_clip)
        gnorm = torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters()
                                                if not n.startswith("encoder.")], tc.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            tau.clamp_(min=1e-3)
        for e in emas.values():
            e.update(model)
        step += 1
        metrics["grad_norm"], metrics["grad_norm_enc"] = gnorm, gnorm_enc
        for kk, v in metrics.items():
            agg[kk] = agg.get(kk, 0.0) + (v.detach() if torch.is_tensor(v) else v)
        if step % tc.log_every == 0:
            n = tc.log_every
            agg = {kk: float(v) / n for kk, v in agg.items()}
            for kk, v in agg.items():
                writer.add_scalar(f"train/{kk}", v, step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            print(f"step {step} " + " ".join(f"{kk}={v:.4g}" for kk, v in agg.items())
                  + f" ({n / (time.time() - t0):.2f} it/s)", flush=True)
            agg, t0 = {}, time.time()
        if sample_every and step % sample_every == 0:
            log_samples(ema.model, val_ds, vocoder, work / "samples", step, device,
                        temperature=tc.get("sample_temperature", 0.5))
        if eval_every and (step % eval_every == 0 or step == tc.steps):
            evaluate(step)
        if step % tc.save_every == 0 or step == tc.steps:
            if tc.get("save_last", True):  # false: exports only, the run cannot be resumed (a pilot on a small disk)
                save_checkpoint(work / "last.pt", model=model.state_dict(), ema=ema.model.state_dict(),
                                tau=float(tau.detach()), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                                rng=rng_state(), epoch=getattr(loader.batch_sampler, "epoch", 0),
                                config=cfg.to_dict(), num_speakers=ds.num_speakers, emas=other_emas(emas))
            for n, (name, e) in enumerate(emas.items()):  # model_ema.pt, and model_ema_<decay>.pt for another EMA
                save_checkpoint(work / ("model_ema.pt" if n == 0 else f"model_ema_{name}.pt"),
                                ema=e.model.state_dict(), config=cfg.to_dict(), num_speakers=ds.num_speakers,
                                step=step, tau=float(tau.detach()), n_mels=ds.dim, stats=dataset_stats(ds))
            if tc.get("keep_snapshots", False):
                shutil.copyfile(work / "model_ema.pt", work / f"model_ema_{step}.pt")
    print(f"done: {work / 'model_ema.pt'}", flush=True)
    (work / "done.json").write_text(json.dumps({"step": step}))

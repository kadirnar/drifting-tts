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

``model.ar.tied`` (#92, latents stored ``patch`` times for the alignment): the backbone reads and the head draws one
latent frame per step, copied over the patch. ``drift.target_noise`` trains on samples of the codec's posterior
instead of its mean (:func:`sample_targets`).

``drift.rollout`` (#101): with probability ``prob`` a batch has ``steps`` (4 to 16) patches of its history drawn
free-running by the model (:meth:`~drifting_tts.models.ar.ARDriftTTS.rollout`), where ``drift.self_history`` has
patches drawn one step after the recording. The targets stay the recording's. ``model.ar.noise_level`` tells the
backbone the level of ``drift.input_noise``.

``drift.cond_dropout`` and ``drift.voice_dropout`` train the null branches of guidance (#99): utterances without
their text, and without any voice (no prompt, no reference, the row of a voice that is not in the table).
``drift.distill`` then trains the backbone alone to give the guided state in one pass (:func:`distill_step`).

``drift.teacher`` (#98): a frozen model draws ``positives`` more positives per row beside the recording, from its
own state of the clean recorded history (:class:`Teacher`): its one-step head, or a multi-step flow-matching head.

``drift.loss: energy`` replaces the drifting field by the energy score (:func:`~drifting_tts.drift.energy_score`, a
proper scoring rule without a temperature; docs/HEAD_REALISM.md). ``drift.window`` adds kernel maps over a window
(#97): the recorded patches before a step and the step's patch through a frozen 2-D Mel-MAE
(:func:`~drifting_tts.models.ar.window_features`), on a share of the rows and samples.

``train.eval_every`` evaluates every EMA inside the loop, on the held-out objective and on synthesised held-out
sentences, and ``train.ema_decays`` keeps more than one EMA (#102, :mod:`drifting_tts.loop_eval`).

``model.ar.prompt`` with ``drift.prompt`` trains voices from a prompt (#95, Kyutai's Pocket TTS recipe): an utterance
is cut at a word boundary, the frames before the cut are history without a text condition, the words after it are
the text, and the loss is taken after the cut (:func:`cut_prompts`). For a share of them the speaker row is the
"prompted" one, so the voice has to come from the prompt.

``model.ar.reference`` with ``drift.reference`` trains voices from a reference (#96): a crop of *another* utterance
of the speaker (the loader's ``ref``), read by the reference encoder, whose tokens stand in front of the backbone's
sequence. ``drift.reference.prob`` of the utterances that have one use it, the others get the null tokens (no
reference: the speaker row alone, a prompt alone, or no voice at all, the null branch of voice guidance); the
speaker row is dropped as for a prompt (``drift.reference.id_dropout``). Both paths train together.
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
from .drift import feature_drift_loss, feature_energy_loss
from .latents import VAE_BACKENDS
from .loop_eval import AudioEval, build_emas, heldout_metrics, load_emas, other_emas, record
from .models.ar import ARDriftTTS, ar_features, history_patches, patchify, unpatchify, window_features
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
                spk: torch.Tensor, pc, space: int, always: torch.Tensor | None = None,
                never: torch.Tensor | None = None) -> dict:
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
    whenever they can be and never keep a row. ``never`` ``[B]``: utterances that get no prompt (their voice is
    dropped, ``drift.voice_dropout``)."""
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
    if never is not None:
        ok &= ~never[:, None]
    texts, mels, prompts = [], [], []
    for b in range(B):
        cand = ok[b].nonzero()[:, 0]
        j = int(cand[torch.randint(len(cand), (1,), device=dev)]) if len(cand) else 0
        f0 = int(start[b, j]) if len(cand) else 0
        if model.tied:  # the copies of a latent frame stay in one patch (the alignment after the cut absorbs it)
            f0 = f0 // k * k
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


def sample_targets(y: torch.Tensor, y_len: torch.Tensor, noise, group: int = 1,
                   std: torch.Tensor | None = None) -> torch.Tensor:
    """Sampled targets (#92): the frames ``[B, C, T]`` plus Gaussian noise, so that the model learns the codec's
    posterior and not its mean (SALAD: UTMOS 4.28 against 3.47, CER 0.74 against 1.89, for a model trained on
    samples instead of means). ``noise``: a standard deviation in normalised units, or ``"posterior"``: the
    posterior's own ``std`` ``[B, C, T]`` (``extract-latents --save-std``). ``group``: frames that are copies of
    one latent frame share a draw (``model.ar.tied``). Everything downstream sees the sample: the alignment, the
    history and the target."""
    B, C, T = y.shape
    eps = torch.randn(B, C, -(-T // group), device=y.device, dtype=y.dtype)
    if group > 1:
        eps = eps.repeat_interleave(group, -1)[..., :T]
    if noise == "posterior":
        if std is None:
            raise ValueError("drift.target_noise: posterior needs the posterior's standard deviations in the batch "
                             "(a dataset of `extract-latents --save-std`)")
        scale = std.to(y.dtype)
    else:
        scale = float(noise)
    return torch.where(sequence_mask(y_len, T)[:, None], y + scale * eps, y)


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


def kernel_loss(f_gen: dict, f_pos: dict, kind: str, tau: torch.Tensor, key_weights=None,
                max_locations: int | None = None) -> tuple[torch.Tensor, dict]:
    """The loss of one set of kernel maps: Kyutai's drifting field at the learned temperature (``field``) or the
    energy score (``energy``). ``info`` holds ``tau_loss`` (0 for the energy score) and the per-map entries."""
    if kind == "energy":
        loss, info = feature_energy_loss(f_gen, f_pos, key_weights, max_locations)
        return loss, {**info, "tau_loss": loss.new_zeros(())}
    if kind != "field":
        raise ValueError(f"drift.loss: {kind!r} (field or energy)")
    return feature_drift_loss(f_gen, f_pos, None, taus=tau, reduce="mean", key_weights=key_weights,
                              max_locations=max_locations)


def distill_step(model: ARDriftTTS, teacher: ARDriftTTS, batch: dict, cfg: Config, device) -> tuple:
    """Guidance distilled into the backbone (#99, ``drift.distill``; Kyutai: "the head is not retrained, or even
    touched"). The frozen ``teacher`` gives, for every step of the recorded history, its guided state
    ``s_null + cfg_scale * (s - s_null)``; the student's backbone is trained to give that state in one pass (mean
    squared error over the utterance's steps). The text encoder and the head are the teacher's and stay frozen, so
    the distilled model generates at ``cfg_scale`` 1 what the teacher generates with guidance. Voices of the
    speaker table; ``guide``: what the null branch is without (:meth:`ARDriftTTS.null_branch`)."""
    dc = cfg.drift.distill
    text, text_len, y, y_len, spk = (batch[k].to(device, non_blocking=True)
                                     for k in ("text", "text_len", "mel", "mel_len", "spk"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=y.is_cuda)
    B, k = y.shape[0], model.patch
    scale, guide = float(dc.get("cfg_scale", 1.5)), dc.get("guide", "text")
    with torch.no_grad():
        y_mask = sequence_mask(y_len, y.shape[-1])[:, None].float()
        with amp:
            h, mu, _, x_mask = teacher.encoder(text, text_len, spk)
        attn, _ = align(mu.float(), x_mask, y, y_mask)
        cond, y = teacher.pad_frames(teacher.frame_condition(h.float(), mu.float(), attn)), teacher.pad_frames(y)
        level = torch.rand(B, device=device) * float(cfg.drift.get("input_noise", 0.0))
        hist = y + level[:, None, None] * torch.randn_like(y)  # the same noisy history for teacher and student
        tokens = teacher._tokens(None, B)
        cp, spk_b, tok_b = teacher.null_branch(teacher.cond_patches(cond), spk, tokens, guide)
        yp = teacher.squeeze(patchify(hist, k))
        x_prev = torch.cat([teacher.bos.expand(B, 1, -1), yp[:, :-1]], 1)
        with amp:
            s = teacher.backbone(torch.cat([x_prev, x_prev]), cp, spk_b, tokens=tok_b).float()
        guided = s[B:] + scale * (s[:B] - s[B:])
    yp = model.squeeze(patchify(hist, k))
    x_prev = torch.cat([model.bos.expand(B, 1, -1), yp[:, :-1]], 1)
    with amp:
        out = model.backbone(x_prev, model.cond_patches(cond), spk, tokens=model._tokens(None, B)).float()
    valid = (torch.arange(out.shape[1], device=device)[None] * k < y_len[:, None]).float()[..., None]
    loss = ((out - guided) ** 2 * valid).sum() / (valid.sum() * out.shape[-1])
    with torch.no_grad():
        gap = ((s[:B] - guided) ** 2 * valid).sum() / (valid.sum() * out.shape[-1])  # what guidance changes
    return loss, {"loss": loss.detach(), "state_mse": loss.detach(), "guided_gap": gap}


class Teacher:
    """The frozen model that draws positives (#98, ``drift.teacher``): a trained ARDrift model (``path``), whose
    backbone gives the state of every row from the clean recorded history, and its one-step head (teacher A,
    self-distillation) or a multi-step flow-matching head trained on those states (``head``: teacher B,
    ``scripts/head_finetune.py --loss flow``)."""

    def __init__(self, tc, device):
        from .models.ar import FlowHead
        from .train import load_tts

        self.model = load_tts(tc.path, device)[0].eval().requires_grad_(False)
        self.positives, self.temperature = int(tc.get("positives", 15)), float(tc.get("temperature", 1.0))
        self.steps, self.flow = int(tc.get("steps", 8)), None
        if tc.get("head"):
            ck = torch.load(tc.head, map_location="cpu", weights_only=False)
            if ck.get("kind") != "flow":
                raise ValueError(f"{tc.head} is not a flow head (scripts/head_finetune.py --loss flow)")
            m = self.model
            self.flow = FlowHead(m.x_in.out_features, m.head.out.out_features, m.head.inp.out_features,
                                 len(m.head.blocks)).to(device).eval().requires_grad_(False)
            self.flow.load_state_dict(ck["head"])

    @torch.no_grad()
    def draw(self, text, text_len, spk, spk_emb, attn, y, idx) -> torch.Tensor:
        """Positives ``[R, P, D]`` for the rows ``idx`` (indices into the flattened steps): the teacher's own text
        condition under the student's alignment ``attn``, its backbone on the recorded history ``y``, its head.
        Voices of the speaker table (a teacher with a reference encoder reads its null tokens)."""
        m, P = self.model, self.positives
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=y.is_cuda)
        with amp:
            h, mu, _, _ = m.encoder(text, text_len, spk, spk_emb)
        cond = m.pad_frames(m.frame_condition(h.float(), mu.float(), attn))
        cond = torch.nn.functional.pad(cond, (y.shape[-1] - cond.shape[-1], 0))  # a prompt in front: no condition
        with amp:
            s = m.teacher_forced(cond, y, spk).flatten(0, 1)[idx]
        base = m.squeeze(patchify(cond[:, : m.n_mels], m.patch).flatten(0, 1)[idx])
        z = torch.randn(s.shape[0] * P, m.head.noise_dim, device=y.device) * self.temperature
        with amp:
            rows = s.repeat_interleave(P, 0)
            res = self.flow.sample(rows, z, self.steps) if self.flow is not None else m.head(rows, z)
        return base[:, None] + res.float().view(s.shape[0], P, -1)


def training_step(model: ARDriftTTS, batch: dict, cfg: Config, device, tau: torch.Tensor,
                  mel_std: float | None = None, detail: bool = False, mae=None, teacher=None) -> tuple:
    """One training step; ``mel_std``: the scalar std of the mel statistics (for ``drift.energy_bands``).
    ``detail``: the metrics also hold every feature map's drift loss, raw field and data mass (``loss/<map>``,
    ``force/<map>``, ``p_data/<map>``), for the evaluation in the loop. ``mae``: the frozen Mel-MAE of
    ``drift.window``; ``teacher``: the :class:`Teacher` of ``drift.teacher``."""
    dc, lc = cfg.drift, cfg.loss
    text, text_len, y, y_len, spk = (batch[k].to(device, non_blocking=True)
                                     for k in ("text", "text_len", "mel", "mel_len", "spk"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=y.is_cuda)
    B = y.shape[0]
    if dc.get("target_noise"):  # a sample of the codec's posterior instead of its mean (#92)
        std = batch.get("mel_std")
        y = sample_targets(y, y_len, dc.target_noise, model.patch if model.tied else 1,
                           None if std is None else std.to(device, non_blocking=True))
    unknown = spk < 0  # voices without a speaker row (a corpus streamed with `speakers: prompt`, #104)
    if unknown.any():
        if not model.prompted and model.ref_enc is None:
            raise ValueError("utterances without a speaker row need a model trained with prompts or references "
                             "(model.ar.prompt, model.ar.reference)")
        spk = torch.where(unknown, torch.full_like(spk, model.prompt_row), spk)  # without a prompt: the null voice
    lead, spk_emb, spk_in = torch.zeros_like(y_len), None, spk  # prompt frames, the encoder's and the backbone's voice
    prompting = model.prompted and dc.get("prompt")
    referencing = model.ref_enc is not None
    dropped = torch.zeros(B, dtype=torch.bool, device=device)  # the speaker row is the prompted one
    has_ref, tokens, ref_vec = dropped, None, None
    no_voice = dropped  # utterances trained without any voice: the null branch of voice guidance (#99)
    if float(dc.get("voice_dropout", 0.0)) > 0:
        if not model.prompted and not referencing:
            raise ValueError("drift.voice_dropout needs the row of a voice that is not in the table "
                             "(model.ar.prompt or model.ar.reference)")
        no_voice = torch.rand(B, device=device) < float(dc.voice_dropout)
    if referencing and batch.get("ref") is not None:  # voices from a reference (#96): another utterance's crop
        rc = dc.get("reference") or {}
        ref, ref_len = batch["ref"].to(device, non_blocking=True), batch["ref_len"].to(device, non_blocking=True)
        has_ref = (ref_len > 0) & (torch.rand(B, device=device) < float(rc.get("prob", 0.8))) & ~no_voice
        with amp:
            tokens, ref_vec = model.reference_tokens(ref, ref_len, has_ref)
    if prompting:  # voices from a prompt (#95): the utterance's own beginning, cut off at a word
        # a voice without a row is cut whenever it can be, unless it has a reference: then the reference alone
        # is one of its conditions too (with a prompt always, the reference would never be read without one)
        cut = cut_prompts(model, text, text_len, y, y_len, spk, dc.prompt,
                          frontend(cfg.model.text.get("language")).SYMBOL_TO_ID[" "], unknown & ~has_ref, no_voice)
        text, text_len, y, y_len, lead = cut["text"], cut["text_len"], cut["mel"], cut["mel_len"], cut["prompt_len"]
        dropped = cut["dropped"]
    if has_ref.any():  # a reference alone: its own draw of the speaker-ID dropout
        p_id = float((dc.get("reference") or {}).get("id_dropout", 0.5))
        dropped = dropped | (has_ref & (lead == 0) & (torch.rand(B, device=device) < p_id))
    if prompting or referencing:
        dropped = dropped | (unknown & ((lead > 0) | has_ref)) | no_voice  # a voice without a row never keeps one
        spk_emb = model.encoder.spk(torch.where(no_voice, torch.full_like(spk, model.prompt_row), spk))
        if prompting:
            with amp:
                vec = model.prompt_vector(cut["prompt"], lead)
            spk_emb = torch.where((dropped & (lead > 0))[:, None], vec.float(), spk_emb)
        if ref_vec is not None:  # with both, the text encoder's voice is the reference's
            spk_emb = torch.where((dropped & has_ref)[:, None], ref_vec.float(), spk_emb)
        spk_in = torch.where(dropped, torch.full_like(spk, model.prompt_row), spk)
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
            own = model.gta(cond, y, spk_in, float(dc.get("self_history_temperature", 0.5)), tokens=tokens).float()
        swap = (torch.rand(B, device=device) < share)[:, None, None]
        swap = swap & (torch.arange(y.shape[-1], device=device)[None, None] >= lead[:, None, None])  # not a prompt
        hist = torch.where(swap, own, y)
    rolled = 0
    rc = dc.get("rollout")
    if rc and float(torch.rand(())) < float(rc.get("prob", 0.5)):  # several free-running patches as history (#101)
        lo, hi = (int(v) for v in rc.get("steps", [4, 16]))
        first, last = int(lead.max()) // k + 1, int((lead + y_len).min()) // k  # steps every utterance has
        n = min(int(torch.randint(lo, hi + 1, ())), last - first)
        if n >= 1:
            t0 = int(torch.randint(first, last - n + 1, ()))
            with amp:
                roll = model.rollout(cond, y, spk_in, t0, n, float(rc.get("temperature", 0.5)), tokens=tokens)
            hist = hist.clone()
            hist[..., t0 * k: (t0 + n) * k] = unpatchify(roll.float(), k, C)
            rolled = n
    with amp:
        s = model.teacher_forced(cond, hist, spk_in, noise, drop, tokens)  # [B, S, dim]
    S = s.shape[1]
    pos = torch.arange(S, device=device)[None] * k
    valid = (pos >= lead[:, None]) & (pos < (lead + y_len)[:, None])  # steps that start inside the utterance
    idx = valid.flatten().nonzero()[:, 0]
    if idx.numel() > dc.max_rows:  # a random subset of the steps (rows) keeps the head's batch bounded
        idx = idx[torch.randperm(idx.numel(), device=device)[: dc.max_rows]]
    rows = s.flatten(0, 1)[idx]  # [R, dim]
    # what the head draws: the patch, or one frame for all of it (model.ar.tied, #92)
    target = model.squeeze(patchify(y, k).flatten(0, 1)[idx])  # [R, k * C] (tied: [R, C])
    base = model.squeeze(patchify(cond[:, :C], k).flatten(0, 1)[idx])  # the aligned prior mean: the head's residual
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
    kf = 1 if model.tied else k  # frames in what the head draws
    f_gen = ar_features(x, kf, C, **opts)
    positives = target[:, None]  # [R, 1, D]: the recording
    if teacher is not None:  # ... and the teacher's samples for the same step (#98)
        if prompting and (lead > 0).any():
            raise ValueError("drift.teacher does not go with prompts (drift.prompt): the teacher needs the cut text")
        drawn = teacher.draw(text, text_len, spk_in, spk_emb, attn, y, idx)
        positives = torch.cat([positives, drawn], 1)
    with torch.no_grad():
        f_pos = ar_features(positives, kf, C, **opts)
    kind = dc.get("loss", "field")
    l_drift, info = kernel_loss(f_gen, f_pos, kind, tau, dc.get("key_weights"))
    wc = dc.get("window")
    if wc:  # kernel maps over a window: the recorded patches before the step, then the step's patch (#97)
        if mae is None or model.tied or k * C != x.shape[-1]:
            raise ValueError("drift.window needs the Mel-MAE (drift.window.mae) and a log-mel model")
        n, rw, gw = int(wc.get("context", 3)), int(wc.get("rows", 256)), int(wc.get("samples", 16))
        hist = history_patches(patchify(y, k), n).flatten(0, 1)[idx[:rw]]  # [Rw, n, k * C]: the recording's
        with amp:
            w_gen = window_features(mae, hist, x[:rw, :gw], k, C)
            with torch.no_grad():
                w_pos = window_features(mae, hist, positives[:rw], k, C)
        l_win, w_info = kernel_loss(w_gen, w_pos, wc.get("loss", "energy"), tau, None, wc.get("max_locations", 32))
        l_drift = l_drift + float(wc.get("weight", 1.0)) * l_win
        info = {**info, **{kk: v for kk, v in w_info.items() if kk != "tau_loss"},
                "tau_loss": info["tau_loss"] + w_info["tau_loss"], "window": l_win.detach() - w_info["tau_loss"]}
    loss = lc.drift * l_drift + lc.prior * l_prior + lc.duration * l_dur
    with torch.no_grad():
        metrics = {"loss": loss.detach(), "drift": l_drift.detach() - info["tau_loss"], "tau_loss": info["tau_loss"],
                   "prior": l_prior.detach(), "duration": l_dur.detach(),
                   "centroid_mse": ((x.mean(1) - target) ** 2).mean(), "sample_std": x.std(1).mean(),
                   "rows": float(R), "tau": tau.detach().reshape(())}
        for name in ("p_data", "spread"):  # the field's data mass; the energy score's spread (1: as the data's)
            vals = [v for kk, v in info.items() if kk.startswith(name + "/")]
            if vals:
                metrics[name] = torch.stack(vals).mean()
        if "window" in info:
            metrics["window"] = info["window"]
        if rc:
            metrics["rollout"] = float(rolled)
        if teacher is not None:  # the teacher's spread beside the student's, and how far its mean is from the data
            metrics["teacher_std"] = positives[:, 1:].std(1).mean()
            metrics["teacher_mse"] = ((positives[:, 1:].mean(1) - target) ** 2).mean()
        if prompting or referencing:
            metrics["prompted"] = (lead > 0).float().mean()
            metrics["id_dropped"] = dropped.float().mean()
            metrics["no_row"] = unknown.float().mean()
        if referencing:
            metrics["referenced"] = has_ref.float().mean()
            metrics["ref_only"] = (has_ref & (lead == 0) & dropped).float().mean()  # the reference is the voice
            metrics["no_voice"] = ((unknown | no_voice) & ~has_ref & (lead == 0)).float().mean()
        if detail:
            metrics.update({kk: v for kk, v in info.items()
                            if kk.startswith(("loss/", "force/", "p_data/", "spread/"))})
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
    distill = None
    if cfg.drift.get("distill"):  # the guided state into the backbone (#99): the encoder and the head stay as they are
        from .train import load_tts

        distill = load_tts(cfg.drift.distill.teacher, device)[0].eval().requires_grad_(False)
        model.encoder.requires_grad_(False)
        model.head.requires_grad_(False)
        print(f"distilling guidance {float(cfg.drift.distill.get('cfg_scale', 1.5)):g} "
              f"({cfg.drift.distill.get('guide', 'text')}) of {cfg.drift.distill.teacher} into the backbone",
              flush=True)
    teacher = None
    if cfg.drift.get("teacher"):  # a frozen model draws positives beside the recording (#98)
        teacher = Teacher(cfg.drift.teacher, device)
        print(f"teacher: {cfg.drift.teacher.path}, {teacher.positives} positives per row"
              + (f" from the flow head {cfg.drift.teacher.head} ({teacher.steps} steps)" if teacher.flow else ""),
              flush=True)
    mae = None
    if cfg.drift.get("window"):  # the frozen Mel-MAE of the window maps (#97)
        from .train_mae import load_mae

        mae = load_mae(cfg.drift.window.mae, device)
        mae.eval().requires_grad_(False)
        print(f"window maps: {cfg.drift.window.mae}, {int(cfg.drift.window.get('context', 3))} patches of history",
              flush=True)

    def evaluate(at: int) -> None:  # every EMA on the held-out objective and on synthesised sentences (#102)
        t = time.time()
        for n, (name, e) in enumerate(emas.items()):
            res = heldout_metrics(lambda b, m=e.model: training_step(m, b, cfg, device, tau, mel_std, detail=True,
                                                                     mae=mae, teacher=teacher)[1], heldout)
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
        if distill is not None:
            loss, metrics = distill_step(model, distill, next(it), cfg, device)
        else:
            loss, metrics = training_step(model, next(it), cfg, device, tau, mel_std, mae=mae, teacher=teacher)
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

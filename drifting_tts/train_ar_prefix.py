"""The training step of ARDrift v2 (#103, :class:`~drifting_tts.models.ar_prefix.ARPrefixTTS`): the text as a prefix,
no alignment, no duration predictor. ``train-ar`` runs it for ``model.kind: ar_prefix``.

One step:

1. the characters of every utterance (and, for a share of them, a crop of another utterance of the speaker as a
   voice prompt: the loader's ``ref``, ``drift.reference``) stand in front of its mel patches;
2. the causal transformer, teacher-forced on the recorded patches plus noise of a random level, gives one state per
   patch and one for the step after the last;
3. at ``drift.max_rows`` random patches the head draws ``drift.gen_per_cond`` samples of the **whole patch** (there
   is no prior mean to add to); the drifting loss, or the energy score (``drift.loss``), on the maps of
   :func:`~drifting_tts.models.ar.ar_features`;
4. the end head: binary cross-entropy at every step, 1 at the step after the last patch (``loss.eos``, 0.1).

The log's ``text_gap / null_steps`` says whether the model reads: how much larger the error of the mean patch is on
the utterances that lost their text (``drift.cond_dropout``) than on the others.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config
from .models.ar import ar_features, patchify
from .models.ar_prefix import ARPrefixTTS, characters, eos_loss


def training_step(model: ARPrefixTTS, batch: dict, cfg: Config, device, tau: torch.Tensor,
                  mel_std: float | None = None, detail: bool = False, **_) -> tuple:
    from .train_ar import kernel_loss

    dc, lc = cfg.drift, cfg.loss
    text, text_len, y, y_len, spk = (batch[k].to(device, non_blocking=True)
                                     for k in ("text", "text_len", "mel", "mel_len", "spk"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=y.is_cuda)
    chars, n = characters(text, text_len)
    B, k, C = y.shape[0], model.patch, model.n_mels
    y = F.pad(y, (0, (-y.shape[-1]) % k))
    steps = -(-y_len // k)  # patches that start inside the utterance
    unknown = spk < 0  # voices without a speaker row
    prompt = prompt_len = None
    dropped = torch.zeros(B, dtype=torch.bool, device=device)
    if model.prompted:
        row = torch.full_like(spk, model.prompt_row)
        if batch.get("ref") is not None:  # the voice from another utterance of the speaker, for a share of them
            rc = dc.get("reference") or {}
            prompt = batch["ref"].to(device, non_blocking=True)
            prompt_len = batch["ref_len"].to(device, non_blocking=True)
            use = (prompt_len > 0) & (torch.rand(B, device=device) < float(rc.get("prob", 0.5)))
            prompt_len = torch.where(use, prompt_len, torch.zeros_like(prompt_len))
            dropped = use & ((torch.rand(B, device=device) < float(rc.get("id_dropout", 0.5))) | unknown)
        spk = torch.where(dropped | unknown, row, spk)  # without a prompt and a row: no voice
    elif unknown.any():
        raise ValueError("utterances without a speaker row need a model trained with prompts (model.ar.prompt)")
    noise = torch.rand(B, device=device) * float(dc.get("input_noise", 0.0))
    p_drop = float(dc.get("cond_dropout", 0.0))
    drop = torch.rand(B, device=device) < p_drop if p_drop > 0 else None
    with amp:
        s = model.teacher_forced(chars, n, y, spk, noise, prompt, prompt_len, drop)  # [B, S + 1, dim]
    S = s.shape[1] - 1
    valid = torch.arange(S, device=device)[None] < steps[:, None]
    idx = valid.flatten().nonzero()[:, 0]
    if idx.numel() > dc.max_rows:
        idx = idx[torch.randperm(idx.numel(), device=device)[: dc.max_rows]]
    rows = s[:, :S].flatten(0, 1)[idx]
    target = patchify(y, k).flatten(0, 1)[idx]
    R, G = rows.shape[0], int(dc.gen_per_cond)
    z = torch.randn(R * G, model.head.noise_dim, device=device)
    with amp:
        x = model.head(rows.repeat_interleave(G, 0), z).float().view(R, G, -1)
    opts = {"detail": bool(dc.get("detail", True)), "energy_bands": dc.get("energy_bands")}
    if opts["energy_bands"]:
        opts["energy_scale"] = 2.0 * float(mel_std)
    f_gen = ar_features(x, k, C, **opts)
    with torch.no_grad():
        f_pos = ar_features(target[:, None], k, C, **opts)
    l_drift, info = kernel_loss(f_gen, f_pos, dc.get("loss", "field"), tau, dc.get("key_weights"))
    l_eos, acc = eos_loss(model.eos(s)[..., 0], steps)
    loss = lc.drift * l_drift + float(lc.get("eos", 0.1)) * l_eos
    with torch.no_grad():
        metrics = {"loss": loss.detach(), "drift": l_drift.detach() - info["tau_loss"], "tau_loss": info["tau_loss"],
                   "eos": l_eos.detach(), "eos_acc": acc, "centroid_mse": ((x.mean(1) - target) ** 2).mean(),
                   "sample_std": x.std(1).mean(), "rows": float(R), "tau": tau.detach().reshape(())}
        for name in ("p_data", "spread"):
            vals = [v for kk, v in info.items() if kk.startswith(name + "/")]
            if vals:
                metrics[name] = torch.stack(vals).mean()
        # does the model read? The error of the mean patch without the text over the error with it, minus 1, on
        # the steps that have utterances of both kinds (`null_steps`: their share; text_gap / null_steps is the
        # mean). 0: the states do not depend on the text.
        gap, both = 0.0, 0.0
        if drop is not None:
            null = drop[idx // S]
            if bool(null.any()) and not bool(null.all()):
                err = ((x.mean(1) - target) ** 2).mean(1)
                gap, both = err[null].mean() / err[~null].mean() - 1.0, 1.0
        metrics["text_gap"], metrics["null_steps"] = gap, both
        if model.prompted:
            metrics["prompted"] = (prompt_len > 0).float().mean() if prompt_len is not None else 0.0
            metrics["id_dropped"] = dropped.float().mean()
        if detail:
            metrics.update({kk: v for kk, v in info.items()
                            if kk.startswith(("loss/", "force/", "p_data/", "spread/"))})
    return loss, metrics

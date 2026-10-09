"""Stochastic prosody predictor: noise + text features + speaker -> per-token durations and pitch in one pass (#39).

The generator of a pitch-conditioned :class:`~drifting_tts.models.tts.DriftingTTS` is trained on ground-truth MAS
durations and ground-truth token pitch, so its deterministic duration / pitch regressors can be replaced at inference
by a sampler without retraining it. :class:`ProsodyNet` maps the frozen encoder states ``h``, the speaker embedding,
the regressors' own predictions and noise to three channels per token:

* ``ld``: the log-duration in frames (MAS durations are >= 1 frame, so ``log d >= 0``; inference rounds ``exp(ld)``);
* ``p``: a continuous token pitch contour (normalised log-F0, unvoiced tokens interpolated from their neighbours);
* a voicing logit: unvoiced tokens get pitch 0, as in :func:`~drifting_tts.models.text_encoder.token_pitch`.

``ld`` and ``p`` are standardised (:class:`ProsodyStats`) and predicted as a residual over the regressors'
predictions. Three training objectives share the network (``kind``):

* ``drift``: per-token and global noise in, one pass; drifting loss on multi-scale feature maps of the sequence
  (:func:`prosody_features`), one positive (the recording's prosody) and ``G`` samples per utterance;
* ``mse``: no noise, regression (the deterministic status quo with the same backbone);
* ``flow``: conditional flow matching on the residual, a few Euler steps at inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..text import PUNCTUATION, SYMBOL_TO_ID, SYMBOLS
from .text_encoder import EncoderLayer

KINDS = ("drift", "mse", "flow")
SPACE_ID = SYMBOL_TO_ID[" "]
PUNCT_IDS = [SYMBOL_TO_ID[c] for c in PUNCTUATION if c != " "]


class ProsodyStats(nn.Module):
    """Standardisation of the two channels and of the word / utterance summary features (set from the data)."""

    def __init__(self, word_dim: int = 0):
        super().__init__()
        self.register_buffer("seq", torch.tensor([[0.0, 1.0], [0.0, 1.0]]))  # (ld, p) x (mean, std)
        self.register_buffer("word", torch.zeros(2, 4) + torch.tensor([[0.0], [1.0]]))  # mean / std of 4 features
        self.register_buffer("utt", torch.zeros(2, 6) + torch.tensor([[0.0], [1.0]]))  # mean / std of 6 features
        if word_dim:  # contextual word features (BERT), per dimension
            self.register_buffer("wfeat", torch.zeros(2, word_dim) + torch.tensor([[0.0], [1.0]]))

    def norm(self, ld: Tensor, p: Tensor) -> Tensor:
        """``[..., N]`` log-durations and pitch -> standardised ``[..., 2, N]``."""
        s = self.seq
        return torch.stack([(ld - s[0, 0]) / s[0, 1], (p - s[1, 0]) / s[1, 1]], -2)

    def denorm(self, y: Tensor) -> tuple[Tensor, Tensor]:
        s = self.seq
        return y[..., 0, :] * s[0, 1] + s[0, 0], y[..., 1, :] * s[1, 1] + s[1, 0]


def word_index(ids: Tensor) -> Tensor:
    """Token ids ``[B, N]`` -> word index ``[B, N]`` (words are separated by space tokens)."""
    return torch.cumsum((ids == SPACE_ID).long(), 1)


def boundary_tokens(ids: Tensor) -> Tensor:
    """Tokens between words ``[..., N]`` bool: spaces, punctuation and the blanks next to them (a pause between words
    lives in their durations; the letters' own durations do not include it)."""
    from ..text import BLANK_ID

    mark = ids == SPACE_ID
    for i in PUNCT_IDS:
        mark = mark | (ids == i)
    left = F.pad(mark, (1, 0))[..., :-1]
    right = F.pad(mark, (0, 1))[..., 1:]
    return mark | ((ids == BLANK_ID) & (left | right))


def sentence_tokens(ids: Tensor, lengths: Tensor, boundary: bool = False) -> Tensor:
    """CPU token ids ``[B, N]`` -> sentence features broadcast to the tokens ``[B, DIM, N]``
    (:func:`drifting_tts.sentence_features.sentence_features` of each text); ``boundary`` appends
    :func:`boundary_tokens` as one more channel."""
    from ..sentence_features import DIM, sentence_features
    from ..text import ids_to_text

    word = word_index(ids)
    out = torch.zeros(ids.shape[0], ids.shape[1], DIM + int(boundary))
    for b in range(ids.shape[0]):
        n = int(lengths[b])
        f = torch.from_numpy(sentence_features(ids_to_text(ids[b, :n].tolist())))
        out[b, :n, :DIM] = f[word[b, :n].clamp_max(f.shape[0] - 1)]
    if boundary:
        out[..., DIM] = (boundary_tokens(ids) & (torch.arange(ids.shape[1])[None] < lengths[:, None])).float()
    return out.transpose(1, 2)


def floor_letters(frames: Tensor, ids: Tensor, min_frames: float = 0.0, rel: float = 0.0,
                  ref_frames: Tensor | None = None) -> Tensor:
    """Raise short letters: a letter (a character token and the blank after it) gets at least ``min_frames`` frames
    and at least ``rel`` times its ``ref_frames`` (e.g. the regressors' durations, same shape as ``frames``); the
    missing frames go to the character token. ``frames`` / ``ids`` ``[B, N]``."""
    table = torch.tensor([len(c) == 1 and c.isalpha() for c in SYMBOLS], device=ids.device)
    letter = table[ids]
    total = frames + F.pad(frames[:, 1:], (0, 1))
    need = torch.full_like(frames, float(min_frames))
    if rel > 0 and ref_frames is not None:
        need = torch.maximum(need, torch.round(rel * (ref_frames + F.pad(ref_frames[:, 1:], (0, 1)))))
    return frames + (need - total).clamp_min(0) * letter


def _masked_mean_std(x: Tensor, m: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    n = m.sum(dim).clamp_min(1)
    mean = (x * m).sum(dim) / n
    var = (((x - mean.unsqueeze(dim)) ** 2) * m).sum(dim) / n
    return mean, (var + 1e-6).sqrt()


def summary_features(y: Tensor, mask: Tensor, word: Tensor, stats: ProsodyStats, standardize: bool = True,
                     n_words: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Word and utterance summaries of standardised sequences ``y`` ``[B, S, 2, N]``.

    Word (``[B, S, W, 4]``): mean ld, mean pitch, pitch range, log total frames. Utterance (``[B, S, 1, 6]``): mean
    and std of ld and of pitch, pitch slope over the utterance, log total frames. Returns both and the valid-word
    mask ``[B, W]``. ``n_words``: ``W`` if known on the host (else read from ``word``, a device sync)."""
    B, S, C, N = y.shape
    m = mask[:, None, :].float()  # [B, 1, N]
    ld_raw, _ = stats.denorm(y)
    frames = torch.exp(ld_raw) * m  # [B, S, N]
    W = n_words or (int(word.max()) + 1 if word.numel() else 1)
    onehot = F.one_hot(word.clamp_min(0), W).float() * mask[..., None].float()  # [B, N, W]
    cnt = onehot.sum(1)  # [B, W]
    wvalid = cnt > 0
    wmean = torch.einsum("bscn,bnw->bswc", y, onehot) / cnt.clamp_min(1)[:, None, :, None]  # [B, S, W, 2]
    big = 1e4
    p = y[:, :, 1]  # [B, S, N]
    sel = onehot.bool()[:, None]  # [B, 1, N, W]
    pmax = torch.where(sel, p[..., None], torch.full_like(p[..., None], -big)).amax(2)  # [B, S, W]
    pmin = torch.where(sel, p[..., None], torch.full_like(p[..., None], big)).amin(2)
    wrange = torch.where(wvalid[:, None], pmax - pmin, torch.zeros_like(pmax))
    wframes = torch.log(torch.einsum("bsn,bnw->bsw", frames, onehot).clamp_min(1.0))
    word_f = torch.cat([wmean, wrange[..., None], wframes[..., None]], -1)  # [B, S, W, 4]

    mean_ld, std_ld = _masked_mean_std(y[:, :, 0], m, -1)
    mean_p, std_p = _masked_mean_std(p, m, -1)
    n = mask.float().sum(-1, keepdim=True).clamp_min(2)  # [B, 1]
    pos = torch.arange(N, device=y.device)[None].float() / (n - 1) - 0.5  # [B, N]
    pos = (pos - (pos * mask).sum(-1, keepdim=True) / n) * mask
    slope = ((p - mean_p[..., None]) * pos[:, None] * m).sum(-1) / (pos**2).sum(-1)[:, None].clamp_min(1e-6)
    total = torch.log(frames.sum(-1).clamp_min(1.0))
    utt_f = torch.stack([mean_ld, std_ld, mean_p, std_p, slope, total], -1)[:, :, None]  # [B, S, 1, 6]
    if standardize:
        word_f = (word_f - stats.word[0]) / stats.word[1]
        utt_f = (utt_f - stats.utt[0]) / stats.utt[1]
    return word_f, utt_f, wvalid


def prosody_features(y: Tensor, mask: Tensor, word: Tensor, stats: ProsodyStats, pools=(3, 9, 27),
                     windows=(8, 32), n_words: int | None = None) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Multi-scale feature maps of standardised prosody sequences ``y`` ``[B, S, 2, N]`` (mask ``[B, N]``).

    Every map is ``[B, S, L, D]`` with a valid-location mask ``[B, L]``; each (utterance, location) is one drift
    problem. Maps: ``tok`` (per-token values), ``d1`` (first differences: smoothness), ``avg<k>`` (masked means over
    ``k`` tokens, stride ``k // 2``), ``win<k>`` (the whole window of ``k`` tokens flattened: local shape), ``word``
    and ``utt`` (standardised summaries, :func:`summary_features`)."""
    B, S, C, N = y.shape
    mf = mask[:, None, None, :].float()
    y = y * mf
    feats, valid = {"tok": y.transpose(2, 3)}, {"tok": mask}
    feats["d1"] = (y[..., 1:] - y[..., :-1]).transpose(2, 3) * mf[..., 1:].transpose(2, 3)
    valid["d1"] = mask[:, 1:] & mask[:, :-1]
    flat, mflat = y.reshape(B * S, C, N), mask[:, None].float()  # [B*S, C, N], [B, 1, N]
    for k in pools:
        stride = max(1, k // 2)
        pad = (0, max(0, k - N))
        s = F.avg_pool1d(F.pad(flat, pad), k, stride) * k
        c = F.avg_pool1d(F.pad(mflat, pad), k, stride) * k  # [B, 1, L]
        L = s.shape[-1]
        feats[f"avg{k}"] = (s.view(B, S, C, L) / c.clamp_min(1)[:, None]).transpose(2, 3)
        valid[f"avg{k}"] = c[:, 0] > 0
    for k in windows:
        stride = max(1, k // 2)
        n_win = max(1, math.ceil((N - k) / stride) + 1)
        total = (n_win - 1) * stride + k
        w = F.pad(y, (0, total - N)).unfold(-1, k, stride)  # [B, S, C, L, k]
        feats[f"win{k}"] = w.permute(0, 1, 3, 2, 4).reshape(B, S, n_win, C * k)
        starts = torch.arange(n_win, device=y.device) * stride
        valid[f"win{k}"] = starts[None] < mask.sum(-1, keepdim=True)
    word_f, utt_f, wvalid = summary_features(y, mask, word, stats, n_words=n_words)
    feats["word"], valid["word"] = word_f, wvalid
    feats["utt"], valid["utt"] = utt_f, torch.ones(B, 1, dtype=torch.bool, device=y.device)
    return feats, valid


def _time_embedding(t: Tensor, dim: int = 64) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class ProsodyNet(nn.Module):
    """Non-causal transformer over tokens; see the module docstring. ``cond`` = ``[h; speaker; regressors]``."""

    def __init__(self, cond_dim: int, kind: str = "drift", d: int = 256, layers: int = 4, heads: int = 4,
                 ffn: int = 1024, dropout: float = 0.0, noise_tok: int = 16, noise_glob: int = 32,
                 out_init: float = 0.1, word_dim: int = 0, word_model: str | None = None, sent_dim: int = 0,
                 ctx_pitch_only: bool = False, pitch_layers: int = 2, ctx_boundaries: bool = False):
        """``word_dim`` / ``sent_dim``: contextual word features (:mod:`drifting_tts.word_features`) and sentence
        features (:mod:`drifting_tts.sentence_features`), appended to ``cond`` in that order. ``ctx_pitch_only``:
        they bypass the trunk, which predicts the log-duration alone, and enter ``pitch_layers`` more layers on top
        of it that predict the pitch and the voicing, so the durations do not depend on them. ``ctx_boundaries``
        (with ``ctx_pitch_only``; the last sentence channel is then :func:`boundary_tokens`): that branch also
        predicts the durations of the tokens between words (phrase breaks), the trunk those of the letters."""
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind, self.noise_tok, self.noise_glob, self.word_dim = kind, noise_tok, noise_glob, word_dim
        self.sent_dim, self.cond_dim, self.ctx_pitch_only = sent_dim, cond_dim, ctx_pitch_only and word_dim + sent_dim > 0
        self.ctx_boundaries = bool(ctx_boundaries and self.ctx_pitch_only and sent_dim)
        ctx_dim = word_dim + sent_dim
        c_in = (cond_dim + (0 if self.ctx_pitch_only else ctx_dim) + (noise_tok if kind == "drift" else 0)
                + (2 if kind == "flow" else 0))
        self.inp = nn.Conv1d(c_in, d, 1)
        g_dim = noise_glob if kind == "drift" else 64 if kind == "flow" else 0
        self.glob = nn.ModuleList([nn.Linear(g_dim, d) for _ in range(layers + 1)]) if g_dim else None
        self.prenet = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2) for _ in range(2)])
        self.prenet_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.layers = nn.ModuleList([EncoderLayer(d, heads, ffn, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1 if self.ctx_pitch_only else 3)
        outs = [self.out]
        if self.ctx_pitch_only:  # the pitch branch: trunk states + context -> pitch, voicing
            self.ctx_in = nn.Conv1d(ctx_dim, d, 1)
            self.pglob = nn.ModuleList([nn.Linear(g_dim, d) for _ in range(pitch_layers)]) if g_dim else None
            self.players = nn.ModuleList([EncoderLayer(d, heads, ffn, dropout) for _ in range(pitch_layers)])
            self.pnorm = nn.LayerNorm(d)
            self.pout = nn.Linear(d, 3 if self.ctx_boundaries else 2)
            outs.append(self.pout)
        with torch.no_grad():
            for o in outs:
                o.weight.mul_(out_init)
                o.bias.zero_()

    def forward(self, cond: Tensor, mask: Tensor, z_tok: Tensor | None = None, z_glob: Tensor | None = None,
                x_t: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        """``cond`` ``[B, C, N]``, ``mask`` ``[B, 1, N]`` -> ``[B, 3, N]``: residual (ld, p) or velocity; voicing."""
        ctx = None
        if self.ctx_pitch_only:
            cond, ctx = cond[:, : self.cond_dim], cond[:, self.cond_dim:]
        parts = [cond]
        g = None
        if self.kind == "drift":
            parts.append(z_tok)
            g = z_glob
        elif self.kind == "flow":
            parts.append(x_t)
            g = _time_embedding(t)
        m = mask.transpose(1, 2)  # [B, N, 1]
        x = self.inp(torch.cat(parts, 1) * mask).transpose(1, 2)
        if g is not None:
            x = x + self.glob[0](g)[:, None]
        for conv, norm in zip(self.prenet, self.prenet_norms):
            x = x + norm(F.gelu(conv((x * m).transpose(1, 2)).transpose(1, 2)))
        x = x * m
        for i, layer in enumerate(self.layers):
            if g is not None:
                x = x + self.glob[i + 1](g)[:, None] * m
            x = layer(x, m)
        if ctx is None:
            return (self.out(self.norm(x)) * m).transpose(1, 2)
        xp = x + self.ctx_in(ctx * mask).transpose(1, 2) * m
        for i, layer in enumerate(self.players):
            if g is not None:
                xp = xp + self.pglob[i](g)[:, None] * m
            xp = layer(xp, m)
        ld, pv = self.out(self.norm(x)), self.pout(self.pnorm(xp))
        if self.ctx_boundaries:  # durations between words from the context branch
            ld = torch.where(ctx[:, -1:].transpose(1, 2) > 0.5, pv[..., :1], ld)
            pv = pv[..., 1:]
        return (torch.cat([ld, pv], -1) * m).transpose(1, 2)


class ProsodyPredictor(nn.Module):
    """A trained :class:`ProsodyNet` with its statistics: samples durations and token pitch for a TTS model."""

    def __init__(self, net_cfg: dict, cond_dim: int):
        super().__init__()
        self.net_cfg = dict(net_cfg)
        self.net = ProsodyNet(cond_dim, **self.net_cfg)
        self.stats = ProsodyStats(self.net.word_dim)
        self.duration_scales: dict[int, float] = {}
        self.temperature: float | None = None  # preferred prosody temperature (train-prosody --calibrate-only)
        self.pitch_temperature: float | None = None  # preferred one of the pitch channel (None: the same)
        self.flow_steps = 8
        self._word_encoder = None

    @property
    def kind(self) -> str:
        return self.net.kind

    @staticmethod
    def condition(tts, h: Tensor, x_mask: Tensor, spk: Tensor, logw_det: Tensor, stats: ProsodyStats,
                  word_tok: Tensor | None = None, sent_tok: Tensor | None = None) -> Tensor:
        """``[h; speaker embedding; standardised regressor predictions (; word features) (; sentence features)]``
        ``[B, C, N]``.

        ``word_tok``: contextual word features broadcast to the tokens ``[B, word_dim, N]`` (standardised here);
        ``sent_tok``: sentence features ``[B, sent_dim, N]`` (:meth:`sent_tokens`, used as they are)."""
        s = tts.encoder.spk(spk)[:, :, None].expand(-1, -1, h.shape[-1])
        pitch_det = tts.pitch_predictor(torch.cat([h, s], 1), x_mask)
        base = stats.norm(logw_det[:, 0], pitch_det[:, 0]) * x_mask
        parts = [h, s, base]
        if word_tok is not None:
            parts.append((word_tok - stats.wfeat[0, :, None]) / stats.wfeat[1, :, None])
        if sent_tok is not None:
            parts.append(sent_tok)
        return torch.cat(parts, 1) * x_mask, base

    def sent_tokens(self, text: Tensor, text_len: Tensor) -> Tensor | None:
        """Sentence features (:func:`drifting_tts.sentence_features.sentence_features`) of the texts behind token
        ids, broadcast to the tokens ``[B, sent_dim, N]`` (``None`` without them)."""
        if not self.net.sent_dim:
            return None
        return sentence_tokens(text.cpu(), text_len.cpu(), self.net.ctx_boundaries).to(text.device, non_blocking=True)

    def word_tokens(self, text: Tensor, text_len: Tensor) -> Tensor | None:
        """Word features of the texts behind token ids, broadcast to the tokens ``[B, word_dim, N]`` (``None``
        without word features)."""
        if not self.net.word_dim:
            return None
        from ..text import ids_to_text
        from ..word_features import WordEncoder

        if self._word_encoder is None:
            self._word_encoder = WordEncoder(self.net_cfg["word_model"], str(text.device))
        texts = [ids_to_text(text[b, : int(text_len[b])].tolist()) for b in range(text.shape[0])]
        feats = self._word_encoder(texts)
        word = word_index(text)
        out = torch.zeros(text.shape[0], text.shape[1], self.net.word_dim, device=text.device)
        for b, f in enumerate(feats):
            out[b] = f.to(text.device)[word[b].clamp_max(f.shape[0] - 1)]
        return out.transpose(1, 2)

    def sample(self, cond: Tensor, base: Tensor, mask: Tensor, temperature: float = 1.0,
               generator: torch.Generator | None = None, steps: int | None = None, spread: float = 1.0,
               mean_samples: int = 16, pitch_temperature: float | None = None,
               pitch_spread: float | None = None) -> tuple[Tensor, Tensor]:
        """Standardised ``(ld, p)`` ``[B, 2, N]`` and the voicing logit ``[B, N]``; noise from ``generator``.

        ``temperature`` scales the input noise. ``spread`` != 1 (output-space temperature) draws ``mean_samples``
        samples in one batch and returns ``mean + spread * (sample - mean)`` for the first one: below 1 it trades
        expressiveness and seed diversity for per-token accuracy (the noise temperature of the drift sampler mostly
        changes the diversity between seeds, not the spread within an utterance).
        ``pitch_temperature`` / ``pitch_spread`` (``None``: the same as for the durations) set the pitch channel and
        the voicing apart: the network sees one noise draw, scaled by each temperature in a second batch row, and the
        durations come from the first row, the pitch from the second (:meth:`_draw`)."""
        temps = (temperature, temperature if pitch_temperature is None else pitch_temperature)
        spreads = (spread, spread if pitch_spread is None else pitch_spread)
        if spreads == (1.0, 1.0) or self.net.kind == "mse":
            return self._draw(cond, base, mask, temps, generator, steps)
        K, B = mean_samples, cond.shape[0]
        y, v = self._draw(cond.repeat(K, 1, 1), base.repeat(K, 1, 1), mask.repeat(K, 1, 1), temps, generator, steps)
        y = y.view(K, B, *y.shape[1:])
        mean = y.mean(0)
        s = torch.tensor(spreads, device=y.device, dtype=y.dtype)[:, None]  # per channel [2, 1]
        return (mean + s * (y[0] - mean)) * mask, v[:B]

    def _draw(self, cond: Tensor, base: Tensor, mask: Tensor, temperature: float | tuple[float, float],
              generator: torch.Generator | None, steps: int | None) -> tuple[Tensor, Tensor]:
        """One sample per row. ``temperature``: one value, or ``(durations, pitch)``; when the two differ, the same
        unit noise is scaled by each and run as a batch of ``2 B`` (log-duration from the first half, pitch and
        voicing from the second), so equal temperatures give exactly the one-pass sample."""
        td, tp = (temperature, temperature) if isinstance(temperature, (int, float)) else temperature
        B, _, N = cond.shape
        net, dev = self.net, cond.device
        split = td != tp and net.kind != "mse"

        def two(x: Tensor) -> Tensor:
            return torch.cat([x, x]) if split else x

        def scaled(z: Tensor) -> Tensor:
            return torch.cat([z * td, z * tp]) if split else z * td

        def pick(y: Tensor) -> Tensor:  # [2B, C, N] -> [B, C, N]: channel 0 from the first half, the rest second
            return torch.cat([y[:B, :1], y[B:, 1:]], 1) if split else y

        if net.kind == "drift":
            z_tok = torch.randn(B, net.noise_tok, N, device=dev, generator=generator)
            z_glob = torch.randn(B, net.noise_glob, device=dev, generator=generator)
            out = pick(net(two(cond), two(mask), scaled(z_tok), scaled(z_glob)))
            return (base + out[:, :2]) * mask, out[:, 2]
        if net.kind == "mse":
            out = net(cond, mask)
            return (base + out[:, :2]) * mask, out[:, 2]
        steps = steps or self.flow_steps
        x = scaled(torch.randn(B, 2, N, device=dev, generator=generator)) * two(mask)
        c2, m2 = two(cond), two(mask)
        for k in range(steps):
            t = torch.full((x.shape[0],), k / steps, device=dev)
            out = net(c2, m2, x_t=x, t=t)
            x = (x + out[:, :2] / steps) * m2
        return (base + pick(x)) * mask, pick(out)[:, 2]

    def frames_and_pitch(self, y: Tensor, voiced_logit: Tensor, mask: Tensor,
                         length_scale: float | Tensor = 1.0) -> tuple[Tensor, Tensor]:
        """Standardised samples -> integer frames ``[B, N]`` (rounded, >= 1) and token pitch ``[B, 1, N]``."""
        ld, p = self.stats.denorm(y)
        frames = torch.round(torch.exp(ld) * length_scale).clamp_min(1) * mask[:, 0]
        pitch = torch.where(voiced_logit > 0, p, torch.zeros_like(p)) * mask[:, 0]
        return frames, pitch[:, None]

    @torch.no_grad()
    def predict(self, tts, text: Tensor, text_len: Tensor, spk: Tensor, temperature: float = 1.0,
                length_scale: float | Tensor = 1.0, generator: torch.Generator | None = None,
                spread: float = 1.0, pitch_temperature: float | None = None,
                pitch_spread: float | None = None, min_letter_frames: float = 0.0,
                rel_letter_floor: float = 0.0) -> tuple[Tensor, Tensor]:
        """Sample frames per token ``[B, N]`` and token pitch ``[B, 1, N]`` for ``tts`` (its frozen encoder and
        regressors give the condition); pass them to :meth:`DriftingTTS.synthesize` as ``durations`` / ``pitch``.
        ``temperature`` / ``spread`` (``pitch_temperature`` / ``pitch_spread``): see :meth:`sample`.
        ``min_letter_frames`` / ``rel_letter_floor``: :func:`floor_letters`, relative to the regressors' durations."""
        h, _, logw, x_mask = tts.encoder(text, text_len, spk)
        cond, base = self.condition(tts, h, x_mask, spk, logw, self.stats, self.word_tokens(text, text_len),
                                    self.sent_tokens(text, text_len))
        y, vlogit = self.sample(cond, base, x_mask, temperature, generator=generator, spread=spread,
                                pitch_temperature=pitch_temperature, pitch_spread=pitch_spread)
        frames, pitch = self.frames_and_pitch(y, vlogit, x_mask, length_scale)
        if min_letter_frames or rel_letter_floor:
            ref = torch.exp(logw[:, 0]) * length_scale
            frames = floor_letters(frames, text, min_letter_frames, rel_letter_floor, ref) * x_mask[:, 0]
        return frames, pitch

    @classmethod
    def load(cls, path, device="cpu", tts=None) -> ProsodyPredictor:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cond_dim = ck["cond_dim"]
        if tts is not None:
            want = tts.encoder.d + tts.encoder.spk.embedding_dim + 2
            if want != cond_dim:
                raise ValueError(f"prosody model {path} expects {cond_dim} condition channels, the TTS gives {want}")
        p = cls(ck["net_cfg"], cond_dim)
        p.net.load_state_dict(ck["ema"])
        p.stats.load_state_dict(ck["stats"])
        p.duration_scales = {int(k): float(v) for k, v in ck.get("duration_scales", {}).items()}
        p.temperature = ck.get("temperature")
        p.pitch_temperature = ck.get("pitch_temperature")
        p.flow_steps = int(ck.get("flow_steps", 8))
        return p.to(device).eval()

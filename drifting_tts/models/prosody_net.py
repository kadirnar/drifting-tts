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
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..text import PUNCTUATION, SYMBOL_TO_ID
from .text_encoder import EncoderLayer

KINDS = ("drift", "mse", "flow")
# what ProsodyPredictor.load reads: a published checkpoint keeps only these (export_checkpoint)
CHECKPOINT_KEYS = ("ema", "stats", "net_cfg", "cond_dim", "duration_scales", "temperature", "flow_steps",
                   "tts_fingerprint")
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
                 out_init: float = 0.1, word_dim: int = 0, word_model: str | None = None):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind, self.noise_tok, self.noise_glob, self.word_dim = kind, noise_tok, noise_glob, word_dim
        c_in = cond_dim + word_dim + (noise_tok if kind == "drift" else 0) + (2 if kind == "flow" else 0)
        self.inp = nn.Conv1d(c_in, d, 1)
        g_dim = noise_glob if kind == "drift" else 64 if kind == "flow" else 0
        self.glob = nn.ModuleList([nn.Linear(g_dim, d) for _ in range(layers + 1)]) if g_dim else None
        self.prenet = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2) for _ in range(2)])
        self.prenet_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.layers = nn.ModuleList([EncoderLayer(d, heads, ffn, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 3)
        with torch.no_grad():
            self.out.weight.mul_(out_init)
            self.out.bias.zero_()

    def forward(self, cond: Tensor, mask: Tensor, z_tok: Tensor | None = None, z_glob: Tensor | None = None,
                x_t: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        """``cond`` ``[B, C, N]``, ``mask`` ``[B, 1, N]`` -> ``[B, 3, N]``: residual (ld, p) or velocity; voicing."""
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
        return (self.out(self.norm(x)) * m).transpose(1, 2)


class ProsodyPredictor(nn.Module):
    """A trained :class:`ProsodyNet` with its statistics: samples durations and token pitch for a TTS model."""

    def __init__(self, net_cfg: dict, cond_dim: int):
        super().__init__()
        self.net_cfg = dict(net_cfg)
        self.net = ProsodyNet(cond_dim, **self.net_cfg)
        self.stats = ProsodyStats(self.net.word_dim)
        self.duration_scales: dict[int, float] = {}
        self.temperature: float | None = None  # preferred prosody temperature (train-prosody --calibrate-only)
        self.flow_steps = 8
        self._word_encoder = None

    @property
    def kind(self) -> str:
        return self.net.kind

    @staticmethod
    def condition(tts, h: Tensor, x_mask: Tensor, spk: Tensor, logw_det: Tensor, stats: ProsodyStats,
                  word_tok: Tensor | None = None) -> Tensor:
        """``[h; speaker embedding; standardised regressor predictions (; word features)]`` ``[B, C, N]``.

        ``word_tok``: contextual word features broadcast to the tokens ``[B, word_dim, N]`` (standardised here)."""
        s = tts.encoder.spk(spk)[:, :, None].expand(-1, -1, h.shape[-1])
        pitch_det = tts.pitch_predictor(torch.cat([h, s], 1), x_mask)
        base = stats.norm(logw_det[:, 0], pitch_det[:, 0]) * x_mask
        parts = [h, s, base]
        if word_tok is not None:
            parts.append((word_tok - stats.wfeat[0, :, None]) / stats.wfeat[1, :, None])
        return torch.cat(parts, 1) * x_mask, base

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
               mean_samples: int = 16) -> tuple[Tensor, Tensor]:
        """Standardised ``(ld, p)`` ``[B, 2, N]`` and the voicing logit ``[B, N]``; noise from ``generator``.

        ``temperature`` scales the input noise. ``spread`` != 1 (output-space temperature) draws ``mean_samples``
        samples in one batch and returns ``mean + spread * (sample - mean)`` for the first one: below 1 it trades
        expressiveness and seed diversity for per-token accuracy (the noise temperature of the drift sampler mostly
        changes the diversity between seeds, not the spread within an utterance)."""
        if spread == 1.0 or self.net.kind == "mse":
            return self._draw(cond, base, mask, temperature, generator, steps)
        K, B = mean_samples, cond.shape[0]
        y, v = self._draw(cond.repeat(K, 1, 1), base.repeat(K, 1, 1), mask.repeat(K, 1, 1), temperature, generator,
                          steps)
        y = y.view(K, B, *y.shape[1:])
        mean = y.mean(0)
        return (mean + spread * (y[0] - mean)) * mask, v[:B]

    def _draw(self, cond: Tensor, base: Tensor, mask: Tensor, temperature: float,
              generator: torch.Generator | None, steps: int | None) -> tuple[Tensor, Tensor]:
        B, _, N = cond.shape
        net, dev = self.net, cond.device
        if net.kind == "drift":
            z_tok = torch.randn(B, net.noise_tok, N, device=dev, generator=generator) * temperature
            z_glob = torch.randn(B, net.noise_glob, device=dev, generator=generator) * temperature
            out = net(cond, mask, z_tok, z_glob)
            return (base + out[:, :2]) * mask, out[:, 2]
        if net.kind == "mse":
            out = net(cond, mask)
            return (base + out[:, :2]) * mask, out[:, 2]
        steps = steps or self.flow_steps
        x = torch.randn(B, 2, N, device=dev, generator=generator) * temperature * mask
        for k in range(steps):
            t = torch.full((B,), k / steps, device=dev)
            out = net(cond, mask, x_t=x, t=t)
            x = (x + out[:, :2] / steps) * mask
        return (base + x) * mask, out[:, 2]

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
                spread: float = 1.0) -> tuple[Tensor, Tensor]:
        """Sample frames per token ``[B, N]`` and token pitch ``[B, 1, N]`` for ``tts`` (its frozen encoder and
        regressors give the condition); pass them to :meth:`DriftingTTS.synthesize` as ``durations`` / ``pitch``.
        ``temperature`` / ``spread``: see :meth:`sample`."""
        h, _, logw, x_mask = tts.encoder(text, text_len, spk)
        cond, base = self.condition(tts, h, x_mask, spk, logw, self.stats, self.word_tokens(text, text_len))
        y, vlogit = self.sample(cond, base, x_mask, temperature, generator=generator, spread=spread)
        return self.frames_and_pitch(y, vlogit, x_mask, length_scale)

    @classmethod
    def load(cls, path, device="cpu", tts=None) -> ProsodyPredictor:
        """``path``: a checkpoint, or a name of :data:`drifting_tts.hub.PROSODY_MODELS` (e.g. ``drift``, downloaded).
        With ``tts``, the condition size is checked, and a checkpoint that records the fingerprint of the encoder it
        was trained on (:func:`tts_fingerprint`) warns when ``tts`` has another one."""
        from ..hub import resolve_prosody

        path = resolve_prosody(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cond_dim = ck["cond_dim"]
        if tts is not None:
            want = tts.encoder.d + tts.encoder.spk.embedding_dim + 2
            if want != cond_dim:
                raise ValueError(f"prosody model {path} expects {cond_dim} condition channels, the TTS gives {want}")
            if ck.get("tts_fingerprint") and ck["tts_fingerprint"] != tts_fingerprint(tts):
                warnings.warn(f"prosody model {path} was trained on the text encoder of another acoustic model "
                              f"(fingerprint {ck['tts_fingerprint']}, this one {tts_fingerprint(tts)}): check its "
                              "durations and pitch (scripts/eval_prosody_tokens.py) before using it", stacklevel=2)
        p = cls(ck["net_cfg"], cond_dim)
        p.net.load_state_dict(ck["ema"])
        p.stats.load_state_dict(ck["stats"])
        p.duration_scales = {int(k): float(v) for k, v in ck.get("duration_scales", {}).items()}
        p.temperature = ck.get("temperature")
        p.flow_steps = int(ck.get("flow_steps", 8))
        return p.to(device).eval()


def tts_fingerprint(tts) -> str:
    """Short hash of the weights a prosody predictor is conditioned on: the text encoder (with the duration
    predictor and speaker table) and the token pitch predictor of a :class:`DriftingTTS`."""
    import hashlib

    h = hashlib.sha256()
    state = {**{f"encoder.{k}": v for k, v in tts.encoder.state_dict().items()},
             **{f"pitch_predictor.{k}": v for k, v in tts.pitch_predictor.state_dict().items()}}
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def export_checkpoint(ck: dict, tts=None) -> dict:
    """A training checkpoint (``train-prosody``'s ``prosody_ema.pt``) -> what :meth:`ProsodyPredictor.load` needs
    (:data:`CHECKPOINT_KEYS`), without the training config, paths or step; with ``tts``, the fingerprint of its
    encoder. Tensors are copied to the CPU, contiguous."""
    out = {k: ck[k] for k in CHECKPOINT_KEYS if k in ck}
    for k in ("ema", "stats"):
        out[k] = {n: t.detach().cpu().contiguous().clone() for n, t in out[k].items()}
    out["net_cfg"] = dict(out["net_cfg"])
    out["duration_scales"] = {int(k): float(v) for k, v in out.get("duration_scales", {}).items()}
    if tts is not None:
        out["tts_fingerprint"] = tts_fingerprint(tts)
    return out

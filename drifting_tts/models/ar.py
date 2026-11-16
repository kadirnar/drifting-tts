"""ARDrift: an autoregressive acoustic model with a one-step drifting head (#42, the Pocket TTS recipe, our own model).

The same text side as :class:`~drifting_tts.models.tts.DriftingTTS` (character encoder, monotonic alignment against a
Gaussian prior, duration predictor), but the mel is generated **left to right**: a causal transformer runs over
patches of ``patch`` mel frames (4 frames = 43 ms), and at every step a small head turns noise into the next patch
in **one** pass, trained with the drifting field (:func:`~drifting_tts.drift.kyutai_drift_loss`, Kyutai's learned
temperature). The backbone sees the previous patches (teacher-forced, noise-augmented in training) and the
frame-aligned text condition of the current and the next ``lookahead`` patches; nothing tells it the pitch, so the
intonation is the model's own. Like Pocket TTS (an autoregressive backbone over continuous latents with a one-step
sampler head), but on this repository's mels (decoded by Vocos v2), with aligned text instead of a text prefix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..alignment import sequence_mask
from ..text import frontend
from .generator import RMSNorm, SwiGLU
from .text_encoder import TextEncoder, durations_to_alignment, expand, frames_to_alignment, rope


def rope_at(x: Tensor, pos: Tensor) -> Tensor:
    """:func:`rope` of one position ``pos`` (a 1-element tensor, so that a CUDA graph can replay it) on
    ``[B, H, 1, Dh]``."""
    half = x.shape[-1] // 2
    freqs = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    ang = pos.float()[:, None] * freqs[None]  # [1, half]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(dim // heads)
        self.k_norm = RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, cache: dict | None = None) -> Tensor:
        """``x`` ``[B, L, D]``. Without ``cache``: causal attention over the sequence. With ``cache`` (a dict per layer,
        inference): ``x`` are the next ``L`` positions, appended to the cached keys and values."""
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        offset = 0 if cache is None else cache.get("n", 0)
        if cache is not None and offset:
            assert L == 1, "a cached step takes one position at a time"
        q, k = rope(self.q_norm(q), offset), rope(self.k_norm(k), offset)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"], cache["n"] = k, v, offset + L
            y = F.scaled_dot_product_attention(q, k, v, is_causal=L > 1)  # L > 1 only from an empty cache
        return self.proj(y.transpose(1, 2).reshape(B, L, D))

    def step_static(self, x: Tensor, pos: Tensor, kc: Tensor, vc: Tensor, mask: Tensor) -> Tensor:
        """One position ``x`` ``[B, 1, D]`` at ``pos`` with preallocated caches ``[B, H, Smax, Dh]`` (written in place)
        and the valid-key mask ``[1, 1, 1, Smax]``: fixed shapes, for CUDA graphs."""
        B, _, D = x.shape
        q, k, v = self.qkv(x).view(B, 1, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope_at(self.q_norm(q), pos), rope_at(self.k_norm(k), pos)
        kc.index_copy_(2, pos, k)
        vc.index_copy_(2, pos, v)
        y = F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask)
        return self.proj(y.transpose(1, 2).reshape(B, 1, D))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = CausalAttention(dim, heads)
        self.mlp = SwiGLU(dim, (int(2 / 3 * dim * mlp_ratio) + 31) // 32 * 32)

    def forward(self, x: Tensor, cache: dict | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), cache)
        return x + self.mlp(self.norm2(x))

    def step_static(self, x: Tensor, pos: Tensor, kc: Tensor, vc: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn.step_static(self.norm1(x), pos, kc, vc, mask)
        return x + self.mlp(self.norm2(x))


class HeadBlock(nn.Module):
    """Residual MLP block modulated by the backbone state (adaLN-zero)."""

    def __init__(self, dim: int, cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada = nn.Linear(cond, 3 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale, gate = self.ada(c).chunk(3, -1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class DriftHead(nn.Module):
    """``(state [N, cond], noise [N, noise_dim]) -> patch residual [N, out]`` in one pass."""

    def __init__(self, cond: int, out: int, dim: int = 512, depth: int = 6, noise_dim: int | None = None):
        super().__init__()
        self.noise_dim = noise_dim or out
        self.inp = nn.Linear(self.noise_dim, dim)
        self.cond_in = nn.Linear(cond, dim)
        self.cond_act = nn.SiLU()
        self.blocks = nn.ModuleList([HeadBlock(dim, cond) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out = nn.Linear(dim, out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, s: Tensor, z: Tensor) -> Tensor:
        c = self.cond_act(s)
        x = self.inp(z) + self.cond_in(s)
        for b in self.blocks:
            x = b(x, c)
        return self.out(self.norm(x))


class FlowHead(nn.Module):
    """A multi-step head for the same backbone states (#98, teacher B): a flow-matching velocity field over the
    patch residual, sampled with a few Euler steps. Same body as :class:`DriftHead` (so it can start from a trained
    one), plus the time as an input of the conditioning. It is a teacher: it draws positives for the one-step head,
    it is not used at inference."""

    def __init__(self, cond: int, out: int, dim: int = 512, depth: int = 6):
        super().__init__()
        self.noise_dim = out
        self.body = DriftHead(cond, out, dim, depth, out)
        self.time = LevelEmbedding(cond)

    def velocity(self, s: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """States ``[N, cond]``, points ``[N, out]`` on the path at times ``t`` ``[N]`` -> velocities ``[N, out]``."""
        return self.body(s + self.time(t).to(s.dtype), x)

    def loss(self, s: Tensor, target: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Rectified-flow loss: on the straight path from noise ``z`` to the residual ``target`` ``[N, out]`` the
        velocity is ``target - z``."""
        z = torch.randn(target.shape, device=target.device, generator=generator)
        t = torch.rand(target.shape[0], device=target.device, generator=generator)
        x = (1 - t[:, None]) * z + t[:, None] * target
        return (self.velocity(s, x.to(s.dtype), t).float() - (target - z)).pow(2).mean()

    @torch.no_grad()
    def sample(self, s: Tensor, z: Tensor, steps: int = 8) -> Tensor:
        """Residuals ``[N, out]`` from the noise ``z`` (already scaled by the temperature): ``steps`` Euler steps."""
        x = z
        for i in range(steps):
            t = torch.full((s.shape[0],), i / steps, device=s.device)
            x = x + self.velocity(s, x.to(s.dtype), t).to(x.dtype) / steps
        return x


class LevelEmbedding(nn.Module):
    """The noise level of the history as an input of the backbone (#101): Fourier features of the level, then an
    MLP whose output starts at zero (a fine-tune starts as the model was)."""

    def __init__(self, dim: int, bands: int = 8):
        super().__init__()
        self.register_buffer("freqs", 2 * math.pi * 2.0 ** torch.arange(bands), persistent=False)
        self.net = nn.Sequential(nn.Linear(2 * bands, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, level: Tensor) -> Tensor:
        """Levels ``[B]`` -> ``[B, dim]``."""
        a = level.float()[:, None] * self.freqs
        return self.net(torch.cat([a.sin(), a.cos()], -1).to(self.net[0].weight.dtype))


def join_prefix(cond: Tensor, prefix: Tensor | None, prefix_cond: Tensor | None, k: int) -> tuple[Tensor, int]:
    """The condition of a generation with ``prefix`` (see :meth:`ARDriftTTS.generate`) and the number of leading
    frames that are context, not output: ``prefix_cond`` in front of ``cond`` and its length, or ``cond`` and 0."""
    if prefix is not None and prefix.shape[-1] % k:
        raise ValueError(f"the prefix has {prefix.shape[-1]} frames: not a multiple of the patch ({k})")
    if prefix_cond is None:
        return cond, 0
    if prefix is None or prefix.shape[-1] != prefix_cond.shape[-1]:
        raise ValueError("prefix_cond needs a prefix of the same number of frames")
    return torch.cat([prefix_cond.to(cond.dtype), cond], -1), prefix_cond.shape[-1]


class PromptEncoder(nn.Module):
    """The speaker vector of a voice given by a prompt (#95): a projection of the prompt's mel patches, averaged
    over the prompt. It stands in for the speaker row of the text encoder and the duration predictor; the backbone
    reads the prompt itself, as its history."""

    def __init__(self, patch_dim: int, spk_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(patch_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, spk_dim)

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        """Patches ``[B, S, patch_dim]`` and their mask ``[B, S]`` -> ``[B, spk_dim]``."""
        h = self.net(patches) * mask[..., None]
        return self.out(h.sum(1) / mask.sum(1, keepdim=True).clamp_min(1))


class RefAttention(nn.Module):
    """Attention of the reference encoder: over the reference's own patches (rotary positions), or from the queries
    to them (``context``); ``mask`` ``[B, L]`` marks the patches that are not padding."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q, self.kv, self.proj = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)

    def forward(self, x: Tensor, mask: Tensor, context: Tensor | None = None) -> Tensor:
        B, L, D = x.shape
        src = x if context is None else context
        q = self.q_norm(self.q(x).view(B, L, self.heads, -1).transpose(1, 2))
        k, v = self.kv(src).view(B, src.shape[1], 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        k = self.k_norm(k)
        if context is None:
            q, k = rope(q, 0), rope(k, 0)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None].bool())
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class RefBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1, self.norm2, self.norm_c = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.attn = RefAttention(dim, heads)
        self.mlp = SwiGLU(dim, (int(2 / 3 * dim * 4) + 31) // 32 * 32)

    def forward(self, x: Tensor, mask: Tensor, context: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), mask, None if context is None else self.norm_c(context))
        return x + self.mlp(self.norm2(x))


class ReferenceEncoder(nn.Module):
    """The voice of a reference recording: another utterance of the speaker, no transcript (#96). ``layers``
    transformer layers over the reference's mel patches, then ``tokens`` learned queries that attend to them (a
    Perceiver resampler, the conditioning encoder of XTTS): the same number of tokens for 3 s as for 10 s. The
    tokens go in front of the backbone's sequence with a segment embedding; their mean, projected, stands in for the
    speaker row of the text encoder and the duration predictor. A voice without a reference has the ``null``
    tokens there (also the null branch of voice guidance)."""

    def __init__(self, patch_dim: int, dim: int, spk_dim: int, width: int = 512, layers: int = 4, heads: int = 8,
                 tokens: int = 32, query_layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(patch_dim, width)
        self.blocks = nn.ModuleList([RefBlock(width, heads) for _ in range(layers)])
        self.queries = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.readout = nn.ModuleList([RefBlock(width, heads) for _ in range(query_layers)])
        self.norm = RMSNorm(width)
        self.out, self.vec = nn.Linear(width, dim), nn.Linear(width, spk_dim)
        self.segment = nn.Parameter(torch.zeros(dim))
        self.null = nn.Parameter(torch.zeros(tokens, dim))

    @property
    def tokens(self) -> int:
        return self.queries.shape[0]

    def null_tokens(self, batch: int) -> Tensor:
        return (self.null + self.segment).expand(batch, -1, -1)

    def forward(self, patches: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Patches ``[B, S, patch_dim]`` and their mask ``[B, S]`` (every row has a patch) -> the tokens
        ``[B, tokens, dim]`` and the speaker vectors ``[B, spk_dim]``."""
        x = self.inp(patches)
        for blk in self.blocks:
            x = blk(x, mask)
        q = self.queries.expand(x.shape[0], -1, -1)
        for blk in self.readout:
            q = blk(q, mask, x)
        q = self.norm(q)
        return self.out(q) + self.segment, self.vec(q.mean(1))


def history_generator(generator: torch.Generator | None, device) -> torch.Generator:
    """The generator of ``history_noise``: its own stream, seeded from the call's, so that the patches' noise is
    drawn as without it."""
    g = torch.Generator(device=device)
    g.manual_seed((generator.initial_seed() if generator is not None else 0) + 7919)
    return g


def patchify(x: Tensor, k: int) -> Tensor:
    """``[B, C, T]`` (T a multiple of ``k``) -> ``[B, T / k, C * k]`` (frame-major inside a patch)."""
    B, C, T = x.shape
    return x.view(B, C, T // k, k).permute(0, 2, 3, 1).reshape(B, T // k, k * C)


def unpatchify(p: Tensor, k: int, c: int) -> Tensor:
    """Inverse of :func:`patchify`: ``[B, S, k * C]`` -> ``[B, C, S * k]``."""
    B, S, _ = p.shape
    return p.view(B, S, k, c).permute(0, 3, 1, 2).reshape(B, c, S * k)


class ARDriftTTS(nn.Module):
    """See the module docstring. ``cfg``: ``model`` section (``text``, ``ar``)."""

    def __init__(self, cfg, num_speakers: int, n_mels: int = 100):
        super().__init__()
        t, a = cfg.text, cfg.ar
        vocab = len(frontend(t.get("language")).SYMBOLS)  # model.text.language: the frontend (default Turkish)
        # model.ar.prompt (#95): a voice can also come from a recording. One more speaker row, the last, stands for
        # "the voice is in the prompt"; the text encoder then gets a vector pooled from the prompt (prompt_vector)
        # model.ar.reference (#96): a voice can also come from another recording of the speaker, read by a reference
        # encoder whose tokens stand in front of the backbone's sequence. The same extra row stands for both
        self.prompted = bool(a.get("prompt", False))
        rc = a.get("reference")
        num_speakers += int(self.prompted or bool(rc))
        self.encoder = TextEncoder(vocab, n_mels=n_mels, d=t.d, heads=t.heads, layers=t.layers, ffn=t.ffn,
                                   dropout=t.dropout, num_speakers=num_speakers, spk_dim=t.spk_dim)
        self.n_mels, self.patch, self.lookahead = n_mels, int(a.patch), int(a.get("lookahead", 2))
        self.pitch_enabled = False  # the intonation is the autoregressive model's own
        # model.ar.tied (#92): the frames of a patch are copies of one codec frame (a latent stored `patch` times
        # for the alignment). The backbone then reads one frame per step and the head draws one frame, which is
        # copied over the patch: the model cannot make the copies disagree. The text condition keeps the fine grid
        self.tied = bool(a.get("tied", False))
        k, cond_ch, dim = self.patch, n_mels + t.d, int(a.dim)
        out = n_mels if self.tied else k * n_mels  # what the backbone reads and the head draws per step
        self.x_in = nn.Linear(out, dim)
        self.c_in = nn.Linear(k * cond_ch * (1 + self.lookahead), dim)
        self.spk = nn.Embedding(num_speakers, dim)
        nn.init.normal_(self.spk.weight, std=0.02)
        self.bos = nn.Parameter(torch.zeros(out))
        self.blocks = nn.ModuleList([Block(dim, int(a.heads), float(a.get("mlp_ratio", 4.0)))
                                     for _ in range(int(a.depth))])
        self.norm = RMSNorm(dim)
        self.head = DriftHead(dim, out, int(a.head_dim), int(a.head_depth), a.get("noise_dim"))
        # model.ar.noise_level (#101): the backbone is told how much noise its history carries (training: the level
        # of drift.input_noise; inference: none, or the level of `history_noise`)
        self.level_in = LevelEmbedding(dim) if a.get("noise_level", False) else None
        self.prompt_enc = PromptEncoder(k * n_mels, t.spk_dim, int(a.get("prompt_dim", 256))) if self.prompted else None
        self.ref_enc = None
        if rc:
            rc = {} if rc is True else dict(rc)
            self.ref_enc = ReferenceEncoder(k * n_mels, dim, t.spk_dim, int(rc.get("dim", 512)),
                                            int(rc.get("layers", 4)), int(rc.get("heads", 8)),
                                            int(rc.get("tokens", 32)), int(rc.get("query_layers", 2)))
        self.duration_scale, self.duration_scales, self.temperature = 1.0, {}, None
        self.fast_generator: ARGenerator | None = None  # set by :meth:`enable_fast`

    def enable_fast(self, max_steps: int = 512, graphs: bool = True) -> None:
        """Generate with :class:`ARGenerator` (preallocated caches, CUDA graphs) from now on."""
        self.fast_generator = ARGenerator(self, max_steps, graphs)

    # ------------------------------------------------------------------------------------------ one frame per step
    def squeeze(self, p: Tensor) -> Tensor:
        """Patches ``[..., k * n_mels]`` -> what the backbone reads and the head draws: the patch itself, or with
        ``model.ar.tied`` the one frame its ``k`` copies stand for (their mean) ``[..., n_mels]``."""
        return p.unflatten(-1, (self.patch, self.n_mels)).mean(-2) if self.tied else p

    def spread(self, f: Tensor) -> Tensor:
        """The inverse of :meth:`squeeze`: a tied model's frame ``[..., n_mels]`` copied over its patch."""
        return f.unsqueeze(-2).expand(*f.shape[:-1], self.patch, f.shape[-1]).flatten(-2) if self.tied else f

    def emit(self, mu: Tensor, s: Tensor, z: Tensor) -> Tensor:
        """One step's patch ``[N, k * n_mels]``: the prior-mean patch ``mu`` plus the head's residual from the
        state ``s`` and the noise ``z``."""
        return self.spread(self.squeeze(mu) + self.head(s, z))

    # ----------------------------------------------------------------------------------------------- conditioning
    def cond_patches(self, cond: Tensor) -> Tensor:
        """Aligned condition ``[B, C, T]`` (T a multiple of ``patch``) -> per-step inputs ``[B, S, C * k * (1 + L)]``:
        each step's own condition and that of the next ``lookahead`` steps (zeros past the end)."""
        p = patchify(cond, self.patch)
        if not self.lookahead:
            return p
        ahead = [F.pad(p[:, i:], (0, 0, 0, i)) for i in range(1, self.lookahead + 1)]
        return torch.cat([p, *ahead], -1)

    def backbone(self, x_prev: Tensor, cpatch: Tensor, spk: Tensor, caches: list[dict] | None = None,
                 tokens: Tensor | None = None, level: Tensor | None = None) -> Tensor:
        """Previous patches ``[B, S, k * n_mels]`` and condition inputs ``[B, S, *]`` -> states ``[B, S, dim]``.
        ``tokens`` ``[B, K, dim]``: positions in front of the sequence (the reference's, #96); with ``caches``,
        given with the first step only. ``level`` ``[B]``: the noise level of ``x_prev`` (``model.ar.noise_level``;
        default none)."""
        h = self.x_in(x_prev) + self.c_in(cpatch) + self.spk(spk)[:, None]
        if self.level_in is not None:
            level = torch.zeros(h.shape[0], device=h.device) if level is None else level
            h = h + self.level_in(level)[:, None].to(h.dtype)
        n = 0
        if tokens is not None:
            h, n = torch.cat([tokens.to(h.dtype), h], 1), tokens.shape[1]
        for i, blk in enumerate(self.blocks):
            h = blk(h, None if caches is None else caches[i])
        return self.norm(h[:, n:])

    def text(self, text: Tensor, text_len: Tensor, spk: Tensor, spk_emb: Tensor | None = None):
        return self.encoder(text, text_len, spk, spk_emb)

    # ------------------------------------------------------------------------------------------- prompted voices
    @property
    def prompt_row(self) -> int:
        """The speaker row of a voice given by a prompt (``model.ar.prompt``)."""
        if not self.prompted and self.ref_enc is None:
            raise ValueError("this model was not trained with voice prompts (model.ar.prompt, model.ar.reference)")
        return self.spk.num_embeddings - 1

    def reference_tokens(self, ref: Tensor | None, ref_len: Tensor | None = None, use: Tensor | None = None,
                         batch: int | None = None) -> tuple[Tensor, Tensor | None]:
        """Normalised reference mels ``[B, n_mels, R]`` (``ref_len`` ``[B]``: their frames, default all; ``use``
        ``[B]``: the rows that have one) -> the tokens in front of the backbone's sequence ``[B, K, dim]`` (the null
        tokens where there is no reference) and the text encoder's speaker vectors ``[B, spk_dim]`` (``None``
        without ``ref``; meaningless where there is no reference)."""
        if ref is None or ref.shape[-1] < self.patch:  # a batch in which no utterance has a reference
            return self.ref_enc.null_tokens(batch if ref is None else ref.shape[0]), None
        p = patchify(ref[..., : ref.shape[-1] // self.patch * self.patch], self.patch)
        if ref_len is None:
            ref_len = torch.full((p.shape[0],), p.shape[1] * self.patch, device=p.device)
        mask = torch.arange(p.shape[1], device=p.device)[None] * self.patch < ref_len[:, None]
        use = mask[:, 0] if use is None else use & mask[:, 0]
        mask = mask | ~use[:, None]  # a row without a reference attends to padding: its tokens are not used
        tokens, vec = self.ref_enc(p, mask)
        return torch.where(use[:, None, None], tokens, self.ref_enc.null_tokens(p.shape[0]).to(tokens.dtype)), vec

    def _tokens(self, tokens: Tensor | None, batch: int) -> Tensor | None:
        """The positions in front of the sequence: ``tokens``, or the null ones of a model with a reference encoder
        (a voice from the speaker table, or from a prompt alone)."""
        if tokens is None and self.ref_enc is not None:
            return self.ref_enc.null_tokens(batch)
        return tokens

    def prompt_vector(self, prompt: Tensor, prompt_len: Tensor | None = None) -> Tensor:
        """Normalised prompt mels ``[B, n_mels, P]`` (``P`` a multiple of ``patch``; ``prompt_len`` ``[B]``: their
        frames, default all) -> the text encoder's speaker vectors ``[B, spk_dim]``."""
        p = patchify(prompt, self.patch)
        if prompt_len is None:
            mask = torch.ones(p.shape[:2], device=p.device)
        else:
            mask = torch.arange(p.shape[1], device=p.device)[None] * self.patch < prompt_len[:, None]
        return self.prompt_enc(p, mask.to(p.dtype))

    @staticmethod
    def frame_condition(h: Tensor, mu: Tensor, attn: Tensor) -> Tensor:
        return torch.cat([expand(mu, attn), expand(h, attn)], 1)

    def pad_frames(self, x: Tensor) -> Tensor:
        return F.pad(x, (0, (-x.shape[-1]) % self.patch))

    # -------------------------------------------------------------------------------------------------- training
    def teacher_forced(self, cond: Tensor, y: Tensor, spk: Tensor, input_noise: Tensor | None = None,
                       drop: Tensor | None = None, tokens: Tensor | None = None) -> Tensor:
        """States ``[B, S, dim]`` for every step, the previous target patches as input (plus ``input_noise`` ``[B]``
        times unit noise: robustness to the model's own, imperfect patches at inference). ``drop`` ``[B]`` (bool):
        utterances whose text condition the backbone does not see (the null condition of guidance). ``tokens``
        ``[B, K, dim]``: the reference's (:meth:`reference_tokens`)."""
        yp = self.squeeze(patchify(y, self.patch))
        x_prev = torch.cat([self.bos.expand(yp.shape[0], 1, -1), yp[:, :-1]], 1)
        if input_noise is not None:
            x_prev = x_prev + input_noise[:, None, None] * torch.randn_like(x_prev)
        cp = self.cond_patches(cond)
        if drop is not None:
            cp = cp * (~drop)[:, None, None].to(cp.dtype)
        return self.backbone(x_prev, cp, spk, tokens=self._tokens(tokens, yp.shape[0]), level=input_noise)

    @torch.no_grad()
    def rollout(self, cond: Tensor, y: Tensor, spk: Tensor, start: int, steps: int, temperature: float = 1.0,
                generator: torch.Generator | None = None, tokens: Tensor | None = None) -> Tensor:
        """``steps`` patches generated free-running from step ``start`` on (#101): the recorded patches before
        ``start`` are the history (one pass), then every patch is drawn after the model's own ones, as at inference.
        ``cond`` ``[B, C, T]`` and ``y`` ``[B, n_mels, T]`` in whole patches -> ``[B, steps, k * n_mels]``."""
        k = self.patch
        cp, mu = self.cond_patches(cond), patchify(cond[:, : self.n_mels], k)
        yp = self.squeeze(patchify(y, k))
        B = yp.shape[0]
        caches = [{} for _ in self.blocks]
        x_prev = torch.cat([self.bos.expand(B, 1, -1), yp[:, :start]], 1)  # the inputs of steps 0 .. start
        s = self.backbone(x_prev, cp[:, : start + 1], spk, caches, self._tokens(tokens, B))[:, -1]
        out = []
        for j in range(steps):
            z = torch.randn(B, self.head.noise_dim, device=cond.device, generator=generator) * temperature
            x = self.emit(mu[:, start + j], s, z)
            out.append(x)
            if j + 1 < steps:
                s = self.backbone(self.squeeze(x)[:, None], cp[:, start + j + 1: start + j + 2], spk, caches)[:, 0]
        return torch.stack(out, 1)

    @torch.no_grad()
    def gta(self, cond: Tensor, y: Tensor, spk: Tensor, temperature: float = 1.0,
            generator: torch.Generator | None = None, tokens: Tensor | None = None) -> Tensor:
        """Ground-truth-aligned ("GTA") generation: every patch drawn by the head (noise at ``temperature``) from the
        state after the *recorded* previous patches, in one teacher-forced pass. ``cond`` ``[B, C, T]`` aligned to
        ``y`` ``[B, n_mels, T]`` -> ``[B, n_mels, T]``, frame for frame with the recording (``finetune-vocoder``)."""
        T, k = y.shape[-1], self.patch
        cond, y = self.pad_frames(cond), self.pad_frames(y)
        s = self.teacher_forced(cond, y, spk, tokens=tokens)  # [B, S, dim]
        mu = patchify(cond[:, : self.n_mels], k)
        z = torch.randn(*s.shape[:2], self.head.noise_dim, device=s.device, generator=generator) * temperature
        x = self.emit(mu.flatten(0, 1), s.flatten(0, 1), z.flatten(0, 1)).view_as(mu)
        return unpatchify(x, k, self.n_mels)[..., :T]

    # ------------------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, cond: Tensor, spk: Tensor, temperature: float = 1.0, generator: torch.Generator | None = None,
                 prefix: Tensor | None = None, cfg_scale: float = 1.0, prefix_cond: Tensor | None = None,
                 tokens: Tensor | None = None, history_noise: float = 0.0, guide: str = "text") -> Tensor:
        """Aligned condition ``[B, C, T]`` -> normalised mel ``[B, n_mels, T]``, one step (patch) at a time with a KV
        cache. ``prefix``: known leading mel frames ``[B, n_mels, Tp]`` (``Tp`` a multiple of ``patch``), kept.
        ``prefix_cond`` ``[B, C, Tp]``: the prefix is then acoustic context that comes *before* ``cond`` (the end of
        the previous utterance and the condition it was generated under): the model continues from it as it
        continues a longer utterance, no noise is drawn for it, and only the ``T`` frames of ``cond`` are returned.
        ``cfg_scale`` != 1: guidance on the backbone state, ``s_null + cfg_scale * (s - s_null)``, the null state
        from the same history without the text (a model trained with ``drift.cond_dropout``). ``tokens``
        ``[B, K, dim]``: the reference's (:meth:`reference_tokens`), in front of the sequence. ``history_noise``
        (#101): Gaussian noise of this level on what the backbone reads of the patches so far, as on the recorded
        history in training (``drift.input_noise``: a level up to 0.3); the output patches stay clean. ``guide``
        (#99): what the null branch of the guidance is without: the ``text``, the ``voice`` (the speaker row and the
        reference's tokens; a model trained with ``drift.voice_dropout``) or ``both``."""
        cond, lead = join_prefix(cond, prefix, prefix_cond, self.patch)
        T = cond.shape[-1]
        cond = self.pad_frames(cond)
        cp = self.cond_patches(cond)
        B, S = cp.shape[:2]
        k, mu = self.patch, patchify(cond[:, : self.n_mels], self.patch)
        guided = cfg_scale != 1.0
        tokens = self._tokens(tokens, B)
        if guided:  # the conditional and the null rows as one batch
            cp, spk_b, tokens = self.null_branch(cp, spk, tokens, guide)
        else:
            spk_b = spk
        caches = [{} for _ in self.blocks]
        prev = self.bos.expand(B, 1, -1)
        out = []
        n_pre = 0 if prefix is None else prefix.shape[-1] // k
        pre = None if prefix is None else patchify(prefix, k)
        hg, level = history_generator(generator, cond.device) if history_noise else None, None
        if history_noise and self.level_in is not None:
            level = torch.full((spk_b.shape[0],), float(history_noise), device=cond.device)
        for t in range(S):
            if history_noise and t:  # the beginning-of-sequence patch is not audio
                prev = prev + history_noise * torch.randn(prev.shape, device=prev.device, generator=hg)
            s = self.backbone(torch.cat([prev, prev]) if guided else prev, cp[:, t: t + 1], spk_b, caches,
                              tokens if t == 0 else None, level)[:, 0]
            if guided:
                s = s[B:] + cfg_scale * (s[:B] - s[B:])
            if t < n_pre:
                x = pre[:, t]
            else:
                z = torch.randn(B, self.head.noise_dim, device=cond.device, generator=generator) * temperature
                x = self.emit(mu[:, t], s, z)
            out.append(x)
            prev = self.squeeze(x)[:, None]
        mel = unpatchify(torch.stack(out, 1), k, self.n_mels)
        return mel[..., lead:T]

    def null_branch(self, cp: Tensor, spk: Tensor, tokens: Tensor | None, guide: str = "text"
                    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """The inputs of a guided generation, the conditional rows and then the null ones: the condition inputs
        ``[2 B, S, *]``, the speaker rows ``[2 B]`` and the tokens ``[2 B, K, dim]`` (``None`` without a reference
        encoder). The null rows are without what ``guide`` names: ``text`` (a zero condition), ``voice`` (the
        prompted row and the null tokens: no voice) or ``both``."""
        if guide not in ("text", "voice", "both"):
            raise ValueError(f"guide: {guide!r} (text, voice or both)")
        cp_null = cp if guide == "voice" else torch.zeros_like(cp)
        spk_null, tok_null = spk, tokens
        if guide != "text":
            spk_null = torch.full_like(spk, self.prompt_row)
            tok_null = None if tokens is None else self.ref_enc.null_tokens(tokens.shape[0]).to(tokens.dtype)
        return (torch.cat([cp, cp_null]), torch.cat([spk, spk_null]),
                None if tokens is None else torch.cat([tokens, tok_null]))

    def step_static(self, prev: Tensor, cpatch: Tensor, spk: Tensor, pos: Tensor, caches: list[tuple[Tensor, Tensor]],
                    mu: Tensor, z: Tensor, cfg: Tensor | None = None, level: Tensor | None = None) -> Tensor:
        """One generation step with fixed shapes (:class:`ARGenerator`): the previous patch ``[B, k * n_mels]``, this
        step's condition input ``[B, *]``, the position ``pos`` ``[1]``, per-layer caches, the prior-mean patch ``mu``
        and the (scaled) noise ``z`` -> the next patch ``[B, k * n_mels]``. ``cfg`` ``[1]``: guidance; ``cpatch``,
        ``spk`` and the caches then have ``2 B`` rows (the conditional ones, then the null ones)."""
        if cfg is not None:
            prev = torch.cat([prev, prev])
        h = self.x_in(self.squeeze(prev)) + self.c_in(cpatch) + self.spk(spk)
        if self.level_in is not None:  # ``level`` [rows]: the noise level of ``prev`` (zeros: none)
            h = h + self.level_in(level).to(h.dtype)
        h = h[:, None]
        smax = caches[0][0].shape[2]
        mask = (torch.arange(smax, device=prev.device) <= pos)[None, None, None]
        for blk, (kc, vc) in zip(self.blocks, caches):
            h = blk.step_static(h, pos, kc, vc, mask)
        s = self.norm(h)[:, 0]
        if cfg is not None:
            B = s.shape[0] // 2
            s = s[B:] + cfg * (s[:B] - s[B:])
        return self.emit(mu, s, z)

    @torch.no_grad()
    def aligned_condition(self, text: Tensor, text_len: Tensor, spk: Tensor, length_scale: float = 1.0,
                          pitch_shift: float = 0.0, durations: Tensor | None = None, pitch: Tensor | None = None,
                          context: int = 0, spk_emb: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if pitch is not None or pitch_shift:
            raise ValueError("ARDrift models generate their own intonation: no token pitch or pitch shift")
        h, mu, logw, x_mask = self.text(text, text_len, spk, spk_emb)
        if context:
            h, mu, logw, x_mask = (t[..., context:] for t in (h, mu, logw, x_mask))
            durations = None if durations is None else durations[:, context:]
        if durations is None:
            attn, y_len = durations_to_alignment(logw, x_mask, length_scale)
        else:
            attn, y_len = frames_to_alignment(durations, x_mask)
        return self.frame_condition(h, mu, attn), y_len

    @torch.no_grad()
    def synthesize(self, text: Tensor, text_len: Tensor, spk: Tensor, cfg_scale: float = 1.0,
                   temperature: float = 1.0, length_scale: float = 1.0, noise_labels: Tensor | None = None,
                   generator: torch.Generator | None = None, attn_window: int | None = None,
                   pitch_shift: float = 0.0, steps: int | None = None, durations: Tensor | None = None,
                   pitch: Tensor | None = None, context: int = 0, prompt: Tensor | None = None,
                   reference: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """As :meth:`DriftingTTS.synthesize` (the DiT's own arguments are accepted and ignored): normalised mels
        ``[B, n_mels, T]`` and their lengths. Durations from the duration predictor (``length_scale``) or given;
        ``cfg_scale``: guidance on the backbone state (:meth:`generate`). ``prompt`` (``model.ar.prompt``, #95):
        normalised mels ``[B, n_mels, P]`` of recordings whose voices the model continues in; ``spk`` is then
        ignored, and no transcript of the prompt is needed. ``reference`` (``model.ar.reference``, #96): normalised
        mels ``[B, n_mels, R]`` of recordings of the voices, read by the reference encoder; with or without a
        ``prompt``."""
        spk_emb = prefix = prefix_cond = tokens = None
        if prompt is not None:
            prompt = prompt[..., : prompt.shape[-1] // self.patch * self.patch]
            spk, spk_emb = torch.full_like(spk, self.prompt_row), self.prompt_vector(prompt)
        if reference is not None:
            tokens, spk_emb = self.reference_tokens(reference)
            spk = torch.full_like(spk, self.prompt_row)
        cond, y_len = self.aligned_condition(text, text_len, spk, length_scale, pitch_shift, durations, pitch, context,
                                             spk_emb)
        if prompt is not None:  # the prompt is history without a text condition
            prefix, prefix_cond = prompt, cond.new_zeros(cond.shape[0], cond.shape[1], prompt.shape[-1])
        if self.fast_generator is not None:
            mel = self.fast_generator(cond, spk, temperature, generator, cfg_scale, prefix=prefix,
                                      prefix_cond=prefix_cond, tokens=tokens)
        else:
            mel = self.generate(cond, spk, temperature, generator, prefix=prefix, cfg_scale=cfg_scale,
                                prefix_cond=prefix_cond, tokens=tokens)
        return mel * sequence_mask(y_len, mel.shape[-1])[:, None], y_len


class ARGenerator:
    """Fast generation for :class:`ARDriftTTS`: preallocated KV caches of ``max_steps`` positions and, on CUDA, one
    CUDA graph of a whole step (backbone and head) per batch size, replayed ``S`` times. The draws are those of
    :meth:`ARDriftTTS.generate` (one ``[B, noise_dim]`` noise per step from the same generator), so the output matches
    it to float precision."""

    def __init__(self, model: ARDriftTTS, max_steps: int = 512, graphs: bool = True):
        self.model, self.max_steps = model, max_steps
        self.graphs = graphs and next(model.parameters()).is_cuda
        self._state: dict[int, dict] = {}

    def _buffers(self, B: int, cdim: int, guided: bool) -> dict:
        if (B, guided) in self._state:
            return self._state[(B, guided)]
        R = 2 * B if guided else B
        m = self.model
        dev, dt = next(m.parameters()).device, next(m.parameters()).dtype
        dim, heads = m.x_in.out_features, m.blocks[0].attn.heads
        st = {"prev": torch.zeros(B, m.patch * m.n_mels, device=dev, dtype=dt),
              "cp": torch.zeros(R, cdim, device=dev, dtype=dt), "spk": torch.zeros(R, dtype=torch.long, device=dev),
              "cfg": torch.ones(1, device=dev, dtype=dt) if guided else None,
              "pos": torch.zeros(1, dtype=torch.long, device=dev),
              "mu": torch.zeros(B, m.patch * m.n_mels, device=dev, dtype=dt),
              "z": torch.zeros(B, m.head.noise_dim, device=dev, dtype=dt),
              "level": torch.zeros(R, device=dev, dtype=dt),
              "caches": [(torch.zeros(R, heads, self.max_steps, dim // heads, device=dev, dtype=dt),
                          torch.zeros(R, heads, self.max_steps, dim // heads, device=dev, dtype=dt))
                         for _ in m.blocks]}

        def run():
            return m.step_static(st["prev"], st["cp"], st["spk"], st["pos"], st["caches"], st["mu"], st["z"],
                                 st["cfg"], st["level"])

        if self.graphs:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                st["out"] = run()
            st["graph"] = g
        st["run"] = run
        self._state[(B, guided)] = st
        return st

    @property
    def lead(self) -> int:
        """Positions in front of every sequence: the reference encoder's tokens (#96)."""
        return 0 if self.model.ref_enc is None else self.model.ref_enc.tokens

    @torch.no_grad()
    def fits(self, frames: int, prefix_frames: int = 0) -> bool:
        """Whether ``frames`` frames after ``prefix_frames`` of context fit the caches."""
        steps = (prefix_frames + frames + (-frames) % self.model.patch) // self.model.patch
        return steps + self.lead <= self.max_steps

    @torch.no_grad()
    def __call__(self, cond: Tensor, spk: Tensor, temperature: float = 1.0,
                 generator: torch.Generator | None = None, cfg_scale: float = 1.0, prefix: Tensor | None = None,
                 prefix_cond: Tensor | None = None, tokens: Tensor | None = None,
                 history_noise: float = 0.0, guide: str = "text") -> Tensor:
        """As :meth:`ARDriftTTS.generate`: aligned condition ``[B, C, T]`` -> mel ``[B, n_mels, T]``."""
        m = self.model
        T = cond.shape[-1]
        if not self.fits(T, 0 if prefix_cond is None else prefix_cond.shape[-1]):
            return m.generate(cond, spk, temperature, generator, prefix, cfg_scale, prefix_cond, tokens,
                              history_noise, guide)
        out = list(self.steps(cond, spk, temperature, generator, cfg_scale, prefix, prefix_cond, tokens,
                              history_noise, guide))
        return unpatchify(torch.stack(out, 1), m.patch, m.n_mels)[..., :T]

    @torch.no_grad()
    def steps(self, cond: Tensor, spk: Tensor, temperature: float = 1.0, generator: torch.Generator | None = None,
              cfg_scale: float = 1.0, prefix: Tensor | None = None, prefix_cond: Tensor | None = None,
              tokens: Tensor | None = None, history_noise: float = 0.0, guide: str = "text"):
        """Yield the patches ``[B, k * n_mels]`` of ``cond`` one step at a time (streaming); at most ``max_steps``
        with the prefix. ``prefix`` / ``prefix_cond``: as :meth:`ARDriftTTS.generate` (kept leading frames are
        yielded as they are; the steps of a context before ``cond`` run, to fill the caches, but are not yielded).
        ``tokens``: the reference's; they take the first positions of the caches, written in one pass."""
        m = self.model
        cond, lead = join_prefix(cond, prefix, prefix_cond, m.patch)
        pre = None if prefix is None else patchify(prefix, m.patch)
        n_pre = 0 if pre is None else pre.shape[1]
        cond = m.pad_frames(cond)
        cp = m.cond_patches(cond)
        B, S = cp.shape[:2]
        K = self.lead
        if S + K > self.max_steps:
            raise ValueError(f"{S + K} steps: longer than the caches ({self.max_steps})")
        mu = patchify(cond[:, : m.n_mels], m.patch)
        guided = cfg_scale != 1.0
        st = self._buffers(B, cp.shape[-1], guided)
        tokens = m._tokens(tokens, B)
        if guided:
            cp, spk_b, tokens = m.null_branch(cp, spk, tokens, guide)
        if K:  # the reference's tokens: positions 0 .. K - 1 of every layer's cache
            h = tokens.to(st["prev"].dtype)
            for blk, (kc, vc) in zip(m.blocks, st["caches"]):
                cache: dict = {}
                h = blk(h, cache)
                kc[:, :, :K].copy_(cache["k"])
                vc[:, :, :K].copy_(cache["v"])
        st["prev"].copy_(m.spread(m.bos).expand(B, -1))
        if guided:
            st["spk"].copy_(spk_b)
            st["cfg"].fill_(cfg_scale)
        else:
            st["spk"].copy_(spk)
        hg = history_generator(generator, cond.device) if history_noise else None
        st["level"].fill_(float(history_noise))
        for t in range(S):
            st["cp"].copy_(cp[:, t])
            st["mu"].copy_(mu[:, t])
            st["pos"].fill_(K + t)
            if history_noise and t:  # noise on what the backbone reads of the patch before (as generate)
                noise = torch.randn(B, 1, m.x_in.in_features, device=cond.device, generator=hg)[:, 0]
                st["prev"].add_(m.spread(history_noise * noise))
            if t >= n_pre:  # a known patch draws no noise (its step only fills the caches)
                st["z"].copy_(torch.randn(B, m.head.noise_dim, device=cond.device, generator=generator) * temperature)
            if self.graphs:
                st["graph"].replay()
                x = st["out"].clone()
            else:
                x = st["run"]()
            if t < n_pre:
                x = pre[:, t].to(x.dtype)
            st["prev"].copy_(x)
            if t * m.patch >= lead:
                yield x


def band_energy(frames: Tensor, scale: float, bands: Sequence[int]) -> Tensor:
    """Log energy of mel bands per frame: normalised log-mel frames ``[..., n_mels]`` -> ``[..., len(bands) - 1]``,
    ``log mean exp(scale * x) / scale`` over each band ``[bands[i], bands[i + 1])``. With ``scale`` twice the mel
    statistics' std, ``scale * x`` is the log power up to a constant, so a band's value follows its loudest bins
    (on a log-mel distance, every bin weighs alike, the noise floor's as much as the speech's: #73)."""
    out = [torch.logsumexp(frames[..., a:b] * scale, -1) - math.log(b - a) for a, b in zip(bands[:-1], bands[1:])]
    return torch.stack(out, -1) / scale


def mae_features(mae, window: Tensor, frames: int) -> dict[str, Tensor]:
    """Feature maps of a frozen :class:`~drifting_tts.models.mae2d.MelMAE2d` on mel windows ``[N, n_mels, W]`` at the
    time columns that cover the last ``frames`` frames: ``{mae_<stage>: [N, L, D]}`` (L: frequency rows x columns)."""
    W, out = window.shape[-1], {}
    for name, f in mae.encode(window).items():  # [N, D, F', T']
        cols = max(1, round(frames * f.shape[-1] / W))
        out[f"mae_{name}"] = f[..., -cols:].flatten(2).transpose(1, 2).float()
    return out


def history_patches(yp: Tensor, context: int) -> Tensor:
    """For every step of ``yp`` ``[B, S, D]`` the ``context`` patches before it, oldest first: ``[B, S, context,
    D]``. Before the first patch the first one stands in (an utterance has no audio before it)."""
    padded = torch.cat([yp[:, :1].expand(-1, context, -1), yp[:, :-1]], 1)  # [B, S + context - 1, D]
    return padded.unfold(1, context, 1).permute(0, 1, 3, 2)


def window_features(mae, hist: Tensor, x: Tensor, k: int, n_mels: int) -> dict[str, Tensor]:
    """Kernel maps over a window (#97): the recorded patches ``hist`` ``[R, n, k * n_mels]`` before a step, then the
    step's patches ``x`` ``[R, S, k * n_mels]`` (generated samples, or the recorded patch), through the frozen
    Mel-MAE; the maps are read where they cover the step's own frames: ``{mae_<stage>: [R, S, L, D]}``. The window
    lets the kernel see how a patch joins the audio before it and how its frames follow each other; gradients
    reach only ``x``."""
    R, S, _ = x.shape
    frames = torch.cat([hist.reshape(R, 1, -1, n_mels).expand(-1, S, -1, -1), x.view(R, S, k, n_mels)], 2)
    feats = mae_features(mae, frames.reshape(R * S, -1, n_mels).transpose(1, 2), k)
    return {name: f.view(R, S, *f.shape[1:]) for name, f in feats.items()}


def ar_features(x: Tensor, k: int, n_mels: int, detail: bool = True, energy_bands: Sequence[int] | None = None,
                energy_scale: float = 1.0) -> dict[str, Tensor]:
    """Drift kernel features of patches ``[R, S, k * n_mels]`` (R rows, S samples): ``patch`` (the whole patch, one
    location), ``frame`` (each frame a location), ``detail`` (each frame minus its frequency-smoothed envelope: the
    harmonic structure) and, with ``energy_bands``, ``energy`` (each frame's band energies, :func:`band_energy`).
    Every map is ``[R, S, L, D]``."""
    from .mae import spectral_detail

    R, S, _ = x.shape
    frames = x.view(R, S, k, n_mels)
    out = {"patch": x[:, :, None]}
    if k > 1:  # a patch of one frame (model.ar.tied) is its own frame map
        out["frame"] = frames
    if detail:
        d = spectral_detail(frames.reshape(R * S, k, n_mels).transpose(1, 2)).transpose(1, 2)
        out["detail"] = d.reshape(R, S, k, n_mels)
    if energy_bands:
        out["energy"] = band_energy(frames, energy_scale, energy_bands)
    return out



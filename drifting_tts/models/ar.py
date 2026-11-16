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
        self.prompted = bool(a.get("prompt", False))
        num_speakers += int(self.prompted)
        self.encoder = TextEncoder(vocab, n_mels=n_mels, d=t.d, heads=t.heads, layers=t.layers, ffn=t.ffn,
                                   dropout=t.dropout, num_speakers=num_speakers, spk_dim=t.spk_dim)
        self.n_mels, self.patch, self.lookahead = n_mels, int(a.patch), int(a.get("lookahead", 2))
        self.pitch_enabled = False  # the intonation is the autoregressive model's own
        k, cond_ch, dim = self.patch, n_mels + t.d, int(a.dim)
        self.x_in = nn.Linear(k * n_mels, dim)
        self.c_in = nn.Linear(k * cond_ch * (1 + self.lookahead), dim)
        self.spk = nn.Embedding(num_speakers, dim)
        nn.init.normal_(self.spk.weight, std=0.02)
        self.bos = nn.Parameter(torch.zeros(k * n_mels))
        self.blocks = nn.ModuleList([Block(dim, int(a.heads), float(a.get("mlp_ratio", 4.0)))
                                     for _ in range(int(a.depth))])
        self.norm = RMSNorm(dim)
        self.head = DriftHead(dim, k * n_mels, int(a.head_dim), int(a.head_depth), a.get("noise_dim"))
        self.prompt_enc = PromptEncoder(k * n_mels, t.spk_dim, int(a.get("prompt_dim", 256))) if self.prompted else None
        self.duration_scale, self.duration_scales, self.temperature = 1.0, {}, None
        self.fast_generator: ARGenerator | None = None  # set by :meth:`enable_fast`

    def enable_fast(self, max_steps: int = 512, graphs: bool = True) -> None:
        """Generate with :class:`ARGenerator` (preallocated caches, CUDA graphs) from now on."""
        self.fast_generator = ARGenerator(self, max_steps, graphs)

    # ----------------------------------------------------------------------------------------------- conditioning
    def cond_patches(self, cond: Tensor) -> Tensor:
        """Aligned condition ``[B, C, T]`` (T a multiple of ``patch``) -> per-step inputs ``[B, S, C * k * (1 + L)]``:
        each step's own condition and that of the next ``lookahead`` steps (zeros past the end)."""
        p = patchify(cond, self.patch)
        if not self.lookahead:
            return p
        ahead = [F.pad(p[:, i:], (0, 0, 0, i)) for i in range(1, self.lookahead + 1)]
        return torch.cat([p, *ahead], -1)

    def backbone(self, x_prev: Tensor, cpatch: Tensor, spk: Tensor, caches: list[dict] | None = None) -> Tensor:
        """Previous patches ``[B, S, k * n_mels]`` and condition inputs ``[B, S, *]`` -> states ``[B, S, dim]``."""
        h = self.x_in(x_prev) + self.c_in(cpatch) + self.spk(spk)[:, None]
        for i, blk in enumerate(self.blocks):
            h = blk(h, None if caches is None else caches[i])
        return self.norm(h)

    def text(self, text: Tensor, text_len: Tensor, spk: Tensor, spk_emb: Tensor | None = None):
        return self.encoder(text, text_len, spk, spk_emb)

    # ------------------------------------------------------------------------------------------- prompted voices
    @property
    def prompt_row(self) -> int:
        """The speaker row of a voice given by a prompt (``model.ar.prompt``)."""
        if not self.prompted:
            raise ValueError("this model was not trained with voice prompts (model.ar.prompt)")
        return self.spk.num_embeddings - 1

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
                       drop: Tensor | None = None) -> Tensor:
        """States ``[B, S, dim]`` for every step, the previous target patches as input (plus ``input_noise`` ``[B]``
        times unit noise: robustness to the model's own, imperfect patches at inference). ``drop`` ``[B]`` (bool):
        utterances whose text condition the backbone does not see (the null condition of guidance)."""
        yp = patchify(y, self.patch)
        x_prev = torch.cat([self.bos.expand(yp.shape[0], 1, -1), yp[:, :-1]], 1)
        if input_noise is not None:
            x_prev = x_prev + input_noise[:, None, None] * torch.randn_like(x_prev)
        cp = self.cond_patches(cond)
        if drop is not None:
            cp = cp * (~drop)[:, None, None].to(cp.dtype)
        return self.backbone(x_prev, cp, spk)

    @torch.no_grad()
    def gta(self, cond: Tensor, y: Tensor, spk: Tensor, temperature: float = 1.0,
            generator: torch.Generator | None = None) -> Tensor:
        """Ground-truth-aligned ("GTA") generation: every patch drawn by the head (noise at ``temperature``) from the
        state after the *recorded* previous patches, in one teacher-forced pass. ``cond`` ``[B, C, T]`` aligned to
        ``y`` ``[B, n_mels, T]`` -> ``[B, n_mels, T]``, frame for frame with the recording (``finetune-vocoder``)."""
        T, k = y.shape[-1], self.patch
        cond, y = self.pad_frames(cond), self.pad_frames(y)
        s = self.teacher_forced(cond, y, spk)  # [B, S, dim]
        mu = patchify(cond[:, : self.n_mels], k)
        z = torch.randn(*s.shape[:2], self.head.noise_dim, device=s.device, generator=generator) * temperature
        x = mu + self.head(s.flatten(0, 1), z.flatten(0, 1)).view_as(mu)
        return unpatchify(x, k, self.n_mels)[..., :T]

    # ------------------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, cond: Tensor, spk: Tensor, temperature: float = 1.0, generator: torch.Generator | None = None,
                 prefix: Tensor | None = None, cfg_scale: float = 1.0, prefix_cond: Tensor | None = None) -> Tensor:
        """Aligned condition ``[B, C, T]`` -> normalised mel ``[B, n_mels, T]``, one step (patch) at a time with a KV
        cache. ``prefix``: known leading mel frames ``[B, n_mels, Tp]`` (``Tp`` a multiple of ``patch``), kept.
        ``prefix_cond`` ``[B, C, Tp]``: the prefix is then acoustic context that comes *before* ``cond`` (the end of
        the previous utterance and the condition it was generated under): the model continues from it as it
        continues a longer utterance, no noise is drawn for it, and only the ``T`` frames of ``cond`` are returned.
        ``cfg_scale`` != 1: guidance on the backbone state, ``s_null + cfg_scale * (s - s_null)``, the null state
        from the same history without the text (a model trained with ``drift.cond_dropout``)."""
        cond, lead = join_prefix(cond, prefix, prefix_cond, self.patch)
        T = cond.shape[-1]
        cond = self.pad_frames(cond)
        cp = self.cond_patches(cond)
        B, S = cp.shape[:2]
        k, mu = self.patch, patchify(cond[:, : self.n_mels], self.patch)
        guided = cfg_scale != 1.0
        if guided:  # the conditional and the null rows as one batch
            cp, spk_b = torch.cat([cp, torch.zeros_like(cp)]), torch.cat([spk, spk])
        else:
            spk_b = spk
        caches = [{} for _ in self.blocks]
        prev = self.bos.expand(B, 1, -1)
        out = []
        n_pre = 0 if prefix is None else prefix.shape[-1] // k
        pre = None if prefix is None else patchify(prefix, k)
        for t in range(S):
            s = self.backbone(torch.cat([prev, prev]) if guided else prev, cp[:, t: t + 1], spk_b, caches)[:, 0]
            if guided:
                s = s[B:] + cfg_scale * (s[:B] - s[B:])
            if t < n_pre:
                x = pre[:, t]
            else:
                z = torch.randn(B, self.head.noise_dim, device=cond.device, generator=generator) * temperature
                x = mu[:, t] + self.head(s, z)
            out.append(x)
            prev = x[:, None]
        mel = unpatchify(torch.stack(out, 1), k, self.n_mels)
        return mel[..., lead:T]

    def step_static(self, prev: Tensor, cpatch: Tensor, spk: Tensor, pos: Tensor, caches: list[tuple[Tensor, Tensor]],
                    mu: Tensor, z: Tensor, cfg: Tensor | None = None) -> Tensor:
        """One generation step with fixed shapes (:class:`ARGenerator`): the previous patch ``[B, k * n_mels]``, this
        step's condition input ``[B, *]``, the position ``pos`` ``[1]``, per-layer caches, the prior-mean patch ``mu``
        and the (scaled) noise ``z`` -> the next patch ``[B, k * n_mels]``. ``cfg`` ``[1]``: guidance; ``cpatch``,
        ``spk`` and the caches then have ``2 B`` rows (the conditional ones, then the null ones)."""
        if cfg is not None:
            prev = torch.cat([prev, prev])
        h = (self.x_in(prev) + self.c_in(cpatch) + self.spk(spk))[:, None]
        smax = caches[0][0].shape[2]
        mask = (torch.arange(smax, device=prev.device) <= pos)[None, None, None]
        for blk, (kc, vc) in zip(self.blocks, caches):
            h = blk.step_static(h, pos, kc, vc, mask)
        s = self.norm(h)[:, 0]
        if cfg is not None:
            B = s.shape[0] // 2
            s = s[B:] + cfg * (s[:B] - s[B:])
        return mu + self.head(s, z)

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
                   pitch: Tensor | None = None, context: int = 0, prompt: Tensor | None = None
                   ) -> tuple[Tensor, Tensor]:
        """As :meth:`DriftingTTS.synthesize` (the DiT's own arguments are accepted and ignored): normalised mels
        ``[B, n_mels, T]`` and their lengths. Durations from the duration predictor (``length_scale``) or given;
        ``cfg_scale``: guidance on the backbone state (:meth:`generate`). ``prompt`` (``model.ar.prompt``, #95):
        normalised mels ``[B, n_mels, P]`` of recordings whose voices the model continues in; ``spk`` is then
        ignored, and no transcript of the prompt is needed."""
        spk_emb = prefix = prefix_cond = None
        if prompt is not None:
            prompt = prompt[..., : prompt.shape[-1] // self.patch * self.patch]
            spk, spk_emb = torch.full_like(spk, self.prompt_row), self.prompt_vector(prompt)
        cond, y_len = self.aligned_condition(text, text_len, spk, length_scale, pitch_shift, durations, pitch, context,
                                             spk_emb)
        if prompt is not None:  # the prompt is history without a text condition
            prefix, prefix_cond = prompt, cond.new_zeros(cond.shape[0], cond.shape[1], prompt.shape[-1])
        if self.fast_generator is not None:
            mel = self.fast_generator(cond, spk, temperature, generator, cfg_scale, prefix=prefix,
                                      prefix_cond=prefix_cond)
        else:
            mel = self.generate(cond, spk, temperature, generator, prefix=prefix, cfg_scale=cfg_scale,
                                prefix_cond=prefix_cond)
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
              "caches": [(torch.zeros(R, heads, self.max_steps, dim // heads, device=dev, dtype=dt),
                          torch.zeros(R, heads, self.max_steps, dim // heads, device=dev, dtype=dt))
                         for _ in m.blocks]}

        def run():
            return m.step_static(st["prev"], st["cp"], st["spk"], st["pos"], st["caches"], st["mu"], st["z"],
                                 st["cfg"])

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

    @torch.no_grad()
    def fits(self, frames: int, prefix_frames: int = 0) -> bool:
        """Whether ``frames`` frames after ``prefix_frames`` of context fit the caches."""
        return (prefix_frames + frames + (-frames) % self.model.patch) // self.model.patch <= self.max_steps

    @torch.no_grad()
    def __call__(self, cond: Tensor, spk: Tensor, temperature: float = 1.0,
                 generator: torch.Generator | None = None, cfg_scale: float = 1.0, prefix: Tensor | None = None,
                 prefix_cond: Tensor | None = None) -> Tensor:
        """As :meth:`ARDriftTTS.generate`: aligned condition ``[B, C, T]`` -> mel ``[B, n_mels, T]``."""
        m = self.model
        T = cond.shape[-1]
        if not self.fits(T, 0 if prefix_cond is None else prefix_cond.shape[-1]):
            return m.generate(cond, spk, temperature, generator, prefix, cfg_scale, prefix_cond)
        out = list(self.steps(cond, spk, temperature, generator, cfg_scale, prefix, prefix_cond))
        return unpatchify(torch.stack(out, 1), m.patch, m.n_mels)[..., :T]

    @torch.no_grad()
    def steps(self, cond: Tensor, spk: Tensor, temperature: float = 1.0, generator: torch.Generator | None = None,
              cfg_scale: float = 1.0, prefix: Tensor | None = None, prefix_cond: Tensor | None = None):
        """Yield the patches ``[B, k * n_mels]`` of ``cond`` one step at a time (streaming); at most ``max_steps``
        with the prefix. ``prefix`` / ``prefix_cond``: as :meth:`ARDriftTTS.generate` (kept leading frames are
        yielded as they are; the steps of a context before ``cond`` run, to fill the caches, but are not yielded)."""
        m = self.model
        cond, lead = join_prefix(cond, prefix, prefix_cond, m.patch)
        pre = None if prefix is None else patchify(prefix, m.patch)
        n_pre = 0 if pre is None else pre.shape[1]
        cond = m.pad_frames(cond)
        cp = m.cond_patches(cond)
        B, S = cp.shape[:2]
        if S > self.max_steps:
            raise ValueError(f"{S} steps: longer than the caches ({self.max_steps})")
        mu = patchify(cond[:, : m.n_mels], m.patch)
        guided = cfg_scale != 1.0
        st = self._buffers(B, cp.shape[-1], guided)
        st["prev"].copy_(m.bos.expand(B, -1))
        if guided:
            cp = torch.cat([cp, torch.zeros_like(cp)])
            st["spk"].copy_(torch.cat([spk, spk]))
            st["cfg"].fill_(cfg_scale)
        else:
            st["spk"].copy_(spk)
        for t in range(S):
            st["cp"].copy_(cp[:, t])
            st["mu"].copy_(mu[:, t])
            st["pos"].fill_(t)
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


def ar_features(x: Tensor, k: int, n_mels: int, detail: bool = True, energy_bands: Sequence[int] | None = None,
                energy_scale: float = 1.0) -> dict[str, Tensor]:
    """Drift kernel features of patches ``[R, S, k * n_mels]`` (R rows, S samples): ``patch`` (the whole patch, one
    location), ``frame`` (each frame a location), ``detail`` (each frame minus its frequency-smoothed envelope: the
    harmonic structure) and, with ``energy_bands``, ``energy`` (each frame's band energies, :func:`band_energy`).
    Every map is ``[R, S, L, D]``."""
    from .mae import spectral_detail

    R, S, _ = x.shape
    frames = x.view(R, S, k, n_mels)
    out = {"patch": x[:, :, None], "frame": frames}
    if detail:
        d = spectral_detail(frames.reshape(R * S, k, n_mels).transpose(1, 2)).transpose(1, 2)
        out["detail"] = d.reshape(R, S, k, n_mels)
    if energy_bands:
        out["energy"] = band_energy(frames, energy_scale, energy_bands)
    return out



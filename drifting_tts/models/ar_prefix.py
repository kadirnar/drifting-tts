"""ARDrift v2 (#103): the text as a prefix instead of a forced alignment.

:class:`~drifting_tts.models.ar.ARDriftTTS` is a Grad-TTS front end with an autoregressive back end: a character
encoder, monotonic alignment, a duration predictor, a prior mean, and a head that draws a residual over that mean.
Its rhythm is decided before the first patch is drawn. Pocket TTS, CALM, DiTAR and VoxCPM give the text as tokens and
let the autoregressive model decide the timing and the end of the utterance; rhythm, pauses and intonation are then
sampled by one model that hears its own output.

The sequence is Kyutai's: ``[voice] [text] [audio]``.

* **voice** (optional, ``model.ar.prompt``): the mel patches of another recording of the speaker, no transcript;
* **text**: the characters (the frontend's symbols, without the blanks the aligner needs), one embedding each;
* **audio**: the patches of ``patch`` mel frames, each drawn in one pass by the drifting head from the state after
  the patches before it. No prior mean: the head draws the patch itself.

With ``model.ar.prenet`` the patches the model reads back go through a narrow pre-net with dropout
(:class:`PreNet`), so that the history alone does not say what comes next.

One causal transformer reads all of it. A second head on the same state says whether the utterance is over (the
step after the last patch; Kyutai's ``eos_loss_weight: 0.1``). The rows of a batch are padded **in front**, so that
the audio starts at the same position in every row; rotary positions are relative, so a row does not see its
padding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..text import BLANK_ID, PAD_ID, frontend
from .ar import Block, DriftHead, patchify, unpatchify
from .generator import RMSNorm

TEXT, VOICE, AUDIO = 0, 1, 2  # segments


class PreNet(nn.Module):
    """A bottleneck on the patches the model reads back (``model.ar.prenet``): narrow layers with dropout, **in
    training and at inference**. Teacher-forced, the patch before a step says most of what the next one is, and a
    model that can read it exactly has little to gain from the text; Tacotron 2 found this bottleneck "essential
    for learning attention", and MELLE, the mel model with the text as a prefix, keeps it (three layers, dropout
    0.5, at inference too)."""

    def __init__(self, inp: int, out: int, dim: int = 256, dropout: float = 0.5, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim if i else inp, dim) for i in range(layers)])
        self.out = nn.Linear(dim, out)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), self.dropout, training=True)
        return self.out(x)


def characters(text: Tensor, text_len: Tensor) -> tuple[Tensor, Tensor]:
    """Token ids interspersed with blanks (``blank c blank c ... blank``, what the aligned models read) -> the
    characters alone ``[B, N]`` and their number ``[B]``."""
    if text.shape[1] > 1 and bool((text[:, 0] == BLANK_ID).all()):
        return text[:, 1::2], text_len // 2
    return text, text_len


def front_pad(seqs: list[Tensor], dim: int) -> tuple[Tensor, Tensor]:
    """Sequences ``[n_b, dim]`` -> ``[B, N, dim]`` padded in front, and the mask of what is not padding ``[B, N]``."""
    n = max(s.shape[0] for s in seqs)
    out = seqs[0].new_zeros(len(seqs), n, dim)
    mask = torch.zeros(len(seqs), n, dtype=torch.bool, device=seqs[0].device)
    for b, s in enumerate(seqs):
        if s.shape[0]:
            out[b, n - s.shape[0]:], mask[b, n - s.shape[0]:] = s, True
    return out, mask


class ARPrefixTTS(nn.Module):
    """See the module docstring. ``cfg``: the ``model`` section (``text.language``, ``ar``)."""

    def __init__(self, cfg, num_speakers: int, n_mels: int = 100):
        super().__init__()
        t, a = cfg.text, cfg.ar
        self.vocab = len(frontend(t.get("language")).SYMBOLS)
        self.n_mels, self.patch, dim = n_mels, int(a.patch), int(a.dim)
        self.prompted = bool(a.get("prompt", False))
        num_speakers += int(self.prompted)  # the last row: "the voice is in the prompt"
        k = self.patch
        self.tok = nn.Embedding(self.vocab, dim)
        self.seg = nn.Parameter(torch.zeros(3, dim))
        pn = a.get("prenet")  # {dim, dropout, layers}: a bottleneck on the history (see PreNet)
        self.x_in = (PreNet(k * n_mels, dim, int(pn.get("dim", 256)), float(pn.get("dropout", 0.5)),
                            int(pn.get("layers", 2))) if pn else nn.Linear(k * n_mels, dim))
        self.spk = nn.Embedding(num_speakers, dim)
        nn.init.normal_(self.spk.weight, std=0.02)
        self.bos = nn.Parameter(torch.zeros(k * n_mels))
        self.blocks = nn.ModuleList([Block(dim, int(a.heads), float(a.get("mlp_ratio", 4.0)))
                                     for _ in range(int(a.depth))])
        self.norm = RMSNorm(dim)
        self.head = DriftHead(dim, k * n_mels, int(a.head_dim), int(a.head_depth), a.get("noise_dim"))
        self.eos = nn.Linear(dim, 1)
        self.max_rate = float(a.get("max_seconds_per_char", 0.35))  # a bound on the length, should the end not come
        self.temperature = None

    @property
    def prompt_row(self) -> int:
        if not self.prompted:
            raise ValueError("this model was not trained with voice prompts (model.ar.prompt)")
        return self.spk.num_embeddings - 1

    # ------------------------------------------------------------------------------------------------- the prefix
    def prefix(self, text: Tensor, text_len: Tensor, spk: Tensor, prompt: Tensor | None = None,
               prompt_len: Tensor | None = None, drop: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """What stands in front of the audio: ``[voice] [text]`` as embeddings ``[B, P, dim]``, padded in front,
        and its mask ``[B, P]``. ``text``: characters ``[B, N]`` with ``text_len``; ``prompt``: normalised mels
        ``[B, n_mels, R]`` with ``prompt_len`` frames (0: none); ``drop`` ``[B]``: rows without their text (the
        null branch of guidance: the text segment's embedding alone, once)."""
        k, rows = self.patch, []
        tok = self.tok(text.clamp_min(0)) + self.seg[TEXT]
        if prompt is not None:
            voice = self.x_in(patchify(prompt[..., : prompt.shape[-1] // k * k], k)) + self.seg[VOICE]
        for b in range(text.shape[0]):
            parts = []
            if prompt is not None:
                n = prompt.shape[-1] // k if prompt_len is None else int(prompt_len[b]) // k
                parts.append(voice[b, :n])
            if drop is not None and bool(drop[b]):
                parts.append(self.seg[TEXT][None].to(tok.dtype))
            else:
                parts.append(tok[b, : int(text_len[b])])
            rows.append(torch.cat(parts))
        out, mask = front_pad(rows, tok.shape[-1])
        return out + self.spk(spk)[:, None].to(out.dtype), mask

    def states(self, prefix: Tensor, mask: Tensor, x_prev: Tensor, spk: Tensor) -> Tensor:
        """The states of the audio positions ``[B, S, dim]``: the prefix (:meth:`prefix`) and the patches before
        each step ``x_prev`` ``[B, S, k * n_mels]`` (the first: :attr:`bos`) through the causal transformer."""
        P, S = prefix.shape[1], x_prev.shape[1]
        audio = self.x_in(x_prev) + self.seg[AUDIO] + self.spk(spk)[:, None]
        h = torch.cat([prefix.to(audio.dtype), audio], 1)
        L = P + S
        keys = torch.cat([mask, torch.ones(mask.shape[0], S, dtype=torch.bool, device=mask.device)], 1)
        causal = torch.ones(L, L, dtype=torch.bool, device=h.device).tril()
        attn = causal[None, None] & keys[:, None, None]
        attn = attn | torch.eye(L, dtype=torch.bool, device=h.device)[None, None]  # a padded position reads itself
        for blk in self.blocks:
            h = blk(h, mask=attn)
        return self.norm(h[:, P:])

    def teacher_forced(self, text: Tensor, text_len: Tensor, y: Tensor, spk: Tensor,
                       input_noise: Tensor | None = None, prompt: Tensor | None = None,
                       prompt_len: Tensor | None = None, drop: Tensor | None = None) -> Tensor:
        """States ``[B, S + 1, dim]`` for the ``S`` patches of ``y`` ``[B, n_mels, S * k]`` and the step after them
        (where the end is said), the recorded patches as the history (plus ``input_noise`` ``[B]`` times unit
        noise)."""
        yp = patchify(y, self.patch)
        x_prev = torch.cat([self.bos.expand(yp.shape[0], 1, -1), yp], 1)
        if input_noise is not None:
            noise = input_noise[:, None, None] * torch.randn_like(x_prev)
            noise[:, 0] = 0  # the beginning-of-sequence patch is not audio
            x_prev = x_prev + noise
        prefix, mask = self.prefix(text, text_len, spk, prompt, prompt_len, drop)
        return self.states(prefix, mask, x_prev, spk)

    # -------------------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, text: Tensor, spk: Tensor, temperature: float = 1.0, generator: torch.Generator | None = None,
                 prompt: Tensor | None = None, max_steps: int | None = None, min_steps: int = 4,
                 eos_threshold: float = 0.5) -> Tensor:
        """Characters ``[1, N]`` -> normalised mel ``[1, n_mels, T]``: one utterance, one patch at a time with a KV
        cache, until the end head says so (``eos_threshold``; never before ``min_steps``) or ``max_steps`` (default:
        :attr:`max_rate` seconds per character)."""
        from ..audio import HOP_LENGTH, SAMPLE_RATE

        if text.shape[0] != 1:
            raise ValueError("generate takes one utterance (rows of different length need padding in front)")
        k = self.patch
        n = torch.tensor([text.shape[1]], device=text.device)
        if max_steps is None:
            max_steps = max(int(text.shape[1] * self.max_rate * SAMPLE_RATE / HOP_LENGTH / k), min_steps + 1)
        prefix, _ = self.prefix(text, n, spk, prompt)
        caches = [{} for _ in self.blocks]
        h = torch.cat([prefix, (self.x_in(self.bos) + self.seg[AUDIO] + self.spk(spk))[:, None].to(prefix.dtype)], 1)
        out = []
        for t in range(max_steps):
            for blk, cache in zip(self.blocks, caches):
                h = blk(h, cache)
            s = self.norm(h[:, -1])
            if t >= min_steps and float(torch.sigmoid(self.eos(s).float())[0, 0]) > eos_threshold:
                break
            z = torch.randn(1, self.head.noise_dim, device=text.device, generator=generator) * temperature
            x = self.head(s, z.to(s.dtype))
            out.append(x)
            h = (self.x_in(x) + self.seg[AUDIO] + self.spk(spk))[:, None]
        if not out:
            return text.new_zeros(1, self.n_mels, 0, dtype=torch.float)
        return unpatchify(torch.stack(out, 1).float(), k, self.n_mels)

    @torch.no_grad()
    def synthesize(self, text: Tensor, text_len: Tensor, spk: Tensor, cfg_scale: float = 1.0,
                   temperature: float = 1.0, generator: torch.Generator | None = None,
                   prompt: Tensor | None = None, **_) -> tuple[Tensor, Tensor]:
        """As the other models' ``synthesize`` for one utterance: token ids as the frontend gives them (with or
        without blanks) -> normalised mel ``[1, n_mels, T]`` and its length. ``prompt``: a recording of the voice
        (``model.ar.prompt``); ``spk`` is then ignored."""
        chars, n = characters(text, text_len)
        if prompt is not None:
            spk = torch.full_like(spk, self.prompt_row)
        mel = self.generate(chars[:, : int(n[0])], spk, temperature, generator, prompt)
        return mel, torch.tensor([mel.shape[-1]], device=mel.device)


def eos_loss(logits: Tensor, steps: Tensor) -> tuple[Tensor, Tensor]:
    """The end head's loss: ``logits`` ``[B, S + 1]`` of every step, ``steps`` ``[B]`` the patches of each
    utterance. The target is 1 at the step after the last patch and 0 before it (later steps are padding). Returns
    the mean binary cross-entropy and the accuracy at 0."""
    pos = torch.arange(logits.shape[1], device=logits.device)[None]
    valid = pos <= steps[:, None]
    target = (pos == steps[:, None]).float()
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    acc = (((logits > 0).float() == target) & valid).sum() / valid.sum()
    return (loss * valid).sum() / valid.sum(), acc

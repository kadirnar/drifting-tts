"""Dataset over the packed log-mels written by :mod:`drifting_tts.prepare`."""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .audio import FRAME_RATE, N_MELS
from .text import PAD_ID, frontend

SCORE_ALIASES = {"mos": "mos_ovrl", "rate": "cps", "bandwidth": "bw_hz"}


def load_scores(root: str | Path) -> dict[int, dict]:
    """``scores.jsonl`` of ``drifting-tts score``: ``index.jsonl`` line number -> score fields (later lines win)."""
    path = Path(root) / "scores.jsonl"
    out: dict[int, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # line cut short by an interrupted run
                    continue
                out.setdefault(r.pop("i"), {}).update(r)
    return out


def parse_filters(filters: dict | None) -> list[tuple[str, float, float]]:
    """``{"max_cer": 0.1, "min_mos": 3.0, "rate": [8, 20]}`` -> ``[(score field, low, high), ...]``.

    Keys are ``min_<field>``, ``max_<field>`` or ``<field>: [low, high]`` (``null`` = open end) over the fields of
    ``scores.jsonl``; ``mos``, ``rate`` and ``bandwidth`` alias ``mos_ovrl``, ``cps`` and ``bw_hz``.
    ``apply_to_val`` and ``require_scores`` are flags, not rules."""
    rules = []
    for key, v in (filters or {}).items():
        if key in ("apply_to_val", "require_scores"):
            continue
        if key.startswith(("min_", "max_")):
            lo, hi = (v, None) if key.startswith("min_") else (None, v)
            key = key[4:]
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = v
        else:
            raise ValueError(f"data.filters: cannot parse {key}: {v!r}")
        rules.append((SCORE_ALIASES.get(key, key), -np.inf if lo is None else float(lo),
                      np.inf if hi is None else float(hi)))
    return rules


def passes(rec: dict | None, rules: list[tuple[str, float, float]], require: bool = False) -> bool:
    """Whether the scores satisfy every rule. A missing score passes, so that unscored utterances are not dropped;
    with ``require`` (``data.filters.require_scores``, #104) it fails: an utterance nobody scored is not known to be
    good."""
    rec = rec or {}
    return all((not require) if rec.get(f) is None else lo <= rec[f] <= hi for f, lo, hi in rules)


class MelStats:
    """Normalisation ``(x - mean) / std``: one scalar for log-mels, one value per channel for VAE latents (lists in
    ``stats.json``; a channel's std is floored at ``min_std`` so collapsed channels stay finite)."""

    def __init__(self, mean: float | list[float], std: float | list[float], min_std: float = 1e-3):
        m, s = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        if m.numel() == 1:
            self.mean, self.std = float(m), float(s)
        else:  # [C, 1]: broadcasts over [..., C, T]
            self.mean, self.std = m.reshape(-1, 1), s.reshape(-1, 1).clamp_min(min_std)

    @classmethod
    def load(cls, root: str | Path) -> MelStats:
        s = json.loads((Path(root) / "stats.json").read_text())
        return cls(s["mean"], s["std"])

    @property
    def per_channel(self) -> bool:
        return isinstance(self.mean, torch.Tensor)

    def _on(self, x: torch.Tensor):
        if not self.per_channel:
            return self.mean, self.std
        return self.mean.to(x.device, x.dtype), self.std.to(x.device, x.dtype)

    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        mean, std = self._on(mel)
        return (mel - mean) / std

    def denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        mean, std = self._on(mel)
        return mel * std + mean

    def to_dict(self) -> dict:
        if not self.per_channel:
            return {"mean": self.mean, "std": self.std}
        return {"mean": self.mean.flatten().tolist(), "std": self.std.flatten().tolist()}


def reference_frames(rc) -> tuple[int, int] | None:
    """``drift.reference`` -> the shortest and the longest reference in frames (``min_seconds``, ``max_seconds``:
    3 to 10 s), or ``None`` without it."""
    if not rc:
        return None
    return int(float(rc.get("min_seconds", 3.0)) * FRAME_RATE), int(float(rc.get("max_seconds", 10.0)) * FRAME_RATE)


def crop_reference(mel: torch.Tensor, lo: int, hi: int, rng=random) -> torch.Tensor:
    """A reference (#96): a random crop of ``lo`` to ``hi`` frames of the normalised mel ``[n_mels, T]`` of another
    utterance (``T >= lo``; all of it when it is shorter than the drawn length)."""
    n = min(rng.randint(lo, hi), mel.shape[1])
    start = rng.randint(0, mel.shape[1] - n)
    return mel[:, start: start + n]


class MelDataset(Dataset):
    """Items: interspersed symbol ids, normalised log-mel ``[n_mels, T]`` and the speaker id.

    ``filters`` (config ``data.filters``, see :func:`parse_filters`) drop utterances by their ``scores.jsonl``
    scores; without that file nothing is dropped. Only the ``train`` split is filtered: held-out splits (``val``)
    are just reported on unless ``filters.apply_to_val`` is set, so evaluation stays comparable across curation
    settings."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        min_quality: float = 0.0,
        min_frames: int = 64,
        max_frames: int = 1500,
        with_f0: bool = False,
        with_audio: bool = False,
        filters: dict | None = None,
        reference: tuple[int, int] | None = None,
    ):
        root = Path(root)
        self.stats = MelStats.load(root)
        st = json.loads((root / "stats.json").read_text())
        self.backend = st.get("backend", "vocos")
        self.language = st.get("language", "tr")  # the text frontend of norm_text (prepare --language)
        # features per frame and frames per second: 100 log-mel bins at 93.75 Hz, or VAE latents (extract-latents)
        self.dim, self.frame_rate = int(st.get("dim", N_MELS)), float(st.get("frame_rate", FRAME_RATE))
        self.latent_repeat = int(st.get("latent_repeat", 1))  # extract-latents: copies of every VAE frame
        self.with_f0 = with_f0
        self._f0_path = root / "f0.bin"
        if with_f0 and not self._f0_path.exists():
            raise FileNotFoundError(f"{self._f0_path} missing: run `drifting-tts prepare --f0`")
        self._f0: np.ndarray | None = None
        self.with_audio = with_audio
        self._audio_path = root / "audio.bin"
        if with_audio and not self._audio_path.exists():
            raise FileNotFoundError(f"{self._audio_path} missing: run `drifting-tts prepare --save-audio`")
        self._audio: np.ndarray | None = None
        rules = parse_filters(filters)
        scores = load_scores(root) if rules else {}
        drop = bool(rules and scores) and (split == "train" or bool(filters.get("apply_to_val")))
        self.items = []
        failed = 0
        with open(root / "index.jsonl") as f:
            for n, line in enumerate(f):
                e = json.loads(line)
                n_tokens = 2 * len(e["norm_text"]) + 1
                if (e["split"] == split and (e.get("quality_score") or 0) >= min_quality
                        and min_frames <= e["frames"] <= max_frames and n_tokens <= e["frames"]):
                    if scores and not passes(scores.get(n), rules, bool(filters.get("require_scores"))):
                        failed += 1
                        if drop:
                            continue
                    self.items.append(e)
        if rules and not scores:
            print(f"data.filters: no {root / 'scores.jsonl'} (run `drifting-tts score`), nothing filtered")
        elif rules:
            hours = sum(e["frames"] for e in self.items) / self.frame_rate / 3600
            total = len(self.items) + failed * drop
            print(f"data.filters ({split}): {failed}/{total} utterances fail; "
                  + (f"kept {len(self.items)} ({hours:.1f} h)" if drop else "not filtered (apply_to_val: false)"))
        self.num_speakers = len(json.loads((root / "speakers.json").read_text()))
        # reference (#96): every item also holds "ref", a crop of another utterance of its speaker (None: the
        # speaker has no other utterance that is long enough)
        self.reference, self.by_speaker = reference, {}
        if reference:
            for n, e in enumerate(self.items):
                if e["frames"] >= reference[0]:
                    self.by_speaker.setdefault(e["spk_id"], []).append(n)
        self._mels_path = root / "mels.bin"
        self._mels: np.ndarray | None = None  # opened lazily (once per dataloader worker)

    @property
    def mels(self) -> np.ndarray:
        if self._mels is None:
            self._mels = np.memmap(self._mels_path, dtype=np.float16, mode="r").reshape(-1, self.dim)
        return self._mels

    @property
    def f0(self) -> np.ndarray:
        if self._f0 is None:
            self._f0 = np.memmap(self._f0_path, dtype=np.float16, mode="r")
        return self._f0

    @property
    def audio(self) -> np.ndarray:
        if self._audio is None:
            self._audio = np.memmap(self._audio_path, dtype=np.float16, mode="r")
        return self._audio

    def __len__(self) -> int:
        return len(self.items)

    def frames(self, i: int) -> int:
        return self.items[i]["frames"]

    def _mel(self, i: int) -> torch.Tensor:
        e = self.items[i]
        mel = torch.from_numpy(np.array(self.mels[e["offset"]: e["offset"] + e["frames"]], dtype=np.float32))
        return self.stats.normalize(mel.T.contiguous())

    def __getitem__(self, i: int) -> dict:
        e = self.items[i]
        item = {
            "text": torch.tensor(frontend(self.language).text_to_ids(e["norm_text"], normalized=True),
                                 dtype=torch.long),
            "mel": self._mel(i),
            "spk": e["spk_id"],
            "index": i,
        }
        if self.reference:
            others = [j for j in self.by_speaker.get(e["spk_id"], ()) if j != i]
            item["ref"] = crop_reference(self._mel(random.choice(others)), *self.reference) if others else None
        if self.with_audio:
            a = self.audio[e["audio_offset"]: e["audio_offset"] + e["audio_samples"]]
            item["audio"] = torch.from_numpy(np.array(a, dtype=np.float32))
        if self.with_f0:
            item["f0"] = torch.from_numpy(np.array(self.f0[e["offset"]: e["offset"] + e["frames"]], dtype=np.float32))
        return item


def collate(batch: list[dict]) -> dict:
    """Pad to the longest item. Returns texts ``[B, N]``, mels ``[B, n_mels, T]`` and lengths."""
    B = len(batch)
    text_len = torch.tensor([b["text"].numel() for b in batch])
    mel_len = torch.tensor([b["mel"].shape[1] for b in batch])
    text = torch.full((B, int(text_len.max())), PAD_ID, dtype=torch.long)
    mel = torch.zeros(B, batch[0]["mel"].shape[0], int(mel_len.max()))
    for i, b in enumerate(batch):
        text[i, : text_len[i]] = b["text"]
        mel[i, :, : mel_len[i]] = b["mel"]
    out = {"text": text, "text_len": text_len, "mel": mel, "mel_len": mel_len,
           "spk": torch.tensor([b["spk"] for b in batch]), "index": torch.tensor([b["index"] for b in batch])}
    if any("ref" in b for b in batch):  # references (#96): [B, n_mels, R], 0 frames where there is none
        refs = [b.get("ref") for b in batch]
        ref_len = torch.tensor([0 if r is None else r.shape[1] for r in refs])
        ref = torch.zeros(B, mel.shape[1], max(int(ref_len.max()), 1))
        for i, r in enumerate(refs):
            if r is not None:
                ref[i, :, : ref_len[i]] = r
        out["ref"], out["ref_len"] = ref, ref_len
    if "audio" in batch[0]:
        n = max(b["audio"].numel() for b in batch)
        audio = torch.zeros(B, n)
        for i, b in enumerate(batch):
            audio[i, : b["audio"].numel()] = b["audio"]
        out["audio"] = audio
    if "f0" in batch[0]:
        f0 = torch.zeros(B, int(mel_len.max()))
        for i, b in enumerate(batch):
            f0[i, : b["f0"].numel()] = b["f0"]
        out["f0"] = f0
    return out


class BucketBatchSampler(Sampler[list[int]]):
    """Batches of similar length with a frame budget, reshuffled every epoch (infinite if ``loop``)."""

    def __init__(self, lengths: list[int], max_frames: int, max_batch: int, bucket: int = 2048, seed: int = 0,
                 drop_last: bool = True):
        self.lengths, self.max_frames, self.max_batch = lengths, max_frames, max_batch
        self.bucket, self.seed, self.drop_last, self.epoch = bucket, seed, drop_last, 0

    def _order(self, rng: random.Random) -> list[int]:
        """The utterances of one epoch: every one once, shuffled."""
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        return order

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = self._order(rng)
        batches = []
        for s in range(0, len(order), self.bucket):
            chunk = sorted(order[s: s + self.bucket], key=lambda i: self.lengths[i])
            cur: list[int] = []
            for i in chunk:
                if cur and (len(cur) + 1 > self.max_batch or (len(cur) + 1) * self.lengths[i] > self.max_frames):
                    batches.append(cur)
                    cur = []
                cur.append(i)
            if cur and not self.drop_last:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._batches()
        self.epoch += 1

    def __len__(self) -> int:
        return len(self._batches())


class WeightedBucketBatchSampler(BucketBatchSampler):
    """:class:`BucketBatchSampler` whose epochs draw ``len(lengths)`` utterances with replacement, in proportion to
    ``weights`` (e.g. :func:`speaker_balance_weights`), instead of taking every utterance once."""

    def __init__(self, lengths: list[int], weights: list[float], **kw):
        super().__init__(lengths, **kw)
        if len(weights) != len(lengths) or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("weights: one non-negative weight per utterance, not all zero")
        self.cum_weights = list(itertools.accumulate(float(w) for w in weights))

    def _order(self, rng: random.Random) -> list[int]:
        return rng.choices(range(len(self.lengths)), cum_weights=self.cum_weights, k=len(self.lengths))


def speaker_balance_weights(speakers: list[int], lengths: list[int], shares: dict | None = None,
                            temperature: float = 1.0) -> list[float]:
    """Per-utterance sampling weights (summing to 1) that give speaker ``s`` the share ``shares[s]`` of the draws and
    split the rest over the other speakers in proportion to ``frames_s ** temperature``, ``frames_s`` being the
    speaker's total length: 1 in proportion to their audio, 0 every speaker equally often. A speaker's utterances
    are drawn equally often, as by :class:`BucketBatchSampler`."""
    shares = {int(k): float(v) for k, v in (shares or {}).items()}
    count, frames = Counter(speakers), Counter()
    for s, n in zip(speakers, lengths):
        frames[s] += n
    if unknown := sorted(set(shares) - set(count)):
        raise ValueError(f"speaker_balance.shares: speakers {unknown} have no utterances")
    rest = [s for s in count if s not in shares]
    fixed = sum(shares.values())
    if min(shares.values(), default=0.0) < 0 or fixed > 1 + 1e-9 or (not rest and fixed <= 0):
        raise ValueError(f"speaker_balance.shares must be non-negative and sum to at most 1, got {shares}")
    z = sum(frames[s] ** temperature for s in rest)
    share = {**{s: max(0.0, 1 - fixed) * frames[s] ** temperature / z for s in rest}, **shares} if rest else \
        {s: v / fixed for s, v in shares.items()}
    return [share[s] / count[s] for s in speakers]

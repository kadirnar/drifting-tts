"""Score utterances for data curation: ASR agreement, speaking rate, DNSMOS, bandwidth, speaker purity.

Reads ``audio.bin`` (``prepare --save-audio``) and ``index.jsonl`` and appends to ``scores.jsonl``, one JSON object
per utterance keyed by its ``index.jsonl`` line number ``i``. Runs are resumable per scorer, and the file is
compacted to one line per utterance at the end (so do not run two ``score`` processes on one dataset at once).

    asr        ``hyp``, ``cer``, ``wer``: Whisper large-v3 (faster-whisper, Turkish, beam 5, batched across clips)
               against ``norm_text``, both normalised and stripped of punctuation as in ``evaluate``
    rate       ``cps``: letters of ``norm_text`` per second of audio
    mos        ``mos_sig``, ``mos_bak``, ``mos_ovrl`` (DNSMOS P.835) and ``mos_p808``: Microsoft's ONNX models from
               ``speechmos``, run on the GPU via ``onnx2torch`` (16 kHz, 9.01 s windows with 1 s hop)
    bandwidth  ``bw_hz``: highest frequency whose long-term spectrum is within ``--bw-db`` dB of its peak (24 kHz)
    spk        ``spk_sim``: cosine of the ECAPA embedding to the speaker's centroid, ``spk_next``: to the closest
               other speaker's centroid; embeddings are cached in ``spk_emb.npy``

``--summary`` prints the score distributions, outlier counts, the hours that survive the filters and per-speaker
counts instead of scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .audio import FRAME_RATE, SAMPLE_RATE
from .config import Config, apply_overrides, load_config
from .data import load_scores, parse_filters, passes
from .evaluate import _plain

try:
    from rapidfuzz.distance.Levenshtein import distance as _levenshtein
except ImportError:  # pragma: no cover - rapidfuzz comes with the `score` and `eval` extras
    _levenshtein = None

SCORERS = ("asr", "rate", "mos", "bandwidth", "spk")
SR16 = 16_000
FIELDS = ("cer", "wer", "cps", "mos_sig", "mos_bak", "mos_ovrl", "mos_p808", "bw_hz", "spk_sim", "spk_next")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default="data/train", help="prepared dataset (needs audio.bin: prepare --save-audio)")
    p.add_argument("--scorers", default=",".join(SCORERS), help="comma-separated subset of " + ",".join(SCORERS))
    p.add_argument("--summary", action="store_true", help="print distributions and filter yield instead of scoring")
    p.add_argument("--config", default=None, help="summary: apply this config's data.filters/min_quality/max_frames")
    p.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE",
                   help="summary: filter rule on top of --config, e.g. max_cer=0.1, min_mos=3, rate=[8,20]")
    p.add_argument("--speakers-tsv", default=None, help="summary: also write the full per-speaker table here")
    p.add_argument("--asr-model", default="large-v3", help="faster-whisper model")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--spk-model", default="speechbrain/spkrec-ecapa-voxceleb", help="SpeechBrain speaker encoder")
    p.add_argument("--bw-db", type=float, default=50.0, help="bandwidth: level below the spectral peak, dB")
    p.add_argument("--batch-size", type=int, default=32, help="clips per GPU batch")
    p.add_argument("--asr-batch-size", type=int, default=16, help="clips per Whisper decode (x beam size sequences)")
    p.add_argument("--workers", type=int, default=4, help="audio loading / resampling processes")
    p.add_argument("--limit", type=int, default=0, help="score only the first N utterances (debugging)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------------- pure functions


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or lists of words)."""
    if _levenshtein is not None:
        return _levenshtein(a, b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rates(ref: str, hyp: str) -> tuple[float, float]:
    """CER and WER of ``hyp`` against ``ref``, both normalised without punctuation (as in ``evaluate``)."""
    r, h = _plain(ref), _plain(hyp)
    return edit_distance(r, h) / max(len(r), 1), edit_distance(r.split(), h.split()) / max(len(r.split()), 1)


def speaking_rate(e: dict) -> float:
    """Letters of ``norm_text`` (no spaces or punctuation) per second of (edge-trimmed) audio."""
    return sum(c.isalpha() for c in e["norm_text"]) / (e["frames"] / FRAME_RATE)


@torch.no_grad()
def bandwidth_hz(wavs: list[torch.Tensor], db: float = 50.0, sr: int = SAMPLE_RATE, n_fft: int = 1024) -> list[float]:
    """Effective bandwidth: the highest frequency at which the long-term power spectrum (averaged over the frames
    within 40 dB of the loudest one) is at most ``db`` below its peak above 80 Hz."""
    x = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
    hop = n_fft // 4
    spec = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft, device=x.device), center=False,
                      return_complex=True).abs().square()  # [B, F, T]
    lens = torch.tensor([len(w) for w in wavs], device=x.device)
    valid = torch.arange(spec.shape[-1], device=x.device)[None] * hop + n_fft <= lens[:, None]
    energy = (10 * torch.log10(spec.sum(1) + 1e-10)).masked_fill(~valid, -1e4)
    active = (energy > energy.amax(-1, keepdim=True) - 40).float()
    lts = 10 * torch.log10((spec * active[:, None]).sum(-1) / active.sum(-1, keepdim=True).clamp_min(1) + 1e-20)
    lo = int(80 * n_fft / sr) + 1
    above = lts >= lts[:, lo:].amax(-1, keepdim=True) - db
    top = above.shape[-1] - 1 - above.flip(-1).float().argmax(-1)
    return (top * sr / n_fft).tolist()


def speaker_purity(emb: np.ndarray, spk: np.ndarray, trim: float = 0.2):
    """Per utterance: cosine to its own speaker's centroid and to the closest other speaker's centroid.

    A centroid is the mean of the speaker's L2-normalised embeddings without the ``trim`` fraction least similar
    to the plain mean (robust to mislabelled clips); an utterance's own embedding is left out of its centroid
    (NaN for a speaker with a single utterance). Returns ``(sim, next_sim, speakers, centroids)``."""
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-8)
    speakers = np.unique(spk)
    cent = np.zeros((len(speakers), e.shape[1]), np.float32)
    sim = np.full(len(e), np.nan, np.float32)
    for k, s in enumerate(speakers):
        idx = np.flatnonzero(spk == s)
        x = e[idx]
        keep = np.ones(len(idx), bool)
        if len(idx) >= 5:
            rank = x @ x.sum(0)
            keep = rank >= np.quantile(rank, trim)
        total = x[keep].sum(0)
        cent[k] = total / max(np.linalg.norm(total), 1e-8)
        loo = total[None] - x * keep[:, None]
        norm = np.linalg.norm(loo, axis=1)
        sim[idx] = np.where(norm > 1e-6, (x * loo).sum(1) / norm.clip(1e-6), np.nan)
    other = e.astype(np.float32) @ cent.T
    other[np.arange(len(e)), np.searchsorted(speakers, spk)] = -np.inf
    next_sim = other.max(1) if len(speakers) > 1 else np.full(len(e), np.nan, np.float32)
    return sim, next_sim, speakers, cent


def _r(x, nd: int = 4):
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


# ------------------------------------------------------------------------------------------------------ scorers


def _torch_whisper_features(fe, device: str):
    """faster-whisper's ``FeatureExtractor`` with ``__call__`` computed in torch on ``device`` (same mel filters,
    reflect-padded STFT, dropped last frame and log scaling)."""
    fb = torch.from_numpy(fe.mel_filters).to(device)
    window = torch.hann_window(fe.n_fft, device=device)

    class Features(type(fe)):
        def __call__(self, waveform: np.ndarray, padding: int = 160, chunk_length=None) -> np.ndarray:
            if chunk_length is not None:
                self.n_samples = chunk_length * self.sampling_rate
                self.nb_max_frames = self.n_samples // self.hop_length
            x = torch.nn.functional.pad(torch.as_tensor(waveform, dtype=torch.float32, device=device), (0, padding))
            spec = torch.stft(x, self.n_fft, self.hop_length, window=window, center=True, pad_mode="reflect",
                              return_complex=True)[..., :-1].abs().square()
            log = (fb @ spec).clamp_min(1e-10).log10()
            return ((torch.maximum(log, log.max() - 8.0) + 4.0) / 4.0).cpu().numpy()

    out = Features.__new__(Features)
    out.__dict__.update(fe.__dict__)
    return out


class AsrScorer:
    """Whisper transcripts -> CER / WER against the reference. Clips up to 30 s are decoded ``batch_size`` at a time
    through ``BatchedInferencePipeline`` (each clip is one window, via ``clip_timestamps``); longer clips are
    VAD-chunked."""

    name, key = "asr", "cer"

    def __init__(self, model: str, device: str, beam_size: int, batch_size: int):
        import faster_whisper

        dev, _, idx = device.partition(":")
        self.model = faster_whisper.WhisperModel(model, device=dev, device_index=int(idx or 0),
                                                 compute_type="float16" if dev == "cuda" else "int8")
        self.batched = (faster_whisper.BatchedInferencePipeline(self.model)
                        if hasattr(faster_whisper, "BatchedInferencePipeline") else None)
        self.opts = dict(language="tr", beam_size=beam_size, without_timestamps=True)
        self.batch_size = batch_size
        if dev == "cuda":  # the numpy log-mel took ~0.3 s per 10 s clip on a busy CPU, 2 ms in torch
            self.model.feature_extractor = _torch_whisper_features(self.model.feature_extractor, device)

    def _clips(self, wavs: list[np.ndarray]) -> list[str]:
        """One batched decode of clips <= 30 s."""
        bounds = np.cumsum([0] + [len(w) for w in wavs])
        clips = [{"start": a / SR16, "end": b / SR16} for a, b in zip(bounds[:-1], bounds[1:])]
        # cap the output length (a hallucination loop would otherwise run the whole batch to 448 tokens)
        max_new = min(440, 32 + int(12 * max(np.diff(bounds)) / SR16))
        segs, _ = self.batched.transcribe(np.concatenate(wavs), clip_timestamps=clips, batch_size=len(wavs),
                                          max_new_tokens=max_new, **self.opts)
        out, starts = [""] * len(wavs), bounds[:-1] / SR16
        for s in segs:
            k = int(np.searchsorted(starts, s.start + 1e-3, side="right")) - 1
            out[k] = f"{out[k]} {s.text.strip()}".strip()
        return out

    def transcribe(self, wavs: list[np.ndarray]) -> list[str]:
        if self.batched is None:  # faster-whisper < 1.1: one clip at a time
            opts = dict(vad_filter=False, condition_on_previous_text=False, **self.opts)
            return [" ".join(s.text.strip() for s in self.model.transcribe(w, **opts)[0]) for w in wavs]
        out = [""] * len(wavs)
        short = [k for k, w in enumerate(wavs) if len(w) <= 30 * SR16]
        for k in sorted(set(range(len(wavs))) - set(short)):
            segs, _ = self.batched.transcribe(wavs[k], batch_size=self.batch_size, **self.opts)
            out[k] = " ".join(s.text.strip() for s in segs)
        s = 0
        while s < len(short):
            group = short[s: s + self.batch_size]
            try:
                texts = self._clips([wavs[k] for k in group])
            except RuntimeError as e:  # e.g. a shared GPU: continue with smaller decode batches
                # CTranslate2 reports some allocation failures as "parallel_for failed: ... invalid device ordinal"
                if not any(m in str(e) for m in ("out of memory", "parallel_for failed")) or len(group) == 1:
                    raise
                self.batch_size = len(group) // 2
                print(f"asr: CUDA out of memory, now decoding {self.batch_size} clips at a time")
                continue
            for k, text in zip(group, texts):
                out[k] = text
            s += len(group)
        return out

    def __call__(self, batch: list[tuple], index: list[dict]) -> list[tuple[int, dict]]:
        hyps = self.transcribe([w16.numpy() for _, _, w16 in batch])
        out = []
        for (i, _, _), hyp in zip(batch, hyps):
            cer, wer = error_rates(index[i]["norm_text"], hyp)
            out.append((i, {"hyp": _plain(hyp), "cer": _r(cer), "wer": _r(wer)}))
        return out


class DnsMos:
    """DNSMOS P.835 (``sig_bak_ovr.onnx``) and P.808 (``model_v8.onnx``), batched over all 9.01 s windows.
    Matches Microsoft's ``dnsmos_local.py`` (non-personalised polynomial fit, P.808 on librosa log-mels)."""

    name, key = "mos", "mos_ovrl"
    LEN = int(9.01 * SR16)
    POLY = {"sig": (-0.08397278, 1.22083953, 0.0052439), "bak": (-0.13166888, 1.60915514, -0.39604546),
            "ovrl": (-0.06766283, 1.11546468, 0.04602535)}

    def __init__(self, device: str, chunk: int = 16):
        import warnings

        import librosa
        import onnx2torch
        import speechmos

        models = Path(speechmos.__file__).parent / "dnsmos_models"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.primary = onnx2torch.convert(str(models / "sig_bak_ovr.onnx")).to(device).eval()
            self.p808 = onnx2torch.convert(str(models / "model_v8.onnx")).to(device).eval()
        self.fb = torch.from_numpy(librosa.filters.mel(sr=SR16, n_fft=321, n_mels=120)).to(device)
        self.window = torch.hann_window(321, device=device)
        self.device, self.chunk = device, chunk

    @classmethod
    def windows(cls, w: np.ndarray) -> np.ndarray:
        """The reference windowing: tile clips shorter than 9.01 s, then 9.01 s windows every second (including its
        float rounding, which drops the windows whose end ``int((h + 9.01) * 16000)`` rounds down)."""
        if not len(w):
            w = np.zeros(cls.LEN, np.float32)
        while len(w) < cls.LEN:
            w = np.concatenate([w, w])
        hops = int(np.floor(len(w) / SR16) - 9.01) + 1
        segs = [w[int(h * SR16): int((h + 9.01) * SR16)] for h in range(hops)]
        return np.stack([s for s in segs if len(s) == cls.LEN])

    def _p808_features(self, x: torch.Tensor) -> torch.Tensor:
        """``librosa.power_to_db(melspectrogram(x, n_fft=321, hop=160, n_mels=120), ref=np.max)``, scaled."""
        spec = torch.stft(x, 321, 160, window=self.window, center=True, pad_mode="constant",
                          return_complex=True).abs().square()
        db = 10 * torch.log10((self.fb @ spec).clamp_min(1e-10))
        db = (db - db.amax((1, 2), keepdim=True)).clamp_min(-80.0)
        return ((db + 40) / 40).transpose(1, 2)

    @torch.no_grad()
    def score(self, wavs: list[np.ndarray]) -> np.ndarray:
        """``[N, 4]``: sig, bak, ovrl, p808 per clip (16 kHz, ``[-1, 1]``)."""
        wins = [self.windows(w) for w in wavs]
        owner = torch.from_numpy(np.repeat(np.arange(len(wavs)), [len(w) for w in wins])).to(self.device)
        x = torch.from_numpy(np.concatenate(wins)).float()
        raw = []
        for s in range(0, len(x), self.chunk):
            xc = x[s: s + self.chunk].to(self.device)
            raw.append(torch.cat([self.primary(xc), self.p808(self._p808_features(xc[:, :-160]))], 1).float())
        raw = torch.cat(raw)
        mos = torch.stack([a * raw[:, k] ** 2 + b * raw[:, k] + c for k, (a, b, c) in enumerate(self.POLY.values())]
                          + [raw[:, 3]], 1)
        total = torch.zeros(len(wavs), 4, device=self.device).index_add_(0, owner, mos)
        return (total / torch.bincount(owner, minlength=len(wavs))[:, None]).cpu().numpy()

    def __call__(self, batch: list[tuple], index: list[dict]) -> list[tuple[int, dict]]:
        mos = self.score([w16.numpy().clip(-1, 1) for _, _, w16 in batch])
        return [(i, {f"mos_{k}": _r(v, 3) for k, v in zip(("sig", "bak", "ovrl", "p808"), m)})
                for (i, _, _), m in zip(batch, mos)]


class Bandwidth:
    name, key = "bandwidth", "bw_hz"

    def __init__(self, device: str, db: float):
        self.device, self.db = device, db

    def __call__(self, batch: list[tuple], index: list[dict]) -> list[tuple[int, dict]]:
        bw = bandwidth_hz([w24.to(self.device) for _, w24, _ in batch], self.db)
        return [(i, {"bw_hz": int(round(b))}) for (i, _, _), b in zip(batch, bw)]


class SpeakerEmbedder:
    """ECAPA-TDNN embeddings into ``spk_emb.npy`` (zero row = not computed yet); ``finish`` turns them into
    ``spk_sim`` / ``spk_next`` for every embedded utterance, since the centroids change as more are added."""

    name = "spk"

    def __init__(self, source: str, device: str, path: Path, n: int):
        from speechbrain.inference.speaker import EncoderClassifier

        dev = "cuda:0" if device == "cuda" else device
        save = os.path.join(torch.hub.get_dir(), "speechbrain", source.replace("/", "--"))
        self.model = EncoderClassifier.from_hparams(source=source, savedir=save, run_opts={"device": dev})
        self.device = dev
        if path.exists():
            self.emb = np.load(path, mmap_mode="r+")
            if self.emb.shape[0] != n:
                raise ValueError(f"{path} has {self.emb.shape[0]} rows for {n} utterances; delete it to recompute")
        else:
            dim = self._embed([0.01 * torch.randn(SR16)]).shape[1]
            self.emb = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, dim))

    def done(self) -> np.ndarray:
        return np.abs(self.emb).sum(1) > 0

    @torch.no_grad()
    def _embed(self, wavs: list[torch.Tensor]) -> np.ndarray:
        x = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).to(self.device)
        lens = torch.tensor([len(w) / x.shape[1] for w in wavs], device=self.device)
        return self.model.encode_batch(x, lens)[:, 0].float().cpu().numpy()

    def __call__(self, batch: list[tuple], index: list[dict]) -> list[tuple[int, dict]]:
        ids = [i for i, _, _ in batch]
        self.emb[ids] = self._embed([w16 for _, _, w16 in batch])
        self.emb.flush()
        return []

    def finish(self, index: list[dict]) -> list[tuple[int, dict]]:
        ids = np.flatnonzero(self.done())
        sim, nxt, _, _ = speaker_purity(np.asarray(self.emb[ids]), np.array([index[i]["spk_id"] for i in ids]))
        return [(int(i), {"spk_sim": _r(s), "spk_next": _r(t)}) for i, s, t in zip(ids, sim, nxt)]


def _one_thread(_) -> None:
    torch.set_num_threads(1)


class _Clips(Dataset):
    """``(index line, 24 kHz waveform, 16 kHz waveform)`` from ``audio.bin``."""

    def __init__(self, root: Path, index: list[dict]):
        self.path, self.index, self.audio = root / "audio.bin", index, None
        self.resample = torchaudio.transforms.Resample(SAMPLE_RATE, SR16)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        if self.audio is None:
            self.audio = np.memmap(self.path, dtype=np.float16, mode="r")
        e = self.index[i]
        w24 = torch.from_numpy(np.array(self.audio[e["audio_offset"]: e["audio_offset"] + e["audio_samples"]],
                                         dtype=np.float32))
        return i, w24, self.resample(w24)


# --------------------------------------------------------------------------------------------------- scoring run


def read_index(root: Path) -> list[dict]:
    with open(root / "index.jsonl") as f:
        return [json.loads(line) for line in f]


def write_scores(root: Path, scores: dict[int, dict]) -> None:
    """Rewrite ``scores.jsonl`` atomically with one line per utterance."""
    tmp = root / "scores.jsonl.tmp"
    with open(tmp, "w") as f:
        for i in sorted(scores):
            f.write(json.dumps({"i": i, **scores[i]}, ensure_ascii=False) + "\n")
    os.replace(tmp, root / "scores.jsonl")


def score(root: Path, names: list[str], args) -> None:
    index = read_index(root)
    ids = range(min(args.limit, len(index)) if args.limit else len(index))
    scores = load_scores(root)
    write_scores(root, scores)  # drops a line cut short by an interrupted run before appending
    audio_names = [n for n in names if n != "rate"]
    if audio_names and not (root / "audio.bin").exists():
        raise FileNotFoundError(f"{root / 'audio.bin'} missing: run `drifting-tts prepare --save-audio`")
    with open(root / "scores.jsonl", "a") as out:
        def emit(records):
            for i, rec in records:
                out.write(json.dumps({"i": i, **rec}, ensure_ascii=False) + "\n")
            out.flush()

        if "rate" in names:
            emit((i, {"cps": _r(speaking_rate(index[i]), 2)}) for i in ids if "cps" not in scores.get(i, {}))
        scorers = []
        for name in audio_names:
            if name == "asr":
                scorers.append(AsrScorer(args.asr_model, args.device, args.beam_size, args.asr_batch_size))
            elif name == "mos":
                scorers.append(DnsMos(args.device))
            elif name == "bandwidth":
                scorers.append(Bandwidth(args.device, args.bw_db))
            elif name == "spk":
                scorers.append(SpeakerEmbedder(args.spk_model, args.device, root / "spk_emb.npy", len(index)))
        pending = {}
        for s in scorers:
            done = s.done() if isinstance(s, SpeakerEmbedder) else None
            pending[s.name] = {i for i in ids if (not done[i] if done is not None else s.key not in scores.get(i, {}))}
            print(f"{s.name}: {len(pending[s.name])} of {len(ids)} utterances to score")
        todo = sorted(set().union(*pending.values()), key=lambda i: index[i]["audio_samples"])
        batches = [todo[k: k + args.batch_size] for k in range(0, len(todo), args.batch_size)]
        loader = DataLoader(_Clips(root, index), batch_sampler=batches, num_workers=args.workers,
                            collate_fn=list, prefetch_factor=4 if args.workers else None, worker_init_fn=_one_thread)
        seconds, audio_s = defaultdict(float), defaultdict(float)
        for batch in tqdm(loader, total=len(batches), desc="score"):
            for s in scorers:
                sub = [b for b in batch if b[0] in pending[s.name]]
                if sub:
                    t = time.perf_counter()
                    emit(s(sub, index))
                    if "cuda" in args.device:
                        torch.cuda.synchronize()
                    seconds[s.name] += time.perf_counter() - t
                    audio_s[s.name] += sum(len(w24) for _, w24, _ in sub) / SAMPLE_RATE
        for s in scorers:
            if isinstance(s, SpeakerEmbedder):
                emit(s.finish(index))
            if seconds[s.name]:
                print(f"{s.name}: {audio_s[s.name] / 3600:.2f} h of audio in {seconds[s.name]:.0f} s "
                      f"({audio_s[s.name] / seconds[s.name]:.0f}x real time)")
    write_scores(root, load_scores(root))
    print(f"wrote {root / 'scores.jsonl'}")


# -------------------------------------------------------------------------------------------------------- summary


def _percentile_table(rows: dict[str, np.ndarray]) -> str:
    qs = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    lines = [f"{'score':<10}{'n':>7}{'mean':>9}" + "".join(f"{'p' + str(q):>9}" for q in qs)]
    for name, v in rows.items():
        v = v[np.isfinite(v)]
        if len(v):
            fmt = ">9.0f" if np.abs(v).max() >= 1000 else ">9.3f"
            lines.append(f"{name:<10}{len(v):>7}" + "".join(f"{x:{fmt}}" for x in [v.mean(), *np.percentile(v, qs)]))
    return "\n".join(lines)


def summarize(root: Path, filters: dict, min_quality: float, max_frames: int, tsv: str | None) -> None:
    index = read_index(root)
    scores = load_scores(root)
    if not scores:
        raise FileNotFoundError(f"no scores in {root / 'scores.jsonl'}: run `drifting-tts score` first")
    hours = np.array([e["frames"] for e in index]) / FRAME_RATE / 3600
    col = {f: np.array([np.nan if scores.get(i, {}).get(f) is None else scores[i][f] for i in range(len(index))])
           for f in FIELDS}
    print(f"{len(index)} utterances, {hours.sum():.1f} h; scored: "
          + ", ".join(f"{f} {np.isfinite(v).sum()}" for f, v in col.items() if f in ("cer", "cps", "mos_ovrl",
                                                                                         "bw_hz", "spk_sim")))
    print(_percentile_table(col))

    lo, hi = np.nanpercentile(col["cps"], [1, 99]) if np.isfinite(col["cps"]).any() else (np.nan, np.nan)
    flags = [("cer > 0.1 (transcript or audio mismatch)", col["cer"] > 0.1),
             ("cer > 0.3", col["cer"] > 0.3),
             (f"cps outside its 1-99th percentile [{lo:.1f}, {hi:.1f}]", (col["cps"] < lo) | (col["cps"] > hi)),
             ("mos_ovrl < 2.5 (noisy)", col["mos_ovrl"] < 2.5),
             ("bw_hz < 10000 (band-limited source, e.g. 16 kHz)", col["bw_hz"] < 10000),
             ("spk_sim < 0.5 (likely mislabelled speaker)", col["spk_sim"] < 0.5),
             ("spk_next > spk_sim (closer to another speaker)", col["spk_next"] > col["spk_sim"])]
    print("\noutliers:")
    for name, m in flags:
        print(f"  {name:<52}{m.sum():>7} utts {hours[m].sum():>8.2f} h")
    worst = [i for i in np.argsort(-np.nan_to_num(col["cer"], nan=-1))[:5] if np.isfinite(col["cer"][i])]
    if worst:
        print("highest cer (reference / whisper):")
    for i in worst:
        print(f"  #{i} cer {col['cer'][i]:.3f}  {index[i]['norm_text'][:100]}\n{'':>{len(str(i)) + 15}}"
              f"{scores[i].get('hyp', '')[:100]}")

    rules = parse_filters(filters)
    base = np.array([(e["quality_score"] or 0) >= min_quality and e["frames"] <= max_frames
                     and 2 * len(e["norm_text"]) + 1 <= e["frames"] for e in index])
    ok = np.array([passes(scores.get(i), rules) for i in range(len(index))])
    train = np.array([e["split"] == "train" for e in index])
    print(f"\nfilters: min_quality {min_quality:g}, max_frames {max_frames if max_frames < 10**9 else 'none'}, "
          + (", ".join(f"{f} in [{a:g}, {b:g}]" for f, a, b in rules) or "no score filters"))
    for f, a, b in rules:
        bad = base & train & ~np.array([passes(scores.get(i), [(f, a, b)]) for i in range(len(index))])
        print(f"  fails {f} in [{a:g}, {b:g}]: {bad.sum()} train utts, {hours[bad].sum():.2f} h")
    keep = base & train & ok
    print(f"  train: {keep.sum()}/{(base & train).sum()} utterances, {hours[keep].sum():.1f}/"
          f"{hours[base & train].sum():.1f} h survive")
    for split in sorted({e["split"] for e in index} - {"train"}):
        held = np.array([e["split"] == split for e in index])
        print(f"  {split}: {(held & ~ok).sum()}/{held.sum()} would fail (not filtered unless "
              "data.filters.apply_to_val)")

    spk = np.array([e["spk_id"] for e in index])
    names = {e["spk_id"]: e["speaker"] for e in index}
    table = []
    for s in np.unique(spk):
        m = (spk == s) & train
        med = {f: np.nanmedian(col[f][m]) if np.isfinite(col[f][m]).any() else np.nan
               for f in ("cer", "mos_ovrl", "bw_hz", "spk_sim")}
        table.append(dict(spk_id=int(s), speaker=names[s], utts=int(m.sum()), kept=int((m & keep).sum()),
                          minutes=60 * hours[m].sum(), kept_minutes=60 * hours[m & keep].sum(), **med))
    kept_min = np.array([r["kept_minutes"] for r in table])
    print(f"\nspeakers: {len(table)}; with kept utterances {(kept_min > 0).sum()}; >= 10 kept minutes "
          f"{(kept_min >= 10).sum()} ({kept_min[kept_min >= 10].sum() / 60:.1f} h); kept minutes per speaker "
          f"p10/p50/p90 {np.percentile(kept_min, 10):.1f}/{np.percentile(kept_min, 50):.1f}/"
          f"{np.percentile(kept_min, 90):.1f}")
    head = f"{'spk_id':>6} {'speaker':<28}{'utts':>6}{'kept':>6}{'min':>7}{'kept_min':>9}" \
           f"{'cer':>7}{'mos':>6}{'bw_khz':>7}{'spk_sim':>8}"
    fmt = lambda r: (f"{r['spk_id']:>6} {str(r['speaker'])[:27]:<28}{r['utts']:>6}{r['kept']:>6}{r['minutes']:>7.1f}"
                     f"{r['kept_minutes']:>9.1f}{r['cer']:>7.3f}{r['mos_ovrl']:>6.2f}{r['bw_hz'] / 1000:>7.1f}"
                     f"{r['spk_sim']:>8.3f}")
    print("most kept minutes:\n" + head)
    for r in sorted(table, key=lambda r: -r["kept_minutes"])[:15]:
        print(fmt(r))
    print("most rejected utterances:\n" + head)
    for r in sorted(table, key=lambda r: r["kept"] - r["utts"])[:10]:
        print(fmt(r))
    if tsv:
        with open(tsv, "w") as f:
            f.write("\t".join(table[0]) + "\n")
            for r in table:
                f.write("\t".join(str(round(v, 4) if isinstance(v, float) else v) for v in r.values()) + "\n")
        print(f"wrote {tsv}")

    if (root / "spk_emb.npy").exists():
        emb = np.load(root / "spk_emb.npy", mmap_mode="r")
        ids = np.flatnonzero(np.abs(emb).sum(1) > 0)
        ids = ids[np.bincount(spk[ids], minlength=spk.max() + 1)[spk[ids]] >= 5]  # reliable centroids only
        if len(np.unique(spk[ids])) > 1:
            _, _, speakers, cent = speaker_purity(np.asarray(emb[ids]), spk[ids])
            a, b = np.triu_indices(len(speakers), 1)
            cc = (cent[a] * cent[b]).sum(1)
            print(f"\nspeaker ids (>= 5 embedded utterances) whose centroids have cosine >= 0.8, likely one person "
                  f"under two ids: {(cc >= 0.8).sum()} of {len(cc)} pairs; closest:")
            for k in np.argsort(-cc)[:10]:
                sa, sb = speakers[a[k]], speakers[b[k]]
                print(f"  {cc[k]:.3f}  {sa} {names[sa]}  ~  {sb} {names[sb]}")


def run(args) -> None:
    root = Path(args.data)
    if args.summary:
        overrides = [f"data.filters.{x}" for x in args.filter]
        cfg = load_config(args.config, overrides) if args.config else apply_overrides(Config(), overrides)
        d = cfg.get("data", {})
        summarize(root, d.get("filters") or {}, d.get("min_quality", 0.0), d.get("max_frames", 10**9),
                  args.speakers_tsv)
        return
    names = [n.strip() for n in args.scorers.split(",") if n.strip()]
    unknown = set(names) - set(SCORERS)
    if unknown:
        raise ValueError(f"unknown scorers {sorted(unknown)}, expected a subset of {SCORERS}")
    score(root, names, args)

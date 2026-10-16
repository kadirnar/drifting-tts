"""Speech restoration with Sidon (``restore-audio``, #71): degraded speech denoised and extended to full band.

Sidon ([sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1), MIT; arXiv 2509.17052) predicts
clean w2v-BERT 2.0 features from degraded speech and decodes them to 48 kHz. It was trained on FLEURS-R, 102 languages
with Turkish among them. The recipe is the one of its demo Space (``sarulab-speech/sidon_demo_beta``):
1. peak 0.9 and a 50 Hz high-pass;
2. 16 kHz, with 1.5 s of padding at the end (here rounded up to whole seconds: the TorchScript networks specialise
   on each new input length, so a few dozen lengths keep them fast, ~100x real time on an RTX 5090);
3. w2v-BERT 2.0 filterbanks per 96 s chunk, each padded by 160 samples, with the last feature frame of the previous
   chunk carried over;
4. the TorchScript feature predictor and decoder, which return 48 kHz audio cut to the input's length.

``restore-audio`` rewrites parquet shards (an ``audio`` column of encoded bytes, as ``prepare`` reads them) with every
clip restored, the other columns unchanged, so ``prepare --parquet-glob`` runs on the output as on the original.
"""

from __future__ import annotations

import argparse
import glob
import io
import time
from pathlib import Path

import numpy as np
import torch

REPO = "sarulab-speech/sidon-v0.1"
FEATURES = "facebook/w2v-bert-2.0"
RATE = 48_000


class Sidon:
    """``restore(wav, sr)``: mono float audio at any rate -> restored float32 audio at :data:`RATE`."""

    def __init__(self, device: str = "cuda", fe=None, decoder=None, preprocessor=None):
        kind = "cuda" if str(device).startswith("cuda") else "cpu"
        if fe is None or decoder is None:
            from huggingface_hub import hf_hub_download

            fe = torch.jit.load(hf_hub_download(REPO, f"feature_extractor_{kind}.pt"), map_location=device)
            decoder = torch.jit.load(hf_hub_download(REPO, f"decoder_{kind}.pt"), map_location=device)
        if preprocessor is None:
            import transformers

            preprocessor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(FEATURES)
        self.fe, self.decoder, self.preprocessor, self.device = fe, decoder, preprocessor, device

    @torch.inference_mode()
    def restore(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import torchaudio

        x = torch.as_tensor(np.asarray(wav, dtype=np.float32)).view(1, -1)
        target = int(RATE / sr * x.shape[1])
        x = 0.9 * x / x.abs().max().clamp_min(1e-8)
        x = torchaudio.functional.highpass_biquad(x, sr, 50)
        x = torchaudio.functional.resample(x, sr, 16_000)
        x = torch.nn.functional.pad(x, (0, 24_000 + (-(x.shape[1] + 24_000)) % 16_000))
        out, cache = [], None
        for chunk in x.view(-1).split(16_000 * 96):
            inputs = self.preprocessor(torch.nn.functional.pad(chunk, (160, 160)), sampling_rate=16_000,
                                       return_tensors="pt")
            feature = self.fe(inputs["input_features"].to(self.device))["last_hidden_state"]
            if cache is not None:
                feature = torch.cat([cache, feature], dim=1)
            out.append(self.decoder(feature.transpose(1, 2)).view(-1)[:-960])
            cache = feature[:, -1:]
        return torch.cat(out).float().cpu().numpy()[:target]


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--parquet-glob", required=True, help="input shards with an 'audio' column of encoded bytes")
    p.add_argument("--out", required=True, help="output directory: one restored shard per input, same file name")
    p.add_argument("--sample-rate", type=int, default=24_000,
                   help="rate of the written FLAC (Sidon's 48 kHz resampled; prepare works at 24 kHz)")
    p.add_argument("--limit", type=int, default=0, help="first N rows per shard only (0: all; a pilot)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def encode_flac(wav: np.ndarray, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(wav, -1, 1), sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def restore_shard(src: str, dst: Path, sidon: Sidon, sample_rate: int, limit: int = 0) -> int:
    """One parquet shard -> ``dst`` with every row's audio restored (FLAC at ``sample_rate``); returns the rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torchaudio

    from .audio import decode_audio

    table = pq.read_table(src)
    if limit:
        table = table.slice(0, limit)
    rows = table.column("audio").to_pylist()
    audio = []
    for row in rows:
        wav, sr = decode_audio(row["bytes"])
        out = torch.from_numpy(sidon.restore(wav, sr))
        if sample_rate != RATE:
            out = torchaudio.functional.resample(out, RATE, sample_rate)
        path = row.get("path") or ""
        audio.append({"bytes": encode_flac(out.numpy(), sample_rate), "path": str(Path(path).with_suffix(".flac"))})
    i = table.schema.get_field_index("audio")
    table = table.set_column(i, pa.field("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                             pa.array(audio, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(dst)
    return len(rows)


def run(args) -> None:
    files = sorted(glob.glob(args.parquet_glob))
    if not files:
        raise SystemExit(f"no files match {args.parquet_glob}")
    out = Path(args.out)
    sidon = Sidon(args.device)
    for f in files:
        dst = out / Path(f).name
        if dst.exists():  # resumable: a shard is written whole or not at all
            print(f"{dst.name}: done before", flush=True)
            continue
        t0 = time.time()
        n = restore_shard(f, dst, sidon, args.sample_rate, args.limit)
        print(f"{dst.name}: {n} clips restored in {time.time() - t0:.0f} s", flush=True)

"""Mimi encode -> decode (the codec of a Pocket TTS config: 32-d continuous latents at 12.5 Hz) of the recordings
written by ``dump_recordings.py``. Runs in the pocket-tts venv on the CPU. The output is shifted by its measured
lag (cross-correlation; 0 for this codec) and cut to the input length.

    python scripts/pocket_tts_gate/mimi_roundtrip.py --config pocket_tts_tr.yaml --root runs/pg_mimi
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states


def lag(ref: np.ndarray, out: np.ndarray, max_lag: int) -> int:
    n = min(len(ref), len(out))
    f = np.fft.rfft(ref[:n], 2 * n).conj() * np.fft.rfft(out[:n], 2 * n)
    cc = np.fft.irfft(f, 2 * n)
    k = int(np.concatenate([cc[-max_lag:], cc[: max_lag + 1]]).argmax())
    return k - max_lag


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root", default="runs/pg_mimi")
    p.add_argument("--sets", nargs="+", default=["studio", "base"])
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model = TTSModel.load_model(config=args.config)
    torch.set_num_threads(args.threads)
    mimi = model.mimi.eval()
    sr = mimi.sample_rate
    steps = int(mimi.encoder_frame_rate / mimi.frame_rate)
    root = Path(args.root)
    summary = {}
    for name in args.sets:
        (root / "mimi" / name).mkdir(parents=True, exist_ok=True)
        lags, t_enc, t_dec, secs, lat = [], 0.0, 0.0, 0.0, []
        for line in open(root / f"{name}.jsonl"):
            e = json.loads(line)
            x, xsr = sf.read(root / "recording" / name / f"{e['ds_index']:05d}.wav", dtype="float32")
            assert xsr == sr
            with torch.no_grad():
                t = time.perf_counter()
                z = mimi.encode_to_latent(torch.from_numpy(x)[None, None])  # [1, T, 32]
                t_enc += time.perf_counter() - t
                t = time.perf_counter()
                state = init_states(mimi, batch_size=1, sequence_length=z.shape[1] * steps + 8)
                y = mimi.decode_from_latent(z, state)[0, 0].numpy()
                t_dec += time.perf_counter() - t
            k = lag(x.astype(np.float64), y.astype(np.float64), sr // 10)
            lags.append(k)
            y = y[max(k, 0):]
            y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
            sf.write(root / "mimi" / name / f"{e['ds_index']:05d}.wav", y, sr, subtype="FLOAT")
            secs += len(x) / sr
            lat.append(z[0])
        z = torch.cat(lat)
        summary[name] = {"utterances": len(lags), "seconds": secs, "encode_rtf_cpu": t_enc / secs,
                         "decode_rtf_cpu": t_dec / secs, "lag_samples": sorted(set(lags)),
                         "frames_per_second": z.shape[0] / secs, "dim": z.shape[1],
                         "latent_std_range": [z.std(0).min().item(), z.std(0).max().item()]}
        print(name, json.dumps(summary[name]), flush=True)
    (root / "mimi_summary.json").write_text(json.dumps({"threads": args.threads, **summary}, indent=1))


if __name__ == "__main__":
    main()

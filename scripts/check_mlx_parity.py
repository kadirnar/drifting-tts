"""Check the MLX vocoders against PyTorch on real mels: SNR of whole-sentence and streamed audio.

    python scripts/check_mlx_parity.py --vocoder bigvgan-base-ft --vocoder vocos-ft            # ports, float32
    python scripts/check_mlx_parity.py --mlx /path/to/mlx --vocoder vocos-ft --out parity.json  # converted files

The mels come from the released acoustic model (PyTorch, CPU) on public sentences. Without ``--mlx`` each PyTorch
vocoder is converted in memory (the port in float32); with it, the converted files are loaded (their storage
precision included). Streaming uses the MLX synthesizer's windows and context. Needs torch and mlx.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEXTS = ["Merhaba, nasılsınız?",
         "İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı.",
         "Yapay zekâ modelleri her geçen gün daha hızlı ve daha verimli hâle geliyor."]


def snr(ref: np.ndarray, out: np.ndarray) -> float:
    """``10 log10(|ref|^2 / |ref - out|^2)`` in dB (inf when identical)."""
    err = float(np.sum((ref.astype(np.float64) - out) ** 2))
    return float("inf") if err == 0 else float(10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / err))


def mlx_vocoder(name: str, torch_vocoder, mlx_dir: Path | None):
    """The MLX vocoder: the converted file in ``mlx_dir``, else ``torch_vocoder`` converted in memory."""
    import mlx.core as mx

    from drifting_tts.mlx.convert import convert_vocoder
    from drifting_tts.mlx.vocoder import KINDS, load_vocoder, vocoder_path

    if mlx_dir is not None:
        config = json.loads((mlx_dir / "config.json").read_text()) if (mlx_dir / "config.json").is_file() else {}
        return load_vocoder(vocoder_path(name, mlx_dir), config.get("vocoder"))
    kind, hparams, weights = convert_vocoder(torch_vocoder)
    model = KINDS[kind](hparams)
    model.load_weights([(k, mx.array(v)) for k, v in weights.items()])
    return model.prepare_for_inference()


def compare(torch_vocoder, mlx_model, mel, chunk_frames: int = 64, first_chunk_frames: int = 24) -> dict:
    """``mel``: unnormalised log-mel ``[1, 100, T]`` (torch). SNRs of the MLX whole and streamed waveforms against
    PyTorch's whole-sentence waveform, and of MLX streamed against MLX whole."""
    import mlx.core as mx
    import torch

    from drifting_tts.fast import stream_vocoder
    from drifting_tts.mlx.synthesize import chunk_windows, decode_window

    with torch.no_grad():
        ref = torch_vocoder(mel)[0].numpy()
        torch_streamed = torch.cat(list(stream_vocoder(torch_vocoder, mel, context=torch_vocoder.context))).numpy()
    x = mx.array(mel.numpy().transpose(0, 2, 1))
    whole = np.array(mx.clip(mlx_model(x.transpose(0, 2, 1))[0], -1, 1))
    hop, context = mlx_model.hop_length, mlx_model.context_frames
    streamed = np.concatenate([np.array(decode_window(mlx_model, x, a, b, context, hop))
                               for a, b in chunk_windows(x.shape[1], chunk_frames, first_chunk_frames)])
    assert whole.shape == streamed.shape == ref.shape, (whole.shape, streamed.shape, ref.shape)
    return {"frames": int(mel.shape[-1]), "mlx_vs_torch_db": snr(ref, whole),
            "mlx_streamed_vs_torch_db": snr(ref, streamed), "mlx_streamed_vs_mlx_db": snr(whole, streamed),
            "torch_streamed_vs_torch_db": snr(ref, torch_streamed), "max_abs_diff": float(np.abs(ref - whole).max())}


def real_mels(model: str | None, texts: list[str], speaker: str = "studio", seed: int = 0) -> list:
    """Unnormalised mels of ``texts`` from the released acoustic model on the CPU (T = 0.3, CFG 2)."""
    from drifting_tts.synthesize import Synthesizer

    if model is None:
        from huggingface_hub import hf_hub_download

        model = hf_hub_download("Vyvo/drifting-tts-tr", "drifting_tts_v3.1.pt")
    synth = Synthesizer(model, "cpu", vocoder="griffin-lim")  # no vocoder weights needed for the mels
    return [m for text in texts for m in synth.mels(text, speaker, cfg_scale=2.0, temperature=0.3, seed=seed)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--vocoder", action="append", help="repeatable (default: bigvgan-base-ft and vocos-ft)")
    p.add_argument("--mlx", type=Path, help="directory of converted MLX weights (default: convert in memory)")
    p.add_argument("--model", help="PyTorch acoustic checkpoint (default: drifting_tts_v3.1.pt from the Hub)")
    p.add_argument("--text", action="append", help="repeatable (default: three public sentences)")
    p.add_argument("--max-frames", type=int, default=None, help="crop each mel (the MLX CPU backend is slow)")
    p.add_argument("--chunk-frames", type=int, default=64)
    p.add_argument("--first-chunk-frames", type=int, default=24)
    p.add_argument("--out", type=Path, help="write the results as JSON")
    args = p.parse_args(argv)

    import torch

    from drifting_tts.vocoder import load_vocoder

    torch.set_grad_enabled(False)
    mels = [m[..., : args.max_frames] for m in real_mels(args.model, args.text or TEXTS)]
    results = {}
    for name in args.vocoder or ["bigvgan-base-ft", "vocos-ft"]:
        torch_vocoder = load_vocoder(name, "cpu")
        model = mlx_vocoder(name, torch_vocoder, args.mlx)
        results[name] = [compare(torch_vocoder, model, mel, args.chunk_frames, args.first_chunk_frames)
                         for mel in mels]
        for r in results[name]:
            print(f"{name}: {r['frames']} frames, MLX vs PyTorch {r['mlx_vs_torch_db']:.1f} dB, streamed "
                  f"{r['mlx_streamed_vs_torch_db']:.1f} dB (vs MLX whole {r['mlx_streamed_vs_mlx_db']:.1f} dB; "
                  f"PyTorch's own streaming {r['torch_streamed_vs_torch_db']:.1f} dB)", flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()

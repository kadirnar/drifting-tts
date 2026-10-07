"""Latency benchmark: time to first audio (TTFA), per-stage times and real-time factor.

TTFA runs from the input text to the first sentence's waveform on the host. Synthesis streams sentence by sentence,
so this is the wait before playback can start.

    python scripts/bench_ttfa.py                       # weights from huggingface.co/Vyvo/drifting-tts-tr
    python scripts/bench_ttfa.py --cuda-kernel --model runs/tts_v3/model_ema.pt --vocoder runs/vocoder_v3/bigvgan_ft.pt
"""

import argparse
import json
import time

import numpy as np
import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer, split_sentences
from drifting_tts.text import normalize, text_to_ids
from drifting_tts.voices import DEFAULT_VOICE, voice_id

TEXTS = {
    "short sentence": "Merhaba, nasılsınız?",
    "long sentence": "İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı.",
    "4-sentence paragraph": ("Toplantı yarın saat 14:30'da, 2. katta başlayacak. "
                             "Lütfen raporlarınızı yanınızda getirin. "
                             "Prof. Dr. Ayşe Yılmaz, 250 TL'lik bağışın tamamının öğrencilere ayrılacağını söyledi. "
                             "Yapay zekâ modelleri her geçen gün daha hızlı ve daha verimli hâle geliyor."),
}


@torch.no_grad()
def run(synth: Synthesizer, text: str, seed: int, speaker: int, temperature: float, cfg: float) -> dict:
    model, dev, sync = synth.model, synth.device, torch.cuda.synchronize
    g = torch.Generator(device=dev).manual_seed(seed)
    spk = torch.tensor([speaker], device=dev)
    sync()
    t0 = time.perf_counter()
    marks, samples = {}, 0
    for k, sentence in enumerate(split_sentences(normalize(text))):
        ids = torch.tensor([text_to_ids(sentence, normalized=True)], device=dev)
        if k == 0:
            sync()
            marks["frontend"] = time.perf_counter() - t0
        mel, _ = model.synthesize(ids, torch.tensor([ids.shape[1]], device=dev), spk, cfg_scale=cfg,
                                  temperature=temperature, generator=g,
                                  length_scale=getattr(model, "duration_scales", {}).get(speaker, model.duration_scale))
        if k == 0:
            sync()
            marks["acoustic"] = time.perf_counter() - t0 - marks["frontend"]
        wav = synth.vocoder(synth.stats.denormalize(mel))[0].cpu()
        samples += wav.numel()
        if k == 0:
            marks["ttfa"] = time.perf_counter() - t0
            marks["vocoder"] = marks["ttfa"] - marks["frontend"] - marks["acoustic"]
            marks["first_audio_s"] = wav.numel() / SAMPLE_RATE
    marks["total"] = time.perf_counter() - t0
    marks["audio_s"] = samples / SAMPLE_RATE
    return marks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=None, help="TTS checkpoint (default: Vyvo/drifting-tts-tr)")
    p.add_argument("--vocoder", default=None, help="fine-tuned BigVGAN-v2 (default: Vyvo/drifting-tts-tr)")
    p.add_argument("--cuda-kernel", action="store_true", help="BigVGAN fused activation kernel")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--speaker", default=DEFAULT_VOICE, help="voice name or speaker ID")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--out", default=None, help="write the results as JSON")
    args = p.parse_args()
    if args.model is None or args.vocoder is None:
        from huggingface_hub import hf_hub_download

        args.model = args.model or hf_hub_download("Vyvo/drifting-tts-tr", "drifting_tts_v3.1.pt")
        args.vocoder = args.vocoder or hf_hub_download("Vyvo/drifting-tts-tr", "bigvgan_v2_ft.pt")
    synth = Synthesizer(args.model, "cuda", vocoder=args.vocoder, cuda_kernel=args.cuda_kernel)
    kw = dict(speaker=voice_id(args.speaker), temperature=args.temperature, cfg=args.cfg)

    cold = run(synth, TEXTS["long sentence"], 0, **kw)  # first call after loading: CUDA / cuDNN initialisation
    for _ in range(10):
        run(synth, TEXTS["4-sentence paragraph"], 1, **kw)
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda_kernel": args.cuda_kernel,
           "cold_ttfa_ms": round(1000 * cold["ttfa"], 1), "rows": {}}
    print(f"{res['gpu']}, cuda kernel: {args.cuda_kernel}, cold first call TTFA {res['cold_ttfa_ms']} ms")
    head = ("input", "TTFA p50", "p90", "acoustic", "vocoder", "1st audio", "total", "RTF")
    print(f"{head[0]:22s} " + " ".join(f"{h:>{w}s}" for h, w in zip(head[1:], (9, 7, 9, 8, 9, 8, 7))))
    for name, text in TEXTS.items():
        runs = [run(synth, text, seed, **kw) for seed in range(args.runs)]

        def get(k: str, runs: list = runs) -> np.ndarray:
            return np.array([r[k] for r in runs])

        row = {"ttfa_ms_p50": 1000 * np.median(get("ttfa")), "ttfa_ms_p90": 1000 * np.percentile(get("ttfa"), 90),
               "acoustic_ms": 1000 * np.median(get("acoustic")), "vocoder_ms": 1000 * np.median(get("vocoder")),
               "first_audio_s": get("first_audio_s").mean(), "total_ms": 1000 * np.median(get("total")),
               "audio_s": get("audio_s").mean()}
        row["rtf"] = row["total_ms"] / 1000 / row["audio_s"]
        res["rows"][name] = {k: round(float(v), 4 if k == "rtf" else 2) for k, v in row.items()}
        print(f"{name:22s} {row['ttfa_ms_p50']:7.1f}ms {row['ttfa_ms_p90']:5.1f}ms {row['acoustic_ms']:7.1f}ms "
              f"{row['vocoder_ms']:6.1f}ms {row['first_audio_s']:8.2f}s {row['total_ms']:6.1f}ms {row['rtf']:7.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

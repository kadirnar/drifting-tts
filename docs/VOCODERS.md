# Vocoders

The model writes BigVGAN-style log-mels (24 kHz, 100 bands, n_fft 1024, hop 256, fmax 12 kHz, uncentred frames,
natural log floored at 1e-5). Any vocoder trained on this mel can turn them into audio without retraining.
`Synthesizer(model, vocoder=<name>)`, `drifting-tts synthesize / benchmark --vocoder <name>` and
`scripts/bench_ttfa.py --vocoder <name>` take any name below, or a checkpoint path.

| name | vocoder | weights |
|---|---|---|
| `bigvgan-v2-ft` | BigVGAN-v2 fine-tuned on this model's mels (the release default) | `Vyvo/drifting-tts-tr`, `bigvgan_v2_ft.pt` |
| `bigvgan-v2` | NVIDIA BigVGAN-v2, 24 kHz, 100 bands, 256x | `nvidia/bigvgan_v2_24khz_100band_256x` |
| `bigvgan-v1` | NVIDIA BigVGAN (v1), 24 kHz, 100 bands | `nvidia/bigvgan_24khz_100band` |
| `bigvgan-base` | NVIDIA BigVGAN-base (v1), 24 kHz, 100 bands | `nvidia/bigvgan_base_24khz_100band` |
| `bigvgan-base-ft` | BigVGAN-base fine-tuned on this model's mels (14 M parameters) | `Vyvo/drifting-tts-tr`, `bigvgan_base_ft.pt` |
| `vocos-ft` | Vocos fine-tuned on this model's mels (13.5 M parameters) | `Vyvo/drifting-tts-tr`, `vocos_ft.pt` |
| `griffin-lim` | mel filterbank inverted by non-negative least squares, then 64 iterations of fast Griffin-Lim | none |
| `vocos` | `charactr/vocos-mel-24khz`, for models trained on Vocos's own mels | `charactr/vocos-mel-24khz` |

- **Checkpoints.** A path is recognised by its keys. `{"generator", "repo", "hparams"}` is a BigVGAN-family
  fine-tune (`drifting-tts finetune-vocoder`), built with the code of its NVIDIA `repo`. `{"vocos", "init", "mel":
  "bigvgan", "head_padding": "same"}` is a Vocos fine-tuned on BigVGAN-style mels: with a `same` ISTFT head, frame
  *i* is centred on sample *i* · 256 + 128, as in BigVGAN, and T frames give T · 256 samples.
- **The NVIDIA repos** ship the same `bigvgan.py`, and their configs use the same mel as ours (v1: `fmax: 12000`;
  v2: `fmax: null`, i.e. 12 kHz). All three load through `build_bigvgan(repo=...)`. The fused anti-aliased activation
  kernel (`cuda_kernel=True`) is the same snake-beta kernel in v1, v1-base and v2. On real mels its output is within
  62–68 dB SNR of the PyTorch path.
- **Streaming.** Each entry has its own window context (`Vocoder.context`, below). Griffin-Lim cannot stream exactly,
  so `stream()` yields each of its sentences in one piece. With `fast=True`, the fixed-size windows of every neural
  vocoder run as CUDA graphs, and a vocoder whose capture fails falls back to eager windows.

## Comparison on Freya-TR-Eval

All rows vocode the same v3.1 mels of the 495 sentences (protocol below).

- **Fine-tuning matters most.** The fine-tuned BigVGAN-v2 is the most natural vocoder: UTMOSv2 2.94 and DNSMOS
  OVRL 3.32. The same network with NVIDIA's weights is the least natural neural vocoder on these mels: UTMOSv2 2.40
  and DNSMOS OVRL 2.81.
- **`bigvgan-base-ft` is the small vocoder to use.** Fine-tuning lifts BigVGAN-base from UTMOSv2 2.79 to 2.91 and
  DNSMOS OVRL from 3.18 to 3.34, the highest OVRL of all rows. That is within 0.03 UTMOSv2 of `bigvgan-v2-ft` with
  one eighth of the parameters, half the vocoder time, first audio after 7.8 ms instead of 12.4 ms, and 16 frames of
  streaming context instead of 32.
- **`vocos-ft` is the fastest.** Its ISTFT head makes the vocoder 20× faster than BigVGAN-v2 (RTF 0.0003), and the
  first audio comes after 4.9 ms, most of it the acoustic model. It is less natural than the BigVGAN fine-tunes
  (UTMOSv2 2.63, DNSMOS OVRL 3.30), with a slightly higher WER (1.56%).
- **WER does not rank vocoders.** Every row lies between 1.20% and 1.56% WER, and the intervals overlap. Even
  Griffin-Lim reaches 1.20%: Whisper on 8 kHz band-matched audio ignores phase artefacts. UTMOSv2 and DNSMOS are what
  separate the vocoders.
- **Griffin-Lim** is intelligible, but it sounds clearly synthetic (UTMOSv2 1.77, DNSMOS P.808 3.47). It is the
  floor that needs no weights.
- **Context.** BigVGAN-v2 needs 28 frames of context for the streamed audio to equal whole-sentence vocoding, v1 needs
  24 and v1-base 16; fine-tuning does not change this. The Vocos backbone sees 3 + 8 × 3 frames, plus 2 for the
  overlap of its ISTFT, so 29 frames are exact; `tests/test_vocoder_registry.py` checks this in float64. `vocos-ft`
  passes 55 dB at 24 frames and levels off at 81–82 dB from 28 frames on.

All rows were measured on an idle GPU.

<!-- vocoders:begin -->
| vocoder | parameters | WER | CER | UTMOSv2 | DNSMOS OVRL | DNSMOS P.808 | vocoder RTF | TTFA short | TTFA long |
|---|---|---|---|---|---|---|---|---|---|
| `bigvgan-v2-ft` | 112.4 M | 1.23% [0.82, 1.68] | 0.24% [0.16, 0.33] | 2.935 | 3.324 | 3.960 | 0.0058 | 12.4 ms | 14.0 ms |
| `bigvgan-v2` | 112.4 M | 1.20% [0.81, 1.65] | 0.24% [0.16, 0.34] | 2.398 | 2.810 | 3.779 | 0.0058 | 12.4 ms | 14.0 ms |
| `bigvgan-v1` | 112.4 M | 1.36% [0.91, 1.84] | 0.26% [0.17, 0.36] | 2.756 | 3.169 | 3.962 | 0.0058 | 12.2 ms | 13.7 ms |
| `bigvgan-base-ft` | 14.0 M | 1.33% [0.90, 1.80] | 0.26% [0.17, 0.36] | 2.906 | 3.335 | 3.959 | 0.0031 | 7.8 ms | 9.4 ms |
| `bigvgan-base` | 14.0 M | 1.41% [0.95, 1.91] | 0.25% [0.17, 0.35] | 2.793 | 3.176 | 3.976 | 0.0032 | 7.8 ms | 9.4 ms |
| `vocos-ft` | 13.5 M | 1.56% [1.13, 2.05] | 0.29% [0.21, 0.38] | 2.627 | 3.302 | 3.887 | 0.0003 | 4.9 ms | 6.4 ms |
| `griffin-lim` | 0 | 1.20% [0.82, 1.63] | 0.24% [0.16, 0.34] | 1.772 | 3.127 | 3.465 | 0.0038 | 14.8 ms | 17.3 ms |

Streamed against whole-sentence audio, SNR in dB (full fp32), by context in frames on each side of a window:

| vocoder | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 40 | 48 | 64 | > 55 dB from | > 90 dB from | used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bigvgan-v2-ft` | 3.9 | 6.6 | 10.4 | 17.3 | 25.7 | 43.3 | 82.2 | 97.6 | 98.4 | 98.9 | 98.9 | 98.1 | 24 | 28 | **32** |
| `bigvgan-v2` | 6.4 | 11.2 | 16.2 | 22.8 | 34.0 | 48.4 | 72.9 | 96.0 | 95.8 | 96.6 | 96.3 | 96.1 | 24 | 28 | **32** |
| `bigvgan-v1` | 5.5 | 8.8 | 16.4 | 24.1 | 38.0 | 58.7 | 94.2 | 99.3 | 100.1 | 100.2 | 99.5 | 100.0 | 20 | 24 | **24** |
| `bigvgan-base-ft` | 5.8 | 10.3 | 21.6 | 74.0 | 92.5 | 92.4 | 92.6 | 92.6 | 92.6 | 92.5 | 92.4 | 92.4 | 12 | 16 | **16** |
| `bigvgan-base` | 6.2 | 10.2 | 21.7 | 77.2 | 93.8 | 93.5 | 93.8 | 93.5 | 93.7 | 93.6 | 93.8 | 93.9 | 12 | 16 | **16** |
| `vocos-ft` | 2.4 | 5.0 | 8.3 | 13.9 | 20.9 | 35.6 | 62.2 | 82.4 | 80.9 | 81.1 | 81.7 | 81.3 | 24 | – | **32** |

| vocoder | kind | streaming | BigVGAN CUDA kernel | CUDA graphs (`fast=True`) |
|---|---|---|---|---|
| `bigvgan-v2-ft` | bigvgan | 32 frames of context | yes | yes |
| `bigvgan-v2` | bigvgan | 32 frames of context | yes | yes |
| `bigvgan-v1` | bigvgan | 24 frames of context | yes | yes |
| `bigvgan-base-ft` | bigvgan | 16 frames of context | yes | yes |
| `bigvgan-base` | bigvgan | 16 frames of context | yes | yes |
| `vocos-ft` | vocos | 32 frames of context | – | yes |
| `griffin-lim` | griffin-lim | one piece per sentence | – | no (eager) |
<!-- vocoders:end -->

## Protocol

- **Same mels for every vocoder.** `scripts/compare_vocoders.py` generates the v3.1 mels of the 495 Freya-TR-Eval
  sentences once, as `drifting-tts benchmark` does: seed = sentence index, `studio` voice, T = 0.3, α = 2. Texts are
  split into sentences as in `Synthesizer`, and the vocoded sentences are joined with 0.15 s pauses. The rows differ
  only in the vocoder.
- **Quality.**
  - WER / CER: Whisper large-v3 (Turkish, beam 5) on audio band-matched to 8 kHz, with `benchmark`'s normaliser.
    Both are corpus-level, with 95% bootstrap intervals.
  - UTMOSv2 on the full band.
  - DNSMOS P.835 OVRL and P.808 (`drifting_tts.score.DnsMos`).
  - The vocoders run in PyTorch, without the BigVGAN CUDA kernel.
- **Speed.** `Synthesizer(fast=True, cuda_kernel=True)` on an RTX 5090 with PyTorch 2.11.
  - Vocoder RTF: vocoder time over every sentence of the set (whole sentences, batch 1), divided by their duration.
  - TTFA: `stream()`, from the text to the first audio on the host. Each value is the median of 50 runs of the
    short and the long sentence of `scripts/bench_ttfa.py`, after warm-up.
- **Streaming context.** `stream_vocoder` vocodes the first 32 frames, then windows with `context` frames on each
  side.
  - The sweep streams the 24 longest sentences with 64-frame windows (four times as many seams as the default 256)
    and compares the result with whole-sentence vocoding.
  - It runs in full fp32 (TF32 off), so the cut context is the only difference.
  - Above 55 dB SNR the pieces count as matching, and above 90 dB they are equal up to the fp32 noise floor. Each
    entry uses the smallest multiple of 8 frames above 90 dB.
  - In normal use, TF32 convolutions (PyTorch's default in cuDNN) limit the agreement of any two BigVGAN runs on
    different window lengths to about 52 dB, whatever the context.

```bash
python scripts/compare_vocoders.py --model drifting_tts_v3.1.pt --table docs/VOCODERS.md \
    --extra vocos-ft=runs/vocos_bigvgan/vocos_ft.pt bigvgan-base-ft=runs/bigvgan_base_ft/bigvgan_ft.pt
```

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
| `revox` | Minori Live — [Revox Vocoder 1.0](https://huggingface.co/minori-live/revox-vocoder-1) (PC-NSF-Vocos, 48 kHz, 4.5 M parameters) on converted mels, with F0 from the Griffin-Lim audio (`revox:<F0 source>[:dio\|harvest]`). **CC BY-NC-SA 4.0: non-commercial use only.** [Below](#revox-vocoder-10-non-commercial) | `minori-live/revox-vocoder-1`, `vocoder.onnx`, downloaded at runtime |

- **Checkpoints.** A path is recognised by its keys. `{"generator", "repo", "hparams"}` is a BigVGAN-family
  fine-tune (`drifting-tts finetune-vocoder`), built with the code of its NVIDIA `repo`. `{"vocos", "init", "mel":
  "bigvgan", "head_padding": "same"}` is a Vocos fine-tuned on BigVGAN-style mels: with a `same` ISTFT head, frame
  *i* is centred on sample *i* · 256 + 128, as in BigVGAN, and T frames give T · 256 samples.
  `{"decoder", "backend", "target_rate"}` is a fine-tuned VAE decoder (`decoder_ft.pt`, `vocoder.arch:
  vae_decoder`) for a model trained on VAE latents: see [LATENTS.md](LATENTS.md#fine-tuning-the-voxcpm2-decoder-on-generated-latents-27).
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
  OVRL 3.32. The same network with NVIDIA's weights is the least natural BigVGAN or Vocos on these mels: UTMOSv2
  2.40 and DNSMOS OVRL 2.81.
- **`bigvgan-base-ft` is the small vocoder to use.** Fine-tuning lifts BigVGAN-base from UTMOSv2 2.79 to 2.91 and
  DNSMOS OVRL from 3.18 to 3.34, the highest OVRL of all rows. That is within 0.03 UTMOSv2 of `bigvgan-v2-ft` with
  one eighth of the parameters, half the vocoder time, first audio after 7.8 ms instead of 12.4 ms, and 16 frames of
  streaming context instead of 32.
- **`vocos-ft` is the fastest.** Its ISTFT head makes the vocoder 20× faster than BigVGAN-v2 (RTF 0.0003), and the
  first audio comes after 4.9 ms, most of it the acoustic model. It is less natural than the BigVGAN fine-tunes
  (UTMOSv2 2.63, DNSMOS OVRL 3.30), with a slightly higher WER (1.56%).
- **WER does not rank vocoders.** Every row lies between 1.20% and 1.71% WER, and the intervals overlap. Even
  Griffin-Lim reaches 1.20%: Whisper on 8 kHz band-matched audio ignores phase artefacts. UTMOSv2 and DNSMOS are what
  separate the vocoders.
- **Griffin-Lim** is intelligible, but it sounds clearly synthetic (UTMOSv2 1.77, DNSMOS P.808 3.47). It is the
  floor that needs no weights.
- **`revox`** (Minori Live's Revox Vocoder 1.0, non-commercial) is the least natural neural vocoder here: UTMOSv2
  2.27 and DNSMOS OVRL 3.10. It needs frame-level F0 and cannot stream; see [below](#revox-vocoder-10-non-commercial).
- **Context.** BigVGAN-v2 needs 28 frames of context for the streamed audio to equal whole-sentence vocoding, v1 needs
  24 and v1-base 16; fine-tuning does not change this. The Vocos backbone sees 3 + 8 × 3 frames, plus 2 for the
  overlap of its ISTFT, so 29 frames are exact; `tests/test_vocoder_registry.py` checks this in float64. `vocos-ft`
  passes 55 dB at 24 frames and levels off at 81–82 dB from 28 frames on.

All rows were measured on an idle GPU, except the `revox` speeds (†). Those were measured on a GPU and CPU shared with
training jobs. Under the same load, `bigvgan-v2-ft` measured RTF 0.0118 and TTFA 32.1 / 39.5 ms, and `griffin-lim`
0.0085 and 50.6 / 40.6 ms: 2–3× their idle figures.

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
| `revox` | 4.5 M | 1.61% [1.17, 2.09] | 0.29% [0.21, 0.40] | 2.270 | 3.100 | 3.813 | 0.0265 † | 84.5 ms † | 177.7 ms † |
| `revox:griffin-lim:harvest` | 4.5 M | 1.71% [1.24, 2.25] | 0.32% [0.22, 0.42] | 2.263 | 3.094 | 3.806 | 0.1769 † | 288.0 ms † | 1035.5 ms † |
| `revox:bigvgan-v2-ft` | 4.5 M | 1.51% [1.06, 1.98] | 0.28% [0.20, 0.37] | 2.260 | 3.094 | 3.815 | 0.0452 † | 121.4 ms † | 288.2 ms † |
| `revox:none` | 4.5 M | 1.51% [1.05, 2.02] | 0.29% [0.19, 0.39] | 2.044 | 2.044 | 3.564 | 0.0108 † | 58.6 ms † | 83.6 ms † |

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
| `revox` (all F0 sources) | revox | one piece per sentence | – | no (eager) |
<!-- vocoders:end -->

## Revox Vocoder 1.0 (non-commercial)

[Revox Vocoder 1.0](https://huggingface.co/minori-live/revox-vocoder-1) by **Minori Live** is a 48 kHz PC-NSF-Vocos:
a Vocos backbone with a neural source-filter excitation, 4.46 M parameters, conditioned on a log-mel and on
frame-level F0. **Its weights are licensed CC BY-NC-SA 4.0: non-commercial use only.**

- The weights are not part of this repository or of `Vyvo/drifting-tts-tr`. `revox` downloads `vocoder.onnx` from
  the original repo, at a pinned revision, when it is loaded. Only ONNX Runtime executes it (`pip install
  "drifting-tts[revox]"`).
- Audio made with it falls under the same non-commercial, share-alike terms. Credit it as "Minori Live — Revox
  Vocoder 1.0" with a link to its repo.

**Verdict: not worth it for this model.**

- **Quality.** On the TTS mels, `revox` scores UTMOSv2 2.27, below every BigVGAN and Vocos row; only Griffin-Lim is
  lower. Its DNSMOS OVRL (3.10) is below even Griffin-Lim's (3.13).
- **Not the conversion.** On real recordings, Revox stays 0.63 UTMOSv2 below `bigvgan-v2-ft` even with ideal
  inputs: its own mel of the recording and the recording's F0. The mel conversion and the F0 source cost little on
  top of that.
- **Cost.** It needs an F0 step (WORLD on Griffin-Lim audio), runs on the CPU and cannot stream. Its license
  excludes commercial use.
- **Its strengths do not apply here.** It offers 48 kHz output and explicit pitch control, but our mels stop at
  12 kHz and the model has no frame-level pitch to give.
- The registry entry stays as an opt-in experiment.

### From our mel to Revox's

- **The two mels.** Revox takes the 128-band Slaney log-mel of 48 kHz audio: n_fft 2048, hop 480, frames centred at
  *k* · 10 ms. Ours is 24 kHz, 100 bands, n_fft 1024, hop 256, with uncentred frames.
- **Same bins, twice the magnitude.** Both STFTs have 23.4375 Hz bins and a 42.7 ms periodic Hann window: the
  2048-point window, sampled every other sample, is the 1024-point one. So for audio band-limited to 12 kHz and
  upsampled ×2, the 48 kHz magnitude is twice the 24 kHz one in the 513 shared bins.
  - On the held-out recordings upsampled with torchaudio's default resampler, the least-squares gain is 2.000 up to
    8 kHz. Above that the resampler rolls off: 1.95 at 8–10 kHz and 1.63 at 10–11 kHz.
  - With a sharp Kaiser resampler (`resample_sharp`), the gain on white noise is 2.000 up to 11 kHz.
- **The conversion** (`RevoxLogMel.convert`):
  1. `griffin-lim`'s NNLS inverts our filterbank to the linear magnitude.
  2. The magnitude is doubled and mapped by Revox's filters; the bins above 12 kHz are empty.
  3. Linear interpolation moves it from frames centred at (*i* + ½) · 256 samples (93.75 Hz) to 100 Hz frames.
  4. Then ln(max(·, 1e-5)).
- **Interpolation domain.** Interpolating in the linear domain commutes with the filterbank. Log-domain
  interpolation (a geometric mean) has the same error, but its bias is negative. Cubic interpolation would cut the
  interpolation error from 0.56 to 0.46 dB, which is small next to the NNLS error.
- **Output rate.** The output is 48 kHz, resampled to 24 kHz for the pipeline. The mel carries nothing above 12 kHz,
  and Revox puts only 3 · 10⁻⁷ of its energy there. Everything downstream (`Synthesizer`, streaming, the web demo,
  the judges) runs at 24 kHz. `Revox.generate` returns the 48 kHz waveform.
- **Streaming.** The graph resets its source phase on every call, so `revox` vocodes each sentence whole
  (`context=None`), like `griffin-lim`.

The converted mel is checked against the Revox mel of the same recordings upsampled to 48 kHz: 100 held-out
recordings, the 105 bands below 11.5 kHz, and the cells within 60 dB of each utterance's peak.

| conversion | mean abs. error | bias | 95th percentile |
|---|---|---|---|
| NNLS + linear interpolation (used) | 0.91 dB | +0.10 dB | 2.85 dB |
| NNLS + log interpolation | 0.92 dB | −0.17 dB | 2.70 dB |
| NNLS alone (on our frames, no time resampling) | 0.65 dB | −0.05 dB | 2.13 dB |
| exact magnitude + linear interpolation | 0.56 dB | +0.14 dB | 1.92 dB |
| exact magnitude + log interpolation | 0.56 dB | −0.12 dB | 1.74 dB |

In copy-synthesis (below), the conversion costs Revox 0.07 UTMOSv2 (2.29 → 2.23) and leaves WER and DNSMOS
unchanged.

### F0

The model predicts pitch per token only, so `revox` takes the frame-level F0 from elsewhere. `revox:<F0
source>[:dio|harvest]` picks the source and the WORLD method:

- `griffin-lim` (the default): WORLD on the Griffin-Lim audio of the same mel. It needs no weights and reuses the
  NNLS magnitude.
- A registry vocoder, e.g. `revox:bigvgan-v2-ft`: WORLD on that vocoder's audio. This is a cascade of two vocoders.
- `none`: every frame is pitch-invalid, so Revox has to read the pitch from the mel.

WORLD's unvoiced frames are passed as reliable unvoiced decisions (`pitch_valid` is true everywhere).

**Voicing is what matters.** Revox cannot voice a frame without F0. With `none`, only 35% of its output frames are
voiced (82% in the recordings), and it sounds hoarse: UTMOSv2 1.94 and DNSMOS OVRL 2.00 in copy-synthesis.

- WORLD dio with its default voicing threshold (`allowed_range` 0.1) voices only 65% of the frames of the
  Griffin-Lim audio of TTS mels, against 92% for harvest. On TTS mels that cost 0.14 UTMOSv2 against harvest.
- dio with `allowed_range` 0.2, the `revox` default, voices 86% of the frames. It matches harvest on 80 TTS
  sentences (UTMOSv2 2.25 vs 2.24) at a twentieth of its cost.
- Filling short unvoiced gaps instead helps less (2.14).

F0 sources against the recording, on the 100 held-out recordings. The reference is the 77% of 10 ms frames where
WORLD harvest and dio agree on the recording (82% of them voiced); harvest alone extends voicing into pauses.

| F0 source (as `revox` runs it) | VDE | GPE | fine error |
|---|---|---|---|
| Griffin-Lim audio, dio (`revox`) | 4.2% | 0.90% | 45 cents |
| Griffin-Lim audio, harvest | 4.5% | 0.93% | 35 cents |
| BigVGAN-v2-ft audio, dio | 5.6% | 1.15% | 43 cents |
| BigVGAN-v2-ft audio, harvest | 5.7% | 0.56% | 34 cents |

VDE: voicing decision error; GPE: gross pitch error, more than 20% off on frames voiced in both; fine error: RMS of
the remaining frames. Griffin-Lim audio is as good an F0 source as BigVGAN-v2-ft's, so the cascade buys nothing.

### Copy-synthesis on held-out recordings

These are the 100 recordings and the judges of `scripts/resynthesis_benchmark.py` (docs/LATENTS.md). The recording
and BigVGAN-v2-ft rows reproduce that table. "Output F0" is WORLD harvest on the output against the reference above.

| system | mel | F0 | WER 8 kHz [95% CI] | CER 8 kHz | UTMOSv2 | DNSMOS OVRL | speaker sim. | output F0: VDE / GPE / cents |
|---|---|---|---|---|---|---|---|---|
| recording | – | – | 6.56% [4.80, 8.58] | 2.18% | 2.953 | 3.268 | – | – |
| `bigvgan-v2-ft` | ours | – | 6.77% [4.95, 8.89] | 2.24% | **2.922** | **3.305** | **0.966** | 5.7% / 0.56% / 34 |
| Revox | its own, from the recording | recording (harvest) | 7.98% [5.96, 10.21] | 2.44% | 2.294 | 2.962 | 0.884 | 7.8% / 0.14% / 21 |
| Revox | converted | recording (harvest) | 8.19% [6.14, 10.50] | 2.57% | 2.227 | 2.966 | 0.871 | 8.0% / 0.13% / 21 |
| `revox` | converted | Griffin-Lim audio, dio | 7.92% [5.96, 10.07] | 2.53% | 2.252 | 3.009 | 0.873 | 7.3% / 1.34% / 44 |
| `revox:griffin-lim:harvest` | converted | Griffin-Lim audio, harvest | 8.53% [6.34, 11.11] | 2.56% | 2.213 | 2.973 | 0.866 | 9.2% / 1.16% / 39 |
| `revox:bigvgan-v2-ft` | converted | BigVGAN-v2-ft audio, dio | 8.32% [6.32, 10.48] | 2.68% | 2.254 | 2.960 | 0.871 | 9.3% / 1.34% / 42 |
| `revox:bigvgan-v2-ft:harvest` | converted | BigVGAN-v2-ft audio, harvest | 8.05% [6.02, 10.26] | 2.69% | 2.245 | 2.966 | 0.864 | 10.1% / 0.85% / 38 |
| `revox:none` | converted | none | 8.39% [6.45, 10.61] | 2.48% | 1.940 | 1.997 | 0.818 | 51.0% / 19.07% / 138 |

- **Revox is below BigVGAN-v2-ft even with ideal inputs.** Given its own mel of the recording and the recording's F0,
  it scores UTMOSv2 2.29 against 2.92, DNSMOS OVRL 2.96 against 3.31 and speaker similarity 0.884 against 0.966. Its
  WER is 7.98% against 6.77%.
- **It follows the F0 it is given** more closely than BigVGAN reproduces the recording's pitch: GPE 0.14% and 21
  cents, against 0.56% and 34 cents.
- **Among WORLD sources, the F0 barely matters on recordings.** UTMOSv2 stays between 2.21 and 2.25, and WER stays
  within the intervals. Its absence matters a lot.
- **No hallucinated band.** Revox puts 3 · 10⁻⁷ of its output energy above 12 kHz on converted mels, and 3 · 10⁻⁶ on
  its own mel of the recordings. The recordings have 2 · 10⁻¹⁰ there.

### On the TTS mels

These are the 495 Freya-TR-Eval sentences of the comparison above. The `revox` rows are in its table.

| vocoder | WER | UTMOSv2 | DNSMOS OVRL | DNSMOS P.808 |
|---|---|---|---|---|
| `bigvgan-v2-ft` | 1.23% [0.82, 1.68] | **2.935** | 3.324 | 3.960 |
| `bigvgan-v2` (NVIDIA's weights) | 1.20% [0.81, 1.65] | 2.398 | 2.810 | 3.779 |
| `revox` (Griffin-Lim audio, dio) | 1.61% [1.17, 2.09] | 2.270 | 3.100 | 3.813 |
| `revox:griffin-lim:harvest` | 1.71% [1.24, 2.25] | 2.263 | 3.094 | 3.806 |
| `revox:bigvgan-v2-ft` | 1.51% [1.06, 1.98] | 2.260 | 3.094 | 3.815 |
| `revox` with dio's default voicing | 1.43% | 2.120 | 2.978 | 3.803 |
| `revox:none` | 1.51% [1.05, 2.02] | 2.044 | 2.044 | 3.564 |
| `griffin-lim` | 1.20% [0.82, 1.63] | 1.772 | 3.127 | 3.465 |

- **Last among the neural vocoders.** `revox` sits between Griffin-Lim and every BigVGAN or Vocos row. It is 0.67
  UTMOSv2 below `bigvgan-v2-ft` and 0.13 below NVIDIA's stock BigVGAN-v2. Its DNSMOS OVRL (3.10) is below
  Griffin-Lim's.
- **The F0 source does not matter, but voicing does.** Griffin-Lim audio with dio or harvest and BigVGAN-v2-ft audio
  are within 0.01 UTMOSv2. dio's default voicing threshold costs 0.15, and no F0 costs 0.23 plus 1.06 in DNSMOS
  OVRL.
- **WER** is 1.5–1.7%, slightly above most other rows, but the intervals overlap.

### Speed

Each stage was timed on the first 20 held-out recordings (134 s of audio): RTX 5090, 3 CPU threads, under the
shared load of the † speeds above. BigVGAN-v2-ft in eager PyTorch measured 0.0229 in the same run.

| stage | runs on | RTF |
|---|---|---|
| NNLS inversion of our mel | GPU | 0.0017 |
| mel conversion (×2, filters, interpolation) | GPU | 0.0006 |
| F0: Griffin-Lim audio | GPU | 0.0033 |
| F0: WORLD dio | CPU, 1 thread | 0.0079 |
| F0: WORLD harvest (instead of dio) | CPU, 1 thread | 0.1523 |
| F0: BigVGAN-v2-ft audio (instead of Griffin-Lim) | GPU | 0.0229 |
| Revox network | ONNX Runtime, CPU, 3 threads | 0.0063 |
| ISTFT | GPU | 0.0017 |
| resampling 48 → 24 kHz | GPU | 0.0013 |
| **`revox`, whole call** | | **0.0195** |
| `revox:none`, whole call | | 0.0093 |

- **The F0 step costs more than the network.** Griffin-Lim plus dio take an RTF of 0.011, against 0.0063 for the
  network. Harvest alone takes 8× the whole `revox` call.
- **The network runs on the CPU.** ONNX Runtime's CUDA provider was 2–7× slower than the CPU on this GPU for 1–10 s
  inputs; that was onnxruntime-gpu 1.22, the last CUDA 12 build, since 1.30 needs CUDA 13. `onnx2torch` cannot
  convert the graph.
- **Slower than BigVGAN overall.** The network alone (0.0063 on 3 CPU threads) is faster than BigVGAN-v2-ft under
  the same load. With the conversion and the F0 step, though, `revox` is about 2× slower than `bigvgan-v2-ft`: RTF
  0.0265 against 0.0118 in the TTS table.
- **No streaming.** The first audio waits for the whole sentence: 84.5 ms for the short sentence and 177.7 ms for
  the long one (†), against 32.1 / 39.5 ms for `bigvgan-v2-ft` under the same load.

### Where it falls short

- **Not the plumbing.** On 20 recordings with the oracle inputs, UTMOSv2 does not improve with a different input
  level: 2.24, 2.27, 2.17 and 1.93 at −6, 0, +6 and +12 dB. It does not improve with other voicing masks either
  (2.24–2.27).
- **Not the alignment.** The output's lag against the recording is ±5 ms, with mixed signs, as for any vocoder that
  makes its own phase.
- **The empty band above 12 kHz.** Revox expects full-band 48 kHz mels. Filling the empty band by extrapolating the
  6–11 kHz slope gains 0.07 UTMOSv2, which is small next to the gap.
- **Where the output differs from the input.** The mel of Revox's output deviates from the input mel about twice as
  much as BigVGAN-v2-ft's does. The gap is largest below 1 kHz (the harmonics), in pauses (the noise floor) and at
  the top of our band. Mean absolute dB, 30 recordings, both measured with Revox's 48 kHz front end:

  | | voiced frames | unvoiced frames | silent frames | < 1 kHz | 8–11.5 kHz bias |
  |---|---|---|---|---|---|
  | Revox (oracle inputs) | 2.87 | 3.13 | 3.97 | 3.53 | −2.18 dB |
  | BigVGAN-v2-ft | 1.61 | 1.65 | 1.78 | 1.67 | −0.01 dB |

- **Domain.** BigVGAN-v2-ft was fine-tuned on this model's mels; Revox was not. Its README does not state its
  training data. It cites singing-voice tools (NSF-HiFiGAN from OpenVPI's SingingVocoders, OpenTune) and targets
  pitch-controlled resynthesis.

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

- **Revox.**
  - The TTS rows come from `compare_vocoders.py` like every other row. The judges get its 24 kHz output.
  - `scripts/revox_benchmark.py` runs on the first 100 recordings of the `val` split, the set of
    `resynthesis_benchmark.py`. It does the conversion check, the F0 sources, the copy-synthesis rows (scored by
    `resynthesis_benchmark.score`, with Revox's 48 kHz output resampled by the judges) and the per-stage speed.

```bash
python scripts/compare_vocoders.py --model drifting_tts_v3.1.pt --table docs/VOCODERS.md \
    --extra vocos-ft=runs/vocos_bigvgan/vocos_ft.pt bigvgan-base-ft=runs/bigvgan_base_ft/bigvgan_ft.pt
python scripts/compare_vocoders.py --model drifting_tts_v3.1.pt --table docs/VOCODERS.md \
    --vocoders revox revox:griffin-lim:harvest revox:bigvgan-v2-ft revox:none
python scripts/revox_benchmark.py --data data/train --out outputs/revox
```

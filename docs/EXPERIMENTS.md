# Experiment log

Every experiment run on this project since the v3.1 release: what was tried, what it gave, what was kept and what was
dropped, and the pitfalls hit on the way. The details live in the topic documents linked from each section. This page
is the index and the place to start before planning new work.

- [Protocols and judges](#protocols-and-judges)
- [Summary: what worked, what did not](#summary-what-worked-what-did-not)
- [1. Acoustic model recipe (before v3.1)](#1-acoustic-model-recipe-before-v31)
- [2. Deployment: latency, WebGPU, MLX](#2-deployment-latency-webgpu-mlx)
- [3. Vocoders](#3-vocoders)
- [4. Audio-VAE latent spaces](#4-audio-vae-latent-spaces)
- [5. "Robotic" prosody: diagnosis and research](#5-robotic-prosody-diagnosis-and-research)
- [6. Pitfalls and engineering notes](#6-pitfalls-and-engineering-notes)
- [7. Open work](#7-open-work)
- [8. Published artefacts](#8-published-artefacts)
- [9. Release v3.2](#9-release-v32)

## Protocols and judges

| protocol | what | used for |
|---|---|---|
| **Freya-495** | all 495 Freya-TR-Eval sentences, `studio` voice, T = 0.3, α = 2 | the vocoder table ([VOCODERS.md](VOCODERS.md)), the release numbers |
| **Freya-100** | first 100 Freya-TR-Eval sentences, speaker 722, T = 0.3, α = 2, seed = sentence index | systems and latent models side by side; `drifting-tts benchmark --num 100 --speaker 722` |
| **Freya-24** | first 24 of those | quick checks of training snapshots |
| **Resynthesis-100** | 100 held-out recordings, encode → decode or mel → vocoder | ceilings of codecs and vocoders (`scripts/resynthesis_benchmark.py`) |
| **Prosody-40** | 40 recordings of the studio voice against 40 generated sentences | F0 statistics; the texts differ, so treat it as a rough comparison |
| **Prosody-val** | the 100 studio `val` recordings against the model's renditions of the same texts (`drifting-tts prosody`) | F0 statistics, DTW F0 correlation, pauses, oracle prosody ([PROSODY.md](PROSODY.md)) |

**Judges:**
- **Intelligibility:** Whisper large-v3 WER / CER on audio band-matched to 8 kHz (the FreyaTTS protocol).
- **Naturalness and cleanliness:** UTMOSv2 and DNSMOS P.835 (SIG / BAK / OVRL) and P.808 on the full band.
- **Speaker similarity:** WavLM-large ECAPA.
- **Speed:** RTF and time to first audio (TTFA) on an RTX 5090.

**Caveats:**
- UTMOSv2 is trained on English and is nearly blind to prosody (flat intonation, rhythm).
- WER does not rank vocoders: even Griffin-Lim reaches the floor.
- Listening is the final judge for naturalness, and none of the experiments below has had a formal listening test yet.

## Summary: what worked, what did not

| area | experiment | outcome | kept? | details |
|---|---|---|---|---|
| latency | CUDA graphs over length buckets + streaming vocoder | TTFA 21–40 ms → **12–14 ms**, bit-identical audio | ✅ default with `fast=True` | [RESULTS.md](RESULTS.md#latency-and-size) |
| latency | TF32 / `torch.compile` | 10.6–11.3 ms, but the output changes | opt-in | [RESULTS.md](RESULTS.md#latency-and-size) |
| browser | ONNX + WebGPU, fp16 weights | first audio 82–135 ms; fp16 rounding fixed by keeping filters and text encoder in fp32 | ✅ | [RESULTS.md](RESULTS.md#in-the-browser-webgpu) |
| Apple | MLX port, small vocoders, prefetch | parity 92–104 dB with PyTorch; Mac timings pending | ✅ | [MLX.md](MLX.md) |
| vocoders | registry: BigVGAN v1 / base / v2, Vocos, Griffin-Lim | drop-in vocoders, one comparison table | ✅ | [VOCODERS.md](VOCODERS.md) |
| vocoders | GTA fine-tunes of BigVGAN-base and Vocos | BigVGAN-base-ft (14 M) ≈ BigVGAN-v2-ft (112 M); Vocos-ft fastest but less natural | ✅ on the Hub | [VOCODERS.md](VOCODERS.md) |
| vocoders | Revox Vocoder 1.0 (third-party, non-commercial) | intelligible, least natural neural vocoder | opt-in only | [VOCODERS.md](VOCODERS.md#revox-vocoder-10-non-commercial) |
| vocoders | GAN-free vocoder with the drifting objective | content learned, realism not (UTMOSv2 1.40–1.84) | ❌ (code kept, experimental) | [DESIGN.md §8](DESIGN.md#8-a-gan-free-vocoder-experimental) |
| vocoders | Vocos-ft trained further: rebalanced losses, multi-scale mel, instantaneous-frequency loss, cosine LR (10k-step pilots) | Freya-100 UTMOSv2 2.63 → 2.92, WER 1.10% → 0.77%, F0 micro-variation 0.45 → 0.40 st (BigVGAN 0.395); same network and streaming | ✅ `configs/vocoder_vocos_v2.yaml`, `vocos-v2`: the vocoder of v3.2 (160k-step run) | [VOCODERS.md](VOCODERS.md#training-vocos-further) |
| vocoders | Vocos with NVIDIA's released MPD + CQT-D instead of its own | UTMOSv2 2.84, no better pitch; 3× slower, 12 GB at batch 8 | ❌ | [VOCODERS.md](VOCODERS.md#training-vocos-further) |
| latents | VoxCPM2 / DAC-VAE latent TTS (10k pilots) | speaks earlier than mels, but noisy with the released decoders | partly | [LATENTS.md](LATENTS.md#tts-pilots-17) |
| latents | longer latent training (10k → 50k) | worse (WER 4.94% → 7.57%) | ❌ | [LATENTS.md](LATENTS.md#longer-training-and-the-released-model) |
| latents | fine-tuning the VAE decoder on generated latents | **the fix for the noise**: DAC-VAE WER 9.55% → 1.32%, UTMOSv2 1.84 → 2.71 | ✅ on the Hub | [LATENTS.md](LATENTS.md#fine-tuning-the-dac-vae-decoder-on-generated-latents-32) |
| prosody | temperature / CFG as prosody knobs | no effect on intonation or rhythm | – | [§5](#5-robotic-prosody-diagnosis-and-research) |
| prosody | oracle prosody A/B: ground-truth token pitch / MAS durations into the frozen DiT | **the token pitch predictor is the bottleneck**: DTW F0 r 0.61 → 0.79 (copy-synthesis ceiling 0.83); predicted pitch is 26% flatter than its targets, durations 41% | – (diagnosis) | [PROSODY.md](PROSODY.md#oracle-prosody-ab-studio-voice) |
| prosody | pitch-deviation gain ×1.2–1.6 | restores the F0 spread (×1.4: 3.78 vs 3.68 st in the recordings), not the contour (r 0.61 → 0.63); CER unchanged, UTMOSv2 2.67 → 2.72 | probe (`drifting-tts prosody`) | [PROSODY.md](PROSODY.md#inference-time-fixes) |
| prosody | punctuation-aware pauses | the 0.15 s joins make the studio voice's sentence pauses 2.3× too long; the measured policy: 0.32 → 0.17 s (recordings 0.14 s), UTMOSv2 2.614 → 2.628 | opt-in | [PROSODY.md](PROSODY.md#pauses) |
| release | **v3.2** = v3.1's acoustic weights + token pitch sampled by the drift predictor (T 0.5; durations stay v3.1's) + Vocos v2 (160k) + punctuation pauses | Freya-495 WER / UTMOSv2: studio 1.33% / 3.021 (v3.1 1.23% / 2.935), male 2.28% / 2.896 (1.74% / 2.814), female 3.99% / 2.722 (3.02% / 2.752); F0 std 3.53 st (recordings 3.68, v3.1 3.21); 89 M parameters instead of 180 M | ✅ `Synthesizer.from_pretrained("v3.2")` | [§9](#9-release-v32), [RESULTS.md](RESULTS.md#v32-sampled-intonation-vocos-v2-punctuation-pauses) |
| release | the same with **sampled durations** too | Freya-495 WER male 5.78%, female 11.28% (studio 1.89%): word slips on new text for the voices with little data | ❌ as the default; opt-in for the studio voice, T ≤ 0.5 | [§9](#9-release-v32) |
| prosody | sampled durations made safe: the letters' durations at prosody T 0.3, pauses and pitch at T 0.5 (inference only; a second row of the sampler's batch, also in the CUDA graphs) | studio Freya-495, 3 seed sets: WER 1.40% vs 1.36% for v3.2 (+0.04 pp; sampled at T 0.5: +0.32 pp), pauses / rate / F0 of the T 0.5 setting, UTMOSv2 +0.015; male / female: their own sampled rhythm reproduces irregular recordings, the studio voice's rhythm borrowed at T 0.3 still costs +0.5 / +0.7 pp | ✅ opt-in (`prosody_duration_temperature=0.3`), the demo's sampled rhythm (studio only) | [PROSODY_MODEL.md](PROSODY_MODEL.md#sampled-rhythm-without-the-slips) |
| latency | the prosody predictor inside the acoustic model's CUDA graphs; graphs captured under `no_grad` | `fast=True` works with the predictor (v3.2 TTFA 5.8–7.4 ms on an idle RTX 5090, v3.1 + BigVGAN-v2-ft 12.3–13.8 ms); `fast=True` memory ~6 GB → 0.65 GB | ✅ | [§9](#9-release-v32) |
| prosody | stochastic prosody predictor (drifting, 8 M) replacing the duration / pitch regressors, DiT frozen (#39) | studio F0 std 3.20 → 3.63 st (recordings 3.68); Freya-100 WER 1.10% → 0.99%, CER 0.22% → 0.22%, UTMOSv2 2.627 → 2.712 at T 0.5 | opt-in (`--prosody`), pending a listening test | [PROSODY_MODEL.md](PROSODY_MODEL.md) |

**Best systems on one protocol** (Freya-100):

| model | vocoder / decoder | WER | CER | UTMOSv2 | DNSMOS OVRL |
|---|---|---|---|---|---|
| **v3.1 (mels)** | BigVGAN-v2-ft (112 M) | **0.66%** | **0.14%** | **2.934** | 3.33 |
| v3.1 (mels) | BigVGAN-base-ft (14 M) | 0.77% | 0.18% | 2.892 | **3.34** |
| v3.1 (mels) | Vocos-ft (13.5 M) | 1.10% | 0.22% | 2.627 | 3.31 |
| v3.1 (mels) | Revox (4.5 M, non-commercial) | 0.77% | 0.16% | 2.244 | 3.10 |
| DAC-VAE latents | DAC-VAE decoder, fine-tuned | 1.32% | 0.30% | 2.710 | 3.26 |
| VoxCPM2 latents | VoxCPM2 decoder, fine-tuned | 1.87% | 0.43% | 2.530 | 3.25 |

v3.1 with BigVGAN-v2-ft remains the reference. The comparison Space plays all six systems from one click.

## 1. Acoustic model recipe (before v3.1)

From [TRAINING.md](TRAINING.md) and [DESIGN.md](DESIGN.md), kept here for completeness:
- **Kyutai's learned-temperature field** beat the paper's fixed temperatures in 10k-step pilots: harmonic contrast
  0.98 vs 0.93, CER 15.4% vs 17.7%. It is `drift.mode: kyutai`.
- **Not adopted:** a prior-mean negative plus texture drift against over-smoothing. Harmonic contrast fell to 0.75
  and CER rose to 22%.
- **Quality filters** on measured scores (Whisper CER, DNSMOS, bandwidth, speaker purity, speaking rate) are part of
  the recipe.
- **Adding a voice by fine-tuning** (v3 → v3.1: 30k steps, the new voice in ~40% of the batches) kept the other
  voices. Their Freya WER even improved: male 1.92% → 1.74%, female 3.78% → 3.02%, partly from per-voice duration
  factors.

## 2. Deployment: latency, WebGPU, MLX

**Time to first audio** ([RESULTS.md](RESULTS.md#latency-and-size), `drifting_tts/fast.py`):

| mode | short | long | paragraph | first call |
|---|---|---|---|---|
| sentence (before) | 21.4 ms | 39.5 ms | 30.9 ms | 282 ms |
| `fast=True` (fp32, bit-identical) | 12.3 ms | 13.9 ms | 13.6 ms | 39 ms |
| `fast` + compile + TF32 | 10.6 ms | 11.3 ms | 11.2 ms | – |

- **Where the time went.** The acoustic model launched ~1,750 small kernels per sentence. CUDA graphs over length
  buckets (tokens padded to multiples of 32, frames to 64, padding masked) bring it to ~4 ms. The rest is the first
  vocoder window, about 60% of TTFA.
- **Streaming.** Windows start with 32 frames, then 256, with per-vocoder context:

  | vocoder | context |
  |---|---|
  | BigVGAN-v2 | 32 frames |
  | BigVGAN v1 | 24 frames |
  | BigVGAN-base | 16 frames |
  | Vocos | 32 frames |

  Measured exactly (> 90 dB against whole-sentence vocoding).
- **Rejected:**
  - TF32 by default: audio SNR 34 dB against fp32.
  - fp16 BigVGAN: overflows, NaN.
  - `torch.compile` of BigVGAN: +17% speed, but the output changes.
- **Small vocoders:** first audio 7.8 ms with BigVGAN-base-ft and 4.9 ms with Vocos-ft (idle GPU).
- **Graph memory.** Until v3.2, `GraphedAcoustic.warmup` captured the graphs with autograd on: every frame bucket's
  outputs kept their activations alive, ~0.25 GB per bucket and ~6 GB for the full set. Captured under `no_grad`
  they take 0.65 GB.

**WebGPU** ([RESULTS.md](RESULTS.md#in-the-browser-webgpu)). Three ONNX graphs; JavaScript repeats each token's
features for its duration.
- **Generic fp16 converters broke the graphs.** fp16 weights are now stored as half with Cast nodes.
- **Rounding.** Rounded anti-aliasing filters cost quality, so BigVGAN's depthwise filters stay fp32 and the web demo
  loads the fp32 text encoder. Audio against PyTorch: SNR 44–47 dB, 385 MB.
- **The browser runs only the TTS model.** Whisper is not part of the page.

**MLX** ([MLX.md](MLX.md), [RESULTS.md](RESULTS.md#mlx)):
- **The acoustic model must stay fp32.** fp16 storage gave 31.9 dB end to end, against 59.9 dB with fp32.
- **Review fixes in the Codex MLX work.**
  - `dtype` now applies to the DiT only. A bf16 duration predictor changed a sentence's frame count, and the CFG
    scale was rounded.
  - `from_pretrained` downloads only the chosen vocoder.
  - The CLI gained `--pitch-shift`.
- **Small vocoders.** BigVGAN-base-ft and Vocos-ft reach 92–104 dB parity with PyTorch, and streamed audio is
  bit-identical to whole-sentence audio. They stay fp32, because fp16 storage gave Vocos only 51.5 dB.
- **Opt-in options, untested on a Mac.** fp16 / bf16 DiT, 8- or 4-bit quantised DiT, and a fused Metal snake
  activation. Parity against the fp32 model:

  | option | mel SNR | waveform SNR |
  |---|---|---|
  | fp16 DiT | 70 dB | 37–55 dB |
  | bf16 DiT | 52 dB | 12–29 dB |
  | 8-bit DiT | 54 dB | 26–34 dB |
  | 4-bit DiT | 29.5 dB | 7–9 dB |

  4-bit is not recommended.
- **Native iOS** lives in a separate repository (`drifting-tts-swift`). This repo keeps only the Python MLX library.

## 3. Vocoders

Details: [VOCODERS.md](VOCODERS.md).

**Freya-495, v3.1 mels:**

| vocoder | params | WER | UTMOSv2 | DNSMOS OVRL | vocoder RTF | TTFA short |
|---|---|---|---|---|---|---|
| BigVGAN-v2-ft | 112 M | 1.23% | **2.935** | 3.324 | 0.0058 | 12.4 ms |
| BigVGAN-v2 (NVIDIA) | 112 M | 1.20% | 2.398 | 2.810 | 0.0058 | 12.4 ms |
| BigVGAN v1 (NVIDIA) | 112 M | 1.36% | 2.756 | 3.169 | 0.0058 | 12.2 ms |
| BigVGAN-base-ft | 14 M | 1.33% | 2.906 | **3.335** | 0.0031 | 7.8 ms |
| BigVGAN-base (NVIDIA) | 14 M | 1.41% | 2.793 | 3.176 | 0.0032 | 7.8 ms |
| Vocos-ft | 13.5 M | 1.56% | 2.627 | 3.302 | **0.0003** | **4.9 ms** |
| Revox (non-commercial) | 4.5 M | 1.61% | 2.270 | 3.100 | 0.0265 (busy GPU) | 84.5 ms |
| drift Vocos, conditional + pooled | 13.5 M | **1.13%** | 1.403 | 3.146 | 0.0003 | 5.1 ms |
| drift Vocos, pooled only | 13.5 M | 1.41% | 1.401 | 3.102 | – | – |
| drift Vocos, `drift_coeff` 3 | 13.5 M | 1.43% | 1.840 | 2.975 | – | – |
| Griffin-Lim | 0 | 1.20% | 1.772 | 3.127 | 0.0038 | 14.8 ms |

**Lessons:**
- **GTA fine-tuning matters more than size.** It lifts BigVGAN-v2 by +0.54 UTMOSv2. BigVGAN-base-ft is within 0.03
  of the 112 M default at one eighth of the size.
- **Vocos-ft is the fastest vocoder** (20× BigVGAN-v2), but less natural. Prosody-40 shows extra short-term F0
  jitter: 0.45 st, against 0.43 for recordings and 0.39 for BigVGAN. This is a likely source of a "buzzy" or
  robotic texture.
- **Revox is a model / domain gap, not our plumbing.** Even with its own mel and the recordings' F0, it stays below
  BigVGAN-ft in copy-synthesis (UTMOSv2 2.29 vs 2.92, speaker similarity 0.88 vs 0.97).
  - The mel conversion is accurate: 0.91 dB against Revox's own mel.
  - The frame-level F0 step costs more than the network.
  - It cannot stream (its source phase resets every call).
- **Vocos's gap was in its training, not its size** ([VOCODERS.md](VOCODERS.md#training-vocos-further)).
  - Copy-synthesis showed it in pitch and periodicity: F0 error 88 vs 52 cents (BigVGAN-v2-ft), periodicity error
    2.5×, rough voiced frames.
  - Continuing the first recipe changed nothing. Rebalancing the losses lifted Freya-100 UTMOSv2 to 2.92 within 5k
    steps: MRD × 1, feature matching × 2, BigVGAN-v2's multi-scale mel × 15 instead of 45 × single-scale, cosine LR.
  - An instantaneous-frequency (phase-advance) loss against the recording then brought the F0 jitter to BigVGAN's
    level.
  - Per-step evaluation needs pitch measures: UTMOSv2 moved most where the pitch measures moved least.
- **GAN-free drift vocoder: negative.** Three 20k-step pilots learned content (WER as low as 1.13%) but not realism.
  - The pairing (conditional and pooled vs pooled only) and fixed noise barely matter.
  - Tripling the drift weight helps (1.40 → 1.84) but stays far below the GAN fine-tune (2.63).
  - Likely limit: frozen discriminator features do not resolve Vocos artefacts. A trained adversary is needed.

## 4. Audio-VAE latent spaces

Details: [LATENTS.md](LATENTS.md).

**Resynthesis ceilings** (Resynthesis-100):

| system | WER (8 kHz) | UTMOSv2 | speaker sim. | first window |
|---|---|---|---|---|
| recording | 6.56% | 2.953 | – | – |
| BigVGAN-v2-ft (mel) | 6.77% | 2.922 | 0.966 | – |
| DAC-VAE | 6.90% | 2.802 | 0.983 | 18.7 ms |
| VoxCPM2 | 7.10% | 2.795 | 0.962 | 4.0 ms |
| VoxCPM1.5 | 7.71% | 2.682 | 0.966 | – |

**Latent TTS pilots** (10k steps each, same recipe, Freya-100, released decoders):

| target | WER | CER | UTMOSv2 | RTF |
|---|---|---|---|---|
| VoxCPM2 latents | 4.94% | 1.39% | 2.052 | 0.0054 |
| DAC-VAE latents | 9.55% | 2.91% | 1.837 | 0.0239 |
| mels | 18.77% | 5.26% | 2.746 | 0.0158 |

- **Latents learn to speak much earlier than mels.** But they sound noisy with the released decoders.
- **Recipe:** latents repeated 4× to 100 Hz, so monotonic alignment has a frame per token; DiT patch 4; a 1-D latent
  MAE for the kernel features; per-channel normalisation.
- **Longer generator training made VoxCPM2 worse.** 10k → 50k steps at a constant LR: WER 4.94% → 7.57%, UTMOSv2
  2.05 → 1.90. No downward trend in CER over the snapshots. The recommended latent models use the 10k checkpoint.

**Why the latent models sounded noisy** (Freya-24, VoxCPM2 decoder):

| latents decoded | UTMOSv2 | DNSMOS OVRL |
|---|---|---|
| real | 2.660 | 3.25 |
| generated, T = 0.3 / T = 0 | 2.084 / 2.277 | 1.45 / 1.48 |
| real × 0.7 | 2.446 | 3.12 |
| real + noise, σ 0.3 / 0.6 | 2.179 / 1.875 | 2.72 / 1.76 |
| **real, smoothed over 3 frames (120 ms)** | 2.011 | **1.42** |

- **The decoder turns temporally smoothed latents into noise.** Generated latents are over-smooth: frame-to-frame
  change 0.62 vs 0.98, per-channel std 0.68 vs 0.97.
- **No cheap fix works.** Rescaling the variance, picking one of the 4 repeats instead of the average, or sweeping
  T / CFG does not help.
- **Mels are robust to this blur;** VAE latents are not.

**The fix: fine-tune the VAE decoder on generated latents** (GTA, `vocoder.arch: vae_decoder`). On Freya-100:

| model | decoder | WER | CER | UTMOSv2 | DNSMOS OVRL |
|---|---|---|---|---|---|
| VoxCPM2 | released | 4.94% | 1.39% | 2.052 | 1.49 |
| VoxCPM2 | fine-tuned 20k | 2.31% | 0.50% | 2.340 | 3.21 |
| VoxCPM2 | **fine-tuned 35k (published)** | 1.87% | 0.43% | **2.530** | 3.25 |
| VoxCPM2 | fine-tuned 40k | 1.76% | 0.42% | 2.410 | – |
| DAC-VAE | released | 9.55% | 2.91% | 1.837 | 1.48 |
| DAC-VAE | fine-tuned 35k / 37.5k | 1.87% / 1.43% | 0.40% / 0.34% | 2.670 / 2.664 | 3.23 / 3.20 |
| DAC-VAE | **fine-tuned 40k (published)** | **1.32%** | **0.30%** | **2.710** | **3.26** |

**Snapshots** (Freya-24 UTMOSv2):

| model | released | 2.5k | 5k | 10k | 20k | 30k | 35k | 40k |
|---|---|---|---|---|---|---|---|---|
| VoxCPM2 | 2.08 | 1.50 | 1.76 | 2.07 | 2.33 | 2.45 | 2.49 | 2.43 |
| DAC-VAE | 1.91 | 1.60 | 1.92 | 2.24 | 2.47 | 2.59 | 2.72 | 2.68 |

- **DNSMOS jumps within the first 2.5k steps.** UTMOSv2 first dips, then climbs to about 35–40k.
- **Pick snapshots on Freya-100, not Freya-24.** They ranked DAC-VAE 37.5k and 40k differently.

**Design notes for decoder fine-tuning:**
- **Weight norm.** Start from the released `weight_g` / `weight_v`. Re-deriving `v` from folded weights made the
  output convolution ~36× too sensitive and damaged clean decoding. That run was discarded.
- **Context.** VoxCPM2's decoder is causal: 12 frames of left context give 81 dB against whole-utterance decoding.
  DAC-VAE's is not: it needs **8 + 8 frames** (104 dB median). One side only is not enough (66–69 dB).
- **DAC-VAE watermark.** Only the audio path (65.3 M) is trained. All watermark weights (14.7 M) stay frozen and
  bit-identical, the watermark is still added at training and inference time, and every export checks this.
  - The fine-tuned watermark is the released generator applied to the new audio, at the same level.
  - No public detector exists yet, so detection has not been confirmed.
- **Clean decoding is kept.** Resynthesis UTMOSv2 changes 2.72 → 2.70 for DAC-VAE (40k) and 2.66 → 2.52 for
  VoxCPM2 (20k).
- **Output band.** Use the fine-tuned decoders at 24 kHz. They were trained against 24 kHz recordings, so nothing
  above 12 kHz gets a training signal.

**Remaining gap to v3.1** (UTMOSv2 2.71 vs 2.93):
- It lies in the generated latents, not the decoder.
- VoxCPM2's F0 spread is narrower than v3.1's on the same sentences (2.48 vs 3.08 semitones), and it speaks ~6%
  faster.
- Generator-side work is open in #27.

## 5. "Robotic" prosody: diagnosis and research

Listening feedback: the voices sound robotic. Measured on Prosody-40 (harvest F0, semitones):

| | F0 std | F0 range 5–95 | F0 movement / 10 ms | F0 micro-variation |
|---|---|---|---|---|
| recordings | **3.67** | **11.8** | 0.63 | 0.43 |
| v3.1 + BigVGAN, T 0.3 / 0.6 / 1.0 | 3.29 / 3.27 / 3.23 | 10.6 / 10.5 / 10.2 | 0.62 / 0.62 / 0.61 | 0.39 / 0.39 / 0.38 |
| v3.1 + BigVGAN, T 0.3, α 1 | 3.30 | 10.7 | 0.63 | 0.41 |
| v3.1 + Vocos | 3.46 | 11.2 | 0.66 | **0.45** |
| DAC-VAE + fine-tuned decoder | 2.73 | 8.8 | 0.62 | 0.40 |

- **Temperature and CFG do not change prosody.** Durations and token pitch come from deterministic MSE regressors,
  so the same text always gets the same average tune and rhythm. The noise only changes spectral texture.
- **The pitch spread is ~10% narrower than in recordings.** It is ~25% narrower for the DAC-VAE model.

**Research summary** (sources in #37):
- **Kyutai's naturalness** (Kyutai TTS / DSM, Moshi, CALM, Pocket TTS) comes from an autoregressive backbone over
  time, long training context (60–150 s), voice prompts and large data.
  - Pocket TTS ships a ~100 M backbone. Its one-step head, which can be a drifting head, samples only an 80 ms
    frame.
  - None of their systems generates a whole utterance in one step.
- **More parameters would not fix "robotic".** The missing randomness is removed before the DiT runs. Kokoro (82 M)
  and StyleTTS 2 reach top naturalness at our size. Scale helps prosody in autoregressive systems with very large
  data.
- **Most likely causes, ranked:**
  1. deterministic, mean-regressed prosody;
  2. no context beyond the sentence, with a fixed 0.15 s pause between sentences;
  3. vocoder periodicity artefacts (Vocos);
  4. a metric that cannot see the problem.

The plan is in issues #37–#42 (below). The owner decided that the vocoder stays as it is for this work.

**Measured on the same texts (#38, [PROSODY.md](PROSODY.md)).** On 100 held-out studio recordings, ground-truth token
pitch through the frozen DiT lifts the DTW F0 correlation with the recording from 0.61 to 0.79 (copy synthesis 0.83)
and the F0 std from 3.19 to 3.81 st (recordings 3.68). The DiT renders the token pitch it is given as faithfully as
copy synthesis, so the deterministic pitch predictor, not the acoustic model, flattens the intonation.

**Stochastic prosody predictor (#39, [PROSODY_MODEL.md](PROSODY_MODEL.md)).** An 8 M sampler, trained with the
drifting objective on multi-scale feature maps of the per-token (log-duration, pitch) sequence, replaces the
regressors at inference; the DiT is unchanged. On 100 held-out studio sentences the F0 std goes from 3.20 to 3.63 st
(recordings 3.68, oracle prosody 3.81) and the internal pauses from 0.45 to 1.48 per utterance (1.39). At prosody
temperature 0.5, Freya-100 stays as intelligible (WER 0.99%, CER 0.22%) and UTMOSv2 rises to 2.712. At T 1 the
sampled durations cost intelligibility (WER 2.63%). Flow matching with the same backbone is the better per-token
model (CRPS), but it is 10–14% flatter and needs 8 network evaluations. BERTurk word features help the pitch
(polar questions) but not the durations.

## 6. Pitfalls and engineering notes

**Training and evaluation:**
- **Never interrupt a long training run.** Diagnostics run alongside it or not at all. Resume from `last.pt` if a run
  stops.
- **Snapshots.** Keep `decoder_ft_<step>.pt` every 2.5k steps (`train.keep_snapshots`). The last step is not always
  the best.
- **Same protocol everywhere.** Reproduce one known row first (v3.1 + BigVGAN-v2-ft: 0.66% / 2.934 on Freya-100)
  before trusting a new table.
- **Speed tables.** Measure on an idle GPU: a busy GPU inflates RTF and TTFA 2–4×.
- **Machine limits.** CPU quota ~7.7 cores: cap `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `NUMBA_NUM_THREADS` at 2–3
  and `num_workers` ≤ 3. Run pytest files one at a time under load; a combined run can exceed long timeouts.
- **Losses on recordings meet digital silence.** A phase loss normalised by the STFT magnitude gave NaN on all-zero
  training segments. Floor every normaliser (`phase_derivative_loss`).
- **Measured GPU memory, not estimates.** NVIDIA's CQT-D at batch 16 × 16,384 samples needs 17 GB next to a Vocos
  generator. Register the peak that `nvidia-smi` shows.
- **Process management.** `pkill -f` / `pgrep -f` patterns can match the shell running them. Kill by PID, or use
  bracket patterns (`[p]attern`). Wait loops that `pgrep` their own pattern never end.

**Publishing:**
- **Sanitise every checkpoint before upload.** Training checkpoints store the config with absolute data paths. Rewrite
  `data.root` / `mae.path` to generic paths, drop training-only state (`taus`), and scan for paths and names.
  Decoder exports (`decoder_ft.pt`) hold tensors only. `scripts/prepare_release.py` (`drifting_tts/publish.py`) does
  it for the acoustic model, Vocos and the prosody predictor: it keeps only what the loaders read, reloads each file
  against its source, and scans the pickle's strings and the file's printable runs for absolute paths and for every
  speaker, show and data-root name of the training data (counts only). The scan also flags the published
  `drifting_tts_v3.1.pt`: its config names the local data folder (`data.root`).
- **A prosody predictor belongs to one text encoder.** It is conditioned on the encoder states and the regressors'
  predictions of the model it was trained on. Its published checkpoint records a fingerprint of those weights, and
  loading it with another acoustic model warns (a fine-tune that moves the encoder needs a token-level check,
  `scripts/eval_prosody_tokens.py`, or a retrained predictor).
- **Stage before uploading.** `DRIFTING_TTS_HUB_DIR=<staging dir>` makes every Hub download (`from_pretrained`, the
  vocoder and prosody registries) read that directory first, so a release runs locally exactly as users will get it.
- **Privacy.** Never publish dataset names, sources, hours, speaker names or recording-derived audio. Only aggregate
  metrics and integer speaker IDs.
- **Third-party licences.**
  - Revox: CC BY-NC-SA 4.0, downloaded at runtime and never redistributed, with attribution.
  - DAC-VAE and VoxCPM2: Apache-2.0.
  - DAC-VAE's watermark stays intact.
- **Hugging Face Spaces (ZeroGPU):** list every runtime dependency in `requirements.txt`. Missing so far:
  - `torchaudio` for VAE resampling;
  - `vocos`;
  - `onnxruntime`, `pyworld` and `setuptools<81` for Revox.

  Test the app locally with the package copied in, then call the live Space once (`gradio_client`) after every
  deploy.
- **Git.** Force-push is blocked in this environment. Bring a branch up to date by merging `main` into it.

## 7. Open work

| issue | topic |
|---|---|
| #37 | tracking: robotic prosody and the research summary |
| #38 | prosody diagnostics: metrics against recordings on the same texts, oracle prosody A/B, punctuation-aware pauses, pitch-gain sweep |
| #39 | stochastic prosody predictor (drifting), with its own temperature |
| #40 | context: word-level features, paragraph mode |
| #41 | SSL (WavLM) adversarial term on the acoustic model |
| #42 | strategic: Kyutai's Pocket TTS recipe on Turkish |
| #27 | latent models: generator-side fixes (over-smooth latents, flat prosody) |
| #17, #19 | latent TTS and the vocoder / latent tracking issue |
| #8 | small vocoders on phones: Mac / iPhone measurements of the MLX ports |

Suggested order: #38 → #39 → #40 / #41. #42 is a separate strategic decision. A native-listener CMOS or MUSHRA
test should gate every naturalness claim.

## 8. Published artefacts

| where | what |
|---|---|
| [`Vyvo/drifting-tts-tr`](https://huggingface.co/Vyvo/drifting-tts-tr) | v3.1 (`drifting_tts_v3.1.pt`); vocoders `bigvgan_v2_ft.pt`, `bigvgan_base_ft.pt`, `vocos_ft.pt`; `onnx/`; `mlx/` (incl. `bigvgan_base_ft.safetensors`, `vocos_ft.safetensors`); v3.2 (planned): `drifting_tts_v3.2.pt` (v3.1's weights, sanitised config), `vocos_v2.pt`, `prosody_drift_v3.2.pt` |
| [`Vyvo/drifting-tts-tr-dacvae`](https://huggingface.co/Vyvo/drifting-tts-tr-dacvae) | DAC-VAE latent model (10k) and the fine-tuned decoder (40k, watermark kept) |
| [`Vyvo/drifting-tts-tr-voxcpm2`](https://huggingface.co/Vyvo/drifting-tts-tr-voxcpm2) | VoxCPM2 latent models (10k, 50k) and the fine-tuned decoder (35k) |
| [`Vyvo/drifting-tts-tr-demo`](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-demo) | Gradio demo of v3.1 |
| [`Vyvo/drifting-tts-tr-webgpu`](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-webgpu) | in-browser WebGPU demo |
| [`Vyvo/drifting-tts-tr-compare`](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-compare) | one click, six systems: v3.1 with four vocoders, VoxCPM2 and DAC-VAE latents |

Scripts used for the tables:

| script | what |
|---|---|
| `scripts/compare_vocoders.py` | vocoder quality, streaming context and latency |
| `scripts/resynthesis_benchmark.py` | codec and vocoder ceilings |
| `scripts/revox_benchmark.py` | Revox |
| `scripts/bench_ttfa.py`, `scripts/bench_mlx_ttfa.py` | latency |
| `scripts/check_mlx_parity.py` | MLX parity |
| `drifting-tts benchmark` | Freya |
| `drifting-tts finetune-vocoder` | GTA fine-tunes: `vocoder.arch: bigvgan` / `vocos` / `vae_decoder`, `vocoder.objective: drift` |

## 9. Release v3.2

**What it is.** v3.1's acoustic weights (`drifting_tts_v3.2.pt`: the same weights with a sanitised config) with
three results of #37 made the default of a new release: the token pitch sampled by the stochastic prosody predictor
trained with drifting (#39, prosody temperature 0.5), Vocos v2 as the vocoder (#47, the long run's final snapshot,
160k steps) and punctuation-aware pauses (#38). The durations stay v3.1's regressors, with v3.1's per-voice factors.
`Synthesizer.from_pretrained("v3.2")`, `drifting-tts synthesize --release v3.2`; v3.1 stays available
(`from_pretrained("v3.1")`), and every default of the explicit API is unchanged.

**Evaluation** (`scripts/eval_release.sh`, [RESULTS.md](RESULTS.md#v32-sampled-intonation-vocos-v2-punctuation-pauses)):
Freya-495 for the three voices, Freya-100 and the prosody protocol, each against v3.1 + BigVGAN-v2-ft (as released)
and v3.1 + vocos-ft.

| | v3.1 + BigVGAN-v2-ft | v3.1 + vocos-ft | v3.2 | v3.2, sampled durations |
|---|---|---|---|---|
| Freya-495 studio WER / UTMOSv2 | 1.23% / 2.935 | 1.56% / 2.627 | 1.33% / 3.021 | 1.89% / 3.028 |
| Freya-495 male WER / UTMOSv2 | 1.74% / 2.814 | 1.61% / 2.335 | 2.28% / 2.896 | 5.78% / 2.953 |
| Freya-495 female WER / UTMOSv2 | 3.02% / 2.752 | 3.25% / 2.091 | 3.99% / 2.722 | 11.28% / 2.801 |
| studio val: F0 std (recordings 3.68) / pauses per utterance (1.39) | 3.21 / 1.58 | 3.36 / 1.58 | 3.53 / 1.54 | 3.56 / 2.35 |
| TTFA `fast` (short / paragraph) | 12.3 / 13.5 ms | 4.9 / 6.0 ms | 5.8 / 7.1 ms | – |

- **Sampled durations were the first candidate.** With Vocos v2 they keep the studio voice usable (1.89%) but give
  the male and female voices word slips on new text (5.78%, 11.28%); the female voice with the first Vocos fine-tune
  gives 10.89%, so the vocoder is not the cause. Sampling only the pitch keeps most of the intonation gain at a small
  intelligibility cost: female 3.99% against 3.27% for v3.1's regressors through Vocos v2.
- **What is left for the male and female voices.** Their WER stays above v3.1 + BigVGAN-v2-ft (2.28% vs 1.74%, 3.99%
  vs 3.02%). Part of it is the vocoder (female, regressors: BigVGAN-v2-ft 3.02%, Vocos v2 3.27%), part the sampled
  pitch.
- **Sampled rhythm after the release** ([PROSODY_MODEL.md](PROSODY_MODEL.md#sampled-rhythm-without-the-slips)).
  Listening, the owner preferred the sampled durations on the studio voice, so the slips were traced: they come
  from the letters' durations at T 0.5, not from the pauses or the speaking rate. Letters at T 0.3 with pauses and
  pitch at T 0.5 keep the liveliness (pauses, rate, F0 of the T 0.5 setting) at v3.2's intelligibility (Freya-495,
  3 seed sets: 1.40% vs 1.36%). The male and female recordings are irregular (female: abrupt starts, 16.5% of the
  letters at ≤ 2 frames) and the sampler reproduces them, so no temperature fixes those voices; borrowing the studio
  voice's rhythm halves their cost but leaves +0.5 / +0.7 pp.

**Dry run** (Freya-100, studio voice, sampled durations, the 10k-step Vocos v2 pilot standing in for the long run's
snapshot, busy shared GPU, `scripts/eval_release.sh` with `STAGES=freya100` on the staged files). Both reference rows
reproduce the known ones exactly:

| system | WER [95% CI] | CER | UTMOSv2 [95% CI] | DNSMOS OVRL | RTF (busy GPU) |
|---|---|---|---|---|---|
| v3.1 + BigVGAN-v2-ft | 0.66% [0.22, 1.22] | 0.14% | 2.934 [2.892, 2.975] | 3.327 | 0.0244 |
| v3.1 + vocos-ft | 1.10% [0.44, 1.89] | 0.22% | 2.627 [2.585, 2.669] | 3.307 | 0.0074 |
| v3.2 candidate (sampled durations, Vocos v2 10k pilot) | 0.66% [0.11, 1.33] | 0.14% | 2.976 [2.941, 3.012] | 3.360 | 0.0092 |

**Engineering:**
- **Names instead of files.** `vocoder="vocos-v2"` and `prosody="drift"` resolve to `vocos_v2.pt` and
  `prosody_drift_v3.2.pt` in `Vyvo/drifting-tts-tr`; `pause="punct"` builds the voice's `PausePolicy`. A release is a
  row of `drifting_tts/hub.py: RELEASES`: a new acoustic model is a one-line change there.
- **`fast=True` with the prosody predictor.** The drift sampler (one pass) runs inside the text encoder's CUDA graph,
  its noise drawn outside in the eager order: same frame counts, mels within float noise of the eager path.
- **Pause edges belong to the duration source.** `PausePolicy` inserts the measured pause minus the edge silence
  of the generated sentences, which was measured with v3.1's regressors. Sampled durations leave longer edges
  (leading + trailing silence of 200 held-out sentences, with the Vocos v2 pilot): studio 0.202 s (v3.1 0.161),
  male 0.184 (0.176), female 0.123 (0.070). `prepare_release.py` measures them and stores them in the prosody
  checkpoint (`pause_edges`); `pause="punct"` uses them only when that predictor's durations are used. v3.2 keeps
  v3.1's durations, so it uses v3.1's table.
- **`Synthesizer.variant`** gives v3.1 next to v3.2 on one acoustic model (the demo's toggle) and replaces the
  attribute swapping of the comparison Space; `prosody_temperature` can be set per call.


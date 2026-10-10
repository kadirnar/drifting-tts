# Results

Detailed results of the current release, **v3.2** (next section), and of v3.1, whose acoustic weights it keeps: the
v3 model plus the fine-tuned `studio` voice. The [README](../README.md) has the summary.

## v3.2: sampled intonation, Vocos v2, punctuation pauses

v3.2 keeps v3.1's acoustic weights and changes what surrounds them:

| | v3.1 | v3.2 |
|---|---|---|
| token pitch (intonation) | deterministic regressor | sampled by an 8.1 M prosody predictor trained with drifting, prosody temperature 0.5 ([PROSODY_MODEL.md](PROSODY_MODEL.md)) |
| durations (rhythm) | deterministic regressor, per-voice factors | the same: v3.1's |
| vocoder | BigVGAN-v2-ft (112.4 M) | Vocos v2 (13.5 M): `vocos-ft` trained 160k steps further with the second recipe ([VOCODERS.md](VOCODERS.md#training-vocos-further)) |
| pause between sentences | 0.15 s | by the sentence's final punctuation, measured per voice ([PROSODY.md](PROSODY.md#pauses)) |
| acoustic model file | `drifting_tts_v3.1.pt` | `drifting_tts_v3.2.pt`: the same weights with a sanitised config |
| parameters at inference | 180.1 M | 89.3 M (67.7 M acoustic + 8.1 M prosody + 13.5 M vocoder) |

```python
tts = Synthesizer.from_pretrained("v3.2", "cuda")   # vocoder="vocos-v2", prosody="drift", prosody_durations="regressor", pause="punct"
v31 = tts.variant(vocoder="bigvgan-v2-ft", prosody=None, pause=0.15)   # v3.1, sharing the acoustic model
```

**Freya-TR-Eval** (Freya-495: all 495 sentences, T = 0.3, α = 2, seed = sentence index, Whisper large-v3 on 8 kHz
audio, UTMOSv2 and DNSMOS P.835 on the full band). "Sampled durations" is the predictor sampling the durations too
(`prosody_durations="sampled"`); it is not the release setting:

| voice | system | WER [95% CI] | CER | UTMOSv2 [95% CI] | DNSMOS OVRL |
|---|---|---|---|---|---|
| studio | v3.1 + BigVGAN-v2-ft (v3.1 as released) | 1.23% [0.82, 1.68] | 0.24% | 2.935 [2.916, 2.954] | 3.317 |
| studio | v3.1 + vocos-ft | 1.56% [1.13, 2.05] | 0.29% | 2.627 [2.605, 2.649] | 3.299 |
| studio | **v3.2** | 1.33% [0.91, 1.80] | 0.27% | 3.021 [3.002, 3.040] | 3.346 |
| studio | v3.2, sampled durations | 1.89% [1.38, 2.50] | 0.36% | 3.028 [3.009, 3.046] | 3.347 |
| male | v3.1 + BigVGAN-v2-ft | 1.74% [1.24, 2.30] | 0.38% | 2.814 [2.792, 2.836] | 3.302 |
| male | v3.1 + vocos-ft | 1.61% [1.13, 2.16] | 0.32% | 2.335 [2.315, 2.356] | 3.251 |
| male | **v3.2** | 2.28% [1.72, 2.91] | 0.45% | 2.896 [2.875, 2.917] | 3.342 |
| male | v3.2, sampled durations | 5.78% [4.97, 6.75] | 1.41% | 2.953 [2.930, 2.975] | 3.386 |
| female | v3.1 + BigVGAN-v2-ft | 3.02% [2.38, 3.71] | 0.70% | 2.752 [2.729, 2.776] | 3.219 |
| female | v3.1 + vocos-ft | 3.25% [2.56, 3.96] | 0.77% | 2.091 [2.068, 2.116] | 3.120 |
| female | **v3.2** | 3.99% [3.27, 4.73] | 0.97% | 2.722 [2.699, 2.745] | 3.236 |
| female | v3.2, sampled durations | 11.28% [10.06, 12.49] | 3.41% | 2.801 [2.775, 2.826] | 3.316 |

- **The v3.1 rows reproduce the published ones** (1.23% / 2.94, 1.74% / 2.81, 3.02% / 2.75).
- **Studio voice:** as intelligible as v3.1 (the intervals overlap), UTMOSv2 3.021 against 2.935.
- **Male and female voices:** v3.2 loses some intelligibility against v3.1 (2.28% vs 1.74%, 3.99% vs 3.02%). For the
  female voice, v3.1's regressors through Vocos v2 give 3.27% / 2.638: sampling the pitch costs about 0.7 points of
  WER there and adds 0.08 UTMOSv2. v3.2 is far above v3.1 + `vocos-ft` for every voice (UTMOSv2 +0.39 to +0.63),
  mostly from Vocos v2 (female voice, v3.1's regressors through Vocos v2: +0.55).
- **Why the durations stay v3.1's.** Sampled durations cost intelligibility on new text for the voices with little
  data: WER 5.78% (male) and 11.28% (female). With the first Vocos fine-tune the female voice gives 10.89%, so the
  durations cause it, not Vocos v2. The errors are single-phone slips where the sampler places very short sounds
  ([PROSODY_MODEL.md](PROSODY_MODEL.md#guard-rails-freya-100)). `prosody_durations="sampled"` stays an opt-in for
  the studio voice at prosody temperature ≤ 0.5 (studio at 0.3: 1.46% / 3.035).
- **Sampled rhythm, made safe (opt-in, studio voice).** The words are lost on the letters' durations, not on the
  pauses: sampling the letters at prosody temperature 0.3 and the pauses and the pitch at 0.5
  (`prosody_duration_temperature=0.3`) gives WER 1.40% against 1.36% for v3.2 over three seed sets (+0.04 pp
  [−0.15, +0.24]; sampled durations at T 0.5: +0.32 pp) with the pauses, speaking rate and intonation of the T 0.5
  setting (held-out studio texts: 2.37 pauses per utterance, F0 std 3.56) and UTMOSv2 3.032 (3.016). The male and
  female voices stay +0.61 / +0.40 pp above v3.2 even with the studio voice's rhythm and their own sentence edges,
  so the demo keeps v3.2 for them ([PROSODY_MODEL.md](PROSODY_MODEL.md#sampled-rhythm-without-the-slips)).

**Freya-100** (the first 100 sentences, studio voice; the protocol of [EXPERIMENTS.md](EXPERIMENTS.md#protocols-and-judges)):

| system | WER [95% CI] | CER | UTMOSv2 [95% CI] | DNSMOS OVRL |
|---|---|---|---|---|
| v3.1 + BigVGAN-v2-ft | 0.66% [0.22, 1.22] | 0.14% | 2.934 [2.892, 2.975] | 3.327 |
| v3.1 + vocos-ft | 1.10% [0.44, 1.89] | 0.22% | 2.627 [2.585, 2.669] | 3.307 |
| **v3.2** | **0.44%** [0.11, 0.90] | **0.11%** | **2.998** [2.955, 3.041] | 3.356 |
| v3.2, sampled durations | 0.77% [0.22, 1.44] | 0.16% | 3.026 [2.984, 3.068] | 3.360 |
| v3.2, sampled durations, 10k-step Vocos v2 pilot (dry run) | 0.66% [0.11, 1.33] | 0.14% | 2.976 [2.941, 3.012] | 3.360 |

**Prosody** (`drifting-tts prosody`, the 100 studio `val` recordings against each system's rendition of their texts,
sentence by sentence; harvest F0 in semitones; [PROSODY.md](PROSODY.md#how-it-is-measured) defines the columns):

| system | F0 std | F0 range | micro | pauses/utt | pause s | syl/s | DTW F0 r | CER | WER | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recording | 3.68 | 11.9 | 0.42 | 1.39 | 0.139 | 6.22 | – | 0.88% | 2.06% | 3.093 | – |
| v3.1 + BigVGAN-v2-ft | 3.21 | 10.1 | 0.41 | 1.58 | 0.320 | 5.90 | 0.589 | 0.44% | 1.98% | 2.870 | 0.924 |
| v3.1 + vocos-ft | 3.36 | 10.8 | 0.47 | 1.58 | 0.316 | 5.90 | 0.606 | 0.50% | 2.21% | 2.614 | 0.931 |
| **v3.2** | 3.53 | 11.3 | 0.44 | 1.54 | 0.175 | 6.01 | 0.579 | 0.42% | 2.02% | 2.990 | 0.940 |
| v3.2, sampled durations | 3.56 | 11.4 | 0.45 | 2.35 | 0.186 | 6.08 | 0.572 | 0.50% | 2.32% | 3.000 | 0.943 |

- **Intonation.** Sampling the pitch raises the F0 standard deviation from 3.21 semitones (v3.1 as released) to 3.53
  (recordings 3.68) and the 5–95% range from 10.1 to 11.3 (11.9); sampled durations add little (3.56 / 11.4). Part
  of the gain over v3.1 is the vocoder: v3.1's regressors through `vocos-ft` give 3.36.
- **Pauses.** Punctuation pauses bring the gap between sentences from 0.32 s to 0.175 s (recordings 0.139 s) and the
  length ratio to the recordings from 1.054 to 1.035. The rhythm inside sentences stays v3.1's (pauses per
  utterance 1.54; sampled durations: 2.35, recordings 1.39).
- **Guard rails.** On these sentences v3.2 is as intelligible as v3.1 (CER 0.42% / WER 2.02% against 0.44% / 1.98%),
  with UTMOSv2 2.990 (2.870) and speaker similarity 0.940 (0.924). The DTW F0 correlation with the particular
  recording drops slightly (0.589 → 0.579): each sample is one plausible tune among several. F0 micro-variation is
  0.44 st (BigVGAN-v2-ft 0.41, recordings 0.42).

**Latency** (`scripts/bench_ttfa.py`, RTX 5090 with nothing else running, studio voice, T = 0.3, α = 2, median of
100 runs after warm-up; `fast`: `Synthesizer(fast=True).stream`, the prosody predictor inside the acoustic model's
CUDA graphs):

| system | mode | TTFA short | TTFA long sentence | TTFA paragraph | RTF short / long / paragraph |
|---|---|---|---|---|---|
| v3.1 + BigVGAN-v2-ft (`--cuda-kernel`) | `fast` | 12.3 ms | 13.8 ms | 13.5 ms | 0.0185 / 0.0098 / 0.0101 |
| v3.1 + vocos-ft | `fast` | 4.9 ms | 6.3 ms | 6.0–6.1 ms | 0.0043 / 0.0015 / 0.0018 |
| v3.2 | `stream` (eager acoustic model) | 14.8 ms | 15.2 ms | 15.2 ms | 0.0119 / 0.0030 / 0.0039 |
| **v3.2** | **`fast`** | **5.8 ms** | **7.4 ms** | **7.1 ms** | **0.0051 / 0.0017 / 0.0020** |

Two rounds of 100 runs agree within 0.1 ms, and the v3.1 + BigVGAN-v2-ft row reproduces the published one
(12.3 / 13.9 / 13.6 ms, [below](#latency-and-size)). v3.2 starts speaking about twice as fast as v3.1 because Vocos's
first window is cheaper than BigVGAN's, and generates 200–600× faster than real time (v3.1: 55–100×). The files were
the staged release (`DRIFTING_TTS_HUB_DIR`, `scripts/bench_ttfa.py --model drifting_tts_v3.2.pt --mode fast --vocoder
vocos-v2 --prosody drift --prosody-durations regressor`).

Earlier, on the GPU shared with two trainings (2 interleaved rounds × 50 runs; durations sampled, the 10k-step Vocos
v2 pilot), every TTFA was about 4× higher, but the rows compare:

| system (busy GPU) | mode | TTFA short | long sentence | paragraph | RTF paragraph |
|---|---|---|---|---|---|
| v3.1 + BigVGAN-v2-ft (`--cuda-kernel`) | `fast` | 50.5–51.1 ms | 56.2–57.4 ms | 56.8–57.0 ms | 0.034 |
| v3.1 + vocos-ft | `fast` | 30.6–30.7 ms | 32.2 ms | 30.9–31.6 ms | 0.0089 |
| v3.2, sampled durations | `stream` (eager acoustic model) | 48.3–48.7 ms | 50.3–50.5 ms | 49.7–49.9 ms | 0.013 |
| v3.2, sampled durations | `fast` | 31.5 ms | 37.5–37.6 ms | 32.6 ms | 0.010 |

- **The prosody predictor in CUDA graphs.** One pass of the drift sampler (`drift` / `mse` kinds, no word features,
  spread 1) runs inside the text encoder's graph; its noise is drawn outside the graph in the eager order, so a seed
  gives the same frame counts as the eager path, and mels equal to 77–207 dB SNR (bit-identical on most sentences;
  the remaining float differences come from attention kernels on padded buckets, as for v3.1's `fast` path on short
  sentences). It adds about 1 ms of TTFA (v3.2 5.8 ms against 4.9 ms for v3.1 + vocos-ft); with the acoustic model
  eager, v3.2 needs 14.8 ms. Before, `fast=True` with a prosody model ran the acoustic model eagerly.
- **Memory of `fast=True`.** The graphs were captured with autograd on, so each frame bucket kept its activations
  alive: about 6 GB of GPU memory for the full set. They are now captured under `no_grad`: 0.65 GB allocated with or
  without the prosody predictor.

**Reproduce:** stage the files with `scripts/prepare_release.py`, then

```bash
scripts/eval_release.sh runs/rel_publish/drifting_tts_v3.2.pt runs/rel_publish/prosody_drift_v3.2.pt \
    runs/rel_publish/vocos_v2.pt runs/rel_eval_v3.2   # Freya-100, Freya-495 x 3 voices, prosody; summary.md
```

## Freya-TR-Eval

The FreyaTTS report's benchmark (arXiv 2607.09530):
- 495 everyday Turkish sentences;
- audio downsampled to 8 kHz;
- Whisper large-v3 (beam 5);
- the same normalisation on both sides.

The model uses T = 0.3 and α = 2, both chosen on a held-out dev split. Nothing was tuned on Freya. Piper and MMS-TTS
were re-run in this harness to check that it matches the report's.

| system | parameters | WER [95% CI] | CER [95% CI] | UTMOSv2 (full band) |
|---|---|---|---|---|
| **drifting-tts v3.1, studio voice** | 67.7 M + 112.4 M vocoder | **1.23%** [0.82, 1.68] | **0.24%** [0.16, 0.33] | **2.94** |
| drifting-tts v3.1, male voice | 67.7 M + 112.4 M vocoder | 1.74% [1.24, 2.30] | 0.38% [0.25, 0.53] | 2.81 |
| drifting-tts v3.1, female voice | 67.7 M + 112.4 M vocoder | 3.02% [2.38, 3.71] | 0.70% [0.54, 0.87] | 2.75 |
| Piper (tr_TR-dfki-medium), this harness | 16 M | 3.76% [3.06, 4.47] | 0.83% | – |
| MMS-TTS (`facebook/mms-tts-tur`), this harness | 36 M | 6.26% [5.42, 7.15] | 1.43% | – |
| Piper, FreyaTTS report | 16 M | 4.4% | 1.1% | – |
| MMS-TTS, FreyaTTS report | 36 M | 6.8% | 1.7% | – |
| FreyaTTS, FreyaTTS report | 183.2 M | 8.0% | 3.0% | – |
| F5-TTS (tr), FreyaTTS report | 336 M | 24.3% | 10.9% | – |
| XTTS-v2, FreyaTTS report | 470 M | 11.1% | 3.9% | – |

The FreyaTTS report puts real human recordings (FLEURS-tr) at 9.7% WER under this pipeline. Its MOS column comes
from a listening study, so it is not comparable with UTMOSv2.

## Voices

| voice | speaker ID | median pitch | Freya WER | UTMOSv2 (Freya) | duration factor |
|---|---|---|---|---|---|
| `studio` (default) | 722 | 103 Hz | **1.23%** | **2.94** | 0.935 |
| `male` | 389 | 104 Hz | 1.74% | 2.81 | 1.150 |
| `female` | 323 | 174 Hz | 3.02% | 2.75 | 1.194 |

- **Selection:** `male` and `female` were picked by measurement among the best-covered training speakers.
  `scripts/select_voices.py` scored each candidate on 30 held-out dev sentences with UTMOSv2 and Whisper CER.
- **`studio`:** added by fine-tuning (next section).
- **All speaker IDs:** [SPEAKERS.md](SPEAKERS.md) scores every one of the 723 IDs on 50 Freya sentences.
- **Duration factors:** each voice has its own factor from `calibrate-durations`. A single global factor had made
  some voices speak 7–17% faster than their recordings.

## Adding the studio voice (v3 → v3.1)

v3 was fine-tuned on its training corpus merged with recordings of one new studio voice, using
`configs/tts_v3_add_voice.yaml`:
- 30k steps at LR 1e-4, 2.7 h on an RTX 5090;
- the new voice made up about 40% of the batches;
- the learned kernel temperature was resumed from v3.

On 100 held-out recordings of the new voice:

| system | CER | WER | speaker sim. | UTMOSv2 |
|---|---|---|---|---|
| recording | 0.88% | 2.06% | – | 3.09 |
| recording → BigVGAN-v2 | 0.94% | 2.17% | 0.979 | 3.17 |
| v3.1, studio voice | **0.44%** | **1.98%** | 0.924 | 2.87 |

The existing voices kept their quality. Their Freya WER went from 1.92% to 1.74% (male) and from 3.78% to 3.02%
(female); part of that gain comes from the per-voice duration factors.

## Latency and size

Under load, with the method in detail: [LATENCY.md](LATENCY.md). With 64 / 128 / 256 requests arriving at once, v3.2
served one request after another gives the last one its first audio after 0.81 / 1.64 / 3.19 s; one batched pass
gives every request its first audio after 71 / 150 / 304 ms.

**Setup:** RTX 5090, PyTorch 2.11, T = 0.3, α = 2, `studio` voice, fine-tuned BigVGAN-v2 with its CUDA kernel
(`--cuda-kernel`). Each value is the median of 100 runs after warm-up, measured with `scripts/bench_ttfa.py --mode
<mode>`. **TTFA** (time to first audio) runs from the input text to the first audio on the host.

| mode | short sentence | long sentence (6.0 s) | 4-sentence paragraph | first call after loading | load |
|---|---|---|---|---|---|
| `sentence`: each sentence vocoded whole | 21.4 ms | 39.5 ms | 30.9 ms | 282 ms | 2.0 s |
| `stream`: `Synthesizer.stream` | 18.4 ms | 19.2 ms | 18.9 ms | 294 ms | 1.9 s |
| **`fast`: `Synthesizer(fast=True).stream`** | **12.3 ms** | **13.9 ms** | **13.6 ms** | **39 ms** | 3.7 s |
| `fast` + `compile=True` + `tf32=True` | 10.6 ms | 11.3 ms | 11.2 ms | 35 ms | 15 s |

How the time to first audio was cut ([drifting_tts/fast.py](../drifting_tts/fast.py)):

- **Streaming vocoder.** Each sentence is still generated in one pass, but the vocoder no longer waits for the
  whole mel:
  - it first vocodes 32 frames (0.34 s), with 32 frames of right context;
  - then windows of 256 frames, with 32 frames of context on each side.

  BigVGAN-v2's receptive field is about 24 frames, so the pieces join into what vocoding the whole sentence gives,
  and TTFA no longer grows with the sentence.
- **CUDA graphs.** The acoustic model launched about 1,750 small kernels per sentence, and half of its 9.5 ms was
  launch overhead.
  - The text encoder and the DiT now run as CUDA graphs, captured once per length bucket: tokens padded to
    multiples of 32, frames to multiples of 64.
  - The padding is masked, so the mel is bit-identical to the eager model's.
  - The acoustic model now takes about 4 ms, and the first vocoder window about 7.7 ms.
- **What is left.** The first vocoder window is limited by the number of kernels in BigVGAN-v2: about 700 sequential
  kernels at ~10 µs each. fp16 overflows in BigVGAN, and `torch.compile` of BigVGAN gained 17% while changing the
  output, so neither is used.
- **Options that change the output slightly.**
  - `compile=True` fuses the DiT with `torch.compile` (one dynamic-shape compilation).
  - `tf32=True` runs its matmuls in TF32; the mel SNR is 74 dB against fp32, and the log-spectral distance of the
    audio is 0.4–0.7 dB.
  - Together they save about 2 ms.

Quality is unchanged. On all 495 Freya-TR-Eval sentences, `sentence` mode and `fast` streaming both give WER 1.25% and
CER 0.24%, with UTMOSv2 2.935 and 2.934 (`drifting-tts benchmark --fast`). Streaming costs more GPU time per
sentence, because of the window overlaps, but it stays 50–100× faster than real time.

| component | parameters |
|---|---|
| text encoder (with duration predictor) | 7.24 M |
| pitch predictor + pitch embedding | 0.40 M |
| DriftDiT generator | 60.05 M |
| **acoustic model** | **67.69 M** |
| BigVGAN-v2 vocoder (inference) | 112.41 M |
| **total at inference** | **180.10 M** |

The 2-D Mel-MAE feature encoder (4.4 M) is used only during training.

## In the browser (WebGPU)

`scripts/export_onnx.py` exports three graphs: the text encoder, the generator and the vocoder. Between them,
JavaScript repeats each token's features for its predicted duration. Each graph matches PyTorch to a relative error
below 4e-5. The whole JavaScript pipeline in `web/tts.js` was also run under onnxruntime-node with the noise of a
PyTorch run. Its text normalisation, token IDs and frame counts were identical, and the audio matched at an SNR of
80–84 dB.

The fp16 copies (`*_fp16.onnx`) store the weights in half precision and compute in fp32. BigVGAN's anti-aliasing
filters stay in fp32. The web demo loads the fp32 text encoder and the fp16 generator and vocoder: the text encoder is
small, yet it accounts for most of the rounding error.

| graphs | download | audio vs PyTorch |
|---|---|---|
| all fp32 | 729 MB | SNR 91 dB |
| web demo: fp32 text encoder, fp16-weight generator and vocoder | 385 MB | SNR 44–47 dB, log-spectral distance 0.6 dB |
| all with fp16 weights | 369 MB | SNR 38–43 dB |

**Setup:** RTX 5090, headless Chrome 153, onnxruntime-web 1.30 WebGPU, the web demo's graphs. Each value is a single
run after the warm-up pass:

| input | audio | first audio | total | acoustic model | vocoder |
|---|---|---|---|---|---|
| "Merhaba, nasılsınız? Bugün hava çok güzel." (2 sentences) | 2.9 s | 110 ms | 247 ms | 103 ms | 135 ms |
| 2-sentence train announcement (female voice) | 7.8 s | 113 ms | 348 ms | 115 ms | 231 ms |
| one sentence | 3.7 s | 97 ms | 194 ms | 69 ms | 124 ms |

The page streams like `Synthesizer.stream`: the vocoder starts with a 32-frame window, and every piece is played as
soon as it is ready. The text-encoder graph is exported without its attention mask, which is all ones for one
utterance; the mask's IsNaN guard ran on the CPU, which cost a GPU round trip in every layer. Together these changes
cut the first audio from 131–172 ms to 97–113 ms, with the GPU otherwise idle in both runs. The text encoder alone went
from 30 to 8 ms. A sentence of a length not seen before sometimes costs about 20 ms more, for kernel compilation.

- **Download:** loading the 385 MB of graphs from the Hub took about 11 s here, plus under 1 s of warm-up for
  shader compilation. Later visits load the graphs from the browser cache.
- **Intelligibility:** Whisper large-v3 transcribed all three browser outputs with 0% CER.
- **Other devices:** laptop and integrated GPUs will be slower. Without WebGPU the page falls back to WASM on the
  CPU, which is far slower than real time.

## MLX

`drifting_tts.mlx` is a port of the inference model to MLX for Apple silicon: the text encoder, the pitch and duration
predictors, DriftDiT, BigVGAN-v2 and the two small vocoders (BigVGAN-base and Vocos fine-tunes). In BigVGAN, the
anti-aliased activations are written in polyphase form and the transposed convolutions as strided phases, so nothing
is interleaved or zero-inserted. Every part was checked against PyTorch with MLX 0.32 on Linux (CPU backend):

| check | result |
|---|---|
| acoustic model, same token ids and noise (real checkpoint) | identical frame counts, mel relative error ≤ 1.4e-5 |
| BigVGAN-v2, fp32 weights (real fine-tuned checkpoint) | SNR 98.9 dB |
| BigVGAN-v2, published weights (fp16, snake parameters fp32) | SNR 59.1 dB |
| whole pipeline, published weights, same noise as a PyTorch run | identical sample count, SNR 59.9 dB |
| Whisper large-v3 on two MLX outputs | 0% CER |
| BigVGAN-base-ft, fp32, mels of the released model (3 sentences) | SNR 92.3–99.7 dB; streamed = whole-sentence MLX audio |
| Vocos-ft, fp32, mels of the released model (3 sentences) | SNR 95.0–104.4 dB; streamed = whole-sentence MLX audio |

The published weights (`mlx/` in the model repo, 496 MB) keep the acoustic model in fp32. Storing it in fp16 as well
lowered the end-to-end SNR to 31.9 dB. The older Linux CPU results above establish numerical parity; the Apple GPU
measurements below were performed separately, with BigVGAN-v2 and before the streaming prefetch; the small vocoders
have not been timed on a Mac yet ([commands](MLX.md#validating-on-a-mac)).

### Apple M2 Pro, 7 October 2026

Apple M2 Pro, 16 GB, macOS 26.5.2; Python 3.11.15, MLX/MLX Metal 0.32.3, NumPy 2.4.6. Published MLX weights at
Hugging Face revision `2de3308045f6f559b2efa2d8cca3749fa3262848`; float32 compute, studio voice, temperature 0.3,
CFG 2. Each input had two warmups followed by ten measurements (seeds 0–9). Separate processes, GPU device,
256 MiB unused allocator cache limit for **both** implementations. No playback, networking or download latency.

| input | original API TTFA p50 / p95 | streaming TTFA p50 / p95 | original total p50 | streaming total p50 |
|---|---|---|---|---|
| short sentence | 150.2 / 163.7 ms | 83.7 / 87.3 ms | 150.5 ms | 242.0 ms |
| long sentence | 701.2 / 709.3 ms | 97.1 / 102.4 ms | 701.2 ms | 991.0 ms |
| 4-sentence paragraph | 2072.5 / 2131.5 ms | 95.5 / 97.9 ms | 2072.6 ms | 3040.0 ms |

The baseline is the unmodified public MLX API at `17c85cff2a0604988f532add8a8e0d13bdae9d87`, which returns the
**complete waveform**. The updated streaming API returns **256 ms of PCM** first, then grows from 128 to 256 to a
maximum of 512 mel frames per chunk. These are caller-visible TTFA measurements; the old API has no streaming
consumer boundary. The improvement is 1.8× / 7.2× / 21.7× in first-delivery latency, not in total computation.
Repeated context increases total generation cost: streaming RTF is 0.165–0.184, versus about 0.115–0.117 for the
original buffered API. All measured warm streaming runs had zero calculated playback deficit.

The first request in the controlled processes (short input, weights already materialized) took 197.4 ms original
and 110.8 ms streaming. Weight loading was 57 ms and 142 ms respectively. These values are **process-cold**, not
machine-cold: the OS and Metal caches were already populated by previous work. An initial exploratory run before
the cache cap showed much larger variability. Full graph compilation remains opt-in because first-use and unseen
shape compilation can cost seconds; see [MLX usage and caveats](MLX.md).

With `--compile`, a separate five-run warm measurement (two warmups per input, otherwise the same settings) gave
TTFA p50 **70.8 / 79.4 / 77.3 ms** and total generation p50 **198.2 / 832.0 / 2506.9 ms** for the same three inputs.
Its TTFA p95 was 72.7 / 81.9 / 80.8 ms, with zero playback deficit. This optional configuration is suitable when
shapes are warmed before serving; it does not promise these times for unseen text lengths.

Three real checkpoint outputs (short, long and paragraph, seed 0) matched the original waveform exactly after
concatenating streamed chunks: identical sample counts and maximum absolute sample difference 0.0 on this machine.
The optional compiled mode retained identical sample counts with SNR 86.3–91.2 dB versus the original, reflecting
small float32 arithmetic differences (maximum absolute sample error 0.000186).
All three named voices were synthesized successfully in an MLX-only environment with no PyTorch installed.
This is numerical/output validation, not a new listening study or a rerun of the full Freya/Whisper benchmark.

The full Mac test suite finished with **382 passed, 5 skipped** (Ruff also passed). It covers PyTorch CPU training
smoke tests, MLX/PyTorch numerical parity, chunk boundaries,
seed stability, compilation, and public streaming behavior. Two CUDA tests require NVIDIA hardware; optional
jiwer and two uncached upstream weight tests are reported as skips. The JavaScript frontend passed 9,394 checks.
Run `python scripts/bench_mlx_ttfa.py` for raw measurements and provenance; see [MLX.md](MLX.md) for the exact API
and benchmark options.

## Spectral detail

The first version (v1) sounded muffled. Its harmonic peaks and valleys along frequency were only 32–56% as deep as
in the recordings. `evaluate --harmonic` measures this on 100 dev sentences, under the ground-truth alignment and
pitch, as a generated / recorded ratio:

| | low band | mid band (≈ 1.2–4.5 kHz) | high band |
|---|---|---|---|
| v1 | 0.56 | 0.32 | 0.44 |
| v3, T = 0.5 | 0.99 | 1.02 | 1.02 |

## Training

![training curves](training_curves_v3.png)

- **Run:** 150k steps in 13.6 h at 3.08 it/s (bf16, compiled generator). Each step uses 16 utterances × 16 one-step
  samples on 2.7 s windows.
- **Feature encoder:** the 2-D Mel-MAE was pretrained beforehand for 60k steps.
- **What to watch:** the drift loss value is constant by construction, so the curves show other signals instead:
  - the distance of the sample mean to the target;
  - the spread across samples, which is stable (no mode collapse);
  - the learned kernel temperature, which annealed from 1.0 to 0.072 (Kyutai: ≈ 0.056);
  - the harmonic contrast of the checkpoints.

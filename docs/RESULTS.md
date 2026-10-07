# Results

Detailed results of the current model, v3.1: the v3 model plus the fine-tuned `studio` voice. The
[README](../README.md) has the summary.

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
predictors, DriftDiT and BigVGAN-v2. In BigVGAN, the anti-aliased activations are written in polyphase form and the
transposed convolutions as strided phases, so nothing is interleaved or zero-inserted. Every part was checked against
PyTorch with MLX 0.32 on Linux (CPU backend):

| check | result |
|---|---|
| acoustic model, same token ids and noise (real checkpoint) | identical frame counts, mel relative error ≤ 1.4e-5 |
| BigVGAN-v2, fp32 weights (real fine-tuned checkpoint) | SNR 98.9 dB |
| BigVGAN-v2, published weights (fp16, snake parameters fp32) | SNR 59.1 dB |
| whole pipeline, published weights, same noise as a PyTorch run | identical sample count, SNR 59.9 dB |
| Whisper large-v3 on two MLX outputs | 0% CER |

The published weights (`mlx/` in the model repo, 496 MB) keep the acoustic model in fp32. Storing it in fp16 as well
lowered the end-to-end SNR to 31.9 dB. The speed on Apple silicon has not been measured. MLX's Linux CPU build ships a
reference BLAS and is far slower than PyTorch on the same CPU, so the timings measured here say nothing about Metal.

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

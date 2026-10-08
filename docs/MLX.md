# MLX inference on Apple silicon

The acoustic model and the vocoder run on the Apple GPU with MLX. Inference needs only MLX, NumPy and Hugging Face
Hub; the PyTorch training stack is unnecessary. Use Python 3.10 or newer in a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx huggingface_hub numpy
pip install --no-deps -e .
python -m drifting_tts.mlx --stream --text "Merhaba, nasılsınız?" --speaker studio --out merhaba.wav
python -m drifting_tts.mlx --stream --vocoder vocos-ft --text "Merhaba, nasılsınız?" --out merhaba_vocos.wav
```

The CLI downloads the published MLX weights once (the acoustic model and only the chosen vocoder), then writes each
audio chunk into the WAV as it becomes ready. The printed first-PCM time excludes model loading and downloading. File
output is not a speaker playback test. The voices are `studio`, `male` and `female`.

## Vocoders

Three of the registry's vocoders ([docs/VOCODERS.md](VOCODERS.md)) have MLX ports. Each is one file in the `mlx/`
folder of the model repo; the two small ones are for phones and laptops, where the vocoder is most of the time to
first audio.

| name | file in `mlx/` | parameters | stored as | streaming context | MLX vs PyTorch, real mels |
|---|---|---|---|---|---|
| `bigvgan-v2-ft` (default) | `vocoder.safetensors` | 112.4 M | float16, snake parameters float32 (225 MB) | 38 frames | 59.1 dB (98.9 dB in float32) |
| `bigvgan-base-ft` | `bigvgan_base_ft.safetensors` | 14.0 M | float32 (56 MB) | 18 frames | 92.3–99.7 dB |
| `vocos-ft` | `vocos_ft.safetensors` | 13.6 M | float32 (54 MB) | 29 frames | 95.0–104.4 dB |

```python
from drifting_tts.mlx import Synthesizer

tts = Synthesizer.from_pretrained(vocoder="bigvgan-base-ft")   # or "vocos-ft"; a .safetensors path also works
```

The parity column is the SNR of the MLX waveform against the PyTorch vocoder on the same mels of the released
acoustic model (three public sentences, MLX's Linux CPU backend; `scripts/check_mlx_parity.py`; BigVGAN-v2: the
earlier measurement in [RESULTS.md](RESULTS.md#mlx)). Streamed MLX audio was bit-identical to whole-sentence MLX
audio for both small vocoders. The listening and ASR quality of each vocoder is in [docs/VOCODERS.md](VOCODERS.md):
`bigvgan-base-ft` is close to BigVGAN-v2 (UTMOSv2 2.91 vs 2.94), `vocos-ft` is the fastest and somewhat lower
(2.63). Float16 storage would halve the small files, but Vocos then kept only 51.5 dB of parity (BigVGAN-v2 keeps
59 dB), so they are stored in float32.

- **BigVGAN-base** uses the same MLX code as BigVGAN-v2 (`drifting_tts/mlx/bigvgan.py`): 8x / 8x / 2x / 2x
  transposed convolutions as polyphase stride-1 convolutions, polyphase anti-aliased snake activations, tanh output.
- **Vocos** (`drifting_tts/mlx/vocos.py`): the ConvNeXt backbone and the `same`-padded ISTFT head of the fine-tune.
  The inverse FFT is `mx.fft.irfft`; overlap-add sums the four 256-sample blocks of each 1024-sample frame by shifting
  (no scatter), then divides by the squared periodic Hann envelope, so frame `i` is centred on sample `i·256 + 128`
  and `T` frames give `T·256` samples, as in PyTorch. The depthwise convolutions are zero-padded explicitly, because
  MLX's Metal depthwise kernel only takes unpadded inputs (speed effect not measured).
- **Files** record their architecture in the safetensors metadata (`drifting_tts.vocoder`), so a vocoder file is
  self-contained. `vocoder.safetensors` of the first release has none; its architecture is read from `config.json`.

## Streaming API

```python
import mlx.core as mx
from drifting_tts.mlx import Synthesizer

# This setting affects the allocator cache of this process, including other MLX users in it.
# It limits unused cached buffers, not total model/working memory.
mx.set_cache_limit(256 * 1024 * 1024)
tts = Synthesizer.from_pretrained()

for audio, info in tts.stream("Merhaba. Bugün hava çok güzel.", speaker="studio"):
    # audio: host-ready, mono float32 PCM at tts.sample_rate (24,000 Hz).
    # Send it directly to your playback queue or network transport here.
    print(len(audio), info["ttfa_seconds"])
```

Keep one synthesizer alive across requests. `tts(text)` remains the full-waveform API and returns only after all
sentences finish. Use `stream()` to obtain the latency improvement. `stream()` is lazy: work starts when the iterator
is advanced, and a slow consumer delays subsequent generation. Do not collect all chunks before starting playback.

The first chunk has 24 mel frames, or 256 ms of PCM. Subsequent chunks grow through 128 and 256 frames up to a
512-frame cap, allowing playback to accumulate a buffer while reducing repeated vocoder work. `first_chunk_frames`
and `chunk_frames` control the first size and maximum size. A cap of 128 keeps all later chunks at 128 frames.
`chunk_frames=None` emits complete sentences. Short final chunks are emitted without padding.

The acoustic model uses bidirectional attention and still generates the complete sentence mel before yielding audio.
The vocoders are noncausal, so each chunk is decoded with left **and** right context. The context is derived from
the architecture (the table above): the receptive field, plus for Vocos the frames whose ISTFT windows overlap the
chunk. Context samples are cropped exactly; no overlap-add, fades, duplicate samples or missing samples are
introduced. Sentence pauses are emitted separately, only between sentences. `context_frames` may increase the
context but cannot go below the derived bound.

With `prefetch=True` (the default), the next chunk of a sentence is queued with `mx.async_eval` before the current
chunk is copied to the host, so the GPU decodes it while the consumer handles the current one (the pattern of
mlx-lm's generation loop). The audio is identical either way; `prefetch=False` restores strictly one chunk per
`next()`. Its effect on Apple GPU timing **needs measurement on a Mac** (`--no-prefetch` in the benchmark).

Per-chunk metadata includes `sentence_index`, `chunk_index`, `is_silence`, `seconds`, `acoustic_seconds`,
`vocoder_seconds`, `ttfa_seconds` and `elapsed_seconds`. Acoustic time is reported on the first chunk of each
sentence; vocoder time is from the start of the current chunk until its PCM is on the host. Elapsed time includes
consumer delays. TTFA includes normalization, acoustic generation, vocoding, evaluation and host transfer, measured
from the first iterator advance.

## Compilation and memory

Full graph compilation is opt-in (`Synthesizer(..., compile=True)` or CLI `--compile`). It can improve warmed
throughput, but each new input shape can trigger compilation. An exploratory short-input run took 6.57 seconds for
its first compiled inference and about 76 ms warmed, versus about 80 ms warmed without full graph compilation.
Those exploratory runs are not a clean machine-cold comparison: Metal and OS caches persist across processes.
Do not include compilation or model downloads inside a low-latency serving request. Warm representative lengths
before accepting requests if you opt in, and measure unseen lengths separately. Compiled graphs capture weights:
construct a new synthesizer after changing weights. The polyphase activation functions are always compiled
(`shapeless=True`, elementwise only).

The CLI and benchmark cap the MLX allocator cache at 256 MiB by default (`--cache-limit-mb`). The library leaves the
caller's global cache setting unchanged. Large cache limits caused substantial latency variability on a 16 GB Mac.
This is not a 256 MiB total-memory claim; active model and activation memory is much larger.

## Precision and quantization

Float32 is the default everywhere. Two opt-in options trade parity for speed or memory; neither was checked by
listening or ASR, so validate before using them in a product. Parity against the float32 MLX model, same noise, two
public sentences, Linux CPU backend:

| option | mel SNR | waveform SNR | note |
|---|---|---|---|
| `dtype=mx.float16` (DiT only) | 69.8–70.3 dB | 37.1–54.7 dB | same frame counts |
| `dtype=mx.bfloat16` (DiT only) | 51.6–52.5 dB | 11.8–28.9 dB | same frame counts |
| `quantize=8` (DiT blocks, affine, groups of 64 / 32) | 53.9–54.1 dB | 26.0–33.9 dB | block weights ~3.5× smaller in memory |
| `quantize=4` | 29.4–29.7 dB | 6.8–8.6 dB | large deviation; not recommended without a listening check |

`dtype` applies to the DiT only. The text encoder, the duration and pitch predictors and the vocoder always run in
float32: in bfloat16 the duration predictor's rounding changed a sentence's frame count, and the snake activations
are precision sensitive. Waveform SNR is much lower than mel SNR because small phase differences add up; it is the
stricter number, and the float16 weight storage that was rejected for the published acoustic model gave 31.9 dB.
Quantization (`quantize=8` / `4`, CLI `--quantize`) replaces the Linear layers of the 12 DiT blocks with
`nn.QuantizedLinear` at load time; the files on disk stay float32. MLX can only quantize Linear and Embedding layers,
so the convolutional vocoders cannot be quantized this way. Whether float16 or quantized matmuls are faster for this
model on an Apple GPU **needs measurement on a Mac** (`--dtype`, `--quantize` in the benchmark).

## Fused Metal activation (opt-in, needs validation on a Mac)

`Synthesizer(..., fused_activations=True)` (CLI `--fused-activations`) runs each of BigVGAN's anti-aliased snake
activations as one `mx.fast.metal_kernel`: every output sample recomputes the 12 upsampled values it needs from the
input, applies the snake and the low-pass filter, and writes once, instead of materialising the 2x signal and its
padded phases. It only acts when a Metal GPU is the default device; elsewhere (and for Vocos) the polyphase MLX code
runs. The kernel's index arithmetic is checked here against the polyphase activation in NumPy
(`tests/test_mlx_bigvgan.py`); the kernel itself could not be compiled or run without a Mac, so it is off by default.
`tests/test_mlx_bigvgan.py::test_fused_activation_kernel_matches_polyphase` runs it on Apple silicon.

## Converting the weights

```bash
# the published layout: model.safetensors, config.json and vocoder.safetensors (BigVGAN-v2)
python -m drifting_tts.mlx.convert --model drifting_tts_v3.1.pt --vocoder bigvgan-v2-ft --out mlx
# the small vocoders: bigvgan_base_ft.safetensors and vocos_ft.safetensors
python -m drifting_tts.mlx.convert --vocoder bigvgan-base-ft --vocoder vocos-ft --out mlx
```

A vocoder is a registry name (the PyTorch checkpoint is read locally or from the Hub), `NAME=PATH` for a local
checkpoint, or a bare BigVGAN-v2 checkpoint path. `--fp32` stores BigVGAN-v2 in float32 too. Reconverting the
released checkpoints reproduces the published `config.json` and `model.safetensors` byte for byte and the published
vocoder tensors exactly; only the new metadata is added.

## Validating on a Mac

```bash
pip install -e ".[dev,bigvgan,mlx]"      # the PyTorch references need torch, vocos and BigVGAN's dependencies
pytest -q tests/test_mlx*.py tests/test_bench_mlx_ttfa.py   # includes the Metal-only kernel tests
python scripts/check_mlx_parity.py --vocoder bigvgan-base-ft --vocoder vocos-ft --vocoder bigvgan-v2-ft
python scripts/check_mlx_parity.py --mlx /path/to/mlx --vocoder vocos-ft --out parity.json   # converted files
```

`tests/test_mlx_vocoders.py::test_released_vocoders_match_pytorch_on_real_mels` runs when the released checkpoints
are in the Hugging Face cache. Then the latency of each configuration, each in a fresh process:

```bash
python scripts/bench_mlx_ttfa.py --runs 10 --warmup 2 --out bigvgan_v2.json
python scripts/bench_mlx_ttfa.py --vocoder bigvgan-base-ft --runs 10 --warmup 2 --out bigvgan_base.json
python scripts/bench_mlx_ttfa.py --vocoder vocos-ft --runs 10 --warmup 2 --out vocos.json
python scripts/bench_mlx_ttfa.py --no-prefetch --runs 10 --warmup 2 --out no_prefetch.json
python scripts/bench_mlx_ttfa.py --fused-activations --runs 10 --warmup 2 --out fused.json
python scripts/bench_mlx_ttfa.py --dtype float16 --runs 10 --warmup 2 --out fp16.json
python scripts/bench_mlx_ttfa.py --quantize 8 --runs 10 --warmup 2 --out q8.json
```


## Reproduce latency measurements

```bash
python scripts/bench_mlx_ttfa.py --runs 10 --warmup 2 --out latency.json
python scripts/bench_mlx_ttfa.py --mode buffered --runs 10 --warmup 2 --out buffered.json
python scripts/bench_mlx_ttfa.py --compile --runs 10 --warmup 2 --out compiled.json
```

Use `--model /path/to/mlx` for local weights or pin `--revision` to a Hugging Face commit. Each configuration should
run in a fresh process with other GPU work idle. Both the default three texts and custom repeatable `--text` inputs
are supported. The benchmark saves every measurement, input, seed, source hash, hardware/software version, memory
usage and settings. Loading/downloading are separate from the first process-cold inference. Each input receives its
own warmup before p50/p95 calculation. The timer stops only when actual NumPy PCM is available on the host.

`playback_deficit_ms` measures how much additional buffering continuous playback would have needed after the first
chunk, based on chunk arrival times and PCM durations. Zero means no measured generation underrun in that run; it
does not measure device, browser, network or operating-system audio latency. RTF measures full generation time
divided by audio duration; values below 1 are faster than real time.

## Implementation references

- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) and
  [compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html): `shapeless=True` suits elementwise
  graphs; slicing and padding break it, so the full models are compiled per shape.
- [mlx-lm's generation loop](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py): `mx.async_eval`
  of the next step before reading the current one, as in the streaming prefetch.
- [`nn.quantize`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html) and
  [memory management](https://ml-explore.github.io/mlx/build/html/python/memory_management.html).
- [MLX's Metal depthwise convolution](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/conv.cpp)
  takes the fast path only without padding.
- [mlx-audio's Vocos](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/codec/models/vocos/vocos.py) and
  [ISTFT](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/dsp.py) overlap-add by scatter; the port here
  shifts blocks instead and uses the periodic window and squared-window envelope of the PyTorch head.
- Reviewed [mlx-audio at commit 70f4add](https://github.com/Blaizzy/mlx-audio/tree/70f4add32911bab6f869b824864ad9f1e24dcb97)
  for the streaming design: [small first chunks and context cropping in Moss TTS](https://github.com/Blaizzy/mlx-audio/blob/70f4add32911bab6f869b824864ad9f1e24dcb97/mlx_audio/tts/models/moss_tts/moss_tts.py),
  [hot-path compilation in VoxCPM2](https://github.com/Blaizzy/mlx-audio/blob/70f4add32911bab6f869b824864ad9f1e24dcb97/mlx_audio/tts/models/voxcpm2/voxcpm2.py),
  [decoder compilation in Qwen3-TTS](https://github.com/Blaizzy/mlx-audio/blob/70f4add32911bab6f869b824864ad9f1e24dcb97/mlx_audio/tts/models/qwen3_tts/qwen3_tts.py).
  mlx-audio's causal decoder caches cannot be substituted for these noncausal vocoders.

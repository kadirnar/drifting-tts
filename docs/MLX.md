# MLX inference on Apple silicon

The acoustic model and BigVGAN run on the Apple GPU with MLX. Inference needs only MLX, NumPy and Hugging Face
Hub; the PyTorch training stack is unnecessary. Use Python 3.10 or newer in a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx huggingface_hub numpy
pip install --no-deps -e .
python -m drifting_tts.mlx --stream --text "Merhaba, nasılsınız?" --speaker studio --out merhaba.wav
```

The CLI downloads the published MLX weights once, then writes each audio chunk into the WAV as it becomes ready.
The printed first-PCM time excludes model loading and downloading. File output is not a speaker playback test.
The voices are `studio`, `male` and `female`.

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
BigVGAN uses noncausal convolutions, so each chunk is decoded with left **and** right context. The context is derived
from the architecture: 38 mel frames in the released model. Context samples are cropped exactly; no overlap-add,
fades, duplicate samples or missing samples are introduced. Sentence pauses are emitted separately, only between
sentences. `context_frames` may increase the context but cannot go below the derived bound.

Per-chunk metadata includes `sentence_index`, `chunk_index`, `is_silence`, `seconds`, `acoustic_seconds`,
`vocoder_seconds`, `ttfa_seconds` and `elapsed_seconds`. Acoustic time is reported on the first chunk of each
sentence; vocoder time is for the current chunk. Elapsed time includes consumer delays. TTFA includes normalization,
acoustic generation, vocoding, evaluation and host transfer, measured from the first iterator advance.

## Compilation and memory

Full graph compilation is opt-in (`Synthesizer(..., compile=True)` or CLI `--compile`). It can improve warmed
throughput, but each new input shape can trigger compilation. An exploratory short-input run took 6.57 seconds for
its first compiled inference and about 76 ms warmed, versus about 80 ms warmed without full graph compilation.
Those exploratory runs are not a clean machine-cold comparison: Metal and OS caches persist across processes.
Do not include compilation or model downloads inside a low-latency serving request. Warm representative lengths
before accepting requests if you opt in, and measure unseen lengths separately.

Compiled graphs capture weights. Treat a compiled synthesizer's model and vocoder as immutable; construct a new
synthesizer after changing weights. Existing fused polyphase activation functions remain compiled in both modes.
Float32 computation remains the default, including the numerically sensitive Snake activations. No weight
quantization or reduced-precision acoustic computation was used for the measurements.

The CLI and benchmark cap the MLX allocator cache at 256 MiB by default (`--cache-limit-mb`). The library leaves the
caller's global cache setting unchanged. Large cache limits caused substantial latency variability on this 16 GB
Mac, so the controlled comparison applies the same limit to the original and updated implementations. This is not
a 256 MiB total-memory claim; active model and activation memory is much larger.

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

Reviewed [mlx-audio at commit 70f4add](https://github.com/Blaizzy/mlx-audio/tree/70f4add32911bab6f869b824864ad9f1e24dcb97).
Useful patterns were [small first chunks and context cropping in Moss TTS](https://github.com/Blaizzy/mlx-audio/blob/70f4add32911bab6f869b824864ad9f1e24dcb97/mlx_audio/tts/models/moss_tts/moss_tts.py),
[hot-path compilation in VoxCPM2](https://github.com/Blaizzy/mlx-audio/blob/70f4add32911bab6f869b824864ad9f1e24dcb97/mlx_audio/tts/models/voxcpm2/voxcpm2.py),
and [decoder compilation in Qwen3-TTS](https://github.com/Blaizzy/mlx-audio/blob/70f4add32911bab6f869b824864ad9f1e24dcb97/mlx_audio/tts/models/qwen3_tts/qwen3_tts.py).
Drifting keeps its existing polyphase BigVGAN implementation and caches its transformed weights and Snake
coefficients after loading. mlx-audio's causal decoder caches cannot be substituted for this noncausal vocoder.

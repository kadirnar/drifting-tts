# Native MLX engine

The `DriftingTTS` Swift package loads the existing Turkish checkpoint without conversion. It includes the
acoustic model, Float32 BigVGAN, a local JavaScriptCore Turkish frontend, verified offline model storage and an
async streaming API. Build with Swift 6.3 / Xcode 26.4 or later. MLX Swift is pinned to 0.32.3.

```swift
import DriftingTTS

let directory = try await ModelStore().prepare()
let synth = try DriftingSynthesizer(modelDirectory: directory)
let metrics = try await synth.synthesize("Merhaba, nasılsınız?") { chunk in
    // Consume chunk.samples: mono Float32 PCM at chunk.sampleRate.
    // Awaiting this callback provides backpressure.
}
print(metrics.ttfaSeconds as Any)
```

Use the [iPhone application](../../ios/README.md) for streaming playback. In an application, initialize the
synthesizer outside the main actor; cancel the request when entering the background. The engine checks
cancellation between bounded vocoder stages, while Metal operations already submitted must finish.

## Tests and CLI

From this directory, Xcode compiles the required Metal library and executes CPU parity tests:

```bash
xcodebuild test -scheme DriftingTTS-Package -destination 'platform=macOS,arch=arm64'
```

The test fixtures compare encoder, pitch conditioning, generator, vocoder, context cropping, seeded RNG and
end-to-end streaming against deterministic Python MLX references. Lifecycle checks exercise invalid options,
cancellation and reuse. Regenerate fixtures from the repository root with `python scripts/make_swift_fixtures.py`.
The [validated CI run](https://github.com/kadirnar/drifting-tts/actions/runs/37675907933) passed the Xcode tests and
the unsigned iOS Release build on 2026-10-07. Local CPU and Metal GPU runs passed all 20 verification metrics.

The `drifting-tts-swift` executable supports:

```bash
drifting-tts-swift --model /path/to/mlx --text 'Merhaba, nasılsınız?' --runs 6 --out speech.wav
drifting-tts-swift --verify /path/to/Tests/DriftingTTSTests/Fixtures --cpu
```

Plain `swift build -c release` compiles the CLI but does not compile Metal shaders. For development with only
Command Line Tools, pass `--metallib /absolute/path/to/mlx.metallib` from a matching MLX 0.32.3 installation.
Even CPU validation initializes MLX's library loader. This workaround validates macOS; iOS requires Xcode.

## Measured native Mac behavior

On Apple M2 Pro / 16 GB, macOS 26.5.2, Swift 6.3.3, using Float32, 24 first frames, 128 later frames and a 32 MiB
allocator cache, five warm requests gave:

| Input | First PCM, median | Total compute/host collection, median | MLX peak active memory |
|---|---:|---:|---:|
| Short sentence, 1.31 s audio | 109.8 ms | 299.1 ms | 1.21 GB |
| Two sentences, 6.73 s audio | 117.0 ms | 1647.0 ms | 1.51 GB |

The first-ever shader-cache request took 734.5 ms to first PCM. Later fresh processes benefited from OS caches;
loading itself was approximately 0.1 s with warm filesystem caches. These measurements exclude download and
speaker playback latency. The app's total time additionally includes playback backpressure. MLX peak memory is
the process-wide active-allocation high-water mark, including model loading, and excludes cache and app overhead.
The [raw results](../../docs/benchmarks/swift-native-m2-pro.json) include exact text, checkpoint revision and runs.

The exported PCM16 waveforms have identical sample counts to Python, with 73.8 / 74.7 dB SNR and maximum absolute
differences below 0.00009, including PCM16 quantization. No weights were quantized. Tiny deterministic fixtures
also passed separately on CPU and Metal GPU. These Mac results do **not** establish iPhone latency or memory fit.

## Phone constraints

- The default chunks bound the steady vocoder input to 204 frames including both 38-frame context margins.
- Sentences split at 120 characters; a 1024 predicted-frame guard bounds acoustic inference. Extreme length
  scaling can still exceed that guard and returns an error asking for shorter sentences.
- MLX cache policy is process-wide. The default 32 MiB controls unused buffers, not total memory.
- One request runs per synthesizer. Cancellation and callback errors leave the instance reusable.
- Simulator inference is rejected. Use a physical device; the app targets iOS 17+.

The design follows [MLX Swift's iOS guidance](https://github.com/ml-explore/mlx-swift/blob/0.32.3/Source/MLX/Documentation.docc/Articles/running-on-ios.md)
and the streaming patterns in [mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift). Its existing VyvoTTS
implementation is a different Qwen-based model, so this package directly ports the Turkish Drifting architecture.

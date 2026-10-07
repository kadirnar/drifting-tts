# Drifting TTS on iPhone

A native SwiftUI application backed by the local `swift/DriftingTTS` package. The acoustic model and BigVGAN run
on the phone through MLX Swift; no Python process, web view or synthesis server is involved. It targets iOS 17+
and starts with memory-conscious settings for iPhone 14 Pro.

## Build and run

1. Open `ios/DriftingTTSApp.xcodeproj` in **Xcode 26.4 or newer**, with its **Swift 6.3** toolchain.
   The checked-in project already references the two local Swift packages. Its workspace `Package.resolved` pins
   MLX Swift to **0.32.3** and Apple's Swift Numerics to **1.1.1**, including their exact commit revisions.
2. Select the `DriftingTTSApp` target, choose your development Team in **Signing & Capabilities**, and change the
   bundle identifier if your team needs a different one.
3. Connect the iPhone, enable Developer Mode if prompted, and select that **physical device** as the destination.
   Build and run the `DriftingTTSApp` scheme. For measurements, use **Edit Scheme → Run → Release**, disable
   **Debug executable**, and keep the app in the foreground.
4. Tap **Modeli hazırla · 496 MB**. After the download, tap **Seslendir**. To check offline operation, turn on
   Airplane Mode after downloading and synthesize another sentence.

To regenerate the checked-in project after editing its specification:

```bash
xcodegen generate --spec ios/project.yml
```

The iOS simulator is suitable for UI work but cannot execute MLX Metal inference. The app rejects model
preparation there before starting a download. A supported Apple silicon Mac running the **Designed for iPad**
destination is another development option. See the [MLX Swift iOS guide](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Documentation.docc/Articles/running-on-ios.md).

## Model storage

`ModelStore` downloads exactly these files from the public `Vyvo/drifting-tts-tr` model, pinned to revision
`2de3308045f6f559b2efa2d8cca3749fa3262848`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 1,376 | `e3a4e05e8354770e45282a26aab84bd7421b7313214498ba71cee7a3935f2cb4` |
| `model.safetensors` | 270,789,403 | `f462bd0647bd6b75773968c7dd243f45882b67383233c64fa61c7dc7cd28d32f` |
| `vocoder.safetensors` | 224,982,354 | `c1dc7b0b731b9ca84b571072cdbf4e68131bc5d199a9dce93d726d55af20c70f` |

URLSession streams downloads to temporary files. Verification reads bounded blocks, then publishes the complete
directory under Application Support. Incomplete downloads are removed on cancellation or failure. A valid cache
is checked locally before any network request. The model directory is excluded from device backups. Text and
generated audio are never uploaded.

## Playback and latency

The synthesizer is an actor; MLX arrays stay on its executor and only `[Float]` PCM crosses to the UI. The
`AVAudioPlayerNode` queues 24 kHz mono buffers through `AVAudioEngine`'s mixer, which handles the hardware sample
rate. It uses the vocoder's context-and-trim output directly, with no extra fades between chunks.

The app starts with a 24-frame first chunk, a 128-frame maximum chunk, and the runtime's 32 MiB MLX buffer cache.
The async chunk callback provides backpressure, keeping queued PCM near two seconds. Completion waits for
`.dataPlayedBack`, so the last chunk is not truncated. Cancellation stops audio immediately and asks inference to
stop between GPU operations; an operation already submitted to Metal must finish. Leaving the foreground,
audio-session interruption and memory pressure also stop playback. A memory warning releases the loaded model
when the active operation has ended, while retaining downloaded files.

The app reports **request-to-first-host-PCM TTFA** and **request-to-first-buffer-scheduled** separately. Neither is
an acoustic measurement at the speaker. Model loading is done by the preparation action and is outside these
figures. The total streaming time includes callback backpressure; use the Swift CLI for compute-only throughput.
**MLX tepe · oturum** is the process-wide high-water mark of active MLX allocations, including model loading and
earlier requests in that process. It excludes reusable cached allocations and other application/OS memory, and is
not reset for each request.

## Memory and iPhone validation

The native Swift implementation was run with the production checkpoint on an **M2 Pro Mac**. With five warm
requests after the first request, median host-PCM TTFA was **109.8 ms** for a short sentence and **117.0 ms** for a
long sentence. The process-wide MLX active-memory peaks were **1.205 GB** and **1.508 GB**, respectively, excluding
the allocator cache and other app memory. Loading already cached model files took **0.105–0.114 seconds** with a
warm OS file cache. The first-ever observed process/kernel run took about **734 ms** to first PCM; cold behavior
must be measured separately from warm repetitions.

These are **Mac native-runtime results**, not iPhone 14 Pro results. They do not establish memory fit or TTFA on
the phone. The model's download size also does not represent peak inference memory. iOS app memory limits vary
over time and are distinct from
the phone's physical RAM; Apple's [`os_proc_available_memory`](https://developer.apple.com/documentation/os/os_proc_available_memory)
describes how to inspect available app headroom.

`DriftingTTSApp/OptionalMemory.entitlements` contains Apple's **Increased Memory Limit** entitlement. It is
deliberately not enabled by default: if your development profile supports it, add the capability in Xcode or set
the target's `CODE_SIGN_ENTITLEMENTS` to this file. It does not guarantee a particular memory allowance on every
device. See [Apple's entitlement documentation](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit).

For device validation, record cold preparation, first request and at least ten warm requests with a short sentence,
a long sentence and a paragraph; compare TTFA, peak memory and continuous playback. Also check cancellation,
interruption, backgrounding, Airplane Mode after download, and repeated requests after a memory warning. Use the
actual iPhone's results when choosing a larger chunk size or cache.

## Validation available in this workspace

The [Apple native CI run](https://github.com/kadirnar/drifting-tts/actions/runs/37675907933) passed on 2026-10-07,
using Xcode 26.4.1 on a macOS 26 ARM64 runner. It compiled the complete app in **Release for generic iOS** with
signing disabled, preserved dependency pins, and passed the native CPU parity/lifecycle, Core and JavaScript
frontend tests. CI runs the iOS build and native tests independently.

On the local M2 Pro, all 20 native numerical/lifecycle checks passed on both CPU and Metal GPU. Real model
synthesis passed for all three voices; the short and two-sentence waveform comparisons against Python are
documented in the [native engine results](../swift/DriftingTTS/README.md). `ModelStore` verified the real cached
checkpoint's sizes/hashes twice and respected cancellation.

**Signing, installation, playback and TTFA on a physical iPhone 14 Pro remain unverified.** This local Mac has only
Command Line Tools, so the iOS build was validated remotely. A full Xcode installation, development Team and
connected physical iPhone are needed for the remaining device checks. An unsigned CI build cannot be installed
directly on the phone.

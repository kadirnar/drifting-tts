# Audio latent spaces

The TTS model generates BigVGAN-v2 log-mels. `drifting_tts.latents` puts three pretrained audio VAEs (DAC-VAE,
VoxCPM2 and VoxCPM1.5) behind the same encode / decode interface as the mels, so that the model can generate their
latents instead (issue #17). This page
measures the quality ceiling of each latent space: held-out recordings are encoded and decoded, with no TTS model
involved.

## Backends

| name | weights (licence) | input | output | frames | dim | samples / frame (in → out) | decoder |
|---|---|---|---|---|---|---|---|
| `bigvgan` | BigVGAN-v2 log-mel + vocoder (MIT) | 24 kHz | 24 kHz | 93.75 Hz | 100 | 256 → 256 | non-causal |
| `dacvae` | [`facebook/dacvae-watermarked`](https://huggingface.co/facebook/dacvae-watermarked) (Apache-2.0) | 48 kHz | 48 kHz | 25 Hz | 128 | 1920 → 1920 | non-causal, watermarked |
| `voxcpm2` | [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) AudioVAE (Apache-2.0) | 16 kHz | 48 kHz | 25 Hz | 64 | 640 → 1920 | causal |
| `voxcpm1.5` | [`openbmb/VoxCPM1.5`](https://huggingface.co/openbmb/VoxCPM1.5) AudioVAE (Apache-2.0) | 44.1 kHz | 44.1 kHz | 25 Hz | 64 | 1764 → 1764 | causal |

```python
from drifting_tts.latents import LatentStats, load_backend, stream_decode

be = load_backend("voxcpm2", "cuda")       # bigvgan (finetuned=...), dacvae, voxcpm2, voxcpm1.5
z = be.encode(wav, sr)                     # [B, 64, T]: the posterior mean; resamples to be.input_rate
wav48 = be.decode(z)                       # [B, T * be.hop_out] at be.output_rate
for piece in stream_decode(be, z, first=8, chunk=64, context=16):
    ...                                    # windowed decoding (see "Speed and streaming")
```

- **Frames and samples:** `encode` resamples `S` samples at `sr` to `n = ceil(S · input_rate / sr)` samples and
  returns `T = ceil(n / hop_in)` frames; the VAEs right-pad the input to whole frames. The mel returns `n // 256`
  frames, as `prepare` stores them. `decode` returns exactly `T · hop_out` samples.
- **Alignment:** frame `t` covers output samples `[t · hop_out, (t + 1) · hop_out)`, the same time span as input
  samples `[t · hop_in, (t + 1) · hop_in)`. The decoded audio has no delay (the cross-correlation peak with the
  recording is at 0 samples for every VAE), so cutting it to `ceil(S · output_rate / sr)` samples aligns it with
  the input.
- **Posterior mean:** `encode` returns the mean; `posterior` also returns the standard deviation. The upstream
  `DACVAE.encode` returns a *sample*.
- **Normalisation:** `LatentStats` holds per-channel means and standard deviations (`normalize`, `denormalize`).

## Implementation notes

- **Vendored model code.** Both VAEs are minimal inference ports of the upstream code (licence headers kept), loaded
  from the original checkpoints with `torch.load(weights_only=True)` and weight norm folded into the weights.
  - The `dacvae` package depends on `descript-audiotools`, which pins `protobuf<3.20`.
  - The `voxcpm` package pulls in about 70 packages (its whole TTS stack, Gradio, FunASR, ModelScope).
  - The ports match the upstream modules within float precision: latents above 100 dB SNR, decoded audio above
    117 dB, in fp32 without TF32.
- **Extras:** `pip install "drifting-tts[dacvae]"` / `"drifting-tts[voxcpm]"` only add `huggingface_hub`. Only the
  VAE weights are downloaded (DAC-VAE 431 MB; `audiovae.pth` + `config.json`: 377 MB for VoxCPM2, 346 MB for
  VoxCPM1.5), never the VoxCPM language models. The Hub revisions are pinned.
- **DAC-VAE watermark.** The decoder adds an AudioSeal-style watermark through a 150 Hz branch with two LSTMs.
  Upstream draws a random 16-bit message for every call. The backend fixes the message (all zeros, or `message=`), so
  decoding is deterministic; the watermark stays in. Two different messages change the output by about −67 dB.
- **VoxCPM2's 48 kHz decoder.** Before each of its six upsampling blocks, the decoder scales and shifts its features
  with per-channel embeddings of a sample-rate bucket, `bucketize(rate, [20000, 30000, 40000])`. The bucket sets the
  bandwidth that the decoder generates (the output is 48 kHz in every case). VoxCPM2 always decodes with 48000, the
  full-band bucket, and so does the backend (`target_rate=` changes it). Median bandwidth of the output on 20
  held-out recordings (highest frequency within 50 dB of the spectral peak; the encoder sees 16 kHz audio, so
  everything above 8 kHz is generated):

  | `target_rate` | 16000 | 22050, 24000 | 32000 | 44100, 48000 |
  |---|---|---|---|---|
  | bucket | 0 | 1 | 2 | 3 |
  | output bandwidth | 8.6 kHz | 11.0 kHz | 14.7 kHz | 15.0 kHz |

## Resynthesis benchmark

```bash
python scripts/resynthesis_benchmark.py --data data/train --vocoder runs/vocoder_v3/bigvgan_ft.pt
```

- **Audio:** 100 held-out recordings (the first 100 of the `val` split). `prepare` stores audio at 24 kHz, so the
  44.1 / 48 kHz encoders see audio band-limited to 12 kHz, and VoxCPM2's encoder downsamples it to 16 kHz.
- **Rows:** the recording itself; BigVGAN-v2 (our fine-tuned `bigvgan_ft.pt`) on the recording's mel; each VAE,
  encoding the posterior mean and decoding it at its own output rate.
- **Intelligibility:** Whisper large-v3 (`score.AsrScorer`, Turkish, beam 5) against the transcripts, with the
  normaliser of `evaluate`. WER / CER are corpus-level with 95% bootstrap intervals, on audio band-matched to
  8 kHz (as in `benchmark`) and on the full band.
- **Naturalness:** UTMOSv2 and DNSMOS P.835, both at 16 kHz.
- **Speaker similarity:** WavLM-Large + ECAPA-TDNN cosine with the original recording.
- **Speed:** RTX 5090, batch 1, fp32 with PyTorch's defaults (TF32 convolutions), eager, after warm-up. The GPU was
  shared with other jobs running at about 90% utilisation. A repeat run moved the timings by up to 2× (first
  window: VoxCPM2 4.0–8.3 ms, DAC-VAE 12.5–19 ms, BigVGAN 28–40 ms), but never changed the ranking. BigVGAN runs
  here without its fused CUDA activation and without the CUDA graphs of `drifting_tts.fast`.

### Quality

| system | in → out | WER 8 kHz [95% CI] | CER 8 kHz [95% CI] | WER full band | CER full band | UTMOSv2 [95% CI] | DNSMOS OVRL / SIG / BAK | speaker sim. | bandwidth |
|---|---|---|---|---|---|---|---|---|---|
| recording | 24 → 24 kHz | 6.56% [4.80, 8.58] | 2.18% [1.38, 3.28] | 6.29% | 2.07% | **2.953** [2.867, 3.037] | 3.268 / 3.569 / 4.015 | – | 11.9 kHz |
| BigVGAN-v2 (fine-tuned) | 24 → 24 kHz | 6.77% [4.95, 8.89] | 2.24% [1.43, 3.34] | 6.70% | 2.20% | **2.922** [2.841, 3.002] | 3.305 / 3.600 / 4.029 | 0.966 | 11.8 kHz |
| DAC-VAE | 48 → 48 kHz | 6.90% [5.04, 8.97] | 2.17% [1.38, 3.25] | 6.70% | 2.20% | 2.802 [2.721, 2.881] | 3.288 / 3.577 / 4.040 | **0.983** | 11.4 kHz |
| VoxCPM2 AudioVAE | 16 → 48 kHz | 7.10% [5.25, 9.22] | 2.17% [1.40, 3.26] | 6.29% | 2.10% | 2.795 [2.709, 2.879] | 3.262 / 3.551 / 4.027 | 0.962 | 15.0 kHz |
| VoxCPM1.5 AudioVAE | 44.1 → 44.1 kHz | 7.71% [5.77, 9.99] | 2.52% [1.68, 3.68] | 6.70% | 2.20% | 2.682 [2.609, 2.752] | 3.262 / 3.549 / 4.037 | 0.966 | 11.1 kHz |

- **Intelligibility:** every latent space keeps it. All WER / CER intervals overlap the recording's.
- **Naturalness:** UTMOSv2 ranks BigVGAN (−0.03 from the recording) above DAC-VAE and VoxCPM2 (−0.15 each) and
  VoxCPM1.5 (−0.27). DNSMOS sees no difference between any of them.
- **Speaker similarity:** DAC-VAE keeps the most of the voice (0.983). VoxCPM2 keeps the least (0.962); its
  encoder only hears the band below 8 kHz.
- **Bandwidth:** VoxCPM2 extends the 12 kHz recordings to 15 kHz; the others reproduce the input band.

### Frames

| system | frame rate | dim | samples / frame (in → out) | delay | active channels | channel means | channel stds | posterior std / channel std |
|---|---|---|---|---|---|---|---|---|
| BigVGAN mel | 93.75 Hz | 100 | 256 → 256 | – | 100 | −8.29 … −2.79 | 1.64 … 2.25 | – |
| DAC-VAE | 25 Hz | 128 | 1920 → 1920 | 0 samples | 128 | −0.83 … 0.77 | 0.66 … 1.03 | 0.005 |
| VoxCPM2 | 25 Hz | 64 | 640 → 1920 | 0 samples | 64 | −0.53 … 1.10 | 1.07 … 1.78 | 0.009 |
| VoxCPM1.5 | 25 Hz | 64 | 1764 → 1764 | 0 samples | 58 | −0.59 … 0.54 | 0.00 … 1.36 | 0.025 |

- **Delay:** the cross-correlation peak of the decoded audio with the recording, on 20 utterances. BigVGAN
  generates its own phase, so its peak is not a delay.
- **Unused channels:** 6 of VoxCPM1.5's 64 channels have collapsed to the prior. Their means are constant
  (standard deviation below 0.001) and their posterior std is 1. Per-channel normalisation (#17) must leave them
  out or clamp their scale.
- **Posterior noise:** the median ratio of the posterior std to the spread of the channel's means. The latents are
  nearly deterministic, so the posterior mean is the natural training target.
- **Scale:** DAC-VAE is close to unit scale per channel; VoxCPM2's channel stds span 1.07–1.78, so it needs the
  per-channel statistics of `LatentStats`.

### Speed and streaming

| system | encode RTF | decode RTF | first window | its decode time | context for > 55 dB | default precision vs fp32 | peak GPU memory |
|---|---|---|---|---|---|---|---|
| BigVGAN | 0.0007 | 0.0182 | 320 ms (30 frames + 23 right context) | 37.7 ms | 240 ms (23 frames each side) | 45.3 dB | 702 MB |
| DAC-VAE | 0.0085 | 0.0236 | 320 ms (8 frames + 4 right context) | 18.7 ms | 160 ms (4 frames each side) | 70.1 dB | 1670 MB |
| VoxCPM2 | 0.0058 | **0.0056** | 320 ms (8 frames) | **4.0 ms** | 480 ms (12 frames, left only) | 65.2 dB | 831 MB |
| VoxCPM1.5 | 0.0090 | 0.0082 | 320 ms (8 frames) | 6.0 ms | 320 ms (8 frames, left only) | 66.0 dB | 1106 MB |

`stream_decode` decodes a first window of 320 ms, then 2.56 s windows. Each window gets `context` frames of left
context, and also of right context unless the decoder is causal. Only the window's own samples are kept.

The table below gives the SNR of the result against decoding the whole utterance, on 20 utterances. It is measured
in fp32 without TF32, so it isolates the receptive field; the number of frames is in parentheses.

| context | BigVGAN | DAC-VAE | VoxCPM2 | VoxCPM1.5 |
|---|---|---|---|---|
| 0 ms | 11.5 dB (0) | 25.8 dB (0) | 21.1 dB (0) | 21.7 dB (0) |
| 40 ms | 15.2 dB (4) | 40.6 dB (1) | 27.3 dB (1) | 27.5 dB (1) |
| 80 ms | 19.5 dB (8) | 49.1 dB (2) | 30.6 dB (2) | 31.9 dB (2) |
| 160 ms | 32.1 dB (15) | 63.5 dB (4) | 36.3 dB (4) | 40.1 dB (4) |
| 240 ms | 68.2 dB (23) | 85.2 dB (6) | 42.4 dB (6) | 45.5 dB (6) |
| 320 ms | 97.0 dB (30) | 98.4 dB (8) | 54.9 dB (8) | 55.4 dB (8) |
| 480 ms | 95.8 dB (45) | 101.9 dB (12) | 80.3 dB (12) | 78.3 dB (12) |
| 640 ms | 96.9 dB (60) | 102.0 dB (16) | 115.7 dB (16) | 112.6 dB (16) |
| 960 ms | 97.1 dB (90) | 104.3 dB (24) | exact (24) | exact (24) |

- **VoxCPM decoders** are causal convolution stacks, so they need no right context. Their first window needs no
  context at all: it sees the same zero padding as the whole utterance. Later windows match at 12 frames of left
  context (80 dB) and are identical from 24 frames on.
- **DAC-VAE** is non-causal and needs 4 frames (160 ms) on each side for 63 dB. Its watermark LSTMs have unbounded
  memory, but the watermark is about 67 dB below the signal, so that does not limit the match.
- **BigVGAN** needs about 23 mel frames on each side, in line with the receptive field that `stream_vocoder` assumes.
  With TF32 convolutions its output here was only 45 dB from fp32, and its windows matched to about 55 dB. The VAEs
  stay 65–70 dB from fp32 under TF32.

## Which latent space for the TTS pilot (#17)

**First VoxCPM2, then DAC-VAE.** Both keep intelligibility and lose the same naturalness at the ceiling
(UTMOSv2 2.80 vs. 2.95 for the recording and 2.92 for BigVGAN). The pilots have to win that back by being easier
to model than mels.

VoxCPM2 goes first:
- **Smaller target.** It has 64 channels against DAC-VAE's 128, for the generator and the latent MAE alike.
- **Known to be generatable.** VoxCPM2 is itself a TTS model that generates these latents.
- **Fastest decoder.** It decodes at RTF 0.006, about 4× faster than DAC-VAE.
- **Causal, so streaming is free.** The first 320 ms decode in about 4 ms with no lookahead, and windows need only
  left context.
- **48 kHz output.**

Its costs:
- the lowest speaker similarity of the VAEs (0.962);
- input heard at 16 kHz, so the 8–12 kHz band of the recordings is regenerated rather than encoded;
- channel scales that need per-channel normalisation.

DAC-VAE is the second pilot:
- **Better at the ceiling.** It has the best speaker similarity (0.983) and keeps the input band.
- **Its costs:** 128 channels, a decoder 4× slower, 4 frames of right context when streaming, and a watermark in
  every output.

VoxCPM1.5 is dominated: it has the lowest UTMOSv2 (2.68), the highest band-matched WER and 6 unused channels.

## TTS pilots (#17)

Three pilots trained the same v3 recipe for 10k steps on the same filtered training set:
- **VoxCPM2 latents** (`configs/tts_latent.yaml`).
- **DAC-VAE latents** (the same config with the DAC-VAE root and its MAE).
- **Mels** (`configs/tts_v3.yaml`), the reference.

The latent pilots differ from the mel pilot in three ways:
- **Frames.** `extract-latents` repeats every 25 Hz latent frame 4 times, so monotonic alignment has a frame per
  token (characters with blanks come at ~29 per second). The generator uses patch 4, so it still sees one token per
  latent frame.
- **Kernel features.** These come from a 1-D latent MAE (`configs/mae_latent.yaml`, 20k steps) instead of the 2-D
  Mel-MAE.
- **Decoding.** `LatentVocoder` averages the repeats back and decodes with the released VAE decoder, which is not
  fine-tuned. The mel pilot uses the fine-tuned BigVGAN-v2.

Evaluation: `drifting-tts benchmark` on the first 100 Freya-TR-Eval sentences, speaker 722, T = 0.3, α = 2.
- WER / CER: Whisper large-v3 on 8 kHz band-matched audio.
- UTMOSv2.
- RTF: the acoustic model plus the decoder or vocoder.

| pilot (10k steps) | WER | CER | sentences with errors | UTMOSv2 | RTF | training speed |
|---|---|---|---|---|---|---|
| **VoxCPM2 latents** | **4.94%** | **1.39%** | 29 / 100 | 2.052 | **0.0054** | 7.6 it/s |
| DAC-VAE latents | 9.55% | 2.91% | 46 / 100 | 1.837 | 0.0239 | 3–4 it/s (GPU shared) |
| mels | 18.77% | 5.26% | 78 / 100 | **2.746** | 0.0158 | 3.0 it/s |

- **Latents learn to speak much faster.** At the same step count, the VoxCPM2 pilot makes a quarter of the mel
  pilot's word errors. The mel pilot's errors are spread over most sentences, not concentrated in a few failures.
- **Naturalness lags.** UTMOSv2 is 2.05 for VoxCPM2 and 1.84 for DAC-VAE, against 2.75 for mels. The decoders'
  resynthesis ceiling explains only part of this: about 2.80 for both VAEs, 2.92 for BigVGAN. Most of the gap is in
  the generated latents. The mel pilot also gains from a vocoder fine-tuned on generated mels, which the VAE decoders
  are not.
- **VoxCPM2 beats DAC-VAE on every column**, although DAC-VAE has the better ceiling: 64 channels are easier to
  generate than 128. The DAC-VAE decoder is also slower (RTF 0.0239 against 0.0054).
- **VoxCPM2 is the faster model.** With patch 4 on 100 Hz frames, the generator sees 25 tokens per second of audio,
  against 47 for mels with patch 2. It trains 2.5× and synthesises 3× faster.

Next steps:
- A longer VoxCPM2 run, to see whether naturalness catches up while the intelligibility lead holds. The released
  mel model was trained far longer than these pilots.
- Fine-tuning the VoxCPM2 decoder on generated latents, the latent counterpart of the GTA vocoder fine-tune.

```bash
drifting-tts extract-latents --data data/train --backend voxcpm2 --out data/train_voxcpm2   # --repeat 4
drifting-tts train-mae --config configs/mae_latent.yaml --workdir runs/mae_voxcpm2 data.root=data/train_voxcpm2
drifting-tts train --config configs/tts_latent.yaml --workdir runs/tts_voxcpm2 train.steps=10000
# DAC-VAE: --backend dacvae, and model.n_mels=128 for the MAE
```

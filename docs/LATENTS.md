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
  Fine-tuning the decoder keeps the watermark frozen and in place (see
  [below](#fine-tuning-the-dac-vae-decoder-on-generated-latents-32)).
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

### Longer training and the released model

The VoxCPM2 pilot was continued from 10k to 50k steps, at the same constant learning rate after warm-up. It was
then evaluated on the same 100 sentences as above, next to the released v3.1 (mels + `bigvgan-v2-ft`):

| model | WER | CER | UTMOSv2 |
|---|---|---|---|
| v3.1, released | **0.66%** | **0.14%** | **2.934** |
| VoxCPM2 latents, 10k steps | 4.94% | 1.39% | 2.052 |
| VoxCPM2 latents, 50k steps | 7.57% | 2.50% | 1.904 |

- **Longer training made the latent model worse.** On four validation sentences sampled every 10k steps, the CER
  stays at 10–15% with no downward trend.
- **The latent objective plateaus early.** The mel model of the same recipe was trained far longer and reached
  0.66% WER, so the plateau is specific to the latent target.
- The released model stays well ahead on every column.

Both latent checkpoints are on the Hub
([`Vyvo/drifting-tts-tr-voxcpm2`](https://huggingface.co/Vyvo/drifting-tts-tr-voxcpm2)).
[`Vyvo/drifting-tts-tr-compare`](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-compare) plays them next to
v3.1.

Open directions:
- **A decaying learning rate**, or a larger drift batch, in case the constant rate keeps the latent model from settling.
- **Fine-tuning the VoxCPM2 decoder on generated latents**, the latent counterpart of the GTA vocoder fine-tune: see
  [below](#fine-tuning-the-voxcpm2-decoder-on-generated-latents-27).
- **Kernel features.** The latent MAE may be the bottleneck. Mel-MAE features of the decoded audio are an
  alternative.

```bash
drifting-tts extract-latents --data data/train --backend voxcpm2 --out data/train_voxcpm2   # --repeat 4
drifting-tts train-mae --config configs/mae_latent.yaml --workdir runs/mae_voxcpm2 data.root=data/train_voxcpm2
drifting-tts train --config configs/tts_latent.yaml --workdir runs/tts_voxcpm2 train.steps=10000
# DAC-VAE: --backend dacvae, and model.n_mels=128 for the MAE
```

## Fine-tuning the VoxCPM2 decoder on generated latents (#27)

Generated VoxCPM2 latents are over-smooth: real latents decoded by the released decoder score DNSMOS OVRL 3.25,
generated ones 1.45, and real latents averaged over 3 frames fall to 1.42 (#27). The released decoder turns the
missing frame-level structure into noise: on generated latents its output has no harmonic structure at all. The
fix is the latent counterpart of the GTA vocoder fine-tune. The decoder is trained to turn teacher-forced generated
latents back into the recordings.

```bash
drifting-tts finetune-vocoder --config configs/vocoder_voxcpm2_decoder.yaml --workdir runs/voxcpm2_decoder \
  data.root=data/train_voxcpm2 tts.path=runs/tts_voxcpm2/model_ema.pt
drifting-tts benchmark --model runs/tts_voxcpm2/model_ema.pt --vocoder runs/voxcpm2_decoder/decoder_ft.pt \
  --num 100 --speaker 722 --temperature 0.3 --cfg 2.0
```

```python
from drifting_tts.synthesize import Synthesizer

synth = Synthesizer("runs/tts_voxcpm2/model_ema.pt", "cuda", vocoder="runs/voxcpm2_decoder/decoder_ft.pt")
```

Without `vocoder`, a latent model still decodes with the released decoder. A registry vocoder name (`bigvgan-v2-ft`)
is refused for a latent model, since there it used to be ignored silently.

### Recipe (`vocoder.arch: vae_decoder`)

- **What is trained.** Only the decoder (45.8M parameters) is trained; the encoder is not used. The recipe is
  BigVGAN-v2's:
  - NVIDIA's released 24 kHz discriminators, MPD + CQT-D (45.6M), with their AdamW state;
  - LSGAN, feature matching and the multi-scale mel L1 (×15);
  - AdamW (0.8, 0.99), LR 5e-5 with a 1000-step warm-up for the decoder and a per-step decay of 0.99999.
- **Inputs.** Batches mix generated and real latents: 80% of them are generated (`gta_prob`) at T = 0.3 and α = 2,
  the inference settings, under the ground-truth alignment and pitch. The remaining 20% are real latents, so that the
  decoder keeps decoding them cleanly. The latents are denormalised and their 4 repeats averaged back: the decoder
  sees native 25 Hz frames, as at inference.
- **Segments.** Each segment is 16 VAE frames (0.64 s) and starts on a VAE frame, i.e. on a multiple of 960 samples
  at 24 kHz. The decoder is causal, so a segment is decoded in a window with 12 frames of left context, or from the
  utterance start, and only the segment's own samples are scored.
  - Against decoding the whole utterance, the windowed output reaches only 16 dB SNR without context, 81 dB with
    12 frames (median) and 145 dB with 24 frames.
- **Output rate.** The decoder generates 48 kHz audio. It is resampled to 24 kHz inside the graph, with the same
  band-limited sinc as `LatentVocoder` at inference, and compared with the 24 kHz recordings. Three reasons:
  - the recordings carry nothing above 12 kHz;
  - the released discriminators are 24 kHz models;
  - the TTS pipeline outputs 24 kHz.

  The cost: the band above 12 kHz gets no training signal, and the fine-tuned decoder drops VoxCPM2's bandwidth
  extension (12–16 kHz at −47 dB against −35 dB for the released decoder, relative to 0–4 kHz). Use it at 24 kHz,
  the default of `LatentVocoder`.
- **Weight norm.** The decoder is trained with weight norm on every convolution, as the VAE was, starting from the
  released `weight_g` / `weight_v`. It is folded back into plain weights on export.
  - These must be the released tensors. Re-deriving `v` from the folded weights gives it norm `g`: 0.015 instead of
    0.56 for the output convolution. The optimizer's first steps then turn the directions about 36× faster.
  - In a first run that did this (LR 1e-4, no warm-up), the mel loss on real latents jumped from 1.14 to 1.8 in 100
    steps and was still about 1.4 after 2.5k steps.
- **Export.** `decoder_ft.pt` holds the decoder state, the backend, `target_rate` (48000), `sample_rate` (24000) and
  the step: tensors and numbers only, which load with `weights_only=True`.

### Results

The TTS model is the same in every latent row: the VoxCPM2-latent model at 10k steps. Only the decoder changes.
The fine-tuned decoder was trained for 20k steps (about 3 hours on an RTX 5090 shared with other jobs), then
continued to 40k, since UTMOSv2 was still rising.

Freya-TR-Eval, first 100 sentences, speaker 722, T = 0.3, α = 2 (`drifting-tts benchmark`):
- WER / CER: Whisper large-v3 on 8 kHz band-matched audio, with 95% intervals.
- DNSMOS P.835 on the same 100 sentences.

| model | decoder / vocoder | WER [95% CI] | CER | UTMOSv2 [95% CI] | DNSMOS SIG / BAK / OVRL | RTF |
|---|---|---|---|---|---|---|
| VoxCPM2 latents | released decoder | 4.94% [3.23, 6.89] | 1.39% | 2.052 [2.007, 2.100] | 1.73 / 2.76 / 1.49 | 0.0103 |
| VoxCPM2 latents | fine-tuned decoder, 20k steps | 2.31% [1.18, 3.60] | 0.50% | 2.340 [2.294, 2.385] | 3.46 / 4.08 / 3.21 | 0.0104 |
| VoxCPM2 latents | **fine-tuned decoder, 35k steps** (published) | **1.87%** [0.97, 2.96] | **0.43%** | **2.530** [2.479, 2.583] | **3.49 / 4.10 / 3.25** | 0.0056 |
| VoxCPM2 latents | fine-tuned decoder, 40k steps | 1.76% [0.99, 2.69] | 0.42% | 2.410 [2.357, 2.462] | – | 0.0055 |
| v3.1, released | `bigvgan-v2-ft` | 0.66% [0.22, 1.22] | 0.14% | 2.934 [2.892, 2.975] | 3.56 / 4.13 / 3.33 | 0.0231 |

The released and 20k rows were timed back to back on the same busy GPU, and the 35k / 40k rows later on a quieter
one. v3.1's RTF comes from its own, earlier run.

**Continued training (20k → 40k steps).** On Freya-24, UTMOSv2 at each snapshot:

| step | 20k | 25k | 30k | 35k | 40k |
|---|---|---|---|---|---|
| UTMOSv2 | 2.33 | 2.47 | 2.45 | 2.49 | 2.43 |

DNSMOS OVRL stays at 3.22–3.27 throughout. On Freya-100, 35k is the most natural decoder (UTMOSv2 2.53, against
2.41 at 40k) at the same WER, within the intervals. It is published as `voxcpm2_decoder_ft.pt` in
[`Vyvo/drifting-tts-tr-voxcpm2`](https://huggingface.co/Vyvo/drifting-tts-tr-voxcpm2).

On the 24 sentences of the diagnosis in #27, DNSMOS OVRL goes from 1.45 with the released decoder to 3.22 with the
fine-tuned one; v3.1 scores 3.34.

- **The noise is gone.** DNSMOS OVRL on generated speech rises from 1.49 to 3.21, within 0.12 of v3.1.
  - On generated latents, the released decoder produces no harmonic structure at all: whisper-like noise inside the
    formants.
  - The fine-tuned decoder restores voicing. Cepstral peak prominence, a periodicity measure, rises from 0.66 to
    0.99 on 24 sentences. v3.1 scores 1.03 (10 sentences); real-latent resynthesis scores 1.34, against 1.27 for the
    recordings.
- **Intelligibility doubles.** WER falls from 4.94% to 2.31% and CER from 1.39% to 0.50%, with no change to the TTS
  model: Whisper copes far better with clean speech.
- **Naturalness improves, but less.** UTMOSv2 gains 0.29 (2.05 → 2.34; the intervals are disjoint), yet stays
  0.6 below v3.1.
  - DNSMOS jumps within the first 2.5k steps. UTMOSv2 first falls, then climbs steadily, and is still rising at 20k
    steps (table below).
- **Speed is unchanged.** The architecture is the released decoder's (RTF 0.0054 on a quieter GPU, table above).

Resynthesis check: real latents of 24 held-out utterances, decoded and compared with their recordings.

| decoder | UTMOSv2 | DNSMOS SIG / BAK / OVRL | multi-scale mel L1 to the recording |
|---|---|---|---|
| (the recordings) | 2.851 | 3.56 / 3.96 / 3.24 | – |
| released | 2.660 | 3.55 / 3.99 / 3.24 | 1.161 |
| fine-tuned, 20k steps | 2.515 | 3.55 / 4.01 / 3.25 | **1.101** |

- **Clean decoding is kept.** DNSMOS is unchanged, and the output is closer to the recordings: mel L1 1.101 against
  1.161.
- **UTMOSv2 loses 0.15.** At 2.5k steps the loss was 0.48, and it was uniform: every one of the 24 utterances was
  worse. It is not a spectral-tilt effect: matching the long-term spectra moves UTMOSv2 by at most 0.04.
- **The mel regression causes it, not the discriminators.** The decoder is asked to rebuild the recordings from
  generated latents, which lack their fine structure. A mel-loss-only fine-tune with no discriminators, 2k steps:
  - loses more on resynthesis (UTMOSv2 1.91), although its mel L1 is the lowest (1.057);
  - cleans generated speech much less (OVRL 2.36).
  The adversarial terms win this naturalness back as training goes on.

Snapshots on the same 24 Freya sentences and 24 held-out utterances:

| steps | Freya UTMOSv2 | Freya DNSMOS SIG / BAK / OVRL | resynthesis UTMOSv2 | resynthesis mel L1 |
|---|---|---|---|---|
| released decoder | 2.084 | 1.69 / 2.65 / 1.45 | 2.660 | 1.161 |
| 2.5k | 1.500 | 3.26 / 4.09 / 3.02 | 2.176 | 1.145 |
| 5k | 1.757 | 3.34 / 4.07 / 3.10 | 2.174 | 1.134 |
| 10k | 2.071 | 3.44 / 4.11 / 3.21 | 2.274 | 1.137 |
| 15k | 2.258 | 3.45 / 4.11 / 3.21 | 2.381 | 1.166 |
| 17.5k | 2.408 | 3.48 / 4.10 / 3.24 | 2.449 | 1.095 |
| 20k | 2.327 | 3.46 / 4.10 / 3.22 | 2.515 | 1.101 |

During training:
- The mel L1 on real-latent segments fell from 1.155 (first 2.5k steps) to 1.05; the released decoder scores about
  1.14 on such segments.
- On generated-latent segments it fell from 2.32 to 2.17. Generated latents differ from the recording in fine
  prosody, so this loss has a floor.
- The discriminator losses rose slightly (MPD 1.69 → 2.0).

Open issues:
- UTMOSv2 is still rising at 20k steps, and longer training may close more of the gap to v3.1.
- The remaining gap is in the generated latents themselves (#27, generator side): the decoder can only invent the
  frame-level detail they lack.
- UTMOSv2 is trained on English and is only a relative proxy, so listen before choosing a checkpoint.

## Fine-tuning the DAC-VAE decoder on generated latents (#32)

The same recipe for the DAC-VAE latent model (the 10k-step pilot above). With the released decoder it scores WER
9.55% and UTMOSv2 1.84 on Freya-100, and its generated latents decode as noise, as VoxCPM2's did: on Freya-24,
DNSMOS OVRL is 1.44, against 3.27 for the resynthesis of real latents.

```bash
drifting-tts finetune-vocoder --config configs/vocoder_dacvae_decoder.yaml --workdir runs/dacvae_decoder \
  data.root=data/train_dacvae tts.path=runs/tts_dacvae/model_ema.pt
drifting-tts benchmark --model runs/tts_dacvae/model_ema.pt --vocoder runs/dacvae_decoder/decoder_ft.pt \
  --num 100 --speaker 722 --temperature 0.3 --cfg 2.0
```

### What changes for DAC-VAE

- **The watermark is kept.** The DAC-VAE decoder adds an AudioSeal-style watermark to its audio: `y + α · w`, where
  `y` is the audio and `w` comes from a generator branch. That branch is `wm_model` (two LSTMs and a message
  embedding at 150 Hz) plus the watermark's down and up layers inside every decoder block. It reads only `y` and
  the 16-bit message.
  - Only the audio path is trained (65.3M parameters): the input convolution and the audio layers of the four
    blocks (`Decoder.audio_path`).
  - The rest of the decoder is frozen (14.7M): all of `wm_model` and the watermark layers of every block. `α` is a
    constant. The audio's own output layer (Snake, the convolution to one channel, tanh) sits inside
    `wm_model.encoder_block.pre` upstream, and it stays frozen with the rest of `wm_model`.
  - The watermark is added in training as at inference: `decode_train` is the released forward pass, so the losses
    are computed on the watermarked audio.
  - `decoder_ft.pt` holds the whole decoder, frozen weights included. Every export checks that they are still the
    released tensors, bit for bit, and refuses to write the file otherwise.
- **Two-sided context.** The decoder is non-causal, so a training window gets context on both sides:
  `context_frames` on the left and `right_context_frames` on the right. Near the utterance start or end the window
  shifts so that it starts or ends with the utterance, as in whole-utterance decoding. The SNR of the windowed
  output against whole-utterance decoding, on 36 segments of held-out utterances (real latents, fp32):

  | context (left / right frames) | 0 / 1 | 1 / 1 | 2 / 2 | 4 / 4 | 6 / 6 | **8 / 8** | 12 / 12 | 16 / 16 |
  |---|---|---|---|---|---|---|---|---|
  | median SNR | 23.8 dB | 35.0 dB | 44.9 dB | 60.3 dB | 83.7 dB | **104.1 dB** | 107.5 dB | 111.1 dB |
  | worst segment | 12.6 dB | 28.4 dB | 36.5 dB | 48.8 dB | 64.9 dB | **90.9 dB** | 91.8 dB | 91.2 dB |

  Context on one side only is not enough: 8 / 4 frames reach 68.9 dB and 4 / 8 frames 65.9 dB (medians). The
  config uses 8 frames (320 ms) on each side, so a training window is 32 frames (1.28 s).
- **Weight norm.** The 29 convolutions of the audio path start from the released `weight_g` / `weight_v`, as for
  VoxCPM2. The watermark's own convolutions have no weight norm. The output layer's convolution has one in the
  release, but it is frozen, so it stays folded like the rest of the watermark branch.
- **Size.** The decoder is larger than VoxCPM2's (80.0M parameters against 45.8M) and has full rather than depthwise
  convolutions. With the same batch (8 segments of 16 frames) a step takes 24 GB and runs at 2.6 it/s on an RTX 5090.

Everything else is the VoxCPM2 recipe: NVIDIA's 24 kHz discriminators with their AdamW state, LSGAN, feature
matching and the mel L1, LR 5e-5 with warm-up and decay, 80% generated latents at T = 0.3 and α = 2, and the loss on
24 kHz audio.

### Results

The TTS model is the DAC-VAE pilot at 10k steps in every DAC-VAE row; only the decoder changes. The decoder was
trained for 40k steps (4.3 hours on an RTX 5090), with a snapshot every 2.5k steps.

Freya-TR-Eval, first 100 sentences, speaker 722, T = 0.3, α = 2 (`drifting-tts benchmark`). WER / CER come from
Whisper large-v3 on 8 kHz band-matched audio, with 95% intervals. DNSMOS P.835 is computed on the same 100 sentences.
All RTFs were measured back to back on the same idle GPU.

| model | decoder / vocoder | WER [95% CI] | CER | sentences with errors | UTMOSv2 [95% CI] | DNSMOS SIG / BAK / OVRL | RTF |
|---|---|---|---|---|---|---|---|
| DAC-VAE latents | released decoder | 9.55% [7.10, 12.24] | 2.91% | 46 / 100 | 1.837 [1.782, 1.893] | 1.76 / 2.92 / 1.48 | 0.0112 |
| DAC-VAE latents | fine-tuned decoder, 35k steps | 1.87% [0.87, 2.96] | 0.40% | 13 / 100 | 2.670 [2.620, 2.719] | 3.49 / 4.06 / 3.23 | 0.0111 |
| DAC-VAE latents | fine-tuned decoder, 37.5k steps | 1.43% [0.55, 2.46] | 0.34% | 10 / 100 | 2.664 [2.620, 2.705] | 3.49 / 4.02 / 3.20 | 0.0111 |
| DAC-VAE latents | **fine-tuned decoder, 40k steps** | **1.32%** [0.44, 2.55] | **0.30%** | 8 / 100 | **2.710** [2.667, 2.752] | 3.52 / 4.07 / 3.26 | 0.0111 |
| VoxCPM2 latents | fine-tuned decoder, 35k steps (published) | 1.87% [0.97, 2.96] | 0.43% | 14 / 100 | 2.530 [2.479, 2.583] | 3.49 / 4.10 / 3.25 | 0.0052 |
| v3.1, released | `bigvgan-v2-ft` | 0.66% [0.22, 1.22] | 0.14% | 6 / 100 | 2.934 [2.892, 2.975] | 3.56 / 4.13 / 3.33 | 0.0119 |

- **The noise is gone.** DNSMOS OVRL rises from 1.48 to 3.26, within 0.07 of v3.1. This is the same jump as for
  VoxCPM2.
- **Intelligibility improves sevenfold.** WER falls from 9.55% to 1.32% and CER from 2.91% to 0.30%, with no change
  to the TTS model. Sentences with errors drop from 46 to 8. This is below VoxCPM2 with its fine-tuned decoder (1.87%),
  though the intervals overlap. v3.1 is still lower (0.66%), and its interval overlaps too.
- **Naturalness.** UTMOSv2 rises from 1.84 to 2.71. That is 0.18 above VoxCPM2 with its fine-tuned decoder and 0.22
  below v3.1; both gaps have disjoint intervals.
- **The order of the latent spaces flips.** With the released decoders, the DAC-VAE pilot was behind VoxCPM2 on
  every column. With fine-tuned decoders it is ahead on WER and UTMOSv2. DAC-VAE's better resynthesis ceiling shows
  once the decoder can handle generated latents.
- **Speed is unchanged:** the architecture is the released decoder's. End to end, the DAC-VAE model runs at
  RTF 0.0111. That is about v3.1's speed and 2.1× slower than VoxCPM2's (0.0052).
- **Snapshot choice.** 40k is best on every Freya-100 column. On Freya-24, 37.5k scored highest (table below), but
  24 sentences move UTMOSv2 by about ±0.05 from one snapshot to the next.

Snapshots on Freya-24 (the first 24 sentences, seeds 0–23) and on the resynthesis of 24 held-out utterances (real
latents, decoded in 2.56 s windows with 8 frames of context on each side):

| steps | Freya UTMOSv2 | Freya DNSMOS SIG / BAK / OVRL | resynthesis UTMOSv2 | resynthesis mel L1 |
|---|---|---|---|---|
| released decoder | 1.907 | 1.72 / 2.84 / 1.44 | 2.719 | 0.753 |
| 2.5k | 1.602 | 3.24 / 4.01 / 2.97 | 2.613 | 0.885 |
| 5k | 1.918 | 3.37 / 4.00 / 3.08 | 2.601 | 0.821 |
| 10k | 2.244 | 3.43 / 4.01 / 3.14 | 2.557 | 0.830 |
| 15k | 2.426 | 3.51 / 4.06 / 3.25 | 2.621 | 0.889 |
| 20k | 2.472 | 3.51 / 4.09 / 3.26 | 2.671 | 0.980 |
| 25k | 2.520 | 3.51 / 4.08 / 3.26 | 2.575 | 0.808 |
| 27.5k | 2.657 | 3.51 / 4.07 / 3.25 | 2.766 | 0.982 |
| 30k | 2.586 | 3.49 / 4.09 / 3.24 | 2.661 | 0.767 |
| 35k | 2.715 | 3.47 / 4.04 / 3.20 | 2.643 | 0.762 |
| 37.5k | 2.744 | 3.47 / 4.02 / 3.19 | 2.678 | 0.771 |
| 40k | 2.683 | 3.51 / 4.06 / 3.24 | 2.696 | 0.792 |

The curve looks like VoxCPM2's, a step ahead of it:
- DNSMOS jumps within the first 2.5k steps.
- UTMOSv2 first falls (1.60 at 2.5k), then climbs. It passes the released decoder at 5k and reaches 2.68–2.74
  over the last 5k steps. VoxCPM2's decoder scored 2.07 at 10k and 2.33 at 20k on the same sentences.

During training:
- **Real latents.** The mel L1 on real-latent segments was about 0.76 with the released decoder. It rose to 1.16
  during the warm-up, averaged 0.87 over the first 5k steps, and was back at 0.75 over the last 10k.
- **Generated latents.** The mel L1 fell from 3.44 (first 50 steps) to 2.37 (first 5k) and 2.24 (last 5k).
- **Discriminators.** Their losses rose slightly (MPD 1.7 → 2.0).

Resynthesis check, on real latents of 24 held-out utterances:

| decoder | UTMOSv2 | DNSMOS SIG / BAK / OVRL | multi-scale mel L1 to the recording |
|---|---|---|---|
| (the recordings) | 2.851 | 3.56 / 3.96 / 3.24 | – |
| released | 2.719 | 3.57 / 4.00 / 3.27 | 0.753 |
| fine-tuned, 40k steps | 2.696 | 3.57 / 4.00 / 3.27 | 0.792 |

- **Clean decoding is kept.** DNSMOS is unchanged and UTMOSv2 loses only 0.02; VoxCPM2's fine-tune lost 0.15 here.
  The mel L1 is 5% higher, and it moves between snapshots (0.76–0.98).
- **The 48 kHz band above 12 kHz is untrained.** The fine-tuned decoder puts more energy at 16–24 kHz:
  −54.5 dB relative to 0–4 kHz, against −68.1 dB for the released decoder. At 12–16 kHz the two are the same (−48.1
  and −47.1 dB). As for VoxCPM2, use the decoder at 24 kHz, the default of `LatentVocoder`.

### Watermark check

The released decoder and the 40k decoder were run on the same latents: real latents of 24 held-out utterances, and
generated latents of the first 24 Freya sentences. The watermark component `w = α · post(h)` and the audio `y` it is
added to were compared at 48 kHz, before the clamp (medians over the 24 utterances).

| | released, real latents | fine-tuned, real latents | released, generated latents | fine-tuned, generated latents |
|---|---|---|---|---|
| watermark RMS | −77.8 dBFS | −77.3 dBFS | −77.0 dBFS | −78.2 dBFS |
| watermark / audio | −57.4 dB | −57.2 dB | −51.4 dB | −56.8 dB |
| message-dependent part / audio | −62.6 dB | −62.1 dB | −56.1 dB | −63.6 dB |
| correlation with the released watermark | – | 0.84 (min 0.68) | – | 0.72 (min 0.63) |
| correlation with the released audio | – | 0.99 | – | 0.01 |
| watermark energy below 12 kHz | 78% | 72% | 70% | 81% |

- **The watermark weights are the released ones.** All 76 watermark tensors (14.7M parameters) are bit-identical to
  the release in every snapshot, from 2.5k to 40k steps; all 86 audio-path tensors changed.
- **The watermark is the released one.** The fine-tuned decoder's watermark equals the released generator applied
  to the fine-tuned audio, `W_released(y_ft)`, exactly (maximum difference 0). The decoded output equals
  `clamp(y + w)` exactly: the watermark is added at inference.
- **Its strength is unchanged.** The watermark keeps the same absolute level, about −77 to −78 dBFS. On generated
  speech it sits 56.8 dB below the audio, and its message-dependent part 63.6 dB below. The released decoder gives
  57.4 and 62.6 dB on real speech. The released decoder's −51.4 dB on generated latents only reflects its quieter,
  noise-like output (−25.5 dBFS, against −21.3 dBFS once fine-tuned).
- **Much of the watermark does not depend on the audio.** On generated latents, the fine-tuned and released outputs
  are uncorrelated (0.01), yet their watermarks still correlate at 0.72.
- **It survives the 24 kHz pipeline.** 70–81% of the watermark's energy is below 12 kHz, so it is kept when the
  output is resampled to 24 kHz.
- **No detector can be run yet.** Meta has not released a detector for this watermark: the model card says the
  detector API will come later. AudioSeal's public detector (`audioseal_detector_16bits`) is a different model. It
  detects AudioSeal's own watermark in 24 / 24 recordings, but flags none of the released DAC-VAE outputs, the
  fine-tuned ones or the recordings (0 / 24 each, mean score 0.002). Detection rates before and after fine-tuning can
  be measured once the detector is out. The evidence above rests on the frozen generator and the component
  analysis.

Open issues:
- **Naturalness is not saturated.** Freya-24 UTMOSv2 is still around 2.7 after 40k steps, and 0.22 below v3.1 on
  Freya-100. Longer training may help a little. As for VoxCPM2, most of the remaining gap is likely in the generated
  latents themselves.
- **Upper band.** The decoder is meant for 24 kHz output; its 16–24 kHz band is not trained.
- **UTMOSv2 is only a relative proxy.** It is trained on English, so listen before choosing a checkpoint.

## All systems on one protocol

Freya-TR-Eval, first 100 sentences, speaker 722, T = 0.3, α = 2, seed = sentence index.
- WER / CER: Whisper large-v3 on 8 kHz band-matched audio.
- UTMOSv2 and DNSMOS OVRL on the full band.

The comparison Space [`Vyvo/drifting-tts-tr-compare`](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-compare)
plays every row.

| model | vocoder / decoder | WER | CER | UTMOSv2 | DNSMOS OVRL |
|---|---|---|---|---|---|
| v3.1 (mels) | `bigvgan-v2-ft` (112 M) | **0.66%** | **0.14%** | **2.934** | 3.33 |
| v3.1 (mels) | `bigvgan-base-ft` (14 M) | 0.77% | 0.18% | 2.892 | **3.34** |
| v3.1 (mels) | `vocos-ft` (13.5 M) | 1.10% | 0.22% | 2.627 | 3.31 |
| v3.1 (mels) | `revox` (4.5 M, non-commercial) | 0.77% | 0.16% | 2.244 | 3.10 |
| DAC-VAE latents, 10k | DAC-VAE decoder, fine-tuned 40k (#32) | 1.32% | 0.30% | 2.710 | 3.26 |
| VoxCPM2 latents, 10k | VoxCPM2 decoder, fine-tuned 35k (#27) | 1.87% | 0.43% | 2.530 | 3.25 |
| DAC-VAE latents, 10k | released DAC-VAE decoder | 9.55% | 2.91% | 1.837 | 1.48 |
| VoxCPM2 latents, 10k | released VoxCPM2 decoder | 4.94% | 1.39% | 2.052 | 1.49 |

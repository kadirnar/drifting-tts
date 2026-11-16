# ARDrift on Mimi latents (Kyutai's Pocket TTS codec)

The 10,000-hour English run ([docs/ENGLISH.md](ENGLISH.md)) repeated in another audio space: the 32-d continuous
latents of Mimi, the codec of Kyutai's [Pocket TTS](https://kyutai.org/blog/2026-09-28-pocket-tts-drifting/), whose
autoregressive model also generates with a one-step drifting head. Same data, same model, same held-out shard; only
the frames the model predicts and the decoder that turns them into audio change.

- [The codec](#the-codec)
- [What changed in the code](#what-changed-in-the-code)
- [Results](#results)
- [Why the Mimi model sounds worse](#why-the-mimi-model-sounds-worse)
- [Fine-tuning Mimi's decoder](#fine-tuning-mimis-decoder)
- [Conclusions](#conclusions)
- [A second attempt: the code (#92)](#a-second-attempt-the-code-92)
- [Commands](#commands)

## The codec

| | |
|---|---|
| weights | the `mimi.` tensors of [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts) `languages/english_2026-09/model.safetensors` (CC BY 4.0, gated: accept the terms once), pinned to a revision |
| code | the [`pocket-tts`](https://pypi.org/project/pocket-tts/) package (MIT), extra `drifting-tts[mimi]` |
| latents | 32-d, continuous (the VAE's mean, no quantisation), 12.5 Hz (1,920 samples per frame) |
| audio | 24 kHz in and out, mono |
| decoder | causal: latent projection, 2x transposed-convolution upsampler, 2-layer transformer (context 250 at 25 Hz), SEANet decoder; 10.3M parameters |

**Round trip on the same 100 held-out recordings** (encode + decode, no TTS model):

| | UTMOSv2 |
|---|---|
| recordings | 3.259 |
| Mimi round trip | 3.132 |
| the recording's log-mel through Vocos v2 (the mel run's vocoder) | 2.926 |

Mimi's ceiling is 0.21 above Vocos v2's. On the 495 held-out utterances below it is 3.177 against 2.908.

## What changed in the code

- **`mimi` backend** (`drifting_tts/latents/mimi.py`): `encode` / `decode` as for the other audio VAEs
  ([docs/LATENTS.md](LATENTS.md)), plus `trainable_decoder` and `decode_train`: decoding with gradients, the
  convolutions from fresh streaming states and the transformer on its stateless causal path. It matches `decode`
  exactly.
- **`extract-latents --parquet-glob`**: encodes a corpus straight from its parquet shards, local or `hf://`, with no
  prepared root and no copy of the audio. Shards download in parallel; worker processes prepare each waveform as
  `prepare` does; the encoder runs on GPU batches of similar length. `--val-glob`, `--language` and
  `--speaker-column` work as in `prepare`.
  - The 10,000 h took 6.1 h on one RTX 5090 (3,474,750 utterances, none skipped, ~115 GB at 50 Hz).
  - `--max-pending` (default 2,048) bounds the waveforms prepared ahead of the encoder. Without it, the worker pool's
    result backlog grew until the container's 60 GB memory limit killed the run.
- **Storage at 50 Hz**: every 12.5 Hz frame is stored 4 times, so characters with blanks (~29 tokens per second) can
  be aligned. A patch of 4 makes one AR step one Mimi frame (80 ms), as in Pocket TTS. The data loader keeps an
  utterance only if its tokens fit in its frames. That drops 1.49% of the training utterances (47.6 h of very fast
  speech, including the ~0.24% whose audio was cut short) and 5 of the 500 held-out ones.
- **`train-ar`** decodes its TensorBoard samples with the codec for latent datasets.
- **Decoder fine-tuning for an AR model**:
  - `ARDriftTTS.gta`: generated latents aligned with the recording. Every patch is drawn after the recorded ones
    (one teacher-forced pass), at the inference temperature, without guidance.
  - `finetune-vocoder` takes an ARDrift model as `tts.path`.
  - A causal decoder's training windows now accept utterances shorter than the window, right-padded after the
    segment, which a causal decoder does not hear. This also applies to VoxCPM2's causal decoder, which before
    skipped those utterances.
  - `train.teacher` and `train.latent_noise`: distillation from the released decoder (recipe 3 below).

## Results

`configs/ar_tts_en_mimi.yaml`: the 10k-h mel run's model (`configs/ar_tts_en_10k.yaml`) on Mimi latents, 150k steps
of ~480 s of speech (about two passes over the data; 8.9 h at ~5 it/s on one RTX 5090). Noise temperature 0.5, no
guidance, the released Mimi decoder.

**Snapshots on val 100** (Whisper large-v3 on the CPU, int8; UTMOSv2; WavLM-ECAPA similarity to the recording's
round trip):

| step | WER | CER | UTMOSv2 | SIM |
|---|---|---|---|---|
| 10k | 3.94% | 1.86% | 2.146 | 0.292 |
| 20k | 3.69% | 1.78% | 2.374 | 0.341 |
| 30k | 3.78% | 1.73% | 2.403 | 0.345 |
| 40k | 3.86% | 1.81% | 2.390 | 0.382 |
| 50k | 3.37% | 1.78% | 2.325 | 0.407 |
| 60k | 3.53% | 1.61% | 2.471 | 0.428 |
| 70k | 3.12% | 1.54% | 2.431 | 0.445 |
| 80k | 3.94% | 1.96% | 2.478 | 0.481 |
| 90k | 3.86% | 1.96% | 2.548 | 0.525 |
| 100k | 3.78% | 1.74% | 2.475 | 0.557 |
| 110k | 3.86% | 2.03% | 2.489 | 0.601 |
| 120k | 4.27% | 2.20% | 2.491 | 0.645 |
| 130k | 3.28% | 1.49% | 2.498 | 0.668 |
| 140k | 3.61% | 1.78% | 2.508 | 0.677 |
| 150k | 3.37% | 1.68% | 2.468 | 0.701 |
| Mimi round trip | 3.28% | 1.61% | 3.146 | – |

Each run drew its own val split from the same held-out shard, so this val 100 shares only 46 utterances with the mel
run's. The 495 utterances below are the same for both.

**495 held-out utterances** (val + dev, 15,181 words; Whisper on the GPU, fp16), against the mel run on the same
utterances:

| model | WER | CER | UTMOSv2 | SIM |
|---|---|---|---|---|
| Mimi, 150k, released decoder | 2.14% | 1.16% | 2.505 | 0.751 |
| Mimi round trip | 1.71% | 0.83% | 3.177 | – |
| **mel + Vocos v2, 200k** (docs/ENGLISH.md) | **1.42%** | **0.65%** | **2.931** | **0.804** |
| mel copy-synthesis (Vocos v2) | 1.56% | 0.72% | 2.908 | – |

- **The mel model is ahead on every column**, with a worse codec ceiling: UTMOSv2 2.93 against 2.51.
- **The Mimi model reaches 79% of its codec's UTMOSv2; the mel model exceeds its vocoder's.** Mimi's round trip is
  0.27 above Vocos v2's, but the Mimi model lands 0.67 below that round trip.
- **Intelligibility** is close to the round trip (2.14% against 1.71%). Speaker similarity rises at every snapshot,
  as in the mel run, 0.29 → 0.70 on val 100.
- **UTMOSv2 stops early**: 2.37 at 20k, 2.47-2.55 from 60k on.

## Why the Mimi model sounds worse

A diagnosis on 60 val utterances with the 40k snapshot, latents in the dataset's normalised space (unit variance per
dimension). It applies known errors to the recordings' own latents and decodes them with the released decoder:

| latents | UTMOSv2 |
|---|---|
| recording | 3.126 |
| + Gaussian noise, std 0.1 | 2.920 |
| + Gaussian noise, std 0.2 | 2.457 |
| + Gaussian noise, std 0.3 | 2.040 |
| + Gaussian noise, std 0.5 | 1.325 |
| scaled x0.9 | 3.069 |
| scaled x0.8 | 2.846 |
| scaled x0.7 | 2.377 |
| smoothed in time (¼ ½ ¼) | 1.803 |
| **the model's, T 0.5** | **2.304** |
| the model's, std matched to the recordings' per dimension | 2.052 |

- **The decoder punishes small latent errors.** Noise of 0.2 standard deviations costs 0.67 UTMOSv2. A mild
  three-tap smoothing costs 1.3.
- **The model's latents have the recordings' statistics**: per-dimension std 1.005 against 1.009, frame-to-frame
  change 0.871 against 0.886. They are not over-smoothed or shrunk; their per-frame error is what the decoder hears.
  Matching their std makes them worse.
- **The noise temperature does not help** (20k snapshot, val 100): T 0 2.238, T 0.3 2.366, T 0.5 2.374, T 0.7 2.365,
  T 0.9 2.344.
- **Vocos on log-mels is forgiving.** The mel model's UTMOSv2 is at its vocoder's ceiling, though its mels carry
  errors too.

## Fine-tuning Mimi's decoder

Fine-tuning a decoder on the model's own latents (GTA) fixed the noise of the DAC-VAE and VoxCPM2 latent models
([docs/LATENTS.md](LATENTS.md)). Three recipes were tried for Mimi. Each takes the 150k model and the fine-tune data:
37,115 utterances (105 h) of `data/en_mimi`, with their 24 kHz audio, from 80 evenly spaced source shards. Rows with
fewer than 8 or more than 25 characters per second were left out, which removes the clipped-audio rows. The fine-tune
data includes the held-out shard as val, not used for training.

Every snapshot is scored on val 100 as above: the model's latents, and the recordings' latents (round trip) through
the fine-tuned decoder. Speaker similarity is left out of these tables: its reference is the round trip through the
same decoder, so it does not compare across decoders.

**Recipe 1: the BigVGAN-v2 recipe**, as for DAC-VAE (`configs/vocoder_mimi_decoder.yaml`):
- NVIDIA's released 24 kHz MPD + CQT-D with their AdamW state, LSGAN, feature matching and the multi-scale mel L1
  x15.
- 80% GTA batches at T 0.5, 20% recorded latents.
- lr 5e-5, batch 16 windows of 8 Mimi frames (0.64 s) after 40 frames of left context.
- 2.1 it/s, 18 GB; 40k steps (5.3 h). UTMOSv2 fell at 2.5k and 5k. The DAC-VAE fine-tune also fell first and
  passed its released decoder from 5k on, so this run went on to 40k.

| step | WER | CER | UTMOSv2 | round-trip UTMOSv2 |
|---|---|---|---|---|
| released decoder | 3.37% | 1.68% | 2.468 | **3.146** |
| 2.5k | 2.96% | 1.41% | 2.281 | 2.895 |
| 5k | 3.12% | 1.59% | 2.203 | 2.711 |
| 7.5k | 3.37% | 1.78% | 2.193 | 2.732 |
| 10k | 3.12% | 1.51% | 2.404 | 2.953 |
| 12.5k | 3.28% | 1.73% | 2.221 | 2.820 |
| 15k | 3.37% | 1.74% | 2.453 | 2.998 |
| 17.5k | 3.45% | 1.76% | 2.376 | 2.992 |
| 20k | 3.37% | 1.74% | 2.271 | 2.854 |
| 22.5k | 3.37% | 1.78% | **2.472** | 2.950 |
| 25k | 3.04% | 1.46% | 2.289 | 2.798 |
| 27.5k | 2.87% | 1.32% | 2.441 | 3.047 |
| 30k | 3.37% | 1.71% | 2.177 | 2.608 |
| 32.5k | 3.20% | 1.64% | 2.470 | 2.980 |
| 35k | 2.87% | 1.41% | 2.406 | 2.991 |
| 37.5k | 3.45% | 1.83% | 2.465 | 2.947 |
| 40k | 2.79% | 1.24% | 2.371 | 2.922 |

UTMOSv2 swings by up to 0.3 between neighbouring snapshots and never clearly passes the released decoder. The 95%
interval of the released decoder's score is 2.39-2.54.

**The 32.5k snapshot on the 495 held-out utterances** (the same latents, each utterance drawn with the same seed, so
only the decoder differs):

| decoder | WER | CER | UTMOSv2 | round-trip UTMOSv2 |
|---|---|---|---|---|
| released | 2.14% | 1.16% | **2.505** | **3.177** |
| recipe 1, 32.5k | 2.07% | 1.06% | 2.474 | 2.997 |

Per utterance, the fine-tuned decoder is 0.031 UTMOSv2 lower (95% bootstrap interval −0.056 to −0.006); it scores
higher on 224 of 495. On the recordings' latents it is 0.180 lower. The WER difference is 11 of 15,181 words.

**Recipe 2: no mel loss.** Kyutai trained Mimi with adversarial and feature-matching losses only. Here the
discriminators first trained alone for 1k steps, then the decoder joined at lr 1e-5. Stopped at 3.5k:

| step | WER | CER | UTMOSv2 | round-trip UTMOSv2 |
|---|---|---|---|---|
| 2.5k | 3.37% | 1.78% | 2.309 | 2.982 |

The mel L1 on the recordings' latents rose from 1.10 to 1.35: without it, the decoder drifts away from the
recordings.

**Recipe 3: distillation, no discriminator** (`train.teacher`, `train.latent_noise`):
- The target is the released decoder's output on the recording's latents (a frozen copy), so on clean latents the
  loss starts at zero.
- The input is the GTA latents (50%) or the recording's latents plus Gaussian noise with a std drawn from U(0, 0.3)
  per utterance (50%).
- Multi-scale mel L1 only, lr 3e-5, 9.5 it/s, 6.5 GB.

| step | WER | CER | UTMOSv2 | round-trip UTMOSv2 |
|---|---|---|---|---|
| 2.5k | 3.04% | 1.44% | 2.405 | 3.114 |
| 5k | 3.12% | 1.44% | 2.198 | 2.823 |
| 7.5k | 2.96% | 1.54% | 2.359 | 2.936 |
| 10k | 3.04% | 1.44% | 2.402 | 3.031 |

The loss falls only a little over 20k steps: on GTA latents from 1.40 to ~1.34, on noisy latents from 0.95 to
~0.86.

- **No recipe beats the released decoder.** At best a snapshot matches it on val 100, and on the 495 the best one is
  measurably lower. Every recipe lowers the round trip. Intelligibility moves within a few words of the released
  decoder.
- **The mel loss on GTA latents hardly falls** (recipe 1: ~1.76 → ~1.65 over 40k steps; DAC-VAE: 3.44 → 2.24). At
  12.5 Hz, a sampled latent frame does not determine the recording's detail, so a decoder cannot learn to map it back.
- **The foreign discriminators pull the decoder off its training equilibrium.** The recipe that worked for DAC-VAE
  started from a decoder at UTMOSv2 1.84 on generated latents; Mimi's starts at 2.47 and has more to lose.

## Conclusions

- **Mimi is the better codec but the worse target for ARDrift.** Its round trip is above Vocos v2, but its decoder
  is far less tolerant of the model's errors than Vocos is of mel errors. On the same 495 utterances the mel model
  stays ahead on WER (1.42% vs 2.14%), UTMOSv2 (2.93 vs 2.51) and similarity (0.80 vs 0.75).
- **The error is in the latents, not in the decoder's weights.** It would have to shrink at the source, for example:
  - adversarial post-training of ARDrift through the frozen decoder (#72);
  - a drift kernel in a feature space (#73).
- **The mel + Vocos model stays the English model.** The Mimi model is only in a private test Space of its own, with
  the released decoder, to listen to (`space_mimi/`, `scripts/deploy_mimi_space.sh`, which refuses a public Space).
  - Its weights are stored in fp16 (353 MB). On val 100 that changes nothing measurable: UTMOSv2 2.471 and WER 3.45%,
    against 2.468 and 3.37% in fp32.
  - Mimi's weights travel beside it: their `mimi.` tensors, 40 MB, CC BY 4.0, read through
    `$DRIFTING_TTS_MIMI_WEIGHTS`. The Space needs no Hub token for the gated repo.
  - The fine-tuned decoders are not published.

## A second attempt: the code (#92)

The run above differs from Kyutai's recipe and from the published continuous-latent models in three things. The
code for each is in the repository and tested on tiny models; **none of it has been run on Mimi**: the Mimi weights
are gated, and neither they, the 10,000 h of latents nor the 150k checkpoint were on the machine this was written
on.

| difference | then | now |
|---|---|---|
| what the head draws | a 128-d patch of four copies of a Mimi frame that nothing ties together; `LatentVocoder` averages them | `model.ar.tied`: the backbone reads one 32-d frame per step, the head draws one, and it is copied over the patch. The text condition keeps its 50 Hz grid |
| the target | the posterior mean | `drift.target_noise`: Gaussian noise on the latents (0.05–0.1 std), or the codec's own posterior (`extract-latents --save-std`, `target_noise: posterior`). The alignment, the history and the target see the sample |
| the decoder | fine-tuned on teacher-forced GTA latents (every recipe lowered the round trip) | `configs/vocoder_mimi_decoder_noise.yaml`: noise of up to 0.3 std on the recorded latents and free-running GTA (#90) |
| the length | 150k steps, UTMOSv2 flat from 60k | `configs/ar_tts_en_mimi_v2.yaml`: 300k steps with the evaluation in the loop and a second EMA (#102) |

- **Why the copies matter.** `scripts/latent_smoothness.py` measures after the averaging, so disagreement between
  the copies has never been seen. `scripts/codec_gate.py --model` now takes an ARDrift model: per position in the
  group, the squared deviation from the group mean; with the judges it also decodes the generated latents twice,
  from the mean of the copies and from the first copy alone. If one copy scores above their mean, the averaging has
  been hiding a head that cannot make four values agree; if the copies agree, tying them changes little, and what
  remains is the per-frame error.
- **A tied model is a smaller problem for the kernel**: 32 dimensions per step instead of 128, the layout of
  Kyutai's head. Its kernel has one map (the frame); the patch and frame maps of the mel model coincide.
- **Prompts** (#95) on a tied model are cut on the grid of the copies, so a patch never holds two latent frames.
- **Storage**: `extract-latents --native` writes every frame once (29 GB instead of 115 GB for the 10,000 h); the
  data loader makes the copies. The first model's data (`data/en_mimi`) works unchanged.
- **What #90 says about step 4**: on mels, free-running GTA batches did not help the English vocoder (UTMOS22 4.10
  / 4.04 / 4.01 for 0 / 50 / 100% of them at 2,500 steps), and recipe 3 above already trained with latent noise of
  0.3. Run a teacher-forced control beside the free-running decoder.

## Commands

```bash
# the second attempt (#92): first the copies of the 150k model, then the tied model on sampled targets
python scripts/codec_gate.py --data data/en_mimi_ft --backends mimi --skip native tolerance phonetic \
    --model runs/ar_en_mimi/model_ema.pt --cfg 1 --temperature 0.5 --out outputs/mimi_copies
drifting-tts train-ar --config configs/ar_tts_en_mimi_v2.yaml --workdir runs/ar_en_mimi_v2
drifting-tts finetune-vocoder --config configs/vocoder_mimi_decoder_noise.yaml --workdir runs/mimi_decoder_noise \
    data.root=data/en_mimi_ft

pip install "drifting-tts[mimi]"   # pocket-tts; accept the terms of kyutai/pocket-tts on the Hub once
drifting-tts extract-latents --parquet-glob "hf://datasets/VoiceHub/echo-clones-10kh-en/data/*.parquet" \
    --val-glob "hf://datasets/SynDataLab-EN/echo-clones-4m-en/data/train-07960.parquet" --backend mimi \
    --language en --speaker-column speaker_id --out data/en_mimi --tf32
drifting-tts train-ar --config configs/ar_tts_en_mimi.yaml --workdir runs/ar_en_mimi
drifting-tts evaluate --model runs/ar_en_mimi/model_ema.pt --split val --num 100 --temperature 0.5 --asr-device cpu

# decoder fine-tuning (data/en_mimi_ft: utterances of data/en_mimi with an audio.bin of their 24 kHz audio)
drifting-tts finetune-vocoder --config configs/vocoder_mimi_decoder.yaml --workdir runs/mimi_dec_ft \
    tts.path=runs/ar_en_mimi/model_ema.pt data.root=data/en_mimi_ft                        # recipe 1
drifting-tts finetune-vocoder --config configs/vocoder_mimi_decoder.yaml --workdir runs/mimi_dec_distill \
    tts.path=runs/ar_en_mimi/model_ema.pt data.root=data/en_mimi_ft train.teacher=true train.latent_noise=0.3 \
    train.gta_prob=0.5 train.adv_loss_coeff=0 train.fm_loss_coeff=0 train.lr=3e-5 train.warmup_steps=500  # recipe 3
drifting-tts evaluate --model runs/ar_en_mimi/model_ema.pt --split val --num 100 --temperature 0.5 \
    --vocoder runs/mimi_dec_ft/decoder_ft.pt

# the private Space: the model in fp16 (publish.export_tts(ck, half=True)) and the voices offered
scripts/deploy_mimi_space.sh <private space> ardrift_en_mimi.pt english_voices.json
```

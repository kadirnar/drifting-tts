# Training guide

## The recipe (v3)

`configs/tts_v3.yaml` is the recipe behind the released model. It extends the v2 config, which adds a 2-D Mel-MAE
with spectral-detail features and pitch conditioning to the first recipe. Each choice was checked in 10k-step
pilots:

| choice | evidence |
|---|---|
| Kyutai's released learned-temperature field (`drift.mode: kyutai`) | pilot: harmonic contrast 0.98 vs 0.93, CER 15.4% vs 17.7% against the paper's fixed temperatures |
| the paper's block features (`every_k_block: 2`), CFG α ∈ [1, 4] | 48 → 80 feature maps; as in the official code |
| G = N = 16 samples / negatives per condition, a 60 M generator | larger per-condition sets beat more conditions at a fixed budget (paper, Table 2) |
| training data filtered on measured quality (`drifting-tts score`) | Whisper CER, DNSMOS, bandwidth, speaker purity, speaking rate |
| location subsampling of large feature maps, a compiled generator | 3.1 it/s on an RTX 5090 |

Tested and **not** adopted:
- **A prior-mean negative plus texture drift against over-smoothing.** It lowered harmonic contrast to 0.75 and
  raised CER to 22% in the pilot.
- **The paper's fixed temperatures.** They lost the A/B against `kyutai`.

## Pipeline

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,eval,bigvgan,score]"   # + a CUDA build of torch
drifting-tts prepare --dataset <hf-dataset-id> --out data/train --backend bigvgan --f0 --save-audio --dev-size 200
drifting-tts train-mae --config configs/mae2d.yaml --workdir runs/mae2d train.steps=60000
drifting-tts score --data data/train                                       # quality scores for the filters
drifting-tts train --config configs/tts_v3.yaml --workdir runs/tts_v3
drifting-tts calibrate-durations --model runs/tts_v3/model_ema.pt --temperature 0.3
drifting-tts finetune-vocoder --config configs/vocoder_bigvgan.yaml --workdir runs/vocoder_v3 \
    tts.path=runs/tts_v3/model_ema.pt train.steps=15000
```

- **Data format:** `prepare` reads a Hugging Face parquet dataset (or local files with `--parquet-glob`). It needs
  `audio` and `text` columns; `speaker` and `quality_score` are optional.
- **Filters:** `data.filters` in the config uses the scores from `drifting-tts score`
  ([EVALUATION.md](EVALUATION.md#data-curation)).

Measured on one RTX 5090 (32 GB). The container had a 7.7-CPU quota, so keep `OMP_NUM_THREADS`/`NUMBA_NUM_THREADS`
≤ 4.

| stage | throughput | time |
|---|---|---|
| 2-D Mel-MAE, 60k steps, batch 64 | 10–35 it/s | ~1.5 h |
| TTS v3, 150k steps (16 conditions × 16 samples, 60 M generator, kyutai) | 3.08 it/s, bf16, compiled | 13.6 h |
| BigVGAN-v2 fine-tuning, 15k steps (4 × 16384 samples) | ~4.7 it/s | ~1 h |

## Adding a voice

A new voice is fine-tuned into a trained model. Its data is mixed with the original training data, so the existing
voices are kept.

```bash
drifting-tts prepare --dataset <new-voice-dataset> --out data/new_voice --backend bigvgan --f0 --save-audio \
    --speaker-name new_voice --val-size 100 --dev-size 100 --val-max-seconds 16
drifting-tts merge-data --out data/train_new_voice data/train data/new_voice --repeat 1 2
drifting-tts train --config configs/tts_v3_add_voice.yaml --workdir runs/tts_v3_new_voice
drifting-tts calibrate-durations --model runs/tts_v3_new_voice/model_ema.pt --temperature 0.3 \
    --speakers studio male female
```

- **Speaker table:** the model grows one row per new speaker, initialised at the mean of the old ones.
- **Kernel temperature:** fine-tuning resumes the learned temperature.
- **Duration factor:** `calibrate-durations` gives every listed voice its own factor.
- **Cost and results:** the `studio` voice (30k steps) took 2.7 h on an RTX 5090. Its results are in
  [RESULTS.md](RESULTS.md#adding-the-studio-voice-v3--v31).

## BigVGAN-v2 vocoder fine-tuning

The vocoder is fine-tuned on the model's own ground-truth-aligned mels, starting from NVIDIA's released generator
**and** discriminators (`configs/vocoder_bigvgan.yaml`):
- **Losses:** LSGAN, feature matching and a multi-scale mel L1, as in the official recipe.
- **Optimiser:** AdamW (0.8, 0.99) at LR 1.35e-5, where the released schedule ended.
- **Precision:** fp32. bf16 audibly hurts the snake activations.

BigVGAN mel frames are uncentred: `F` frames correspond to `F·256` samples starting at `s·256`. `--cuda-kernel`
builds BigVGAN's fused activation for the local GPU. It is 2.4–2.9× faster at inference, with SNR ≥ 51.6 dB against
the PyTorch path.

## GAN-free vocoder fine-tuning (drifting, experimental)

`vocoder.objective: drift` fine-tunes a Vocos or a BigVGAN-base **without a discriminator**. The drifting field in a
frozen feature space replaces the adversarial loss ([DESIGN.md §8](DESIGN.md#8-a-gan-free-vocoder-experimental)):

```bash
drifting-tts finetune-vocoder --config configs/vocoder_drift_vocos.yaml --workdir runs/vocoder_drift_vocos \
    tts.path=runs/tts_v3/model_ema.pt
drifting-tts finetune-vocoder --config configs/vocoder_drift_bigvgan_base.yaml --workdir runs/vocoder_drift_base \
    tts.path=runs/tts_v3/model_ema.pt
```

- **Generator:** `G(mel, z)`. `vocoder.noise_channels` Gaussian channels are appended to the mel; their input weights
  start at zero. `drift.samples` waveforms are drawn per mel segment.
- **Feature space** (`features`, frozen): the released BigVGAN-v2 MPD and CQT-D (every layer), log-|STFT| patches,
  optionally an SSL encoder (`features.ssl`).
- **Loss:** `mel_loss_coeff` × multi-scale mel L1 + `drift_coeff` × drift loss, with the `conditional` and `pooled`
  pairings weighted by `drift.pairing`. `mel_loss_coeff: 0` gives a pure-drift ablation.
- **Inference:** `z = 0`. The export (`vocos_ft.pt` / `bigvgan_ft.pt`) has the GAN fine-tunes' layout plus
  `noise_channels`. `Vocoder` and `load_bigvgan` fold the noise channels away, so it loads and streams like any
  fine-tuned vocoder. `Vocoder(noise_seed=…)` uses fixed noise instead (evaluation only).
- **What to watch:** `mel`; `spread`, the across-sample std of the log-mel (it starts at 0 and must grow, otherwise
  the noise is ignored); `pool_force_*` / `cond_force_*`, the raw drift norms (the drift loss itself is near-constant).

Measured on an RTX 5090 that was shared with two other training runs (so the throughput is a lower bound):

| config | samples per step | memory | throughput |
|---|---|---|---|
| Vocos (`vocoder_drift_vocos.yaml`) | 4 segments × 4 samples × 16384 | 4.0 GB | ~1.0 it/s |
| BigVGAN-base, compiled (`vocoder_drift_bigvgan_base.yaml`, `train.batch_size=2`) | 2 × 4 × 8192 | 5.7 GB | 0.7–1.2 it/s |

The drift computation (51 feature maps × 2 pairings per step) takes about half of the step.

## SSL adversarial fine-tuning (experimental, #41)

`slm.enabled` adds StyleTTS 2's SLM adversary (arXiv 2306.07691) to the drift objective, for fine-tuning a trained
model (`drifting_tts/slm.py`, `configs/tts_v3_slm.yaml`). It is off by default, and the step is then bit-identical
to the plain recipe.

```bash
drifting-tts train --config configs/tts_v3_slm.yaml --workdir runs/slm_pilot        # from v3.1, 25k steps
```

- **Fake input:** `slm.crops_per_cond` generated crops per condition, either extra samples at CFG scale `slm.alpha`
  and noise temperature `slm.temperature` (drawn in the same generator call), or with `alpha: null` a random subset
  of the drift samples. They are denormalised and vocoded by the **frozen** `vocos-ft` through its differentiable
  ISTFT head (`Vocoder.differentiable`). `slm.trim_frames` frames are dropped at both ends: crop-edge artefacts of
  the vocoder reach 12–16 frames (−18 dB at frame 12, −33 dB at 16, against whole-utterance vocoding). The audio is
  resampled to 16 kHz and encoded by a **frozen** WavLM (`microsoft/wavlm-base-plus`, all 13 hidden states).
- **Real input:** the recorded audio of the same crops (`real: audio`, BigVGAN framing: `F` frames ↔ `F·256`
  samples from `s·256`), or the real mel crop through the same vocoder (`real: vocoded`), which hides the vocoder's
  own artefacts from the discriminator.
- **Discriminator:** StyleTTS 2's `WavLMDiscriminator` (1-D convolutions over the 13 × 768 stacked channels, 1.2 M
  parameters), LSGAN, its own AdamW (`disc_lr`, betas (0, 0.99)). It trains alone for `disc_warmup` steps.
- **Generator terms:** `weight` × LSGAN and `fm_weight` × the L1 distance between the WavLM states of each generated
  crop and its real segment (the crops are generated under the ground-truth alignment and pitch, so they line up).
  The gradient with respect to the generated crops is computed `chunk` crops at a time inside the step and handed
  back as a surrogate loss, so WavLM's activations (~0.12 GB per crop) never sit on top of the drift graph.
  `grad_clip` caps the norm of that gradient.
- **Logged:** `slm_disc`, `slm_d_real`, `slm_d_fake` (LSGAN targets 1 / 0), `slm_adv`, `slm_fm`, `slm_grad_norm_x`
  (the weighted gradient w.r.t. the crops, before the cap).
- **Exports:** `train.keep_snapshots` keeps `model_ema_<step>.pt`; `train.init_calibration` copies the init
  checkpoint's per-voice duration factors and temperature, so that snapshots compare with v3.1 on Freya-100.
- `scripts/slm_probe.py` measures peak memory, step rate and the gradient norms of the drift, adversarial and
  feature-matching terms (`--grad-norms`).

**Gradient norms at v3.1** (`slm_probe.py --grad-norms`, 2 crops per condition at CFG 1, median of 4 batches;
unweighted terms, generator parameters):

| conditions | discriminator | D(real) / D(fake) | drift | LSGAN | WavLM L1 | LSGAN / drift |
|---|---|---|---|---|---|---|
| 2 | fresh | 0.05 / 0.05 | 1.11 | 0.61 | 1.24 | 0.5 |
| 2 | 50 D-only steps | 0.70 / 0.13 | 1.13 | 44.3 | 2.62 | 40 |
| 2, `real: vocoded` | 50 D-only steps | 0.43 / 0.42 | 1.13 | 9.6 | 2.82 | 9 |
| 4 | fresh | 0.05 / 0.05 | 0.69 | 0.40 | 0.86 | 0.5 |
| 4 | 50 D-only steps | 0.86 / 0.17 | 0.73 | 19.6 | 1.61 | 28 |

- **The discriminator separates recordings from vocoded generated crops within 50 steps;** it does not separate
  vocoded real from vocoded generated crops in that time. Its early signal is mostly the vocoder's own artefacts.
- **The adversarial gradient is orthogonal to the drift gradient** (|cos| < 0.07) and, once the discriminator has
  warmed up, 1–2 orders of magnitude larger. StyleTTS 2 scales its SLM gradients down for the same reason.
  `configs/tts_v3_slm.yaml` therefore uses `weight: 0.01` and `grad_clip: 0.01` (the drift gradient w.r.t. the
  crops is ~0.03 at 16 × 16).
- On the encoder the ratio is 0.15 (fresh) and 7–9 (warmed up); its gradients are clipped separately.

**Memory** (peak allocated, compiled generator, `slm_probe.py --steps 30`; the RTX 5090 was shared with three other
jobs, so no step rate is quoted):

| conditions × samples | plain | + SLM, 2 extra crops / condition | + SLM, subset of 2 |
|---|---|---|---|
| 1 × 16 | 3.05 GB | 3.56 GB | – |
| 2 × 16 | 4.56 GB | 5.13 GB | 4.98 GB |
| 4 × 16 | 7.51 GB | – | – |
| 16 × 16 (linear extrapolation) | ~25.3 GB | ~26.7 GB | ~25.7 GB |

The SLM adds its frozen models (0.41 GB), the extra generator samples (~0.03 GB each) and a transient of ~0.12 GB
per crop in a chunk while only the generator graph is alive.

**Pilot (negative so far).** Setup:
- Fine-tuned from v3.1 for 12k steps with `configs/tts_v3_slm.yaml`, at 10 conditions × 16 samples.
- The vocoder in the loop was a frozen snapshot of the improved Vocos (20k steps of the Vocos v2 long run).
- Snapshots were evaluated with the same vocoder on Freya-100 (studio voice, T 0.3, α 2, the scripts of
  `vocoder_quality.py`).
- The control is the identical fine-tune without the adversary.

Measured cost and runs:
- **Cost:** 19.1 GB vs 17.9 GB (nvidia-smi); 1.7 vs 2.3 it/s on a shared GPU, so −30% throughput.
- **`real: audio`** was stopped at 2k steps, by the rule "stop if the discriminator gap stays > 0.6":
  - D(real) 0.96 / D(fake) 0.03 after the 1k-step warm-up, then 0.98 / 0.02 by 2k;
  - the acoustic model cannot remove what the discriminator keys on.
- **`real: vocoded`:**
  - the gap grew slowly, 0.27 → 0.48 over 12k steps, and the crop-gradient cap bound in > 99% of the steps;
  - drift, prior, `across_sample_std` (0.209) and τ tracked the control to the third digit.

| model | WER | CER | UTMOSv2 | DNSMOS OVRL | F0 micro-variation | periodicity |
|---|---|---|---|---|---|---|
| v3.1 | 0.66% | 0.14% | 2.918 | 3.344 | 0.398 | 0.708 |
| SLM, `real: audio`, 2k | 0.77% | 0.18% | 2.929 | 3.350 | 0.392 | 0.707 |
| SLM, `real: vocoded`, 2k / 6k / 12k | 0.55 / 0.66 / 0.77% | 0.14 / 0.16 / 0.16% | 2.928 / 2.951 / 2.942 | 3.354 / 3.353 / 3.347 | 0.391 / 0.400 / 0.387 | 0.708 / 0.709 / 0.709 |
| control, 2k / 6k / 12k | 0.55 / 0.55 / 0.99% | 0.13 / 0.14 / 0.19% | 2.935 / 2.887 / 2.942 | 3.355 / 3.342 / 3.350 | 0.388 / 0.394 / 0.395 | 0.710 / 0.708 / 0.709 |

**SLM minus control on the same sentences and seeds** (UTMOSv2, 95% bootstrap CI):
- 2k: −0.007 [−0.052, +0.042]
- 6k: +0.065 [+0.015, +0.114]; the control's 6k snapshot is its lowest.
- 12k: −0.000 [−0.043, +0.043]

The control's own snapshots vary by ±0.03 UTMOSv2. Word errors are 5–9 of 911 throughout.

At this strength the adversary changes nothing that Freya-100's judges can see. The push it applies is about a
quarter of the drift's.

## What to look at while training

- **`train/centroid_mse`** should keep decreasing.
- **`train/across_sample_std`** should stay roughly flat. A drop towards 0 means collapse; growth means divergence.
- **`train/tau_mean`** (`drift.mode: kyutai`) should anneal after a brief rise.
- **`train/pitch`** is the token pitch MSE in normalised log-F0 units. `train/force_*` are the raw drift norms; the
  drift loss itself is constant by construction.
- **Audio samples** are written to `runs/<run>/samples/` every `sample_every` steps.

## Judging the result

Tune on the `dev` split and report on `val` once:

```bash
M=runs/tts_v3/model_ema.pt
drifting-tts evaluate --model $M --harmonic --split dev --num 200 --temperature 0.3 0.5 0.7 1.0
drifting-tts evaluate --model $M --split dev --num 100 --temperature 0.3 0.5 0.7 --cfg 1.0 1.5 2.0
drifting-tts calibrate-durations --model $M --temperature 0.3      # store the chosen temperature
drifting-tts benchmark --model $M --speaker studio                 # Freya-TR-Eval
```

**Listen** above all: UTMOSv2 is trained on English and is only a relative proxy.

# The v2 recipe

`configs/tts_v2.yaml` adds the 2-D Mel-MAE with spectral-detail kernel features and FastPitch-style pitch
conditioning. `configs/tts_v2_k4.yaml` adds 4-step drifting with on-policy rollout. `scripts/train_v2.sh` runs the
v2 pipeline end to end.

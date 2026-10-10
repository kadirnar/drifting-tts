# Stochastic prosody predictor (#39)

v3.1 takes its per-token durations and token pitch from two deterministic MSE regressors
(`models/text_encoder.py: DurationPredictor`, reused as the pitch predictor). Every text therefore gets the
conditional-mean tune and rhythm, and the noise temperature of the generator does not change prosody
([EXPERIMENTS.md §5](EXPERIMENTS.md#5-robotic-prosody-diagnosis-and-research)). The DiT was trained on ground-truth
MAS durations and ground-truth token pitch, so a sampler can replace the regressors at inference **without retraining
the DiT**. This page describes that sampler, trained with the drifting objective, and compares it with a regression
and a flow-matching baseline on the same backbone.

- [Design](#design)
- [Results](#results)
- [Usage](#usage)
- [Sampled rhythm without the slips](#sampled-rhythm-without-the-slips)
- [Word-level context (#40)](#word-level-context-40)
- [Notes and pitfalls](#notes-and-pitfalls)
- [Phase 2: conditional intonation (#40)](#phase-2-conditional-intonation-40)

## Design

**Targets** (`drifting-tts prosody-cache`, ~2 min on the GPU). The frozen v3.1 text encoder (eval mode, fp32) is run
over the training corpus: the training split with the `data.filters` of `configs/tts_v3.yaml` (40,069 distinct
utterances; the studio voice is listed twice, as in training) and `val` / `dev` (300 + 300). Per token it stores the
MAS duration (`align(mu, x_mask, y, y_mask)` as in `train.py`; always ≥ 1 frame, 47% of tokens have exactly one),
the token pitch of `token_pitch` with the model's `lf0_stats` (0 for unvoiced tokens), the number of voiced frames and
the regressors' predictions.

**Model** (`models/prosody_net.py`, 8.1 M parameters). Inputs per token: the frozen encoder states `h` (192), the
speaker embedding (64), the two regressor predictions (standardised), per-token noise (16 channels) and a global noise
vector (32, added to every layer). A 1×1 input projection, two residual convolutions and four RoPE transformer
layers (d 256, the text encoder's `EncoderLayer`) predict three channels per token:

| channel | target | at inference |
|---|---|---|
| log-duration | `log d` of the MAS frames, standardised | `round(exp(·) × voice factor)`, at least 1 frame |
| pitch | a continuous token contour: voiced tokens' pitch, unvoiced tokens interpolated from their neighbours | used where the voicing flag is on |
| voicing logit | voiced frames > 0 (binary cross-entropy, weight 0.1) | unvoiced tokens get pitch 0, as the DiT saw in training |

The interpolated contour keeps the pitch channel smooth, so the pooled feature maps below see the melody instead of
the voicing pattern. The two prosody channels are predicted as a residual over the regressors, with a small output
layer at initialisation (the first samples are close to v3.1's prosody).

**Objective** (`train_prosody.py`). For each utterance, G = 16 samples are generated in one pass, and the recording's
prosody is the single positive. The kernel space has to see joint structure, otherwise samples become independent
per token and jittery. So the drifting loss (`drift.py: kyutai_drift_loss`: one learned temperature τ starting at 1.0,
per-row distance normalisation, product-form field) is applied on nine feature maps of the standardised sequence
(`prosody_features`). Each location is one drift problem, and the loss is averaged over maps:

| map | location | features |
|---|---|---|
| `tok` | token | log-duration, pitch |
| `d1` | token pair | first differences (smoothness) |
| `avg3`, `avg9`, `avg27` | windows of 3 / 9 / 27 tokens, stride k/2 | masked means |
| `win8`, `win32` | windows of 8 / 32 tokens, stride k/2 | the whole window flattened (local contour and rhythm shape) |
| `word` | word (space-separated) | mean log-duration, mean pitch, pitch range, log word length; standardised |
| `utt` | utterance | mean and std of both channels, pitch slope, log total length; standardised |

Padding locations stay in the batch with weight 0 (`kyutai_drift_loss(row_weight=...)`). Selecting the valid rows
with boolean indexing cost one host sync per map, and that made a step 2.7× slower on the shared GPU.

**Baselines on the same backbone and data:** `mse` (no noise, regression on the same targets: the deterministic status
quo with a bigger network) and `flow` (conditional flow matching on the residual, 8 Euler steps at inference, the
time embedding replaces the global noise).

**Training.** AdamW (0.9, 0.999), LR 3e-4 constant after 500 warm-up steps, no weight decay, gradient clip 3,
EMA 0.999 (Kyutai's recipe), 12 utterances × 16 samples per step. The drift run uses ~6 GB and runs at ~5 it/s on a
shared RTX 5090.

**Two temperatures.** `temperature` scales the input noise, as in Kyutai's head. In the drift sampler it changes the
diversity between seeds, but hardly the spread within an utterance: the zero-noise output already has realistic
spread (see below). `spread` (output-space temperature) draws 16 samples in one batch and returns
`mean + spread × (sample − mean)`. Below 1 it gives a monotone trade between expressiveness and per-token accuracy.

## Results

All rows use the frozen v3.1 model. Training: 12k steps for the drift samplers (dev metrics were flat from 6k; the
run was stopped there to free the shared GPU), 20k for MSE and flow matching. Checkpoints were selected on `dev`, the
temperatures on `dev`; the tables report `val` + `dev` (300 + 300 held-out utterances, 254 speakers, of which 200
are studio-voice utterances). Sampled rows use 8 seeds.

### Token level (no audio)

Each predictor is compared with the targets the DiT was trained on (MAS durations, token pitch) on the held-out
utterances (`scripts/eval_prosody_tokens.py`). The spreads are within-utterance standard deviations relative to the
recordings, so 1 means as varied as the recordings. A "letter" is a character and the blank after it. Jitter is
the mean |Δ pitch| between neighbouring voiced tokens, relative to the recordings. Reversals are local pitch-direction
changes of more than 0.5 semitone per voiced letter. CRPS is the continuous ranked probability score over the seeds
(lower is better). It is a proper score, so deterministic and stochastic predictors are comparable (for one sample
it is the MAE). W1 is the Wasserstein distance between the pooled pitch deviations. Length ratio uses each
predictor's per-voice factors.

**val+dev studio (722)**

| predictor | T | pitch spread | word pitch spread | letter dur spread | word dur spread | pitch jitter | reversals / letter | pitch r | letter dur r | pitch CRPS | log-dur CRPS | W1 pitch | length ratio | seed div. pitch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recordings | – | 1 | 1 | 1 | 1 | 1 | 0.371 | 1 | 1 | 0 | 0 | 0 | 1 | – |
| v3.1 regressors | – | 0.732 | 0.738 | 0.578 | 0.948 | 0.797 | 0.364 | 0.704 | 0.691 | 0.301 | 0.408 | 0.113 | 1.013 | 0 |
| MSE (same backbone) | – | 0.788 | 0.832 | 0.803 | 0.997 | 0.779 | 0.359 | 0.784 | 0.727 | 0.267 | 0.352 | 0.086 | 1.008 | 0 |
| flow matching | 0.5 | 0.776 | 0.795 | 0.912 | 1.014 | 0.816 | 0.362 | 0.756 | 0.695 | 0.220 | 0.256 | 0.092 | 0.925 | 0.098 |
| flow matching | 1 | 0.897 | 0.923 | 0.967 | 1.006 | 0.895 | 0.359 | 0.676 | 0.611 | 0.191 | 0.226 | 0.042 | 0.986 | 0.227 |
| **drift** | 0.5 | 0.912 | 0.890 | 0.944 | 1.010 | 1.041 | 0.400 | 0.670 | 0.640 | 0.220 | 0.258 | 0.030 | 0.969 | 0.181 |
| **drift** | 1 | 0.978 | 1.008 | 1.005 | 1.010 | 1.008 | 0.374 | 0.594 | 0.545 | 0.200 | 0.240 | 0.011 | 1.005 | 0.302 |
| drift, spread 0.8 | 1 | 0.896 | 0.922 | 0.960 | 1.015 | 0.957 | 0.380 | 0.645 | 0.583 | 0.203 | 0.242 | 0.041 | 0.959 | 0.235 |
| drift + BERTurk | 1 | 0.970 | 0.964 | 1.011 | 1.014 | 1.002 | 0.376 | 0.639 | 0.556 | 0.190 | 0.240 | 0.015 | 1.003 | 0.296 |
| drift + BERTurk, spread 0.8 | 1 | 0.909 | 0.912 | 0.947 | 1.001 | 0.937 | 0.370 | 0.672 | 0.601 | 0.192 | 0.242 | 0.036 | 0.958 | 0.239 |

**val+dev (all speakers)**

| predictor | T | pitch spread | word pitch spread | letter dur spread | word dur spread | pitch jitter | reversals / letter | pitch r | letter dur r | pitch CRPS | log-dur CRPS | W1 pitch | length ratio | seed div. pitch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recordings | – | 1 | 1 | 1 | 1 | 1 | 0.310 | 1 | 1 | 0 | 0 | 0 | 1 | – |
| v3.1 regressors | – | 0.593 | 0.594 | 0.436 | 0.823 | 0.705 | 0.295 | 0.522 | 0.512 | 0.425 | 0.564 | 0.162 | 0.955 | 0 |
| MSE (same backbone) | – | 0.667 | 0.733 | 0.626 | 0.872 | 0.626 | 0.261 | 0.579 | 0.530 | 0.402 | 0.490 | 0.130 | 0.818 | 0 |
| flow matching | 0.5 | 0.662 | 0.714 | 0.785 | 0.929 | 0.627 | 0.247 | 0.546 | 0.485 | 0.330 | 0.362 | 0.133 | 0.819 | 0.139 |
| flow matching | 1 | 0.862 | 0.900 | 0.908 | 0.950 | 0.805 | 0.269 | 0.450 | 0.396 | 0.289 | 0.318 | 0.054 | 0.953 | 0.321 |
| **drift** | 0.5 | 0.913 | 0.957 | 0.937 | 0.960 | 0.981 | 0.320 | 0.436 | 0.426 | 0.319 | 0.351 | 0.032 | 0.952 | 0.291 |
| **drift** | 1 | 0.989 | 1.048 | 0.980 | 0.967 | 1.010 | 0.307 | 0.379 | 0.348 | 0.297 | 0.326 | 0.009 | 0.993 | 0.418 |
| drift, spread 0.8 | 1 | 0.869 | 0.926 | 0.898 | 0.943 | 0.912 | 0.305 | 0.420 | 0.379 | 0.301 | 0.328 | 0.048 | 0.913 | 0.329 |
| drift + BERTurk | 1 | 0.971 | 0.976 | 0.978 | 0.972 | 1.036 | 0.323 | 0.435 | 0.360 | 0.283 | 0.325 | 0.013 | 1.002 | 0.413 |
| drift + BERTurk, spread 0.8 | 1 | 0.873 | 0.881 | 0.879 | 0.930 | 0.937 | 0.314 | 0.468 | 0.394 | 0.285 | 0.328 | 0.050 | 0.923 | 0.332 |

- **The regressors are flat.** v3.1's token pitch has 59% of the recordings' within-utterance spread (73% for
  the studio voice), and its letter durations 44% (58%). A bigger regressor trained the same way (MSE) barely helps.
- **Both samplers restore the spread.** The drift sampler matches the recordings' distributions almost exactly at
  T = 1: spreads 0.97–1.05, jitter 1.01, reversals 0.307 per letter (recordings 0.310), W1 0.009, length 0.99 without
  calibration. Flow matching stays 10–14% narrow and smoother than the recordings (jitter 0.81–0.90).
- **Per-token accuracy.** Flow matching beats plain drift on CRPS (pitch 0.289 vs 0.297, log-duration 0.318 vs
  0.326; studio 0.191 vs 0.200) and correlation. The drift samples vary more independently of the text: with one
  positive per text, τ sets the spread (Notes). BERTurk word features (#40) give drift the best pitch CRPS (0.283 all
  speakers, 0.190 studio) at the same distributional match.
- **Temperature.** For flow matching, T scales the spread (T 0.5: pitch spread 0.66, jitter 0.63; that is MSE-flat).
  For drift, T mostly scales the seed diversity (pitch 0.29 → 0.42 between T 0.5 and 1). The spread changes much
  less (0.91 → 0.99), because the zero-noise output already has realistic spread. `spread 0.8` is the knob that
  narrows drift (pitch spread 0.87, CRPS within 0.004).

### Audio: studio voice, 100 held-out `val` utterances

`scripts/eval_prosody_audio.py` adds the samplers as systems to `drifting-tts prosody` (prosody-eval's runner, same
metrics and recordings). It uses one pass over the whole text, T = 0.3, α = 2, vocos-ft, seed = utterance index,
harvest F0 in semitones. `onepass` is v3.1 as released (regressors); `oracle-both` gets the recording's MAS
durations and token pitch. `-pitch` samples only the token pitch and keeps the regressors' durations.

| system | F0 std | F0 range | move | micro | reversals/s | pauses/utt | syl/s | DTW F0 r | render flat | seed F0 spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recordings | 3.68 | 11.9 | 0.63 | 0.42 | 10.9 | 1.39 | 6.22 | – | 1.00 | – |
| v3.1 (regressors) | 3.20 | 10.2 | 0.67 | 0.47 | 11.3 | 0.45 | 6.10 | 0.609 | 1.19 | 1.39 |
| oracle prosody | 3.81 | 12.2 | 0.71 | 0.50 | 11.6 | 1.39 | 6.22 | 0.804 | 1.05 | – |
| MSE | 3.38 | 10.8 | 0.67 | 0.47 | 11.1 | 0.45 | 6.16 | 0.656 | 1.13 | – |
| flow matching, T 1 | 3.59 | 11.5 | 0.70 | 0.50 | 11.4 | 0.98 | 6.30 | 0.585 | 1.05 | 2.32 |
| drift, T 1 | 3.77 | 12.1 | 0.70 | 0.50 | 11.4 | 1.65 | 6.14 | 0.518 | 1.05 | 2.59 |
| drift, T 1, spread 0.8 | 3.62 | 11.5 | 0.71 | 0.50 | 11.6 | 1.27 | 6.42 | 0.542 | 1.07 | – |
| drift, T 0.5 (factors of T 1) | 3.63 | 11.4 | 0.72 | 0.52 | 11.6 | 1.44 | 6.40 | 0.562 | 1.06 | – |
| **drift, T 0.5 (final checkpoint)** | 3.63 | 11.4 | 0.71 | 0.51 | 11.5 | 1.48 | 6.21 | 0.565 | 1.06 | 2.19 |
| drift, T 1, pitch only | 3.78 | 12.0 | 0.71 | 0.50 | 11.4 | 0.53 | 6.10 | 0.518 | 1.07 | 2.28 |
| drift + BERTurk, T 1 | 3.78 | 12.2 | 0.71 | 0.51 | 11.4 | 1.70 | 6.21 | 0.551 | 1.05 | 2.60 |
| drift + BERTurk, T 1, pitch only | 3.75 | 12.0 | 0.71 | 0.49 | 11.1 | 0.48 | 6.11 | 0.561 | 1.06 | – |

- **Intonation range is fixed.** The F0 std goes from 3.20 (v3.1) to 3.77 (drift, T 1) or 3.63 (T 0.5), against
  3.68 for the recordings and 3.81 with oracle prosody. The 5–95% range goes from 10.2 to 12.1 / 11.4 (recordings
  11.9). With factors calibrated at T 0.5, the final checkpoint also matches the recordings' speaking rate (6.21 vs
  6.22 syllables/s). The DiT no longer
  has to stretch a flat input: "render flat" (std of the realised token F0 over std of the conditioning token pitch)
  drops from 1.19 to 1.05, as with oracle prosody. Flow matching gets about two thirds of the way (3.59 / 11.5).
- **Pauses.** Sampled durations put pauses inside the utterance (1.48–1.65 per utterance, recordings 1.39, v3.1
  0.45). The pitch-only mode keeps v3.1's pause pattern.
- **Different, not worse, tunes.** DTW log-F0 correlation with the specific recording drops (0.61 → 0.52; oracle
  0.80), because each sample is one plausible tune among many. Seed diversity of F0 rises from 1.39 to 2.59 st.
- **Micro-variation** (0.50) and reversals/s (11.4) are slightly above the recordings (0.42 / 10.9) for every one-pass
  system, oracle included (0.50 / 11.6). They come from the DiT + Vocos rendering, not from the predictors.

### Guard rails: Freya-100

`drifting-tts benchmark --num 100 --speaker 722 --vocoder vocos-ft` (T 0.3, α 2, sentence by sentence, Whisper
large-v3 on 8 kHz band-matched audio, UTMOSv2 full band). The v3.1 row reproduces the published one exactly.

| prosody | WER | CER | UTMOSv2 | RTF (shared GPU) |
|---|---:|---:|---:|---:|
| v3.1 (regressors) | 1.10% [0.44, 1.89] | 0.22% | 2.627 [2.585, 2.669] | 0.0153 |
| flow matching, T 1 | 1.10% [0.44, 1.86] | 0.24% | 2.742 [2.695, 2.788] | 0.0272 |
| flow matching, T 0.7 | 0.77% [0.22, 1.51] | 0.19% | 2.695 [2.641, 2.744] | 0.0220 |
| flow matching, T 1, pitch only | 0.77% [0.22, 1.52] | 0.19% | 2.723 [2.683, 2.761] | 0.0210 |
| drift, T 1 | 2.63% [1.61, 3.88] | 0.58% | 2.688 [2.642, 2.733] | 0.0182 |
| drift, T 1, spread 0.8 | 1.87% [0.89, 2.98] | 0.48% | 2.702 [2.652, 2.754] | 0.0148 |
| drift, T 0.7 | 1.21% [0.54, 2.00] | 0.26% | 2.706 [2.654, 2.759] | 0.0147 |
| drift, T 0.5 (factors of T 1) | 0.66% [0.11, 1.32] | 0.14% | 2.722 [2.675, 2.771] | 0.0200 |
| **drift, T 0.5, factors calibrated at T 0.5 (final)** | 0.99% [0.33, 1.73] | 0.22% | 2.712 [2.668, 2.757] | 0.0147 |
| drift, T 1, pitch only | 0.77% [0.11, 1.55] | 0.13% | 2.693 [2.647, 2.738] | 0.0144 |
| drift + BERTurk, T 1 | 4.50% [2.53, 6.69] | 1.09% | 2.647 [2.594, 2.699] | 0.0294 |
| drift + BERTurk, T 0.5 | 1.65% [0.76, 2.67] | 0.34% | 2.669 [2.621, 2.717] | 0.0242 |
| drift + BERTurk, T 1, pitch only | 0.66% [0.22, 1.21] | 0.16% | 2.643 [2.594, 2.688] | 0.0231 |

- **The prosody temperature decides intelligibility.** With durations sampled at T 1, the drift sampler loses
  intelligibility on these everyday sentences (WER 2.63%). The errors are single-phoneme slips such as duymak →
  doymak, oldu → olu, ağrıyor → arıyor and salona → salonu. The recordings have 10% of their letters at ≤ 2 frames,
  and the samplers reproduce that share (v3.1: 0.1%, because `ceil` and log-mean regression avoid short letters).
  The DiT renders those short phones less reliably on out-of-domain text, and more so when they land in the wrong
  place. At T 0.5 the samples are more typical: WER 0.66–0.99% and CER 0.14–0.22% (two calibrations), within the v3.1
  intervals; T 0.7 lies in between (1.21%).
- **UTMOSv2 rises with every character-level sampler** (2.69–2.74, at or above the upper end of the v3.1
  interval [2.585, 2.669]), including the pitch-only modes (drift 2.693, flow 2.723). The BERTurk pitch-only row
  stays at v3.1's level (2.643). UTMOSv2 is a weak judge of intonation (EXPERIMENTS.md), so these are guard rails,
  not evidence of naturalness.
- Flow matching at T 1 is as intelligible as v3.1 with the highest UTMOSv2 (2.742), but it is flatter
  (F0 std 3.59) and costs 8 network evaluations.

### Turkish polar questions (#40 diagnostic)

17 held-out utterances with a "?" and a mI question word (`scripts/eval_question_pitch.py`, token pitch, semitones;
8 seeds). `pre-mI` is the mean pitch of the word before mI relative to the utterance mean. `fall` is the pitch of the
last word minus that of the pre-mI word. Turkish polar questions mostly peak before mI and end low (§4.2 of the
research memo).

| predictor | pre-mI (st) | fall (st) | ends low |
|---|---:|---:|---:|
| recordings | +0.05 | −4.24 | 71% |
| v3.1 regressors | −1.17 | −0.95 | 65% |
| MSE | −0.49 | −1.93 | 71% |
| flow matching, T 1 | −0.98 | −1.82 | 71% |
| drift, T 1 | −1.57 | −0.52 | 62% |
| drift + BERTurk, T 1 | −0.13 | −2.37 | 68% |

This is indicative only (17 questions), but it is the one place where word context visibly helps. Character-level
predictors miss the pre-mI peak, and the BERTurk variant gets closest on both measures.

### Training dynamics

| run | step | pitch spread | word pitch spread | log-dur spread | jitter | pitch r | W1 pitch | length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drift | 2k | 0.81 | 0.79 | 0.88 | 1.04 | 0.46 | 0.077 | 0.92 |
| drift | 4k | 0.96 | 0.98 | 0.95 | 1.03 | 0.40 | 0.016 | 0.98 |
| drift | 12k | 1.00 | 1.05 | 0.99 | 1.03 | 0.38 | 0.009 | 0.98 |
| drift + BERTurk | 12k | 0.97 | 0.98 | 0.99 | 1.04 | 0.44 | 0.013 | 1.00 |
| flow matching | 20k | 0.87 | 0.91 | 0.92 | 0.82 | 0.45 | 0.053 | 0.94 |
| MSE | 20k | 0.67 | 0.73 | 0.66 | 0.63 | 0.58 | 0.131 | 0.81 |

(`dev`, T = 1, 4 seeds, uncalibrated lengths.) τ falls from 1.0 to 0.11 within 3k steps and stays at 0.10–0.11
(Kyutai: ≈ 0.056). The kernel mass on the data (`p_data`) rises from 0.05 to ≈ 0.45, and the sample spread stays
at ≈ 0.62 standardised units throughout. There is no collapse and no divergence. Seed diversity is stable from 4k
steps.

## Usage

**Release v3.2 samples only the token pitch** (`prosody="drift"`, the published `prosody_drift_v3.2.pt`, at
prosody temperature 0.5, `prosody_durations="regressor"`): the new intonation with v3.1's rhythm, durations and
per-voice factors. On Freya-100 (studio) sampled durations at T 0.5 had looked safe (WER 0.99%, CER 0.22%, v3.1
1.10% / 0.22%), but the release evaluation on all 495 sentences and three voices did not hold that up: with Vocos v2,
sampled durations give WER 1.89% (studio), 5.78% (male) and 11.28% (female), against 1.33%, 2.28% and 3.99% with the
pitch only (v3.1 + BigVGAN-v2-ft: 1.23%, 1.74%, 3.02%; [RESULTS.md](RESULTS.md#v32-sampled-intonation-vocos-v2-punctuation-pauses)).
The voices with little training data cannot render the sampled short sounds on new text. Sampled durations
(`prosody_durations="sampled"`, the sampler's own voice factors from `train-prosody --calibrate-only
calibrate.temperature=0.5`) stay an opt-in for the studio voice; with the letters' durations at prosody temperature
0.3 (`prosody_duration_temperature=0.3`) they are as intelligible as the release
([below](#sampled-rhythm-without-the-slips)). The explicit API keeps the regressors by default.

```bash
# 1. targets of a trained pitch-conditioned model (~2 min)
drifting-tts prosody-cache --model runs/release/drifting_tts_v3.1.pt --data data/train \
    --out runs/pm_cache/targets_v31.pt
# 2. train (configs/prosody_drift.yaml; net.kind=mse / flow for the baselines)
drifting-tts train-prosody --workdir runs/pm_drift tts=runs/release/drifting_tts_v3.1.pt \
    cache=runs/pm_cache/targets_v31.pt train.batch_size=12
# 3. per-voice duration factors and the preferred prosody temperature (training utterances only)
drifting-tts train-prosody --workdir runs/pm_drift --calibrate-only tts=... cache=... calibrate.temperature=0.5
# 4. synthesise with it (opt-in; the default stays the deterministic regressors)
drifting-tts synthesize --model runs/release/drifting_tts_v3.1.pt --prosody runs/pm_drift/prosody_ema.pt \
    --temperature 0.3 --cfg 2 --vocoder vocos-ft --text "..."
#    --prosody-temperature T (default: the stored one), --prosody-spread S, --prosody-durations regressor
# 5. guard rails with it
drifting-tts benchmark --model runs/release/drifting_tts_v3.1.pt --num 100 --speaker 722 --vocoder vocos-ft \
    --prosody runs/pm_drift/prosody_ema.pt
```

```python
synth = Synthesizer("drifting_tts_v3.1.pt", vocoder="vocos-ft", prosody="prosody_ema.pt")  # T from the checkpoint
wav, _ = synth(text, speaker="studio", cfg_scale=2.0, temperature=0.3, seed=0)
synth = Synthesizer.from_pretrained("v3.2")   # prosody="drift", pitch only, vocos-v2, pause="punct"
wav, _ = synth(text, speaker="studio", cfg_scale=2.0, temperature=0.3, seed=0, prosody_temperature=0.7)  # per call
```

- **Seeds.** The prosody noise is drawn from the same seeded generator as the DiT noise, before it, so a seed
  fixes the whole rendition.
- **Speaking rate.** With sampled durations the sampler uses its own per-voice duration factors (`train-prosody
  --calibrate-only`: the median recorded / sampled length on training utterances, never on the evaluation splits)
  instead of the v3.1 factors, which compensate the regressors' log-domain bias and `ceil`. With
  `prosody_durations="regressor"` (v3.2) the v3.1 factors stay. `length_scale` applies on top either way.
- **`fast=True`.** A one-pass sampler (`drift`, `mse`; no word features, spread 1) runs inside the text encoder's
  CUDA graph (`drifting_tts/fast.py`), its noise drawn outside the graph in the eager order: the same draws and
  mels within float noise of the eager path (with sampled durations, a rounding can flip on that noise: 9 of 366
  test sentences got one frame more or less). Flow matching, BERTurk features and `spread` != 1 run eagerly, with
  only the streaming vocoder windows in CUDA graphs.
- **Publishing.** `scripts/prepare_release.py` keeps what `ProsodyPredictor.load` reads (no training config, cache
  or TTS paths) and records a fingerprint of the text encoder: loading the predictor with another acoustic model
  warns. It also stores the edge silence of the predictor's generated sentences per voice (`pause_edges`, studio
  0.20 s against 0.16 s with the regressors), which `pause="punct"` subtracts from the measured pauses when the
  sampled durations are used.
- **Cost.** One 165-token sentence on the shared (busy) RTX 5090: text encoder 6.6 ms, + drift sampler 9.9 ms in
  total (one pass), flow matching with 8 Euler steps 35.7 ms, drift + BERTurk 48 ms (BERT dominates). Busy-GPU
  numbers, 2–4× above an idle GPU; the drift sampler adds a few milliseconds to time-to-first-audio.
- **Not ported:** ONNX / WebGPU and MLX still use the regressors.

## Sampled rhythm without the slips

Listening, the owner preferred v3.2 with **sampled durations** on the studio voice: pauses inside sentences and a
rhythm closer to the recordings. The demo offers it as an opt-in, but on Freya-495 it costs intelligibility (Vocos
v2, T 0.3, α 2, prosody T 0.5): WER 1.89% against 1.33% for the release (pitch only) on the studio voice, 5.78%
against 2.28% (male) and 11.28% against 3.99% (female). This section finds where the slips come from and removes
them for the studio voice without retraining: **the letters' durations at prosody temperature 0.3, the pauses and the
pitch at 0.5.**

### Where the slips come from

Freya-495 for each voice, the renditions reproduced exactly (same seeds and draws) to read the durations behind every
word error (the analysis scripts are in `runs/agents/rhythm/scripts`, outside the repository; the WERs come from
the same Whisper transcripts).

- **Not the speaking rate.** The sampled renditions are as long as the pitch-only ones (total length 0.99 / 1.00 /
  1.04 of them for studio / male / female); both sets of per-voice factors come from the same kind of calibration on
  training utterances.
- **The letters, not the pauses.** Sampling the durations of the letters (a character and the blank after it) and
  of the tokens between words (spaces, punctuation and the blanks next to them, where pauses live) at different
  temperatures separates the two (studio, one seed set, paired with the release on the same sentences and seeds;
  95% bootstrap intervals over sentences; the three-seed check is below):

| letters | pauses | WER | ΔWER vs pitch only | sentences with a pause ≥ 0.1 s | syllables/s |
|---|---|---:|---|---:|---:|
| regressors | regressors (the release) | 1.33% | – | 0.6% | 6.11 |
| T 0.5 | T 0.5 (sampled durations, the demo's opt-in) | 1.89% | +0.56 pp [+0.15, +1.01] | 11.3% | 6.24 |
| T 0.5 | T 0.3 | 1.66% | +0.33 pp [−0.03, +0.70] | 6.1% | 6.25 |
| T 0.3 | T 0.3 | 1.33% | +0.00 pp [−0.33, +0.33] | 6.7% | 6.28 |
| **T 0.3** | **T 0.5** | 1.36% | +0.03 pp [−0.33, +0.39] | 10.7% | 6.27 |
| T 0 | T 0.5 | 1.41% | +0.08 pp [−0.26, +0.43] | 10.9% | 6.29 |
| regressors | T 0.5 | 1.33% | +0.00 pp [−0.25, +0.26] | 10.5% | 6.15 |

  The letters at T 0.5 cost the words; the pauses do not. Letters at T 0.3 with pauses at T 0.5 keep both the
  pauses (10.7% of these short everyday sentences get one, as with T 0.5) and the release's intelligibility.
- **Where the short letters land.** The studio recordings have 9.9% of their letters at ≤ 2 frames, and the sampler
  keeps that share at every temperature: on Freya 41–43% of the words get such a letter at T 0.3 and at T 0.5 (0.3%
  with the regressors). At T 0.5 the words with a short letter fail more often than the others (2.04% against 1.26%
  word errors); at T 0.3 they do not (1.31% against 1.21%). The noise at T 0.5 puts short letters where the text does
  not support them; at T 0.3 they stay where the zero-noise output, which already has the recordings'
  letter-duration spread (phase 2), puts them. The failing words are spread over the sentence (first word 5.1%
  against 4.2% with the regressors, middle words 1.4% against 0.9%).
- **Male and female: the sampler reproduces their recordings, and those are irregular.** MAS durations of the
  training recordings against the sampler on 200 Freya sentences:

| | letters ≤ 2 frames | jitter (mean \|Δ log\| of neighbouring letters) | first letter ≤ 2 frames | leading blank ≤ 2 frames |
|---|---:|---:|---:|---:|
| studio: recordings / sampled (T 0.5) | 9.9% / 9.1% | 0.60 / 0.55 | 6% / 0% | 15% / 2% |
| male: recordings / sampled | 10.7% / 8.1% | 0.71 / 0.69 | 19% / 6% | 14% / 9% |
| female: recordings / sampled | 16.5% / 14.8% | 0.76 / 0.77 | 46% / 57% | 100% / 100% |
| regressors (any voice) | ≤ 0.05% | 0.34–0.35 | 0% | 0–100% |

  The female recordings start abruptly: the leading blank has at most two frames in all of them (one in most), and
  the first letter is cut to ≤ 2 frames in 46%. The sampler reproduces that, and the generator renders it as a dropped or changed first
  sound: 26% of the female first words fail with sampled durations (pitch only: 13.5%; studio: 4–5%), and the words
  inside the sentence fail 3× as often as with the regressors (7.9% against 2.4%). The temperature hardly changes
  these distributions (female, T 0 / 0.3 / 0.5: 13.4 / 13.9 / 14.8% short letters), so lowering it does not fix
  these voices (female, durations at T 0.3: 8.34%).

### Male and female voices

Inference-only remedies, Freya-495, one seed set, paired with the pitch-only release of each voice:

| durations of the male / female voice | male WER | Δ [95% CI] | female WER | Δ [95% CI] |
|---|---:|---|---:|---|
| regressors (the release) | 2.28% | – | 3.99% | – |
| its own, T 0.5 (sampled durations) | 5.78% | +3.50 [+2.67, +4.39] | 11.28% | +7.29 [+6.12, +8.46] |
| its own, T 0.3 | 4.32% | +2.05 [+1.34, +2.80] | 8.34% | +4.35 [+3.31, +5.41] |
| its own, T 0.5, leading blank ≥ 5 and first letter ≥ 4 frames | | | 9.72% | +5.73 [+4.53, +6.94] |
| regressors for the letters, its own pauses at T 0.5 | 3.22% | +0.95 [+0.38, +1.54] | 5.40% | +1.41 [+0.69, +2.16] |
| the studio voice's rhythm at its rate, T 0.5 | 3.48% | +1.20 [+0.57, +1.86] | 5.19% | +1.20 [+0.31, +2.10] |
| the same, its leading blank from its regressor | 3.30% | +1.02 [+0.38, +1.68] | 5.29% | +1.30 [+0.36, +2.24] |
| the same, its leading blank and first letter from its regressor | | | 4.81% | +0.82 [−0.08, +1.73] |
| **the studio voice's rhythm at its rate, T 0.3** | 2.76% | +0.49 [−0.08, +1.05] | 4.70% | +0.72 [−0.13, +1.49] |
| the same, its leading blank and first letter from its regressor | 3.09% | +0.82 [+0.23, +1.43] | 4.55% | +0.56 [−0.26, +1.32] |
| the same, also the sentence end (final punctuation and blank) | 2.86% | +0.59 [−0.03, +1.25] | 4.32% | +0.33 [−0.47, +1.16] |
| regressors with a leading blank ≥ 5 frames (pitch only) | | | 3.94% | −0.05 [−0.60, +0.53] |

Three seed sets (as for the studio voice below) for the remedy that works, borrowing the studio voice's rhythm at
T 0.3:

| voice | durations | WER [95% CI] | ΔWER vs v3.2 [95% CI] | UTMOSv2 | ΔUTMOSv2 [95% CI] |
|---|---|---|---|---:|---|
| male | v3.2 (regressors) | 2.52% [2.17, 2.88] | – | 2.895 | – |
| male | studio rhythm, factor from his recordings' length (1.50) | 3.06% [2.71, 3.43] | +0.54 [+0.18, +0.91] | 2.831 | −0.064 [−0.078, −0.049] |
| male | the same with his own sentence edges, factor from his regressors' length (1.46; the checkpoint) | 3.13% [2.75, 3.51] | +0.61 [+0.26, +0.97] | 2.843 | −0.052 [−0.066, −0.038] |
| female | v3.2 (regressors) | 4.15% [3.72, 4.57] | – | 2.718 | – |
| female | studio rhythm, factor from her recordings' length (1.33) | 4.92% [4.46, 5.39] | +0.77 [+0.26, +1.28] | 2.751 | +0.033 [+0.018, +0.048] |
| female | the same with her own sentence edges | 4.41% [3.97, 4.84] | +0.26 [−0.22, +0.74] | 2.734 | +0.016 [+0.003, +0.031] |
| female | own edges, factor from her regressors' length (1.30; the checkpoint) | 4.55% [4.13, 5.02] | +0.40 [−0.07, +0.87] | 2.767 | +0.049 [+0.034, +0.062] |

- **Borrowing the studio voice's rhythm** (its sampler conditioned on speaker 722, scaled to the voice's length on
  its training utterances; the pitch stays the voice's own) is the only remedy that removes most of the cost. It
  leaves +0.5–0.6 pp for the male voice and +0.3–0.4 pp for the female voice (with her own sentence edges), above
  the 0.2–0.3 pp the studio voice reaches. The male voice also loses UTMOSv2 (−0.05 to −0.06): the studio rhythm
  stretched to his length slows his articulation (4.60 against 4.90 syllables/s on Freya with the checkpoint's
  factor, 4.46 with the recordings' one).
- **Sentence edges matter for the female voice.** Her recordings start abruptly; with the studio rhythm her
  generator renders a full-length first consonant as an extra syllable (`geçen → ilçen`, `bahar → kulahar`; first
  words fail in 18–19% of the sentences against 13.9% with the regressors). Taking her leading blank, first letter
  and sentence end from her own regressor brings the cost from +0.77 to +0.26–0.40 pp. For the male voice it
  changes nothing measurable. Floors alone (leading blank ≥ 5, first letter ≥ 4 frames) on her own sampled
  durations recover only 1.6 of 7.3 pp, and a longer leading blank leaves the pitch-only voice unchanged (3.94%).
- **The voices' own pauses also cost words** (regressor letters, own pauses: +0.95 / +1.41 pp), unlike the studio
  voice's (+0.00 pp): their tokens between words include the blank after a word's last letter, and their sampled
  values are as irregular as their letters.
- **Retraining was not tried** (a speaker-pooled duration model, a penalty on short letters): even the best-modelled
  rhythm of the corpus, the studio voice's at T 0.3 with their own edges, costs these voices 0.4–0.6 pp, which
  points at how their generator renders varied durations rather than at their duration model. A speaker-pooled
  sampler would sit between their own rhythm and the studio's.

So **sampled rhythm stays a studio-voice option**; for the male and female voices the demo keeps falling back to
v3.2 (pitch only). The borrowed studio rhythm is in the recommended checkpoint for them (`rhythm` below): the
female voice is close (+0.40 pp [−0.07, +0.87]) and could be offered after listening; the male voice is not.

### Recommendation: letters at T 0.3, pauses and pitch at T 0.5

Studio voice, Freya-495 with three seed sets (seed = sentence index + 0 / 1000 / 2000: 1,485 renditions, 11,733
words), paired with the release on the same sentences and seeds:

| durations | WER [95% CI] | CER | UTMOSv2 | ΔWER vs v3.2 [95% CI] | ΔUTMOSv2 [95% CI] | sentences with a pause | syllables/s |
|---|---|---:|---:|---|---|---:|---:|
| v3.2 (regressors) | 1.36% [1.12, 1.60] | 0.26% | 3.016 | – | – | 0.6% | 6.11 |
| sampled, T 0.5 (the demo's opt-in so far) | 1.67% [1.40, 1.96] | 0.32% | 3.026 | +0.32 pp [+0.09, +0.54] | +0.010 [−0.002, +0.021] | 11.1% | 6.24 |
| sampled, T 0.3 | 1.50% [1.23, 1.77] | 0.28% | 3.030 | +0.14 pp [−0.06, +0.36] | +0.014 [+0.003, +0.025] | 6.0% | 6.27 |
| **letters T 0.3, pauses T 0.5** | **1.40%** [1.15, 1.66] | 0.27% | **3.032** | **+0.04 pp** [−0.15, +0.24] | +0.015 [+0.003, +0.026] | 11.6% | 6.26 |

Held-out studio `val` (`drifting-tts prosody`, the 100 recordings against each system's rendition of their texts,
sentence by sentence, as in [RESULTS.md](RESULTS.md#v32-sampled-intonation-vocos-v2-punctuation-pauses)):

| system | F0 std | F0 range | pauses/utt | pause s | syl/s | DTW F0 r | CER | WER | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recording | 3.68 | 11.9 | 1.39 | 0.139 | 6.22 | – | 0.88% | 2.06% | 3.093 | – |
| v3.2 (regressors) | 3.53 | 11.3 | 1.54 | 0.175 | 6.01 | 0.579 | 0.42% | 2.02% | 2.990 | 0.940 |
| v3.2, sampled durations at T 0.5 | 3.56 | 11.4 | 2.35 | 0.186 | 6.08 | 0.572 | 0.50% | 2.32% | 3.000 | 0.943 |
| **v3.2, letters T 0.3, pauses T 0.5** | 3.56 | 11.4 | 2.37 | 0.184 | 6.08 | 0.569 | 0.46% | 2.32% | 3.019 | 0.944 |

- **Intelligibility as the release.** Over three seed sets the recommended setting is +0.04 pp from v3.2 (the
  interval reaches +0.24 pp), where the demo's sampled durations at T 0.5 cost +0.32 pp. One seed set is too noisy
  for differences of this size: T 0.5 cost +0.56, +0.21 and +0.18 pp on the three.
- **The rhythm the owner liked is kept.** Pauses inside the sentence (2.37 per held-out utterance against 2.35 at
  T 0.5; Freya: 11.6% of the sentences against 11.1%), speaking rate (6.08 syllables/s against 6.08) and intonation
  (F0 std 3.56 against 3.56) are those of the T 0.5 setting. The letters' durations keep almost all of their spread
  at T 0.3 (phase 2, `dev`: letter-duration spread 0.934 of the recordings' at T 0.3, 0.941 at T 0.5, 0.430 for the
  regressors).
- **UTMOSv2** is +0.015 above v3.2 (interval above 0), as for every sampled-duration setting.
- **Pauses at T 0.5 are not worse than at T 0.3** (1.40% against 1.50%, within the noise), so nothing argues for
  sampling them colder.

The checkpoint `runs/rh_final/prosody_drift_v3.2_rhythm.pt` (local, not published) is the release's
`prosody_drift_v3.2.pt` with this operating point stored: preferred duration temperature 0.3, the per-voice factors
calibrated for it (studio 1.053, the release file's 1.0445 were calibrated with every duration at T 0.5), the male
and female voices borrowing the studio rhythm with their own sentence edges (factors 1.460 / 1.300, see below) and
the edge silence of its sentences for `pause="punct"` (studio 0.199 s, male 0.220 s, female 0.108 s). Through
`drifting-tts benchmark` (the package path, seed set 0) it gives studio WER 1.38%, CER 0.25%, UTMOSv2 3.032, as the
exploratory runner (1.36% / 3.040).

### Usage

```python
synth = Synthesizer.from_pretrained("v3.2")                   # pitch only (the release)
rhythm = synth.variant(prosody=synth.prosody, prosody_durations="sampled", prosody_duration_temperature=0.3)
wav, _ = rhythm(text, speaker="studio", cfg_scale=2.0, temperature=0.3, seed=0)   # pitch and pauses at T 0.5
```

```bash
drifting-tts synthesize --release v3.2 --prosody-durations sampled --prosody-duration-temperature 0.3 --text "..."
drifting-tts benchmark --model drifting_tts_v3.2.pt --vocoder vocos-v2 --prosody drift --pause punct \
    --prosody-durations sampled --prosody-duration-temperature 0.3 --speaker studio
# store the operating point in a checkpoint: the preferred duration temperature, a voice -> rhythm table and the
# factors calibrated for both (training utterances only)
drifting-tts train-prosody --workdir runs/pm_drift --calibrate-only tts=... cache=... calibrate.temperature=0.5 \
    calibrate.duration_temperature=0.3 "calibrate.rhythm={389: 722, 323: 722}"
```

- **`prosody_duration_temperature`** (`ProsodyPredictor.sample(duration_temperature=)`): the noise temperature of
  the letters' durations. The tokens between words (`boundary_tokens`: spaces, punctuation and the blanks next to
  them) keep the prosody temperature of the call, like the pitch. The same unit noise runs as a second row of the
  sampler's batch (as on the phase-2 branch), so a seed still fixes the rendition and the cost is one more row.
- **`rhythm`** (checkpoint key, `calibrate.rhythm`): voice → speaker whose durations it samples (all of them, its
  pauses included, at the duration temperature). The voice keeps its own pitch and its own sentence edges (the
  leading blank, the first letter and the final punctuation with its blank take its regressor durations:
  `edge_tokens`), and its `duration_scales` entry brings the borrowed rhythm to the length of its own regressor
  durations on its training utterances (the release's speaking rate; matching its recordings' length instead slows
  the male voice's articulation, since his recordings have long pauses).
- **`fast=True`** runs the second row inside the encoder's CUDA graph (one encoder pass per speaker, as the eager
  path). Against the eager path on 366 sentences (3 voices × 61 texts × 2 prosody temperatures), the frame counts
  agree on 364 with the recommended checkpoint (one sentence, at both temperatures, is one frame longer) and the mels
  within 72 dB SNR; the one-row graph of the release with sampled durations agrees on 357 (that predates this
  change: the rounding of a few durations flips on float noise of the padded buckets).

## Word-level context (#40)

**Features** (`drifting_tts/word_features.py`, `prosody-cache --word-model`). `dbmdz/bert-base-turkish-cased`
(BERTurk, 111 M, MIT) runs over the normalised text. Words are its space-separated pieces, which are exactly the
segments of `word_index`. A word's vector is the mean over its sub-word pieces of the mean of the last four hidden
layers. It is standardised per dimension and broadcast to the word's character and blank tokens as 768 extra input
channels. The uncased BERTurk is unusable here: it strips the diacritics of our lower-case text. The cache adds
783k word vectors (fp16, ~10 min on the shared GPU). The model is frozen (no fine-tuning), and everything else
equals the drift run (12k steps).

**A/B against the character-only drift sampler:**

| | pitch CRPS (all / studio) | pitch r (all / studio) | pitch spread | jitter | pre-mI / fall (st) | Freya-100 WER, T 1 / T 0.5 / pitch only |
|---|---|---|---|---|---|---|
| drift | 0.297 / 0.200 | 0.379 / 0.594 | 0.99 | 1.01 | −1.57 / −0.52 | 2.63% / 0.66% / 0.77% |
| drift + BERTurk | **0.283 / 0.190** | **0.435 / 0.639** | 0.97 | 1.04 | **−0.13 / −2.37** | 4.50% / 1.65% / 0.66% |
| recordings | 0 | 1 | 1 | 1 | +0.05 / −4.24 | – |

- **Word context makes the sampled tune more text-specific.** Pitch CRPS improves by 5%, correlation rises by
  0.05–0.06, and the polar-question shape (the pitch before mI, the final fall) moves towards the recordings, at the
  same distributional match.
- **It does not help the durations on out-of-domain text.** The BERTurk sampler's sampled durations are worse for
  intelligibility on Freya-100 (WER 4.50% at T 1, 1.65% at T 0.5, against 2.63% / 0.66–0.99% without it). With the regressors' durations (pitch only) it is as clean
  as v3.1.
- **Cost:** BERTurk adds ~40 ms per sentence on the busy GPU and a `transformers` dependency at inference. A smaller
  cased encoder (ELECTRA-small-tr, 13.7 M, MIT) and fine-tuning (Kenter et al. 2020) are the obvious next steps;
  neither was tried here ([phase 2](#phase-2-conditional-intonation-40) tries ELECTRA-small and uses this sampler
  for the pitch only).

## Notes and pitfalls

- **One positive per text sets the drift sampler's spread through τ, not through the data.** Per row, distances are
  normalised by the samples' own mean distance, so the balance between the attraction to the single positive and
  the repulsion between siblings is scale-free. The learned τ (1.0 → 0.10 within 3k steps, then flat; Kyutai: ≈ 0.056)
  ends up giving realistic *marginal* spread, but the samples depend less on the text than flow matching's
  (lower per-token correlation, higher CRPS). `spread < 1` at inference is the cheap correction.
- **The noise temperature of the drift sampler is a diversity knob, not an expressiveness knob.** Its zero-noise
  output already has the recordings' within-utterance spread (pitch std ratio 0.83 at 2k steps against 0.62 for
  MSE); lowering T shrinks the differences between seeds. Flow matching behaves like a diffusion model: lower T
  is flatter.
- **Host syncs dominate small-model training on a shared GPU.** Boolean row selection (`x[mask]`) per feature map
  and `int(tensor.max())` cost ~0.35 s per step under time-slicing (0.57 → 0.21 s per step after removing them).
- **MAS durations are noisy per token.** The split of a character's time between it and its neighbouring blanks is
  arbitrary (47% of all tokens get exactly one frame), so token-level duration statistics look much flatter for
  any predictor than letter- or word-level ones. The tables report letter (character + following blank) and word
  levels as well.
- **Rounding.** The regressors' `ceil(exp(logw))` adds about half a frame per token (a predicted blank of 1.05 frames
  becomes 2), which the v3.1 per-voice factors partly compensate. The samplers round to the nearest frame instead,
  with their own factors.
- **Turkish BERT.** The normalised text is lower-case, but `dbmdz/bert-base-turkish-uncased` strips diacritics
  (`gelmiş` → `gelmis`, `değişik` → `degis ##ik`). The cased model keeps them on lower-case input.

## Phase 2: conditional intonation (#40)

v3.2 (PR #49) is v3.1 + the drift sampler above at prosody temperature 0.5 + Vocos v2 + punctuation pauses. Phase 2
asks for intonation closer to what a reader does with *this* text, while keeping the recordings' spread and v3.2's
intelligibility. Every row uses the frozen v3.1 acoustic model. Audio rows use the Vocos v2 preview
(`runs/voc_p3/vocos_ft_10000.pt`, the vocoder of the v3.2 dry run), so they are not comparable with the `vocos-ft`
rows above.

> **Since these measurements** (merged with #52): the "v3.2" rows here are the dry-run setting, durations and pitch
> both sampled at T 0.5. v3.2 as released keeps v3.1's durations and samples only the token pitch
> ([EXPERIMENTS.md §9](EXPERIMENTS.md#9-release-v32)). The separate temperatures of section 1 now use #52's API: the
> prosody temperature is the pitch's (`prosody_temperature`, `--prosody-temperature`), and `duration_temperature`
> (`Synthesizer(prosody_duration_temperature=)`, `--prosody-duration-temperature`) sets the letters' durations apart.
> This phase measured them with its first version (`pitch_temperature`), in which the pauses followed the durations'
> temperature (now the pitch's, as in [Sampled rhythm without the slips](#sampled-rhythm-without-the-slips)) and the
> output spread could be set per channel (`pitch_spread`, one row below; dropped). The second pitch predictor, the
> sentence features and the context branch are unchanged; with `fast=True` they run eagerly.

- **Separate temperatures work without retraining.** The same noise run at two temperatures (one more row in the
  sampler's batch) gives durations and pitch their own temperatures; equal temperatures reproduce the one-pass
  sample. This is #52's `duration_temperature`. Raising only the pitch temperature widens the F0 (pitch T 1: F0 std
  3.38 → 3.60) but lowers the contour correlation (DTW F0 r 0.578 → 0.529), at no significant cost in
  intelligibility. It is a dial, not a better tune.
- **Vocos v2 narrows the measured F0** (copy synthesis 3.55 st, recordings 3.68, `vocos-ft` copy synthesis 3.69).
  Under it v3.2 sits at 3.38 st and v3.1 at 2.93.
- **Duration safety comes from the duration temperature, not from a floor.** A 3-frame letter floor recovers 0.4 of
  the 2.2 pp of WER that T 1 costs and lengthens speech. Keeping the durations at T ≤ 0.5, now independent of the
  pitch temperature, is enough; no retraining was needed.
- **Word context makes the pitch more text-specific; where it enters matters more than the encoder.** A pitch-only
  branch (ELECTRA-small or BERTurk, with sentence-type features) keeps the durations context-free and v3.2's
  intelligibility, but captures only part of the gain. The BERTurk sampler used for the pitch only, with v3.2's
  durations, is the best: per-token pitch r 0.494 (v3.2 0.436), DTW F0 r 0.601–0.613 (v3.2 0.578; v3.1 0.617 at a
  much narrower F0 std of 2.93), held-out polar questions at pre-mI −0.05 / fall −2.59 (recordings −0.22 / −3.35;
  v3.2 −1.51 / −0.28), Freya-495 WER 1.92% (v3.2 1.84%, n.s.).
- **G = 32 / 20k steps** gives a slightly better BERTurk sampler (pitch r 0.494 against 0.485, held-out questions
  closer to the readers, UTMOSv2 level), and τ settles at 0.072 instead of 0.10.
- **Questions are better, not solved.** On new questions no system raises the word before mI above the sentence mean
  (readers +0.87 st; best −0.8), and every system ends wh-questions lower than statements, unlike the readers.
- **Candidate for v3.3:** durations of v3.2's sampler + pitch of the G32 BERTurk sampler, both at T 0.5
  ([Recommendation](#recommendation-candidate-for-v33)). Listening has to confirm it.

### What was added

| piece | where | what |
|---|---|---|
| separate temperatures | #52's `ProsodyPredictor.sample(duration_temperature=)`, `Synthesizer(prosody_duration_temperature=)`, `--prosody-duration-temperature` | the letters' durations at their own noise temperature, the pitch (and the voicing, the pauses) at the prosody temperature; inference only (measured here with this phase's first version, see the note above) |
| a second predictor for the pitch | `Synthesizer(prosody_pitch=)`, `set_prosody(pitch=)`, `--prosody-pitch-model` | the durations from `--prosody`, the token pitch from another sampler (e.g. one with word features) at the prosody temperature and spread, drawn after the first one's noise; runs eagerly |
| letter floor | `floor_letters`, `predict(min_letter_frames=, rel_letter_floor=)` | a letter (character + following blank) gets at least N frames, or a ratio of the regressors' duration |
| sentence features | `drifting_tts/sentence_features.py` | 18 rule-based features per word: sentence type, mI host / particle / after, wh-word / after, sentence-final, comma, positions |
| context branch | `net.ctx_pitch_only`, `net.pitch_layers`, `net.ctx_boundaries` | word and sentence features bypass the trunk and feed two more layers that predict the pitch (and, with `ctx_boundaries`, the durations between words) |
| centroid loss | `loss.centroid` | MSE of the mean of the G samples against the recording |
| training on a shared GPU | `train.accum`, `train.max_gpu_gb`, peak memory in the log | gradient accumulation; a cap for the caching allocator |
| diagnostics | `scripts/eval_sentence_prosody.py`, `scripts/prosody_diagnostic_tr.jsonl` | intonation by sentence type and phrase breaks, on held-out recordings and on 153 Turkish sentences written for this |

The diagnostic set has 20 statement / polar-question minimal pairs (`Ali dün akşam yemek yaptı.` / `… yaptı mı?`),
5 mI placement series (`Ayşe mi dün sinemaya gitti?` … `Ayşe dün sinemaya gitti mi?`) with their statements, 20
wh-questions, 8 tag and 8 alternative questions, 15 exclamations, 12 lists, 15 long comma-free sentences (15–20 words)
and 10 focus constructions.

### What readers do, by sentence type

Token pitch of the recordings (MAS durations, dio token pitch, the continuous contour on the tokens of voiced
letters, semitones; `scripts/eval_sentence_prosody.py --heldout --reference train`). `final`: the last word minus the
sentence mean. `pre-mI`: the word before the (last) mI particle minus the sentence mean; `fall`: the last word minus
that word. `breaks`: word boundaries inside a sentence whose blanks, spaces and punctuation last ≥ 16 frames (0.17 s;
on the studio `val` recordings that gives 1.6 per utterance, against 1.39 audio pauses of ≥ 0.1 s).

| recordings (training split) | sentences | statement final | polar final | pre-mI | fall | ends low | wh final | wh-word peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| studio (722) | 22,638 | −3.68 | −4.08 | +0.87 | −4.95 | 99% | −3.04 | +1.92 |
| all voices | 73,277 | −2.98 | −3.13 | +0.47 | −3.60 | 83% | −2.87 | +1.82 |

- Polar questions peak on the word before mI and fall onto the particle (the studio voice: +0.87 st, then −4.95 st),
  as the phonetics literature says (research memo §4.2). Wh-questions peak on the wh-word and end higher than
  statements.
- Phrase breaks inside comma-free statements grow with length: studio 0.08 per sentence below 8 words, 0.26 for 8–13,
  0.88 from 14 words; all voices 0.26 / 0.83 / 1.97.
- In the training data (with the studio voice's repeat) 2.1% of the sentences are polar questions, 2.7% wh-questions
  and 0.1% exclamations.

### 1. Separate temperatures for durations and pitch (inference only)

The sampler's noise (16 per-token and 32 global channels) enters one trunk that predicts both channels, so it cannot be
scaled per channel without retraining. What works without retraining: the same unit noise runs at two temperatures as
one batch of two rows; the pitch and the voicing come from the first row (the prosody temperature), the
log-durations from the second (`duration_temperature`, #52). Equal temperatures reproduce the one-pass sample, and
the two channels stay coupled through the shared noise direction. The cost is one more row in the sampler's batch.
These rows were measured with this phase's first version, in which every duration (the pauses too) followed the
duration temperature; with #52's API the pauses keep the prosody temperature. The last row used a per-channel
output spread (`pitch_spread`), which was dropped when merging.

**Token level** (`dev`, all speakers, v3.2's sampler, 8 seeds):

| T durations | T pitch | pitch spread | word pitch spread | pitch r | pitch CRPS | letter dur spread | letter dur r | log-dur CRPS | short letters where the recording has ≥ 4 frames | length |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.1 regressors | | 0.597 | 0.595 | 0.525 | 0.418 | 0.430 | 0.502 | 0.568 | 0.000 | 0.957 |
| 0 | 0 | 0.829 | 0.824 | 0.478 | 0.448 | 0.931 | 0.477 | 0.512 | 0.078 | 0.948 |
| 0.3 | 0.3 | 0.867 | 0.882 | 0.462 | 0.346 | 0.934 | 0.453 | 0.391 | 0.077 | 0.954 |
| **0.5** | **0.5** (v3.2) | 0.920 | 0.962 | 0.431 | 0.315 | 0.941 | 0.421 | 0.353 | 0.080 | 0.966 |
| 0.5 | 0.7 | 0.960 | 1.018 | 0.401 | 0.302 | 0.941 | 0.421 | 0.353 | 0.080 | 0.966 |
| 0.5 | 1 | 0.998 | 1.057 | 0.374 | 0.294 | 0.941 | 0.421 | 0.353 | 0.080 | 0.966 |
| 0.3 | 0.7 | 0.960 | 1.018 | 0.401 | 0.302 | 0.934 | 0.453 | 0.391 | 0.077 | 0.954 |
| 0 | 1 | 0.998 | 1.057 | 0.374 | 0.294 | 0.931 | 0.477 | 0.512 | 0.078 | 0.948 |
| 1 | 1 | 0.998 | 1.057 | 0.374 | 0.294 | 0.981 | 0.344 | 0.327 | 0.092 | 1.006 |
| 0.5 | 1, pitch spread 0.8 (dropped option) | 0.874 | 0.930 | 0.416 | 0.295 | 0.940 | 0.419 | 0.354 | 0.081 | 0.961 |

**Audio** (studio `val`, 100 utterances, one pass, T 0.3, α 2, Vocos v2 preview, harvest F0; `eval_prosody_audio.py`
systems now named `drift-T<pitch t>-D<duration t>`, with the pauses at the pitch temperature; these rows ran with the
pauses at the duration temperature):

| system | F0 std | F0 range | move | micro | reversals/s | pauses/utt | syl/s | DTW F0 r | F0 RMSE | length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recordings | 3.68 | 11.9 | 0.63 | 0.42 | 10.9 | 1.39 | 6.22 | – | – | – |
| copy synthesis (the recording's mel → Vocos v2) | 3.55 | 11.6 | 0.66 | 0.45 | 11.6 | 1.49 | 6.22 | 0.830 | 2.01 | – |
| v3.1 (regressors) | 2.93 | 9.6 | 0.62 | 0.40 | 11.5 | 0.52 | 6.11 | 0.617 | 2.94 | 1.017 |
| v3.2 (T 0.5) | 3.38 | 10.9 | 0.67 | 0.46 | 11.7 | 1.46 | 6.20 | 0.578 | 3.22 | 1.006 |
| durations T 0.5, pitch T 0.7 | 3.44 | 11.2 | 0.67 | 0.45 | 11.7 | 1.42 | 6.20 | 0.564 | 3.30 | 1.006 |
| durations T 0.5, pitch T 1 | 3.60 | 11.8 | 0.68 | 0.46 | 11.7 | 1.40 | 6.20 | 0.529 | 3.50 | 1.006 |
| durations T 0.3, pitch T 0.7 | 3.46 | 11.2 | 0.67 | 0.46 | 11.8 | 1.11 | 6.29 | 0.558 | 3.32 | 0.991 |

**Freya-100** (studio voice, T 0.3, α 2, Vocos v2 preview, Whisper large-v3 on 8 kHz band-matched audio, UTMOSv2 on the
full band; paired bootstrap differences against v3.2). The v3.2 row reproduces the v3.2 dry run exactly (0.66% /
0.14% / 2.976).

| system | WER [95% CI] | CER | UTMOSv2 [95% CI] | ΔWER vs v3.2 [95% CI] | ΔUTMOSv2 vs v3.2 [95% CI] |
|---|---|---:|---|---|---|
| v3.1 regressors | 0.77% [0.22, 1.35] | 0.16% | 2.922 [2.887, 2.955] | +0.11 pp [−0.66, +0.79] | −0.054 [−0.100, −0.008] |
| **v3.2** (T 0.5) | 0.66% [0.11, 1.33] | 0.14% | 2.976 [2.941, 3.012] | – | – |
| durations T 0.5, pitch T 0.7 | 0.88% [0.23, 1.61] | 0.18% | 2.983 [2.940, 3.022] | +0.22 pp [−0.22, +0.66] | +0.007 [−0.031, +0.046] |
| durations T 0.5, pitch T 1 | 0.99% [0.33, 1.78] | 0.22% | 3.008 [2.962, 3.052] | +0.33 pp [−0.22, +0.99] | +0.032 [−0.022, +0.089] |

- **The pitch temperature costs no significant intelligibility.** Pitch T 0.7 / 1 add 2–3 word errors out of 911
  (not significant) and leave UTMOSv2 level (+0.007 / +0.032, both intervals include 0).
- **Each channel follows its own temperature.** The pitch metrics depend only on the pitch temperature, the duration
  metrics only on the duration temperature.
- **The pitch temperature is a dial between spread and accuracy, not a better tune.** From T 0 to 1 the pitch spread
  goes 0.83 → 1.00 and the per-token r 0.48 → 0.37; CRPS improves only because the samples spread out. In the audio,
  pitch T 0.7 / 1 raise the F0 std from 3.38 to 3.44 / 3.60 and lower the DTW F0 r from 0.578 to 0.564 / 0.529.
- **Vocos v2 narrows the measured F0 spread.** Copy synthesis through it gives 3.55 st (recordings 3.68, `vocos-ft`
  copy synthesis 3.69 in [PROSODY.md](PROSODY.md#oracle-prosody-ab-studio-voice)). Generated speech loses more: v3.1
  3.20 → 2.93 and v3.2 3.63 → 3.38 against the `vocos-ft` rows above. Its lower periodicity jitter (micro-variation
  0.45 against 0.49 for `vocos-ft` copy synthesis) removes some of the spread `vocos-ft` added. Under Vocos v2, v3.2
  is about 5% below copy synthesis.
- **Lower duration temperatures keep the rhythm varied.** Letter-duration spread stays at 0.93 even at T 0 (the
  zero-noise output is already varied), the per-letter r rises (0.42 / 0.45 / 0.48 at T 0.5 / 0.3 / 0), and there are
  fewer internal pauses (1.46 → 1.11 per utterance at T 0.3; recordings 1.39).

### 2. Duration safety

The Freya-100 slips of the drift sampler at T 1 (phase 1: `oldu → olu`, `koyuydu → koyudu`, `oyununu → oyunun`,
`salona → salonu`, `ağrıyor → arıyor`) sit on letters of 2 frames (or a final vowel cut short); at T 0.5 the same
letters mostly get 3–4 frames. The studio recordings themselves have 9.9% of their letters at ≤ 2 frames, spread over
all characters (b 29%, d 22%, g 21%, vowels 7–14%), so a floor also removes recording-like short letters.

**Token level** (`dev`, all speakers, v3.2's sampler at T 0.5, 4 seeds):

| floor | letters ≤ 2 frames, recording ≥ 4 | letter dur spread | letter dur r | log-dur CRPS | length |
|---|---:|---:|---:|---:|---:|
| none | 0.080 | 0.946 | 0.414 | 0.353 | 0.964 |
| letters ≥ 3 frames | 0.020 | 0.827 | 0.403 | 0.373 | 0.987 |
| ≥ 0.5 × the regressors' letter | 0.079 | 0.944 | 0.415 | 0.353 | 0.964 |
| ≥ 0.6 × the regressors' letter | 0.073 | 0.939 | 0.418 | 0.354 | 0.966 |

**Freya-100** (as in section 1; the sampler of v3.2 at T 1, so that slips are frequent enough to measure):

| system | WER [95% CI] | CER | UTMOSv2 | ΔWER vs v3.2 [95% CI] |
|---|---|---:|---:|---|
| v3.2 (T 0.5) | 0.66% [0.11, 1.33] | 0.14% | 2.976 | – |
| T 1 | 2.85% [1.73, 4.06] | 0.61% | 2.975 | +2.20 pp [+1.11, +3.38] |
| T 1, letters ≥ 3 frames | 2.41% [1.30, 3.71] | 0.53% | 2.960 | +1.76 pp [+0.68, +2.98] |

- **A floor is not the fix.** Raising every letter to ≥ 3 frames removes 3/4 of the "wrongly short" letters but
  recovers only 0.44 pp of the 2.2 pp that T 1 costs, lengthens speech (+2.4% at T 0.5) and narrows the
  letter-duration spread (0.95 → 0.83). The slips at T 1 are more than short letters: its errors also include
  sound changes on long letters (`duymak → doymak`, `yıllarıma → yıldırıma`). A floor relative to the regressors
  barely acts.
- **The safety that works is the duration temperature.** With the channels decoupled (section 1), the durations can
  stay at T 0.5 (or lower) whatever the pitch temperature; T 0.3 also gives more accurate letter durations and fewer
  internal pauses. No retraining was needed, so none was done.

### 3. Context for the pitch: word and sentence features

**Design.** Phase 1 found that BERTurk word features make the sampled pitch more text-specific but hurt the sampled
durations on out-of-domain text (Freya-100 WER 4.50% at T 1; its errors include pauses inside words: `birdenbire →
birden bire`, `sabah → sa ba`). The context branch keeps the durations out of their reach. The trunk (the phase-1
network without the context) predicts the log-durations. The word and sentence features enter only two more
transformer layers on top of the trunk, which predict the pitch and the voicing (`net.ctx_pitch_only`, 11.9 M
parameters instead of 8.1 M). With `net.ctx_boundaries` that branch also predicts the durations of the tokens between
words (spaces, punctuation and the blanks next to them), i.e. the phrase breaks, while the letters' own durations
stay the trunk's. Both channels are still drawn from one noise sample and trained with the same drifting loss.

**Word encoders.** `dbmdz/electra-small-turkish-cased-discriminator` (13.7 M parameters, 12 layers of 256, MIT,
revision `6e024916`; its cased tokenizer keeps the diacritics of the lower-case normalised text) and
`dbmdz/bert-base-turkish-cased` (111 M, MIT). A word vector is the mean over its pieces of the mean of the last four
layers, as in phase 1. The ELECTRA features of the 783k training words took ~7 minutes on the shared GPU.

**Sentence features** (`drifting_tts/sentence_features.py`, 18 per word, rule-based on the normalised text): the
sentence type (statement, polar question with mI, wh-question, other question, exclamation) with two question flags
(tag `…, değil mi?`; alternative `… yoksa …`); the word's role in a question (the host of mI, i.e. the word before the
particle; the mI word; after the last mI; a wh-word; after the wh-word); the last word of the sentence; a comma after
the word; positions (in the sentence, words to its end, the sentence's position in the text, last sentence).

**Runs** (everything else as the phase-1 drift run; G samples per utterance, B utterances per step):

| run | word features | sentence features | context feeds | B × G | other | steps |
|---|---|---|---|---|---|---:|
| drift (v3.2) | – | – | – | 12 × 16 | | 12k |
| BERTurk for everything (phase 1) | BERTurk | – | the whole network | 12 × 16 | | 12k |
| `p2_electra_pitch` | ELECTRA-small | yes | pitch branch | 8 × 16 | | 12k |
| `p2_electra_pitch_c1` | ELECTRA-small | yes | pitch branch | 6 × 16 | `loss.centroid=1` | 8k |
| `p2_bert_pitch` | BERTurk | yes | pitch branch | 6 × 16 | | 8k |
| `p2_electra_ctxb` | ELECTRA-small | yes | pitch branch + durations between words | 6 × 16 | | 8k |
| `p2_bert_g32` | BERTurk | – | the whole network (used for the pitch only) | 3 × 32 | | 20k |

B8 × G16 needs about 7 GiB at its longest batches (8.0 GB reserved in nvidia-smi without a cap; under
`train.max_gpu_gb=6.5` it ran out of memory after 2.9k steps), so the later runs use B6 to stay under 7 GB on the
shared GPU.

**Token level** (`val` + `dev`, durations T 0.5 and the voice factors of each checkpoint, 8 seeds). Columns as in
the phase-1 tables; pitch spread and r against the recordings' token pitch:

| sampler | pitch T | all speakers: pitch spread | pitch r | pitch CRPS | studio: pitch spread | pitch r | pitch CRPS | letter dur r (all) | log-dur CRPS (all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.1 regressors | – | 0.593 | 0.522 | 0.425 | 0.732 | 0.704 | 0.301 | 0.512 | 0.564 |
| drift (v3.2) | 0.5 | 0.913 | 0.436 | 0.319 | 0.912 | 0.670 | 0.220 | 0.426 | 0.352 |
| drift (v3.2) | 0.7 | 0.953 | 0.410 | 0.304 | 0.932 | 0.641 | 0.206 | 0.426 | 0.352 |
| drift (v3.2) | 1 | 0.989 | 0.379 | 0.297 | 0.978 | 0.594 | 0.200 | 0.426 | 0.352 |
| BERTurk for everything | 0.5 | 0.985 | 0.485 | 0.309 | 0.934 | 0.694 | 0.206 | 0.427 | 0.343 |
| BERTurk for everything | 0.7 | 0.982 | 0.463 | 0.292 | 0.949 | 0.671 | 0.195 | 0.427 | 0.343 |
| BERTurk for everything | 1 | 0.971 | 0.435 | 0.283 | 0.970 | 0.639 | 0.190 | 0.427 | 0.343 |
| ELECTRA + sentence, pitch branch | 0.5 | 0.924 | 0.451 | 0.312 | 0.905 | 0.678 | 0.211 | 0.423 | 0.349 |
| ELECTRA + sentence, pitch branch | 0.7 | 0.957 | 0.427 | 0.296 | 0.930 | 0.646 | 0.199 | 0.423 | 0.349 |
| ELECTRA + sentence, pitch branch | 1 | 0.992 | 0.400 | 0.288 | 0.982 | 0.604 | 0.194 | 0.423 | 0.349 |
| ELECTRA + sentence + centroid loss (B6, 8k) | 0.5 | 0.872 | 0.459 | 0.318 | 0.884 | 0.677 | 0.220 | 0.410 | 0.344 |
| ELECTRA + sentence + centroid loss (B6, 8k) | 0.7 | 0.906 | 0.440 | 0.300 | 0.910 | 0.659 | 0.206 | 0.410 | 0.344 |
| ELECTRA + sentence + centroid loss (B6, 8k) | 1 | 0.937 | 0.417 | 0.291 | 0.955 | 0.627 | 0.198 | 0.410 | 0.344 |
| BERTurk + sentence, pitch branch (B6, 8k) | 0.5 | 0.920 | 0.447 | 0.315 | 0.876 | 0.681 | 0.214 | 0.406 | 0.343 |
| BERTurk + sentence, pitch branch (B6, 8k) | 0.7 | 0.946 | 0.434 | 0.298 | 0.915 | 0.660 | 0.202 | 0.406 | 0.343 |
| BERTurk + sentence, pitch branch (B6, 8k) | 1 | 0.973 | 0.415 | 0.288 | 0.973 | 0.627 | 0.196 | 0.406 | 0.343 |
| ELECTRA + sentence, pitch + boundary durations (B6, 8k) | 0.5 | 0.922 | 0.419 | 0.314 | 0.838 | 0.649 | 0.220 | 0.402 | 0.342 |
| ELECTRA + sentence, pitch + boundary durations (B6, 8k) | 0.7 | 0.956 | 0.412 | 0.300 | 0.890 | 0.636 | 0.205 | 0.402 | 0.342 |
| ELECTRA + sentence, pitch + boundary durations (B6, 8k) | 1 | 0.989 | 0.398 | 0.292 | 0.964 | 0.609 | 0.198 | 0.402 | 0.342 |
| BERTurk for everything, G 32 (B3, 20k) | 0.5 | 0.939 | 0.494 | 0.317 | 0.959 | 0.700 | 0.211 | 0.426 | 0.343 |
| BERTurk for everything, G 32 (B3, 20k) | 0.7 | 0.945 | 0.470 | 0.297 | 0.957 | 0.678 | 0.198 | 0.426 | 0.343 |
| BERTurk for everything, G 32 (B3, 20k) | 1 | 0.963 | 0.433 | 0.286 | 0.970 | 0.638 | 0.192 | 0.426 | 0.343 |

**Held-out recordings, in context** (`val` + `dev`, all speakers, each utterance sampled in one pass, durations
T 0.5, 8 seeds; 23 polar and 34 wh-questions). Each sampler with its own durations; with v3.2's durations, a BERTurk
pitch keeps its pitch columns and gets v3.2's breaks:

| system | statement final | polar final | pre-mI | fall | ends low | wh final | wh-word peak | breaks / comma-free sentence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recordings | −3.02 | −3.57 | −0.22 | −3.35 | 83% | −3.46 | +1.58 | 0.65 |
| v3.1 regressors | −2.54 | −1.51 | −1.03 | −0.48 | 65% | −2.47 | +1.10 | 0.04 |
| v3.2 sampler | −3.20 | −1.78 | −1.51 | −0.28 | 52% | −3.07 | +0.83 | 0.56 |
| v3.2 sampler, pitch T 0.7 | −3.18 | −1.79 | −1.29 | −0.50 | 54% | −2.99 | +0.72 | 0.56 |
| BERTurk for everything (phase 1) | −3.37 | −2.40 | −0.10 | −2.30 | 70% | −3.46 | +1.15 | 0.68 |
| BERTurk for everything, pitch T 0.7 | −3.29 | −2.22 | −0.12 | −2.10 | 68% | −3.39 | +1.15 | 0.68 |
| ELECTRA + sentence features, pitch branch | −3.29 | −2.25 | −0.91 | −1.34 | 62% | −3.29 | +1.38 | 0.55 |
| the same, pitch T 0.7 | −3.24 | −2.38 | −0.81 | −1.57 | 64% | −3.14 | +1.50 | 0.55 |
| ELECTRA + sentence + centroid loss (B6, 8k) | −3.08 | −1.86 | −0.70 | −1.16 | 64% | −3.14 | +1.25 | 0.38 |
| the same, pitch T 0.7 | −3.11 | −1.97 | −0.69 | −1.28 | 68% | −3.16 | +1.44 | 0.38 |
| BERTurk + sentence, pitch branch (B6, 8k) | −3.18 | −2.59 | −0.50 | −2.09 | 69% | −3.36 | +1.10 | 0.42 |
| the same, pitch T 0.7 | −3.25 | −2.65 | −0.46 | −2.19 | 71% | −3.45 | +1.30 | 0.42 |
| ELECTRA + sentence, pitch + boundary durations (B6, 8k) | −2.94 | −2.18 | −1.02 | −1.15 | 66% | −3.01 | +1.31 | 0.70 |
| the same, pitch T 0.7 | −3.08 | −2.15 | −1.10 | −1.05 | 64% | −3.04 | +1.27 | 0.70 |
| BERTurk for everything, G 32 (B3, 20k) | −3.60 | −2.64 | −0.05 | −2.59 | 74% | −3.34 | +1.51 | 0.47 |
| the same, pitch T 0.7 | −3.50 | −2.43 | −0.23 | −2.21 | 72% | −3.34 | +1.38 | 0.47 |

**Audio, studio `val`** (as in section 1):

| system | F0 std | F0 range | pauses/utt | syl/s | DTW F0 r | F0 RMSE | length |
|---|---:|---:|---:|---:|---:|---:|---:|
| recordings | 3.68 | 11.9 | 1.39 | 6.22 | – | – | – |
| copy synthesis (Vocos v2) | 3.55 | 11.6 | 1.49 | 6.22 | 0.830 | 2.01 | – |
| v3.1 regressors | 2.93 | 9.6 | 0.52 | 6.11 | 0.617 | 2.94 | 1.017 |
| v3.2 sampler | 3.38 | 10.9 | 1.46 | 6.20 | 0.578 | 3.22 | 1.006 |
| v3.2 sampler, pitch T 0.7 | 3.44 | 11.2 | 1.42 | 6.20 | 0.564 | 3.30 | 1.006 |
| ELECTRA + sentence, pitch branch | 3.41 | 11.1 | 1.60 | 6.21 | 0.596 | 3.16 | 1.005 |
| the same, pitch T 0.7 | 3.48 | 11.3 | 1.65 | 6.21 | 0.577 | 3.25 | 1.005 |
| durations of v3.2 + pitch of BERTurk for everything | 3.45 | 11.2 | 1.40 | 6.20 | 0.613 | 3.11 | 1.006 |
| the same, pitch T 0.7 | 3.50 | 11.4 | 1.41 | 6.20 | 0.596 | 3.19 | 1.006 |
| BERTurk + sentence, pitch branch (B6, 8k) | 3.32 | 10.7 | 1.27 | 6.24 | 0.585 | 3.14 | 1.004 |
| ELECTRA + sentence, pitch + boundary durations (B6, 8k) | 3.23 | 10.3 | 1.25 | 6.23 | 0.562 | 3.21 | 1.002 |
| durations of v3.2 + pitch of the G32 BERTurk sampler | 3.49 | 11.3 | 1.52 | 6.22 | 0.601 | 3.16 | 1.004 |
| the same, pitch T 0.7 | 3.51 | 11.4 | 1.51 | 6.22 | 0.588 | 3.21 | 1.004 |
| the G32 BERTurk sampler alone (its own durations) | 3.51 | 11.4 | 1.33 | 6.22 | 0.610 | 3.12 | 1.003 |

**Freya-100** (as in section 1; durations T 0.5, pitch T 0.5 unless noted):

| system | WER [95% CI] | CER | UTMOSv2 [95% CI] | ΔWER vs v3.2 [95% CI] | ΔUTMOSv2 vs v3.2 [95% CI] |
|---|---|---:|---|---|---|
| v3.2 sampler | 0.66% [0.11, 1.33] | 0.14% | 2.976 [2.941, 3.012] | – | – |
| ELECTRA + sentence, pitch branch | 0.33% [0.00, 0.75] | 0.10% | 2.972 [2.935, 3.013] | −0.33 pp [−1.00, +0.11] | −0.005 [−0.051, +0.042] |
| the same, pitch T 0.7 | 0.77% [0.22, 1.43] | 0.14% | 2.992 [2.957, 3.028] | +0.11 pp [−0.67, +0.88] | +0.015 [−0.037, +0.062] |
| BERTurk + sentence, pitch branch (B6, 8k) | 1.54% [0.76, 2.51] | 0.34% | 2.968 [2.927, 3.007] | +0.88 pp [+0.00, +1.88] | −0.008 [−0.057, +0.040] |
| ELECTRA + sentence, pitch + boundary durations (B6, 8k) | 1.43% [0.68, 2.27] | 0.34% | 2.995 [2.951, 3.037] | +0.77 pp [−0.11, +1.65] | +0.019 [−0.033, +0.073] |
| durations of v3.2 + pitch of BERTurk for everything | 0.99% [0.34, 1.72] | 0.22% | 2.978 [2.936, 3.019] | +0.33 pp [−0.22, +0.90] | +0.001 [−0.047, +0.048] |
| the same, pitch T 0.7 | 0.99% [0.33, 1.70] | 0.21% | 2.963 [2.926, 2.999] | +0.33 pp [−0.22, +0.89] | −0.014 [−0.057, +0.029] |
| durations of v3.2 + pitch of the G32 BERTurk sampler (section 4; the first 100 of its Freya-495 run) | 1.10% [0.42, 1.89] | 0.30% | 2.960 [2.923, 2.998] | +0.44 pp [−0.22, +1.19] | −0.016 [−0.061, +0.029] |

**Freya-495** (all 495 sentences, studio voice, same protocol; paired against v3.2 on the same sentences):

| system | WER [95% CI] | CER | UTMOSv2 [95% CI] | ΔWER vs v3.2 [95% CI] | ΔUTMOSv2 vs v3.2 [95% CI] |
|---|---|---:|---|---|---|
| v3.2 sampler | 1.84% [1.36, 2.37] | 0.38% | 2.993 [2.974, 3.012] | – | – |
| ELECTRA + sentence, pitch branch | 1.51% [1.06, 1.98] | 0.32% | 2.999 [2.980, 3.018] | −0.33 pp [−0.83, +0.15] | +0.006 [−0.016, +0.027] |
| durations of v3.2 + pitch of BERTurk for everything (G16) | 1.94% [1.44, 2.50] | 0.37% | 2.968 [2.948, 2.988] | +0.10 pp [−0.36, +0.56] | −0.025 [−0.047, −0.002] |
| durations of v3.2 + pitch of the G32 BERTurk sampler | 1.92% [1.43, 2.45] | 0.41% | 2.988 [2.968, 3.007] | +0.08 pp [−0.36, +0.53] | −0.005 [−0.029, +0.018] |

**Diagnostic set** (`scripts/prosody_diagnostic_tr.jsonl`, studio voice, each sentence alone as `Synthesizer` sends
it, durations T 0.5). Token level: 4 seeds; audio: 2 seeds through Vocos v2, harvest F0 per word from the frames that
were used, final F0 = median of the last 25 voiced frames against the sentence median. Reference: the studio
recordings of the training split (long comma-free statements of ≥ 14 words: 0.88 breaks). `host boost`: the pitch of
the mI host minus that of the same word in the matching statement (minimal pairs and placement series); the readers'
numbers imply about +4.5 st for a final mI (+0.87 against −3.68).

| system | statement final | polar final | pre-mI | fall | mI inside: pre-mI | wh final | wh-word peak | long sentences: breaks | host boost, mI last | host boost, mI inside |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recordings, studio (training split) | −3.68 | −4.08 | +0.87 | −4.95 | – | −3.04 | +1.92 | 0.88 | ≈ +4.5 | – |
| v3.1 regressors | −4.48 | −4.94 | −2.67 | −2.27 | +2.09 | −4.74 | +1.86 | 0.00 | +1.81 | +0.42 |
| v3.2 sampler | −3.76 | −4.63 | −2.39 | −2.24 | +1.72 | −4.38 | +1.96 | 0.32 | +1.38 | +0.47 |
| v3.2 sampler, pitch T 0.7 | −3.96 | −4.80 | −2.59 | −2.21 | +1.71 | −4.57 | +1.92 | 0.32 | +1.37 | +0.29 |
| BERTurk for everything | −3.76 | −5.00 | −1.58 | −3.42 | +2.21 | −4.07 | +2.20 | 0.93 | +2.18 | +1.03 |
| ELECTRA + sentence, pitch branch | −4.17 | −5.07 | −2.10 | −2.97 | +2.43 | −4.68 | +2.41 | 0.78 | +2.07 | +0.82 |
| the same, pitch T 0.7 | −4.21 | −5.03 | −1.97 | −3.06 | +2.50 | −4.73 | +2.41 | 0.78 | +2.24 | +0.87 |
| durations of v3.2 + pitch of BERTurk for everything (2 seeds) | −3.70 | −5.01 | −1.74 | −3.27 | +2.21 | −3.94 | +1.88 | 0.23 | +1.96 | +0.97 |
| the same, pitch T 0.7 (2 seeds) | −3.93 | −4.98 | −1.72 | −3.26 | +2.25 | −4.18 | +1.93 | 0.23 | +2.22 | +0.92 |
| durations of v3.2 + pitch of the G32 BERTurk sampler (2 seeds) | −4.07 | −4.23 | −1.61 | −2.62 | +2.01 | −4.42 | +2.44 | 0.37 | +2.47 | +0.67 |
| ELECTRA + sentence + centroid loss (B6, 8k) | −3.49 | −3.91 | −0.81 | −3.10 | +1.62 | −3.79 | +2.22 | 0.15 | +2.67 | +0.51 |
| BERTurk + sentence, pitch branch (B6, 8k) | −3.76 | −4.22 | −1.18 | −3.04 | +2.13 | −4.19 | +2.24 | 0.22 | +2.58 | +0.84 |
| ELECTRA + sentence, pitch + boundary durations (B6, 8k) | −3.24 | −3.92 | −1.64 | −2.28 | +1.93 | −3.51 | +2.12 | 0.18 | +1.60 | +0.71 |
| BERTurk for everything, G 32 (B3, 20k) | −4.02 | −4.05 | −1.74 | −2.31 | +2.25 | −4.42 | +2.62 | 0.38 | +2.28 | +0.91 |

Audio (studio, 2 seeds):

| system | statement final F0 | polar final F0 | pre-mI | fall | wh final F0 | wh-word peak | pauses, long sentences | pauses, lists | pauses not at a comma, lists |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.1 regressors | −6.7 | −4.5 | −2.68 | −3.41 | −7.4 | +1.98 | 0.00 | 0.17 | 0.00 |
| v3.2 sampler | −5.6 | −4.5 | −2.19 | −3.78 | −6.4 | +1.77 | 0.17 | 0.67 | 0.04 |
| v3.2 sampler, pitch T 0.7 | −5.7 | −4.4 | −2.14 | −3.85 | −6.2 | +1.50 | 0.17 | 0.71 | 0.04 |
| BERTurk for everything | −5.4 | −4.7 | −1.41 | −4.74 | −5.8 | +2.20 | 0.63 | 0.50 | 0.00 |
| ELECTRA + sentence, pitch branch | −5.9 | −4.5 | −2.16 | −4.22 | −6.5 | +2.76 | 0.30 | 0.67 | 0.00 |
| the same, pitch T 0.7 | −5.9 | −4.2 | −1.89 | −4.01 | −6.1 | +2.94 | 0.30 | 0.62 | 0.00 |
| BERTurk + sentence, pitch branch (B6, 8k) | −5.6 | −3.4 | −1.14 | −4.46 | −6.1 | +2.72 | 0.17 | 0.42 | 0.04 |
| ELECTRA + sentence, pitch + boundary durations (B6, 8k) | −4.1 | −4.0 | −1.74 | −3.38 | −5.0 | +2.24 | 0.17 | 0.50 | 0.00 |
| ELECTRA + sentence + centroid loss (B6, 8k) | −5.6 | −4.3 | −1.10 | −4.80 | −6.3 | +2.50 | 0.10 | 0.38 | 0.04 |
| durations of v3.2 + pitch of BERTurk for everything | −5.1 | −4.9 | −1.58 | −4.71 | −5.8 | +2.02 | 0.17 | 0.71 | 0.04 |
| the same, pitch T 0.7 | −5.2 | −4.8 | −1.66 | −4.43 | −6.0 | +1.97 | 0.17 | 0.71 | 0.04 |
| durations of v3.2 + pitch of the G32 BERTurk sampler | −5.6 | −3.7 | −1.42 | −4.25 | −6.4 | +2.51 | 0.30 | 0.50 | 0.00 |
| the same, pitch T 0.7 | −5.6 | −4.0 | −1.52 | −4.00 | −6.1 | +2.68 | 0.33 | 0.50 | 0.00 |

- **Word context is what moves the conditional intonation; the pitch-only branch captures part of it.** At pitch
  T 0.5 the per-token r of the pitch rises from 0.436 (v3.2) to 0.451 with the ELECTRA branch and 0.485 with
  BERTurk for everything (all speakers; studio 0.670 → 0.678 / 0.694), at a pitch spread of 0.92 / 0.99. On the
  audio the DTW F0 correlation with the recording rises from 0.578 to 0.596 (ELECTRA branch) and 0.601–0.613
  (BERTurk pitch, G32 / G16), i.e. back to v3.1's 0.617 but with v3.2's intonation range (F0 std 3.41–3.49 against
  v3.1's 2.93).
- **Polar questions move towards the readers but do not get there.** On the held-out questions the pre-mI pitch goes
  from −1.51 (v3.2) to −0.91 (ELECTRA branch) and −0.10 (BERTurk; recordings −0.22), the fall onto the particle from
  −0.28 to −1.34 / −2.30 (recordings −3.35). On the new questions of the diagnostic set every system still puts the
  host below the sentence mean (pre-mI −2.4 → −2.1 / −1.6, best −0.8 with the centroid loss; readers +0.87): the
  host is raised against the matching statement by +1.4 (v3.2) to +2.2–2.7 st, about half of what the readers'
  numbers imply. Questions are 5% of
  the training sentences; listening has to say whether this is audible.
- **Wh-questions** get a clearer wh-word peak with context (+1.96 → +2.4 st on the diagnostic set, readers +1.92),
  but every system ends them lower than statements, unlike the readers (−3.04 against −3.68).
- **A pitch branch uses the word features less well than a network that sees them from its input.** On `dev` at
  8k steps and T 1 the pitch r is 0.381 for v3.2's sampler, 0.399 for the ELECTRA branch, 0.419 for the BERTurk
  branch and 0.435 for BERTurk for everything. The encoder matters less than where the features enter.
- **Durations.** With the durations kept context-free, intelligibility stays at v3.2's level: on Freya-100 the
  ELECTRA branch has 0.33% WER (v3.2 0.66%, v3.1 0.77%) and v3.2's durations with BERTurk's pitch 0.99%
  (+0.33 pp [−0.22, +0.90]); on Freya-495 1.51% and 1.94% against 1.84%. The two B6 runs (BERTurk branch, boundary
  branch) are at 1.4–1.5% on Freya-100 (+0.8–0.9 pp, at the edge of significance); their context-free trunks were
  trained for 8k steps at B6 only.
- **Phrase breaks from the context branch** (`ctx_boundaries`) bring the held-out breaks to the readers' rate (0.70
  per comma-free sentence; recordings 0.65, v3.2 0.56), but not those of the long diagnostic sentences (0.18–0.43
  against 0.88 for the readers; 15 sentences, noisy), and the shared branch lowers the pitch accuracy (r 0.419 at
  T 0.5).
- **The centroid loss** narrows the samples (pitch spread 0.87 against 0.93 at T 0.5) for a higher r (0.459 against
  0.451, all speakers), much like a lower temperature. It gives the best question shapes on the diagnostic set
  (pre-mI −0.81, host boost +2.67) and on the held-out questions among the ELECTRA runs (pre-mI −0.70), which may be
  the same pull towards the conditional mean; it was not combined with the BERTurk sampler.
- **Sentence features alone were not tested** (no run without word features); in every run with them the
  question features are present, so their separate effect is unknown.

### 4. G = 32 negatives and 20k steps

The best pitch source, the BERTurk sampler (BERTurk word features for the whole network, used for the pitch only),
retrained with 32 samples per utterance: B3 × G32 = 96 samples per step (the samples per step of B6 × G16), 20k
steps, 28 minutes at 12 it/s, 3.8 GiB peak. Everything else as phase 1 (`p2_bert_g32`).

- **τ anneals lower with more negatives:** 0.072 at 20k (phase 1 with G16: 0.10; Kyutai: ≈ 0.056).
- **Token level** (tables above, pitch T 0.5): per-token r 0.494 against 0.485 for the G16 run (all speakers; studio
  0.700 against 0.694), smoother (jitter 1.015 against 1.051), CRPS level (0.317 / 0.211 against 0.309 / 0.206),
  narrower across all speakers (pitch spread 0.94 against 0.985; studio 0.96 against 0.93).
- **Held-out questions:** pre-mI −0.05 (recordings −0.22), fall −2.59 (−3.35), 74% end low (83%), wh-word peak +1.51
  (+1.58): the closest to the readers of all runs. On the diagnostic set its questions are like the G16 run's
  (pre-mI −1.74 against −1.58).
- **Audio and guard rails** (as the pitch source of v3.2's durations, pitch T 0.5): F0 std 3.49, DTW F0 r 0.601
  (G16: 3.45 / 0.613; v3.2 3.38 / 0.578); Freya-495 WER 1.92% against 1.84% for v3.2 (+0.08 pp [−0.36, +0.53]) and
  UTMOSv2 level (−0.005 [−0.029, +0.018]), where the G16 pitch costs a small but significant −0.025 [−0.047, −0.002].
- **Verdict:** slightly better per-token accuracy, held-out question shapes and UTMOSv2, slightly lower DTW F0 r
  (within the spread of these numbers); the same model in effect, not a step change. The comparison changes three
  things at once (G 16 → 32, B 12 → 3, 12k → 20k steps), so the effect of the negatives alone is not isolated.

### Recommendation (candidate for v3.3)

**Durations from v3.2's sampler, token pitch from the G32 BERTurk sampler**, both at prosody temperature 0.5:

```bash
drifting-tts synthesize --model runs/release/drifting_tts_v3.1.pt --vocoder <vocos-v2> --pause-policy punct \
    --prosody runs/pm_drift_final/prosody_ema.pt --prosody-temperature 0.5 \
    --prosody-pitch-model runs/p2_bert_g32/prosody_ema.pt --text "..."
```

```python
synth = Synthesizer("drifting_tts_v3.1.pt", vocoder=..., prosody="runs/pm_drift_final/prosody_ema.pt",
                    prosody_temperature=0.5, prosody_pitch="runs/p2_bert_g32/prosody_ema.pt")
```

Against v3.2 on the same vocoder:

| | v3.2 | candidate |
|---|---:|---:|
| token pitch r, all speakers / studio (T 0.5) | 0.436 / 0.670 | 0.494 / 0.700 |
| token pitch spread, all speakers / studio | 0.91 / 0.91 | 0.94 / 0.96 |
| DTW F0 r with the recording (studio `val`, audio) | 0.578 | 0.601 |
| F0 std (recordings 3.68, Vocos v2 copy synthesis 3.55) | 3.38 | 3.49 |
| held-out polar questions: pre-mI / fall (recordings −0.22 / −3.35) | −1.51 / −0.28 | −0.05 / −2.59 |
| diagnostic polar questions: pre-mI / host boost (readers +0.87 / ≈ +4.5) | −2.39 / +1.38 | −1.61 / +2.47 |
| Freya-495 WER / UTMOSv2 | 1.84% / 2.993 | 1.92% / 2.988 (both n.s.) |
| prosody networks / word encoder | 8.1 M / – | 8.1 M + 8.3 M / BERTurk 111 M |

- **Cost:** a second 8.3 M sampler and BERTurk (111 M parameters, `transformers`) per sentence; phase 1 measured
  ~40 ms of BERT per sentence on the busy GPU. It runs eagerly: with `fast=True` only the vocoder uses CUDA graphs
  (see the notes).
- **Durations:** the candidate samples them (studio voice measured). v3.2 as released keeps the regressors'
  durations; for the male and female voices sampled durations cost words
  ([EXPERIMENTS.md §9](EXPERIMENTS.md#9-release-v32)). `--prosody-durations regressor` gives v3.2's durations with
  the BERTurk pitch (not measured).
- **Lighter alternative:** the ELECTRA branch (`p2_electra_pitch`, one 11.9 M sampler + ELECTRA-small 13.7 M): the
  best intelligibility (Freya-495 1.51%, −0.33 pp n.s.) with a smaller gain in conditional pitch (r 0.451, DTW F0 r
  0.596, held-out pre-mI −0.91).
- **The pitch temperature** stays at 0.5: higher values widen the F0 a little (3.49 → 3.51 at 0.7) at a lower
  contour correlation.
- **Listening decides.** Samples of v3.1, v3.2, the ELECTRA branch and both BERTurk-pitch variants on the
  diagnostic questions, lists, long sentences, Freya sentences and two short paragraphs, for the three voices and
  two seeds, are local in `runs/p2_samples/` (`index.md`).

### Phase 2 notes and pitfalls

- **Compare F0 statistics only under the same vocoder.** Vocos v2 lowers the measured F0 std of generated speech
  by about 0.25 st against `vocos-ft` (copy synthesis: 0.14 st), so phase-1 and phase-2 audio rows are not comparable.
- **GPU memory.** The context branch at B8 × G16 reserved up to 8.0 GB uncapped (nvidia-smi); under
  `train.max_gpu_gb=6.5` it ran out of memory on a long batch after 2.9k steps. B6 × G16 (5.8 GB in nvidia-smi) and
  B3 × G32 (4.6 GB) fit the cap at 9–12 it/s.
- **Release integration (#49, #52).** The CUDA-graph prosody path (`fast.graphable`) takes one-pass samplers without
  word or sentence features (both are computed from the text on the host). A second pitch predictor also runs
  eagerly (`Synthesizer` warns and graphs only the vocoder). Separate temperatures run in the graphs (#52's
  duration row).
- **Word features at inference** are computed from the normalised sentence (ELECTRA-small or BERTurk, `transformers`);
  the sentence features need no model.
- **Male and female voices** (diagnostic set, token level, 4 seeds): the female voice (323), which has no questions in
  its training data, ends polar questions with a rise in every system (fall +1.9 st for v3.1, +3.7 for v3.2's
  sampler, +3.7 for the ELECTRA branch, +1.4 for BERTurk for everything); the male voice (389) ends them low (84–90%
  of the samples, v3.1 12%). These are the voices' learned styles; neither has held-out questions to compare with.

### Phase 2: reproduce

```bash
# word features of ELECTRA-small in a copy of the prosody cache (~7 min on the shared GPU)
cp runs/pm_cache/targets_v31.pt runs/p2_cache/targets_v31_electra.pt
drifting-tts prosody-cache --model runs/release/drifting_tts_v3.1.pt --out runs/p2_cache/targets_v31_electra.pt \
    --word-model dbmdz/electra-small-turkish-cased-discriminator
# context branch: word + sentence features for the pitch only (B8 x G16 needs ~8 GB; B6 fits a 6.5 GiB cap)
drifting-tts train-prosody --workdir runs/p2_electra_pitch tts=runs/release/drifting_tts_v3.1.pt \
    cache=runs/p2_cache/targets_v31_electra.pt train.batch_size=8 \
    net.word_dim=1 net.sent_dim=1 net.ctx_pitch_only=true calibrate.temperature=0.5
#   variants: net.ctx_boundaries=true, loss.centroid=1.0, cache=runs/pm_cache/targets_v31_bert.pt (BERTurk),
#   drift.gen_per_cond=32 train.batch_size=3 train.steps=20000 (G = 32)
# store the preferred temperatures (pitch: calibrate.temperature, letters' durations: calibrate.duration_temperature)
# and the voice factors at that operating point
drifting-tts train-prosody --workdir runs/p2_electra_pitch --calibrate-only tts=... cache=... \
    calibrate.temperature=0.7 calibrate.duration_temperature=0.5
# synthesis with separate temperatures (the defaults: the checkpoint's)
drifting-tts synthesize --model runs/release/drifting_tts_v3.1.pt --prosody runs/p2_electra_pitch/prosody_ema.pt \
    --prosody-temperature 0.7 --prosody-duration-temperature 0.5 --vocoder runs/voc_p3/vocos_ft_10000.pt --text "..."
# durations from v3.2's sampler, token pitch from the BERTurk sampler (the phase-2 candidate)
drifting-tts synthesize --model runs/release/drifting_tts_v3.1.pt --prosody runs/pm_drift_final/prosody_ema.pt \
    --prosody-pitch-model runs/p2_bert_g32/prosody_ema.pt --prosody-temperature 0.5 \
    --vocoder runs/voc_p3/vocos_ft_10000.pt --pause-policy punct --text "..."
# the BERTurk sampler with G = 32 (B3 x G32 fits the 6.5 GiB cap, 12 it/s)
drifting-tts train-prosody --workdir runs/p2_bert_g32 tts=runs/release/drifting_tts_v3.1.pt \
    cache=runs/pm_cache/targets_v31_bert.pt net.word_dim=1 train.batch_size=3 drift.gen_per_cond=32 \
    train.steps=20000 train.max_gpu_gb=6.5 calibrate.temperature=0.5
# token level with pitch temperatures (the durations at 0.5), intonation by sentence type (`@<T>,<T durations>`), the
# diagnostic set (token level; --audio adds synthesis with the vocoder and harvest F0), the audio systems with a
# duration temperature of their own (`-D`)
python scripts/eval_prosody_tokens.py --cache runs/pm_cache/targets_v31.pt --tts runs/release/drifting_tts_v3.1.pt \
    --prosody drift=runs/pm_drift_final/prosody_ema.pt --temperatures 0.5 0.7 1.0 --duration-temperatures 0.5 --scales \
    --out runs/p2_eval/tokens
python scripts/eval_sentence_prosody.py --heldout --reference train --systems v31=regressors \
    v32=runs/pm_drift_final/prosody_ema.pt@0.5 cand=runs/p2_electra_pitch/prosody_ema.pt@0.7,0.5 --out runs/p2_eval/heldout.json
python scripts/eval_sentence_prosody.py --texts scripts/prosody_diagnostic_tr.jsonl --speakers 722 --seeds 4 \
    --systems v31=regressors v32=runs/pm_drift_final/prosody_ema.pt@0.5 --audio --vocoder runs/voc_p3/vocos_ft_10000.pt \
    --out runs/p2_eval/diag.json
python scripts/eval_prosody_audio.py --prosody-model drift=runs/pm_drift_final/prosody_ema.pt \
    --systems recording copy onepass drift-T0.5 drift-T0.7-D0.5 drift-T1-F3 --vocoder runs/voc_p3/vocos_ft_10000.pt \
    --asr none --sv none --mos none --out runs/p2_audio/val722
```

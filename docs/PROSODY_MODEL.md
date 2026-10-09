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
- [Word-level context (#40)](#word-level-context-40)
- [Notes and pitfalls](#notes-and-pitfalls)

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

FREYA_TABLE

FREYA_NOTES

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

**Recommended setting** (from the tables above): the drift sampler at prosody temperature 0.5 with its own voice
factors (`train-prosody --calibrate-only calibrate.temperature=0.5` stores both, so `--prosody` alone selects it).
It keeps Freya-100 intelligibility (WER / CER within the v3.1 intervals), raises UTMOSv2 beyond the v3.1 interval and
gives the held-out studio sentences the recordings' intonation range. `--prosody-durations regressor` (pitch only)
is the conservative option: v3.1's rhythm and UTMOSv2, the new intonation. The default stays the regressors until
a listening test.

```bash
# 1. targets of a trained pitch-conditioned model (~2 min)
drifting-tts prosody-cache --model runs/release/drifting_tts_v3.1.pt --data data/tr12_eleven \
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
```

- **Seeds.** The prosody noise is drawn from the same seeded generator as the DiT noise, before it, so a seed
  fixes the whole rendition.
- **Speaking rate.** The sampler has its own per-voice duration factors (`train-prosody --calibrate-only`: the median
  recorded / sampled length on training utterances, never on the evaluation splits). It replaces the v3.1 factors,
  which compensate the regressors' log-domain bias and `ceil`. `length_scale` still applies on top.
- **`fast=True`.** The CUDA-graph acoustic path (`drifting_tts/fast.py`) covers only the regressors. With `prosody`
  set, the acoustic model runs eagerly, and only the streaming vocoder windows use CUDA graphs.
- **Cost.** One 165-token sentence on the shared (busy) RTX 5090: text encoder 6.6 ms, + drift sampler 9.9 ms in
  total (one pass), flow matching with 8 Euler steps 35.7 ms, drift + BERTurk 48 ms (BERT dominates). Busy-GPU
  numbers, 2–4× above an idle GPU; the drift sampler adds a few milliseconds to time-to-first-audio.
- **Not ported:** ONNX / WebGPU and MLX still use the regressors.

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
| drift | 0.297 / 0.200 | 0.379 / 0.594 | 0.99 | 1.01 | −1.57 / −0.52 | 2.63% / 0.66% / PITCH_ONLY_DRIFT |
| drift + BERTurk | **0.283 / 0.190** | **0.435 / 0.639** | 0.97 | 1.04 | **−0.13 / −2.37** | 4.50% / BERT_T05 / 0.66% |
| recordings | 0 | 1 | 1 | 1 | +0.05 / −4.24 | – |

- **Word context makes the sampled tune more text-specific.** Pitch CRPS improves by 5%, correlation rises by
  0.05–0.06, and the polar-question shape (the pitch before mI, the final fall) moves towards the recordings, at the
  same distributional match.
- **It does not help the durations on out-of-domain text.** At T 1 the BERTurk sampler's sampled durations are
  worse for intelligibility on Freya-100 (WER 4.50%). With the regressors' durations (pitch only) it is as clean
  as v3.1.
- **Cost:** BERTurk adds ~40 ms per sentence on the busy GPU and a `transformers` dependency at inference. A smaller
  cased encoder (ELECTRA-small-tr, 13.7 M, MIT) and fine-tuning (Kenter et al. 2020) are the obvious next steps;
  neither was tried here.

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

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

*(filled in below)*

## Usage

```bash
# 1. targets of a trained pitch-conditioned model (~2 min)
drifting-tts prosody-cache --model runs/release/drifting_tts_v3.1.pt --data data/tr12_eleven \
    --out runs/pm_cache/targets_v31.pt
# 2. train (configs/prosody_drift.yaml; net.kind=mse / flow for the baselines)
drifting-tts train-prosody --workdir runs/pm_drift tts=runs/release/drifting_tts_v3.1.pt \
    cache=runs/pm_cache/targets_v31.pt train.batch_size=12
# 3. synthesise with it (opt-in; the default stays the deterministic regressors)
drifting-tts synthesize --model runs/release/drifting_tts_v3.1.pt --prosody runs/pm_drift/prosody_ema.pt \
    --prosody-temperature 1.0 --temperature 0.3 --cfg 2 --vocoder vocos-ft --text "..."
```

```python
synth = Synthesizer("drifting_tts_v3.1.pt", vocoder="vocos-ft", prosody="prosody_ema.pt", prosody_temperature=1.0)
wav, _ = synth(text, speaker="studio", cfg_scale=2.0, temperature=0.3, seed=0)
```

- **Seeds.** The prosody noise is drawn from the same seeded generator as the DiT noise, before it, so a seed
  fixes the whole rendition.
- **Speaking rate.** The sampler has its own per-voice duration factors (`train-prosody --calibrate-only`: the median
  recorded / sampled length on training utterances, never on the evaluation splits). It replaces the v3.1 factors,
  which compensate the regressors' log-domain bias and `ceil`. `length_scale` still applies on top.
- **`fast=True`.** The CUDA-graph acoustic path (`drifting_tts/fast.py`) covers only the regressors. With `prosody`
  set, the acoustic model runs eagerly, and only the streaming vocoder windows use CUDA graphs.
- **Not ported:** ONNX / WebGPU and MLX still use the regressors.

## Word-level context (#40)

*(filled in below)*

## Notes and pitfalls

*(filled in below)*

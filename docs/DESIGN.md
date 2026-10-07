# Design: one-step TTS with Drifting Models

This document explains how *Generative Modeling via Drifting* (Deng et al., 2026,
[arXiv 2602.04770](https://arxiv.org/abs/2602.04770), code: [lambertae/drifting](https://github.com/lambertae/drifting))
is mapped to text-to-speech. It also lists the follow-up work that shaped the choices.

## 1. The method in one paragraph

A generator `f_θ` maps noise `ε` to a sample `x = f_θ(ε)` in **one** forward pass. Training does
not denoise. It moves the generator's pushforward distribution `q` toward the data distribution `p`
with a *drifting field*:

```
V_{p,q}(x) = E_{y+~p, y-~q}[ k(x,y+) k(x,y-) (y+ - y-) ] / (Z_p Z_q),   k(x,y) = exp(-||x-y|| / τ)
```

The field is anti-symmetric (`V_{p,q} = -V_{q,p}`). It therefore vanishes when `q = p`
(Prop. 3.1). The network regresses onto a frozen, drifted copy of its own output:

```
L = || f_θ(ε) - stopgrad( f_θ(ε) + V(f_θ(ε)) ) ||²
```

The field is computed in the multi-scale feature space of a frozen encoder `φ`. It uses several
temperatures, softmax normalisation over both axes, mean-distance scale normalisation,
per-temperature force normalisation and self-masking. Classifier-free guidance is learned *during
training*: unconditional real samples are added as negatives with weight `(α-1)(G-1)/N`, and the
generator is conditioned on `α`. Inference stays at 1-NFE.

## 2. From ImageNet classes to TTS conditions

| Paper (class-conditional ImageNet) | This repo (TTS) |
|---|---|
| condition = class label | condition = frame-aligned text (prior mean `μ_y` ⊕ text features `h_y`) + speaker |
| `x` = SD-VAE latent 32×32×4 | `x` = 100-bin log-mel (Vocos 24 kHz), random 256-frame windows |
| positives = memory bank of the class (64–128) | the single target window + `P-1` perturbed views |
| negatives = 64 generated samples of the class | `G = 8` generated samples of the same condition (self-masked) |
| CFG negatives = unconditional data bank | real windows of *other* utterances from an unconditional bank |
| latent-MAE ResNet features | Mel-MAE 1-D ResNet features + raw-mel features |
| DiT-B/2 generator with style tokens | DriftDiT (23 M) with the same style / register / CFG embeddings |

The central difficulty: **each text has exactly one recording.** Per-location drift problems are
therefore built per condition. The `G` generated windows for a text compete (repulsion) and are
attracted to the target window and its views, one drift problem per (condition, feature location),
as in the `(b f) x d` layout of the official trainer. Conditional generation papers with one target
per condition converge on this recipe. DriftWorld uses one positive with `N` samples per condition,
and DriftTTS uses a positive pool of the target plus Gaussian views.

The text-to-frame alignment comes from monotonic alignment search against a Gaussian prior
(Grad-TTS / Matcha-TTS). Generated and target frames are therefore time-aligned, and per-location
features compare like with like. Turkish orthography is close to phonemic, so characters are the
input symbols.

## 3. Training step

1. The text encoder gives hidden states `h`, prior mean `μ` and log-durations. MAS aligns `μ` to
   the target mel. The prior NLL and the duration MSE are computed.
2. The same random 256-frame window is cut from the target mel and from the aligned condition
   `[μ_y; h_y]`.
3. `α ~ p(α) ∝ α^-3` on `[1, 3]` is sampled (10% `α = 1`), and `G = 8` samples are generated per
   condition: `x = μ_y + DriftDiT(z, cond, spk, α)`.
4. The Mel-MAE features of generated, positive and unconditional windows are computed. The drift
   loss is applied on every feature map (per-location, windowed mean/std, global mean/std, raw mel),
   with temperatures `{0.02, 0.05, 0.2}`.
5. `loss = drift + prior + duration`. The encoder and the generator are gradient-clipped
   separately (2.0), with AdamW (β = 0.9 / 0.95) and EMA 0.999.

Inference: text → durations (× calibrated `duration_scale`) → `[μ_y; h_y]` → **one** DriftDiT evaluation
(noise temperature 0.5 works best) → Vocos.

## 4. Deviations from the official code and why

- **Residual on the prior mean** (`x = μ_y + G(·)`). With a zero-initialised output layer the
  official generator starts with a constant output. Through the GroupNorm layers of the frozen
  feature encoder, a constant mel gives gradients of about 1e14 in the first steps. Starting from
  `μ_y` keeps the features well conditioned.
- **Drift loss averaged over feature maps** instead of summed. The ~40 feature maps would otherwise
  dwarf the prior and duration losses.
- **Separate gradient clipping** for the text encoder and the generator, for the same reason.
- **Affinity floor** (`1e-6` in the official code) is configurable. On very low-dimensional
  multi-modal toys it adds a uniform moment-matching term that freezes training, and `0` recovers
  both modes. For high-dimensional speech features the official value is kept.

## 5. Learned temperature (Kyutai) vs. the official field

Kyutai trained the Pocket TTS sampler head with drifting
([blog, 2026-09-28](https://kyutai.org/blog/2026-09-28-pocket-tts-drifting/)), also with one positive
per condition: one 32-d codec latent per frame, given the autoregressive context. They report:

- the paper's fixed temperature set fails (UTMOS 2.17), and so does τ fixed at its converged value (collapse);
- a temperature learned as a data-vs-siblings classifier works. It starts wide and anneals to ≈ 0.056,
  and the start value barely matters (0.05 to 10);
- distances normalised per frame (per drift problem) reach the same quality in 200k instead of 300k steps;
- 128 rows × 32 negatives per step, and noise temperature 0.3 at inference.

The blog's pseudo-code is a simplification. The released code
([pocket-tts](https://github.com/kyutai-labs/pocket-tts) `training/modules/samplers.py`, class `Drifting`)
runs a different field. The repo has both:

| | `drift.mode: learned` (blog pseudo-code) | `drift.mode: kyutai` (released code) |
|---|---|---|
| field | one softmax over {data ∪ siblings} | two-sided softmax, product form (as `official`) |
| τ loss | −log p(data), mean over all candidates | −log p(data) of the best candidate per row |
| τ | log τ, start 1.0 | raw τ ≥ 1e-3, start 1.0 (blog's best) or 10 (released config) |
| per-row normalisation | distances | distances and regression coordinates |

**Why the pseudo-code diverges with one positive.** Take a sample `x_i` with union-softmax weights `w₀` on
the data `y⁺` and `w_k` on its siblings (`Σ w_k = 1 − w₀`, weighted mean `ȳ_i`):

```
V_i = w₀ (y⁺ − x_i) − Σ_k w_k (y_k − x_i) = (1 − 2w₀) x_i + w₀ y⁺ − (1 − w₀) ȳ_i
```

The coefficient on the sample's own position is the repulsion mass minus the attraction mass. While the
data holds less than half of the kernel mass (`w₀ < ½`: early in training, or whenever τ is wide), it is
positive and the samples drift apart. Distances and the field are renormalised every step, so this
outward term is scale-free and compounds (pilot: spread → 17.6). The mean-over-candidates τ loss adds to
it: most candidates have a sibling as nearest neighbour, so τ grows (pilot: 1.0 → 2.1). This is not
specific to our near-delta targets. Kyutai's 32–64 samples per positive make the imbalance larger, and
the toy diverges just as fast when each step's single positive is drawn from a conditional with spread.

**The product form cancels it.** The weights are `W⁺_ij = A⁺_ij Σ_k A⁻_ik` and `W⁻_ik = A⁻_ik Σ_j A⁺_ij`.
Attraction and repulsion therefore carry the same mass for every sample, and the coefficient on `x_i` is
exactly 0. What remains is `V_i ∝ ȳ⁺_i − ȳ⁻_i`: a translation towards the data relative to the siblings,
whatever τ is. This is the field of `official` and of Kyutai's code.

Toy of `tests/test_drift.py`: 16 conditions × 8 samples in 4-D, 800 Adam steps at lr 1e-3 (τ after 4k
steps in brackets).

| | 1 positive (delta) | target + 7 views | 1 positive drawn from N(target, 0.5²) |
|---|---|---|---|
| `official` | converges (centroid MSE 0.004) | converges (0.002) | converges, spread 0.52 |
| `learned` | diverges | converges (0.001), τ rises 1.0 → 1.2 | diverges |
| `kyutai` | converges (0.0006), τ 1.0 → 1.09 → 0.93 (0.15) | converges (0.002), τ → 0.90 (0.04) | converges, spread 0.48, τ → 0.98 (0.15) |
| **real data pilot**, `official` | centroid 0.58 → 0.27 in 2.5k steps, spread stable ≈ 0.33 | | |
| **real data pilot**, `learned` | centroid 0.47 → **156**, spread → 17.6, τ 1.0 → 2.1 by step 1.2k | | |

So the pilot shows that the *blog pseudo-code* diverges here. It does not show that Kyutai's recipe does.

`drift.mode: kyutai` (`kyutai_drift_loss`) follows the released code line by line. It is extended to our
rows of `P` positives (target + views), `G` samples and weighted CFG negatives:

- each row (condition × feature location) divides its distances by its mean generated-to-{siblings, positives}
  distance, self pairs excluded, and its regression coordinates by that mean / √D. The CFG negatives are
  left out of the scale, so it does not depend on α;
- the product-form field at one τ is divided by the mean over rows of its per-row RMS;
- τ is trained only by `−mean_rows max_i log Σ_{j ∈ pos} softmax_j(−d_ij / τ + log w_j)`. In each row, the
  sample that puts the most kernel mass on the positives sets τ. The candidates are the positives, the
  siblings (self masked) and the CFG negatives with multiplicity `w`;
- there is one global τ (`drift.kyutai.per_feature_tau: true` gives one per feature map), in its own AdamW
  group with the model's LR and schedule and no weight decay, starting at `drift.kyutai.tau_init`. Under
  Adam a raw τ moves by at most ≈ lr per step, so at lr 2e-4 annealing from 1.0 takes ≳ 5k steps, and from 10
  ≳ 50k steps.

Use `drift.pos_views=1` for Kyutai's single positive. What still differs from Kyutai is structural:
Mel-MAE features instead of a 32-d latent, `G = 8` instead of 32–64 negatives, and 16 windows instead of
128 utterances (each hundreds of frame problems) per step.

**Real-data A/B.** Two 10k-step pilots with the v2 recipe and the later fixes (2-D MAE, spectral
detail, pitch, block features, CFG [1, 4], BigVGAN mels, batch 16 × G 8) differ only in `drift.mode`:

| | harmonic contrast, T = 0.5 (low / mid / high) | CER, T = 0.7, α = 1 / 2 | speaker sim. | UTMOSv2 |
|---|---|---|---|---|
| `official` | 0.93 / 0.94 / 0.90 | 17.7% / 15.8% | 0.39 | 1.50 |
| `kyutai` | **0.98 / 0.98 / 0.96** | **15.4% / 13.3%** | **0.43** | 1.50 |

τ rose to 1.19 and then annealed (0.56 at 10k steps, 0.072 after 150k steps of the v3 run, Kyutai: ≈ 0.056).
Kyutai's released field is therefore the better choice here as well, and `configs/tts_v3.yaml` uses it.
`official` stays the code default so that older configs reproduce.

## 6. What we monitor

The drift loss value is roughly constant by construction, because forces are normalised per
temperature. The useful signals are:

- `force_τ`: the raw squared drift norm before normalisation.
- `tau_mean` / `p_data` (`kyutai`, `learned`): the learned temperature and the kernel mass on the data. With
  `kyutai`, τ should rise briefly and then anneal. A steadily growing τ means the siblings stay closer than
  the data.
- `centroid_mse`: the mean of the `G` samples vs. the target.
- `across_sample_std`: collapse detector.
- prior / duration losses.
- ASR CER of synthesised validation sentences.

## 7. Related work consulted

- DriftTTS (arXiv 2610.03390): drift on 192-frame mel segments, MelMAE + raw mel features, positive
  pool = target + Gaussian views (σ = 0.05), Grad-TTS conditioning. It uses 4-step on-policy
  rollouts for its best quality. Its 1-NFE number (MOS 3.00 vs 4.18) is the 4-step model run for one
  step; no dedicated one-step model is reported.
- Speech enhancement with drifting (arXiv 2604.24199) and DriftSE (2609.12252): frame-pooled drift in
  SSL feature space. DriftSE shows that pairing is essential for content: unpaired training raises WER
  from 7.09% to 20.55% (VoiceBank-DEMAND denoising).
- DriftWorld (arXiv 2607.15065): one positive and `N` samples per condition, per-patch drift. Its
  "accentuation" negative (the CFG analogue) is the current ground-truth frame, with weight `(α−1)(N−1)`,
  not an unconditional bank.
- One target per condition: only Ada3Drift (2603.11984) shows that a batch-marginal, drift-only field
  fails (17.0% success vs. 60.5% for regression). GDM (2604.19736) uses a regression anchor by design
  and never trains drift-only. CoDrift (2608.23939) shows the converse: its per-condition single-positive
  field works, and adding a pooled marginal field helps (+6.8 points).
- Kyutai Pocket TTS with drifting (blog, 2026-09-28): drifting replaces the LSD one-step flow head of
  an autoregressive continuous-latent TTS at parity (WER 0.96%, UTMOS 4.32), with Mimi latents as the
  kernel space, learned τ, per-frame normalisation, 128 rows × 32 negatives, latent CFG and noise
  temperature 0.3. Drifting needed about 2× the training cost of LSD. The released code differs from
  the blog's pseudo-code: it uses the product-form field and a best-candidate τ loss, and its config
  starts τ at 10 with 64 negatives (§5).
- Theory and estimators: Sinkhorn drifting (2603.12366), W-Flow (2605.11755; bias of diagonal
  masking), ABC minibatch correction (2604.27239), the friction analysis (2604.18194; with one target
  the spread around it is set by `τ`).

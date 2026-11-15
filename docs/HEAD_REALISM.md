# Is ARDrift's head realistic? What the judges miss, and a test that sees it (#105, #97, #85)

Listening says ARDrift sounds robotic. UTMOSv2, WER and speaker similarity score it at the level of the recordings
([ENGLISH.md](ENGLISH.md)), so none of them can guide work on that sound. This page records what was measured on the
released English model, what the measurements rule out, and the test that does separate the model from the
recordings.

- [Setup](#setup)
- [What is ruled out](#what-is-ruled-out)
- [The test that sees it: a classifier two-sample test](#the-test-that-sees-it-a-classifier-two-sample-test)
- [What the classifier sees](#what-the-classifier-sees)
- [Why the drifting kernel does not see it](#why-the-drifting-kernel-does-not-see-it)
- [Where the head is miscalibrated: the fast part of the patch](#where-the-head-is-miscalibrated-the-fast-part-of-the-patch)
- [The mechanism: one positive, unequal directions](#the-mechanism-one-positive-unequal-directions)
- [Fine-tuning the head alone](#fine-tuning-the-head-alone)
- [What is not settled](#what-is-not-settled)
- [A panel of judges: where the audible gap is](#a-panel-of-judges-where-the-audible-gap-is)
- [Commands](#commands)

## Setup

- **Model:** `Vyvo/drifting-tts-en` (`ardrift_en.pt`: 200k steps, 178 M parameters; `bigvgan_en.pt`).
- **Data:** the held-out source shard 07,960 (500 utterances, 1.4 h, texts never trained on, voices seen). 381 of them
  are at most 16 s and go through the model's alignment.
- **Renditions of one utterance:**
  - *teacher-forced*: every patch drawn by the head from the state after the **recorded** history (the regime the
    head is trained in);
  - *free-running*: the model on its own output, under the recording's own durations (MAS), so recording and
    rendition have the same text at every frame;
  - *synthesis*: free-running with the predicted durations.

## What is ruled out

| question | measurement | result |
|---|---|---|
| Is the loss Kyutai's? | `kyutai_drift_loss` against `pocket-tts/training/modules/samplers.py` (class `Drifting`), line by line | the same field, scale (`n / (n - 1)`), self mask, batch-mean RMS and best-candidate temperature loss |
| Does the head collapse to the mean? | spread between two samples of a state over their distance to the recorded patch (1 = calibrated) | T 1.0: **0.98** (patch), 0.99 (frame), 0.98 (detail); spread-skill 0.98 in every band |
| Is the mel blurred? | variance of the mel's Laplacian (VarL), RMS of the harmonic detail | 0.91–1.09 and 0.98–1.02 of the recordings' |
| Do the patches join? | frame-to-frame change across patch boundaries over the change inside a patch | free-running 1.00–1.06, recordings 1.015 (teacher-forced 1.23–1.37: its patches are independent draws) |
| Is the intonation flat? | F0 std, range, movement, micro-variation, energy std (60 utterances, 2 seeds) | within 1% of copy-synthesis under the recording's durations |

- **The operating temperature narrows the head.** At T 0.5 the spread / error ratio is 0.68 and at T 0.3 it is 0.46.
  The recording's frame-to-frame motion inside a patch exceeds all 32 samples in 16% of the states at T 1.0 (3%
  expected), in 49% at T 0.5 and in 63% at T 0.3.
- **The predicted durations shorten the speech.** Synthesis is 10% shorter than the recordings (5.47 s of speech
  against 6.04 s), with more pauses (2.57 against 1.82 per utterance) that are shorter in total (2.09 s against
  2.55 s). The F0 correlation with the recording along a DTW path falls from 0.66 (recording's durations) to 0.59;
  copy-synthesis reaches 0.80.
- **#97's first step is answered:** the patch boundaries are not where the model differs from the recordings.

## The test that sees it: a classifier two-sample test

A small CNN (4 convolutions, 0.4 M parameters) is trained to tell windows of the recordings' mels from the same
windows of the model's mels (C2ST, [Lopez-Paz & Oquab](https://arxiv.org/abs/1610.06545)). Utterances are split
80 / 20; the accuracy is on the held-out 20%. 0.5 means the two cannot be told apart.

Controls: only windows of speech (every frame's mean level above −0.5), the model's mel clamped at the recordings'
log-mel floor, Gaussian dither of 0.02 on both sides (the stored recordings are float16).

| rendition | 4 frames (43 ms) | 16 frames (170 ms) |
|---|---|---|
| the recording against itself (null) | 0.50 | 0.50 |
| free-running, T 1.0 | 0.67 | 0.91 |
| free-running, T 0.5 (the operating point) | **0.88** | **0.97** |
| teacher-forced, T 1.0 | 0.80 | 0.97 |
| the aligned prior mean | 1.00 | 1.00 |

**One whole patch of the head** (windows on the patch grid, so no patch boundary is inside), by what the classifier
is shown:

| rendition | whole patch | harmonic detail | envelope | frame-to-frame change |
|---|---|---|---|---|
| teacher-forced, T 1.0 | 0.81 | 0.86 | 0.55 | **0.90** |
| teacher-forced, T 0.5 | 0.86 | 0.93 | 0.71 | 0.94 |
| free-running, T 1.0 | 0.82 | 0.86 | 0.61 | 0.91 |
| free-running, T 0.5 | 0.91 | 0.93 | 0.68 | 0.95 |

- **A single patch gives the model away**, in the regime the head is trained in and at the temperature it is trained
  at. Shown both renditions of one window, the classifier picks the recording 92–96% of the time.
- **It is the fine structure, not the envelope.** The spectral envelope alone is near chance (0.55); the harmonic
  detail and how the frames follow each other are what differ.
- **History noise is not the cause:** teacher-forced with noise 0 / 0.15 / 0.3 on the history gives 0.81 / 0.79 /
  0.78. Another classifier seed gives 0.82.
- **A lower temperature makes single frames recognisable:** frames in random order are told apart at 0.65 (T 1.0)
  and 0.86 (T 0.5).

## What the classifier sees

Statistics of whole patches of speech (45,000 patches; teacher-forced and free-running agree to the third digit):

| inside one patch | recordings | model, T 1.0 | model, T 0.5 |
|---|---|---|---|
| correlation of the harmonic detail of neighbouring frames | 0.735 | 0.775 | 0.780 |
| correlation of one frame-to-frame change with the next | 0.294 | 0.355 (+21%) | 0.424 (+44%) |
| RMS of the frame-to-frame change of the harmonic detail | 0.106 | 0.096 (−9%) | 0.092 (−13%) |
| correlation of the change between neighbouring mel bins | 0.436 | 0.485 (+11%) | 0.466 |

The model's frames follow each other too smoothly: the detail of neighbouring frames is more alike than in speech,
and the motion is closer to a straight line through the patch. The frame-to-frame irregularity of a voice (jitter,
shimmer, breath) is what is missing, which is heard as a buzzing, mechanical timbre.

## Why the drifting kernel does not see it

- **The kernel's maps:** the whole patch (400-d), each frame (100-d), each frame's harmonic detail (100-d)
  (`ar_features`). The two frame maps look at one frame at a time: nothing in them depends on how frames follow each
  other.
- **The patch map** is an L2 distance in 400 dimensions. The mean squared distance between two patches is the sum
  of the variances of their 400 values (plus the squared distance of the means): it does not depend on how the
  values are correlated, so neither does the scale the kernel divides by. The correlation between frames only
  changes how much the distances fluctuate around that mean. In that space the head is calibrated (0.98) and its
  samples are still told from the data.
- **This is the published failure of drifting on raw spectra.**
  - [Deng et al.](https://arxiv.org/abs/2602.04770), Sec. 5.2: "we were unable to make our method work on ImageNet
    without a feature encoder".
  - [DriftTTS](https://arxiv.org/abs/2610.22960), Table 2: frame-wise drifting on the raw mel reaches WER 2.72 /
    UTMOSv2 2.137 where drifting on 80-frame segments in mel-MAE features reaches 1.80 / 3.931; "many frames do not
    connect naturally with their neighbors".
  - Kyutai drift on raw latents and name the reason: Mimi's latents are distilled from WavLM, so "plain L2 distance
    between Mimi latents is a valid kernel"
    ([blog](https://kyutai.org/blog/2026-09-28-pocket-tts-drifting/)).
- **Rows per step.** Kyutai's ablation: 64 utterances per step never leave the plateau (UTMOS 3.77 at 250k steps),
  128 leave it at 150k, 256 at 100k: "batch size is part of the objective". ARDrift draws 2,048 patches from about
  48 utterances per step: 87 s of speech in the loss, against about 20 minutes.

## Where the head is miscalibrated: the fast part of the patch

The overall ratio of 0.98 hides where the spread is. Spread-skill (1 = calibrated) for each coefficient of a cosine
transform along the patch's four frames, on patches of speech of 100 held-out utterances, T 1.0
(`drift_calibration.py --active -0.5`):

| temporal frequency inside the patch | share of the error energy | mel | harmonic detail |
|---|---|---|---|
| the mean of the four frames | 59% | 1.03 | 1.03 |
| their trend | 25% | 0.97 | 0.96 |
| mid | 11% | 0.89 | 0.85 |
| their alternation (fastest) | 5% | **0.80** | **0.76** |

- **The head spends its variance on slow modes.** The two slow coefficients carry 84% of the error energy and set
  the overall ratio; in the fastest one the samples have 64% of the variance they should have (57% in the harmonic
  detail). The sample deviations of neighbouring frames correlate at 0.65 / 0.71 / 0.76 where the recordings'
  residuals correlate at 0.55 / 0.65 / 0.70.
- **Rescaling the spread per coefficient removes most of the full-view tell.** With every coefficient of the
  patch's 2-D cosine transform rescaled around the sample mean (gains from training rows: 0.99 / 1.06 / 1.16 / 1.27
  by temporal frequency), the one-patch C2ST falls from 0.82 to 0.61 on the whole patch and from 0.86 to 0.80 on the
  harmonic detail. The frame-to-frame-change view stays at 0.89: that part is not a matter of variances.

## The mechanism: one positive, unequal directions

A Gaussian toy with the patch's energy profile (four bands with 59 / 25 / 11 / 5% of the variance, one positive and
32 samples per row, a generator with one scale per band, **started at the truth**; `tests/test_drift.py` has a
smaller version). Variance per band after training, truth = 1:

| objective | largest band | | | smallest band |
|---|---|---|---|---|
| Kyutai's field, one kernel over all bands, τ fixed at 0.3 | 0.85 | 0.75 | 0.49 | 0.08 |
| the same, τ fixed at 0.08 (the English model's learned value is 0.085) | 0.96 | 0.93 | 0.86 | **0.66** |
| the same, τ learned (settles at 0.016 in the toy) | 1.00 | 0.99 | 0.96 | 0.90 |
| Kyutai's field, one kernel per band | 0.99 | 0.99 | 0.99 | 0.99 |
| energy score, one kernel per band | 0.99 | 0.99 | 0.99 | 0.99 |
| **the English head, measured** (the squares of the table above) | 1.05 | 0.95 | 0.79 | **0.64** |

- **The data distribution is not a fixed point of the one-positive field.** Started at the truth, the generator
  leaves it: the less of the distance a direction carries, the more it shrinks, and the wider the kernel, the
  stronger the effect. At the temperature the English model learned, the toy ends where the head is.
- **It is a property of the estimator, not of this implementation.** The field's softmax is normalised over the
  one positive and the siblings together; with a single positive that weighting is biased
  ([Analytical Correction for Subsampling Bias in Drifting Models](https://arxiv.org/abs/2604.27239): "suppresses
  diversity"; [Sinkhorn-Drifting](https://arxiv.org/abs/2603.12366): zero drift does not imply `p = q`).
- **Why Kyutai do not see it.** Their kernel space is the codec latent normalised per dimension: 32 directions that
  carry about the same share of the distance. In such a space the bias is a small uniform shrinkage (the "one kernel
  per band" rows). A raw log-mel patch is far from it: 59% of the residual energy is in the mean of the four frames.
- **The energy score** (`drift.energy_score`: `E|x − y| − ½ E|x − x'|`) is a strictly proper scoring rule: with one
  observation per condition its expected gradient is zero only at the data distribution, and it has no temperature.
  [Engression](https://arxiv.org/abs/2307.00835) trains conditional generators with it,
  [SLED](https://arxiv.org/abs/2505.13181) the per-step head of a speech language model. It has the field's
  ingredients (attraction to the observation, repulsion between the samples) without the normalisation.

## Fine-tuning the head alone

`scripts/head_finetune.py`: the text encoder and the backbone frozen, their states computed once on 4,000 training
utterances (11 h, 8 shards; 500,000 patches), the head trained 2,500 steps from the released weights at 2,048 states
x 32 samples per step, lr 1e-4, EMA 0.999. Every run sees the same rows and differs only in the objective. Held-out
utterances as above; C2ST accuracies repeat to about ±0.02 between runs of the same head.

| head | fastest band, mel / detail | one patch, whole / detail / envelope / change | free-running T 1, 4 frames / 16 | UTMOSv2, T 0.5 / T 1 |
|---|---|---|---|---|
| released | 0.80 / 0.76 | 0.81 / 0.86 / 0.55 / 0.90 | 0.67 / 0.91 | 2.887 / 2.633 |
| control: the training's objective | 0.79 / 0.74 | 0.80 / 0.85 / 0.57 / 0.90 | 0.71 / 0.90 | – |
| Kyutai's field + frame-to-frame change maps | 0.79 / 0.77 | 0.70 / 0.84 / 0.56 / 0.89 | 0.70 / 0.90 | 2.852 / 2.652 |
| Kyutai's field + time-frequency tiles | 0.81 / 0.77 | 0.77 / 0.84 / 0.59 / 0.90 | 0.73 / 0.89 | – |
| Kyutai's field + whitened cosine bands | 0.82 / 0.78 | 0.73 / 0.82 / 0.55 / 0.89 | 0.69 / 0.90 | – |
| energy score, the training's maps + whitened bands | **0.99** / 0.90 | 0.68 / 0.76 / 0.69 / 0.89 | 0.62 / 0.86 | 2.816 / 2.532 |
| energy score, + Mel-MAE feature maps | 0.94 / 0.88 | **0.51** / **0.72** / 0.59 / 0.89 | **0.56** / 0.86 | 2.852 / 2.595 |

(UTMOSv2: val 100 through NVIDIA's BigVGAN-v2 as released, fine-tuned on neither head; recordings 3.258,
copy-synthesis 3.294. WER stays at 2.1–2.6% for every head.)

- **More of the same changes nothing**, and the test repeats.
- **New kernel maps do not help while the field is Kyutai's.** Frame-to-frame changes, tiles of 4 frames x 10 bins
  and whitened bands leave the fastest band at 0.79–0.82. In a map where the samples are too narrow the learned
  temperature rises (0.36 in the fastest whitened band, 0.07 in the slowest), and a wide kernel moves every sample
  by the same vector: it does not spread them. The joint whitened patch as the only map narrowed the samples further
  (stopped at 800 steps: sample spread 0.104 → 0.101, τ 0.22).
- **The energy score calibrates the head** in 700 steps: fastest band 0.80 → 0.99, the recording's motion exceeds
  all 32 samples in 3.8% of the states (16.2% before, 3.0% expected; all patches, pauses included), its harmonic
  depth in 3.1% (7.2%), and
  free-running mels at T 1 have the recordings' texture (VarL 0.93 → 1.00, change RMS 0.98 → 1.01).
- **With the feature maps of a mel masked autoencoder the whole-patch tell is gone**: 0.81 → 0.51 for one
  teacher-forced patch, 0.67 → 0.56 for four free-running frames, harmonic detail of 16 frames 0.91 → 0.80. The
  MAE (`configs/mae2d.yaml`, 4.2 M parameters, 16-frame windows, 10k steps on the same 11 h) sees the patch after
  three recorded patches of history; its five stages are five more maps. This run used 512 states x 16 samples per
  step, so its bands are less converged (0.94).

## What is not settled

- **UTMOSv2 does not follow.** Sentence by sentence against the released head, the energy head scores 0.07 lower
  at T 0.5 (95% interval −0.12 to −0.02) and 0.10 lower at T 1 (−0.16 to −0.04); with the MAE maps the difference is
  −0.04 at both temperatures and within its interval (−0.08 to +0.01). UTMOSv2 also prefers T 0.5 to T 1 for every
  head by 0.25, i.e. it prefers the narrower samples, and TTSDS2 reports its correlation with listeners at 0.39.
  Whether a calibrated head is heard as less robotic or as noisier is a listening question:
  `scripts/listening_test.py` makes the blind A/B page. No listening result exists yet.
- **The frame-to-frame-change view stays at 0.88–0.90** for every head, also after the oracle rescaling. Part of it
  is the aligned prior mean: the model's change at token boundaries over the change inside a token is 2.92–3.05
  against the recordings' 2.82, and the change is more alike across neighbouring mel bins (0.61 against 0.56).
- **The envelope view of 16 free-running frames got worse** with the energy score (0.62 → 0.74–0.77).
- **Only the head was trained.** The backbone never saw the new head's patches as its history, 2,500 steps is
  short, and the whitening scale was measured once. `train-ar` is unchanged.
- **English only**: the Turkish corpus was not available on this machine.

## A panel of judges: where the audible gap is

Listening to the two fine-tuned heads against the released one (blind A/B, 14 pairs each): hard to tell apart. So
the question went to seven published predictors that were trained on different data for different questions
(`scripts/judge_panel.py`): UTMOSv2, UTMOS22, SCOREQ (synthetic-speech domain), NISQA, DNSMOS P.835, SQUIM and
Audiobox Aesthetics. The 100 val sentences, the same speaker and seed for every system, each system compared with the
released head sentence by sentence (`*`: the 95% interval of the difference excludes 0).

| system | UTMOSv2 | UTMOS22 | SCOREQ | NISQA | DNSMOS OVRL | SQUIM | Audiobox PQ |
|---|---|---|---|---|---|---|---|
| recording | 3.27 | 4.20 | 4.28 | 4.63 | 3.29 | 4.33 | 7.49 |
| recording's mel through NVIDIA's BigVGAN-v2 | 3.31 | 4.16 | 4.24 | 4.63 | 3.30 | 4.30 | 7.42 |
| recording's mel through the released, fine-tuned BigVGAN-v2 | 3.33 | 4.12 | 4.23 | 4.66 | 3.31 | 4.30 | 7.53 |
| **released head, NVIDIA's BigVGAN-v2, T 0.5** (the baseline) | 2.91 | 3.45 | 3.61 | 4.46 | 3.12 | 4.17 | 6.95 |
| energy-score head | 2.84* | 3.40 | 3.56* | 4.47 | 3.08* | 4.21* | 6.86* |
| energy-score head with MAE maps | 2.87 | 3.43 | 3.58 | 4.45 | 3.10 | 4.20 | 6.90* |
| released head, the recording's durations | 2.80* | 3.36* | 3.52* | 4.45 | 3.12 | 4.13 | 6.90 |
| released head, T 0 | 2.96 | 3.63* | 3.81* | 4.52 | 3.21* | 4.16 | 7.12* |
| released head, T 0.3 | 2.96 | 3.55* | 3.73* | 4.51* | 3.16* | 4.15 | 7.05* |
| released head, T 0.7 | 2.81* | 3.30* | 3.43* | 4.36* | 3.05* | 4.16 | 6.78* |
| released head, T 1.0 | 2.64* | 2.98* | 3.04* | 4.21* | 2.88* | 4.16 | 6.37* |
| **released head, the released fine-tuned BigVGAN-v2, T 0.5** | **3.38*** | **4.13*** | **4.25*** | **4.63*** | **3.29*** | **4.32*** | **7.51*** |

- **The panel hears what UTMOSv2 alone barely hears.** Through a vocoder that has never seen the model's mels the
  model is 0.75 below its recordings on UTMOS22, 0.67 on SCOREQ and 0.55 on Audiobox PQ; all seven judges agree.
  Copy-synthesis is at the recordings' level, so the mel front end and the vocoder are not the limit: the generated
  mel is.
- **The fine-tuned heads are not better.** The energy-score head is lower on four judges, higher on one; with the MAE
  maps the differences are within their intervals except Audiobox (−0.05). Calibrating the spread of the head moved
  the distribution tests above, and nothing a judge hears.
- **The head's noise is heard as damage.** Every judge but SQUIM falls from T 0 to T 1, UTMOS22 by 0.65. T 0.3 and
  T 0 score above the operating point of 0.5 through this vocoder.
- **The recording's own durations do not help**: they score slightly lower than the predicted ones. The judges do
  not find the rhythm lacking.
- **The vocoder fine-tuned on the model's own mels closes the whole gap.** With it the released model is at the
  recordings' level on all seven judges (UTMOS22 4.13 against 4.20, SCOREQ 4.25 / 4.28, Audiobox PQ 7.51 / 7.49):
  +0.68, +0.64 and +0.56 over NVIDIA's generator on the same mels. Nothing tried on the acoustic model comes near.
- **With that vocoder nothing else moves the judges.** Against the released system (released head, fine-tuned
  vocoder, T 0.5): the recordings are higher on UTMOS22 (+0.07) and lower on UTMOSv2 (−0.11), the other five within
  their intervals; T 0 and T 0.3 are the same as T 0.5 (T 0: SCOREQ +0.03), T 0.7 is lower on two judges and T 1 on
  five; the energy-score head with MAE maps is higher on four of seven, none significantly (Audiobox PQ −0.01).
- **What it means for the Turkish model.** Its vocoder (Vocos v2) was fine-tuned on another model's mels and has
  never seen ARDrift's (#89). `configs/vocoder_bigvgan_tr_gta.yaml` is the English recipe for it; it has not been
  run (the Turkish corpus was not available here), and UTMOSv2, the only judge the Turkish tables use, is the least
  sensitive of the seven to this gap (0.36 where UTMOS22 sees 0.75).

### The Turkish release through each vocoder

The same check on the public Turkish model (`Vyvo/drifting-tts-tr`, the DiT v3.1): the first 100 Freya-TR-Eval
sentences in the three built-in voices, the same mel through each vocoder, against the BigVGAN-v2 fine-tuned on that
model's mels (300 sentences; no recordings of these texts exist, and the judges were trained on English, so only the
differences are read).

| vocoder | UTMOSv2 | UTMOS22 | SCOREQ | NISQA | DNSMOS OVRL | SQUIM | Audiobox PQ |
|---|---|---|---|---|---|---|---|
| BigVGAN-v2 fine-tuned on the model's mels (v3.1's) | 2.86 | 3.08 | 3.60 | 4.70 | 3.30 | 4.42 | 7.37 |
| Vocos v2, fine-tuned on the model's mels | 2.86 | 2.98* | 3.52* | 4.71 | 3.32* | 4.42 | 7.32* |
| Vocos-ft, the earlier fine-tune | 2.39* | 2.59* | 3.07* | 3.85* | 3.24* | 4.40* | 7.13* |
| **NVIDIA's BigVGAN-v2, not adapted** | **2.19*** | **1.47*** | **1.95*** | **3.50*** | **2.61*** | **4.26*** | **6.51*** |
| release v3.2 (Vocos v2, sampled pitch, punctuation pauses) | 2.91* | 2.92* | 3.52* | 4.69 | 3.33* | 4.41* | 7.41* |

- **An unadapted vocoder costs 1.6 points here**, twice the English gap: every judge, every voice (UTMOS22 −1.4 to
  −1.7, SCOREQ −1.5 to −1.8).
- **Among the adapted ones, BigVGAN-v2 is ahead of Vocos v2** on UTMOS22, SCOREQ and Audiobox PQ, most on the male
  voice (UTMOS22 3.10 against 2.90, SCOREQ 3.57 / 3.39); on the studio voice they tie (SCOREQ 4.20 / 4.21), and
  UTMOSv2, the judge of the Turkish tables so far, does not see the difference (2.86 / 2.86).
- **The voices differ more than the vocoders**: studio 3.35 / 4.20 (UTMOS22 / SCOREQ), male 3.10 / 3.57, female
  2.78 / 3.02, in the order of their hours of training data.
- **For the Turkish ARDrift** this is the case that was not measured: Vocos v2 adapted to this DiT's mels, decoding
  another model's. Between the first two rows and the fourth lies everything #89 can gain.

## Commands

```bash
# calibration of the head in the kernel's own spaces, mel texture, patch boundaries
python scripts/drift_calibration.py --model ardrift_en.pt --data data/en_val --num 100 --temperatures 1.0 0.5 0.3
# classifier two-sample test: free-running windows, then whole teacher-forced patches
python scripts/c2st_mel.py --model ardrift_en.pt --data data/en_val --mode free --temperature 0.5 \
    --windows 4 16 --views full detail envelope delta --active -0.5
python scripts/c2st_mel.py --model ardrift_en.pt --data data/en_val --mode gta --temperature 1.0 \
    --windows 4 --stride 4 --views full detail envelope delta --active -0.5
python scripts/c2st_mel.py --model ardrift_en.pt --data data/en_val --mode null      # the test's floor: 0.50
# spread-skill by temporal frequency, on patches of speech
python scripts/drift_calibration.py --model ardrift_en.pt --data data/en_val --active -0.5 --no-free
# the head alone: the states once (--cache), then one objective per run; the tests take --head
python scripts/head_finetune.py --model ardrift_en.pt --data data/en_ft --cache rows.pt --out runs/head_control
python scripts/head_finetune.py --model ardrift_en.pt --data data/en_ft --cache rows.pt --out runs/head_energy \
    --maps patch frame detail wdct --loss energy
drifting-tts train-mae --config configs/mae2d.yaml --workdir runs/mae16 data.root=data/en_ft \
    model.speaker_head=false "model.mask_patch=[10,4]" train.crop_frames=16 train.steps=10000
python scripts/head_finetune.py --model ardrift_en.pt --data data/en_ft --cache rows_c3.pt --context 3 \
    --mae runs/mae16/mae_ema.pt --maps patch frame detail wdct mae --loss energy --rows 512 --accum 2 --samples 16 \
    --max-locations 32 --out runs/head_energy_mae
python scripts/c2st_mel.py --model ardrift_en.pt --head runs/head_energy_mae/head.pt --data data/en_val \
    --mode gta --temperature 1.0 --windows 4 --stride 4 --active -0.5
# a blind A/B page from two folders of wavs with the same file names; then the saved answers
python scripts/listening_test.py --a wavs/released --b wavs/energy --name-a released --name-b energy --out ab.html
python scripts/listening_test.py --score listening_answers.json
# seven judges on folders of wavs with the same file names, each system against the baseline sentence by sentence
python scripts/judge_panel.py --systems recording=wav/recording released=wav/released new=wav/new \
    --baseline released --out outputs/panel                    # pip install -e ".[eval,panel]"
python scripts/judge_panel.py --systems ft=wav/bigvgan_ft vocos=wav/vocos_v2 --baseline ft \
    --groups studio male female --out outputs/panel_tr          # also per voice: file names start with it
# the Turkish vocoder on ARDrift's own mels (#89), then the panel on its snapshots
drifting-tts finetune-vocoder --config configs/vocoder_bigvgan_tr_gta.yaml --workdir runs/tr_bigvgan_gta
# audio-level prosody of recording / copy-synthesis / recording's durations / predicted durations
python scripts/prosody_gap.py --model ardrift_en.pt --vocoder bigvgan_en.pt --data data/en_val --num 60 \
    --temperatures 0.5 1.0 --seeds 2 --out outputs/prosody_gap
```

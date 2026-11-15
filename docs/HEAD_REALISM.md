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
# audio-level prosody of recording / copy-synthesis / recording's durations / predicted durations
python scripts/prosody_gap.py --model ardrift_en.pt --vocoder bigvgan_en.pt --data data/en_val --num 60 \
    --temperatures 0.5 1.0 --seeds 2 --out outputs/prosody_gap
```

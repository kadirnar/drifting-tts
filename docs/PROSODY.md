# Prosody: measurements, oracle prosody and inference-time fixes

Issue #38 (part of #37). Inference only: the released v3.1 model, the `vocos-ft` vocoder, T = 0.3, α = 2 unless a row
says otherwise. The question: **how much do the deterministic prosody predictors cost?**

- [Summary](#summary)
- [How it is measured](#how-it-is-measured)
- [The predictors against the training targets](#the-predictors-against-the-training-targets)
- [Oracle prosody A/B (studio voice)](#oracle-prosody-ab-studio-voice)
- [Inference-time fixes](#inference-time-fixes)
- [Pauses](#pauses)
- [Paragraph mode](#paragraph-mode)
- [Attention window on long texts](#attention-window-on-long-texts)
- [Seed diversity](#seed-diversity)
- [Second voice (male, 389)](#second-voice-male-389)
- [Reproduction checks](#reproduction-checks)
- [Commands](#commands)

## Summary

- **The token pitch predictor is the bottleneck of intonation.** On 100 held-out studio recordings, feeding the
  ground-truth token pitch into the frozen DiT lifts the DTW F0 correlation with the recording from 0.61 to 0.79 and
  cuts the F0 RMSE from 3.0 to 2.3 st; copy synthesis through the same vocoder, the ceiling, is at 0.83 / 2.0 st. The
  F0 std goes from 3.19 to 3.81 st (recordings 3.68). Ground-truth durations add nothing to intonation (r 0.62) but
  restore the pauses and the rhythm.
- **Both predictors regress to the mean.** The predicted token pitch varies 26% less than its training targets
  (r 0.71) and the predicted letter durations 41% less (r 0.72); for the male voice, even on its training texts, 37%
  and 49%. Every pause inside an utterance gets 0.10 ± 0.01 s.
- **The DiT is not the problem.** It renders the token pitch it is given as faithfully as copy synthesis (r 0.84, std
  ratio 1.05), and its within-token F0 movement is like the recordings'. Vocos adds its known periodicity jitter.
- **Inference-time knobs do not replace a better predictor.** A pitch gain of ×1.4 restores the F0 spread (3.78 st)
  but not the contour (r 0.63, RMSE 3.0 → 3.2 st); UTMOSv2 rises a little (2.67 → 2.72) at the same
  intelligibility. Duration gains, the guidance scale α ∈ {1, 1.5, 2} and the DiT attention window do not change
  prosody. Seeds change only the fine F0 detail (0.32 st per frame, the size of the micro-variation), never the tune
  or the timing.
- **Even UTMOSv2, nearly blind to prosody, moves with it.** Released pipeline 2.614, the same model in one pass 2.674,
  ground-truth prosody 2.756: above copy synthesis (2.738), at the same intelligibility (CER* 0.37–0.59%).
- **Pauses: the fixed 0.15 s join is wrong in both directions.** The studio voice pauses 0.13 s at an internal full
  stop and its generated sentences already carry 0.16 s of edge silence, so the released joins give 0.32 s gaps
  (2.3×); the male voice pauses 0.85 s. The opt-in punctuation policy (measured per voice) brings the studio gaps to
  0.17 s (CER 0.50% → 0.40%, UTMOSv2 2.614 → 2.628).
- **Next step:** a stochastic token pitch predictor (#39), durations second. Targets on this table: DTW F0 r towards
  0.79, pitch flatness towards 1, a seed spread well above 0.32 st, CER / UTMOSv2 unchanged.

## How it is measured

`drifting-tts prosody` (`drifting_tts/prosody_eval.py`) takes held-out utterances of one voice (default: `val`, the
studio voice 722, 100 utterances; 98 of them have two or more sentences, 12.1 s on average) and synthesises each
recording's own text with every *system*. Seed = dataset index, as in `evaluate`. The metrics live in
`drifting_tts/prosody.py`, so other work (the stochastic predictor of #39) can reuse them.

**Systems:**

| system | durations | token pitch | how |
|---|---|---|---|
| `recording` | – | – | the recording (reference) |
| `copy` | – | – | the recording's mel through `vocos-ft` (copy-synthesis ceiling) |
| `predicted` | predicted | predicted | the released pipeline: `Synthesizer`, sentence by sentence, 0.15 s silence between sentences |
| `onepass` | predicted | predicted | the whole text in one pass (no sentence split); the pipeline of the oracle rows |
| `oracle-dur` | ground truth (MAS) | predicted | one pass; the mel has exactly the recording's frame count |
| `oracle-pitch` | predicted | ground truth | one pass |
| `oracle-both` | ground truth | ground truth | one pass |

The DiT was trained with ground-truth MAS durations and ground-truth token pitch, so the oracle rows are in
distribution. Ground truth is computed as in training: MAS between the model's own prior `mu` and the recording's mel
(`models/text_encoder.py: align`), token pitch from `f0.bin` (dio) with the model's `lf0_stats` (`token_pitch`). The
per-voice `duration_scales` apply to predicted durations only.

**Audio-level** (per utterance, then averaged; WORLD harvest F0 every 10 ms, 60–500 Hz, semitones relative to the
utterance median). F0 std, 5–95% range, skew and kurtosis, movement (mean |ΔF0| between voiced neighbours, st per
10 ms) and micro-variation (mean |F0 − 5-frame moving average|) use exactly the definitions of the Prosody-40
diagnosis ([EXPERIMENTS.md §5](EXPERIMENTS.md#5-robotic-prosody-diagnosis-and-research)). Added: F0 CV (in Hz),
**local pitch reversals** per voiced second (sign changes of the slope of the 50 ms-smoothed contour, slopes below
0.05 st per 10 ms keep the previous sign), voiced %, internal pauses (runs ≥ 100 ms of 10 ms frames 35 dB below the
utterance's 95th-percentile level, between the first and last speech frame) and the speaking rate (syllables, one per
vowel, per second of speech).

**Against the recording of the same text.** MFCC (1–20, mean-normalised) dynamic time warping, then along the path:
the Pearson correlation of log-F0 (`DTW F0 r`) and its RMSE in semitones with each utterance relative to its own median
(`F0 RMSE`: contour shape, not register); the ratio of the speech durations (`dur ratio`, generated / recording).

**On the token alignment** (systems whose alignment is known: the recording with its MAS durations, the one-pass
rows with the durations they used). `in-token F0 std`: mean std of F0 inside a token; `in-token share`: the share of
the F0 variance that lies inside tokens. `render r` / `render flat`: the F0 of the audio averaged per token against
the token pitch it was conditioned on (correlation; std ratio audio / conditioning). They tell whether the acoustic
model and vocoder reproduce the token pitch they are given. For the recording they compare harvest on the audio with
the dio-based training targets, i.e. the agreement of two F0 trackers (the ceiling of this measure).

**Token level, no audio** (`token_report`): the durations and token pitch a system uses against the ground truth.
Durations per letter (its token plus the blank after it) in log-frames, pitch on the tokens with voiced frames, in
semitones: per-utterance Pearson r, the **flatness ratio** (std predicted / std ground truth), the MAE and the
coefficient of variation of the letter durations.

**Guard rails:** Whisper large-v3 CER / WER (full band, as `evaluate`; band-matched to 8 kHz for Freya texts),
UTMOSv2, WavLM-ECAPA similarity to the recording. `CER*` / `WER*` leave out one studio text (index 254) that repeats
two sentences verbatim: Whisper drops the repetition in the recording and in every one-pass rendition (116 character
errors each, about 60% of those rows' errors), which hides the differences between systems.

**Seed diversity:** K = 8 seeds for 10 texts. The F0 tracks are aligned to the first seed's (frame by frame for equal
lengths, else by MFCC-DTW); the spread is the std across seeds per frame voiced in all, in semitones (median over
frames, so that rare octave errors do not dominate; the mean is also given). Also the CV of the per-rendition F0 std
and of the length, and the std of the token log-durations.

## The predictors against the training targets

Studio voice, the 100 `val` utterances (`token_report` of the `onepass` row). Durations per letter (token + following
blank) in log-frames; pitch on the tokens with voiced frames, in semitones.

| | r | flatness (std pred / GT) | MAE | CV of letter durations, pred / GT | bias |
|---|---|---|---|---|---|
| duration predictor | 0.72 | **0.59** | 0.26 (log) | 0.30 / 0.48 | length ×1.02 |
| token pitch predictor | 0.71 | **0.74** | 1.95 st | – | +0.21 st |

- **Both predictors regress to the mean.** The predicted letter durations vary 41% less than the MAS durations of the
  same recordings, the predicted token pitch 26% less.
- **Pauses inside an utterance are almost constant.** The predicted duration of a mark plus the space after it is
  0.10 ± 0.01 s at commas and 0.10 ± 0.02 s at full stops; the recordings have 0.12 ± 0.05 s and 0.11 ± 0.06 s.
  So the one-pass model rarely makes a pause ≥ 100 ms (0.45 per utterance, against 1.39 in the recordings).

## Oracle prosody A/B (studio voice)

Studio voice, the 100 `val` utterances (`runs/pe_studio_val`). Audio level:

| system | F0 std | range | skew | kurt | F0 CV | move | micro | reversals/s | voiced % | pauses/utt | pause s | syl/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recording | 3.68 | 11.9 | 0.38 | 1.08 | 0.228 | 0.63 | 0.42 | 10.9 | 93.8 | 1.39 | 0.139 | 6.22 |
| copy | 3.69 | 11.8 | 0.41 | 0.83 | 0.229 | 0.69 | 0.49 | 11.7 | 93.4 | 1.46 | 0.140 | 6.22 |
| predicted | 3.36 | 10.8 | 0.27 | 1.24 | 0.206 | 0.67 | 0.47 | 11.5 | 92.3 | 1.58 | 0.316 | 5.90 |
| onepass | 3.19 | 10.2 | 0.35 | 1.36 | 0.196 | 0.67 | 0.47 | 11.3 | 94.8 | 0.45 | 0.115 | 6.10 |
| oracle-dur | 3.15 | 10.1 | 0.37 | 1.58 | 0.194 | 0.67 | 0.48 | 11.8 | 94.0 | 1.27 | 0.137 | 6.22 |
| oracle-pitch | 3.81 | 12.2 | 0.42 | 0.81 | 0.237 | 0.71 | 0.50 | 11.2 | 94.0 | 0.57 | 0.115 | 6.11 |
| oracle-both | 3.81 | 12.2 | 0.43 | 0.87 | 0.237 | 0.71 | 0.50 | 11.6 | 93.2 | 1.39 | 0.141 | 6.22 |

Against the recording, on the token alignment, and the guard rails (CER / WER: Whisper large-v3, full band, corpus
level; UTMOSv2; SIM: WavLM-ECAPA cosine with the recording):

| system | DTW F0 r | F0 RMSE | dur ratio | in-token F0 std | render r | render flat | CER | WER | CER* | WER* | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recording | – | – | – | 0.70 | 0.887 | 1.000 | 0.88% | 2.06% | 0.30% | 1.44% | 3.093 | – |
| copy | 0.834 | 2.03 | 1.000 | 0.77 | 0.843 | 1.017 | 0.90% | 2.21% | 0.31% | 1.59% | 2.738 | 0.969 |
| predicted | 0.606 | 3.09 | 1.055 | – | – | – | 0.50% | 2.21% | 0.49% | 2.09% | 2.614 | 0.931 |
| onepass | 0.610 | 3.01 | 1.019 | 0.65 | 0.874 | 1.184 | 0.96% | 2.32% | 0.37% | 1.71% | 2.674 | 0.928 |
| oracle-dur | 0.615 | 2.97 | 1.000 | 0.73 | 0.863 | 1.162 | 1.14% | 2.84% | 0.56% | 2.24% | 2.644 | 0.935 |
| oracle-pitch | 0.794 | 2.33 | 1.019 | 0.68 | 0.840 | 1.056 | 0.95% | 2.28% | 0.36% | 1.59% | 2.686 | 0.931 |
| oracle-both | 0.803 | 2.26 | 1.000 | 0.79 | 0.837 | 1.050 | 1.17% | 2.62% | 0.59% | 2.01% | 2.756 | 0.938 |

- **The token pitch predictor is the bottleneck of intonation.** With the predicted token pitch, the F0 contour
  correlates r = 0.61 with the recording (DTW) and is 3.0 st off (RMSE). Ground-truth token pitch through the same
  frozen DiT gives r = 0.79 and 2.3 st; copy synthesis, the ceiling of this vocoder, gives 0.83 and 2.0 st. The token
  pitch alone closes about 80% of the gap to the ceiling (r: (0.79 − 0.61) / (0.83 − 0.61)), both oracles 86%.
- **The global F0 spread follows the same split.** F0 std 3.19 st with predicted pitch (−13% against the recording's
  3.68), 3.81 with ground-truth pitch; durations do not change it (oracle durations: 3.15).
- **The DiT renders the token pitch it is given; it does not compress it.** The F0 of the generated audio, averaged
  per token, follows the ground-truth token pitch with r = 0.84 and a std ratio of 1.05. That is the agreement of copy
  synthesis (0.84, 1.02); two F0 trackers on the recording itself agree to 0.89. With the predicted (flat) token pitch
  the DiT even widens it (std ratio 1.18). Within-token F0 movement is also like the recording's (std 0.65–0.79 st
  against 0.70; the vocoder adds a little: copy 0.77).
- **The duration predictor costs rhythm and pauses, not intonation.** Oracle durations reproduce the recording's
  pauses (1.27 per utterance of 0.14 s, against 1.39 of 0.14 s) and speaking rate (6.22 syllables/s), where the
  predicted durations make 0.45 pauses per utterance.
- **Vocos adds periodicity jitter, as found before.** Micro-variation 0.42 → 0.49 and local pitch reversals 10.9 →
  11.7 per voiced second from the recording to its copy synthesis; every generated row sits at the copy's level.

## Inference-time fixes

All one-pass rows of the same run; `onepass` is the reference (predicted durations and pitch).

**Pitch-deviation gain** (`pitch-gain-g`: the predicted token pitch scaled around its utterance mean by g on the
tokens that are voiced in ≥ 80% of the training targets: all but ç p s t, sentence-final marks and blanks between
two non-voiced tokens; precision 0.95 / recall 0.94 against the targets' voicing):

| system | F0 std | range | kurt | reversals/s | pitch flat | DTW F0 r | F0 RMSE | CER | WER | CER* | WER* | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| onepass | 3.19 | 10.2 | 1.36 | 11.3 | 0.736 | 0.610 | 3.01 | 0.96% | 2.32% | 0.37% | 1.71% | 2.674 | 0.928 |
| pitch-gain-1.2 | 3.52 | 11.3 | 0.92 | 11.2 | 0.874 | 0.619 | 3.09 | 1.06% | 2.43% | 0.48% | 1.82% | 2.710 | 0.929 |
| pitch-gain-1.4 | 3.78 | 12.2 | 0.47 | 11.0 | 1.013 | 0.629 | 3.16 | 1.04% | 2.54% | 0.45% | 1.86% | 2.716 | 0.928 |
| pitch-gain-1.6 | 4.12 | 13.3 | 0.20 | 10.9 | 1.152 | 0.621 | 3.36 | 1.02% | 2.51% | 0.43% | 1.82% | 2.742 | 0.925 |
| oracle-pitch | 3.81 | 12.2 | 0.81 | 11.2 | 1.000 | 0.794 | 2.33 | 0.95% | 2.28% | 0.36% | 1.59% | 2.686 | 0.931 |

- **A gain restores the global statistics, not the contour.** g = 1.4 matches the recording's F0 std and range (3.78
  / 12.2 against 3.68 / 11.9), but the contour correlation barely moves (0.61 → 0.63) and the RMSE grows (3.01 → 3.16
  st): the predictable part of the melody gets louder, the missing (unpredictable) part is still missing.
- **The guard rails hold, and UTMOSv2 likes it.** CER* stays within 0.37–0.48% and UTMOSv2 climbs with the gain
  (2.674 → 2.710 / 2.716 / 2.742), even past the recordings' spread at g = 1.6, so UTMOSv2 cannot pick the gain. On
  Freya-100 (single sentences, band-matched ASR) g = 1.2 / 1.4 give WER 0.88% / 0.99% (1.10% without) and UTMOSv2
  2.645 / 2.676 (2.627). g = 1.4 is the best probe here (closest to the recordings' spread, highest contour
  correlation); whether listeners prefer it is open.

**Durations** (`dur-gain-g`: letter log-durations scaled around their mean, total length kept; `dur-mix-0.5`: the
geometric mean of predicted and ground-truth durations):

| system | dur r | dur flat | letter CV | len ratio | pauses/utt | syl/s | DTW F0 r | CER | WER | CER* | WER* | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| onepass | 0.719 | 0.592 | 0.296 | 1.017 | 0.45 | 6.10 | 0.610 | 0.96% | 2.32% | 0.37% | 1.71% | 2.674 | 0.928 |
| dur-gain-1.2 | 0.723 | 0.692 | 0.344 | 1.013 | 0.45 | 6.13 | 0.614 | 0.93% | 2.39% | 0.34% | 1.71% | 2.661 | 0.929 |
| dur-gain-1.4 | 0.722 | 0.771 | 0.392 | 1.008 | 0.48 | 6.16 | 0.608 | 0.89% | 2.21% | 0.30% | 1.52% | 2.671 | 0.930 |
| dur-mix-0.5 | 0.909 | 0.637 | 0.319 | 1.077 | 0.74 | 5.76 | 0.616 | 0.93% | 2.21% | 0.35% | 1.59% | 2.602 | 0.927 |
| oracle-dur | 1.000 | 1.000 | 0.482 | 1.000 | 1.27 | 6.22 | 0.615 | 1.14% | 2.84% | 0.56% | 2.24% | 2.644 | 0.935 |

- **A duration gain makes the letters vary more (CV 0.30 → 0.39 at g = 1.4) without making them more right** (r 0.72
  either way). The audio metrics and the guard rails do not move. Half the oracle (r 0.91) does not change
  intonation either; it slows the speech by 6% (the geometric mean of a ceiled prediction and the MAS durations).
- **Oracle durations with predicted pitch are slightly worse than either alone** (WER* 2.24%, UTMOSv2 2.644 against
  1.71% / 2.674): the predicted token pitch belongs to the predicted rhythm.

**Guidance** (`predicted@cfg<α>`, T 0.3, the released split pipeline):

Studio `val` (sentence by sentence, 0.15 s pauses):

| system | F0 std | range | micro | DTW F0 r | CER | WER | CER* | WER* | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| predicted@cfg1 | 3.40 | 10.8 | 0.47 | 0.601 | 0.44% | 2.09% | 0.43% | 1.97% | 2.611 | 0.931 |
| predicted@cfg1.5 | 3.36 | 10.8 | 0.47 | 0.604 | 0.44% | 2.09% | 0.43% | 1.97% | 2.612 | 0.931 |
| predicted | 3.36 | 10.8 | 0.47 | 0.606 | 0.50% | 2.21% | 0.49% | 2.09% | 2.614 | 0.931 |

Freya-100 (ASR band-matched to 8 kHz):

| system | F0 std | range | micro | CER | WER | UTMOSv2 |
|---|---:|---:|---:|---:|---:|---:|
| predicted@cfg1 | 3.56 | 11.6 | 0.46 | 0.22% | 1.10% | 2.612 |
| predicted@cfg1.5 | 3.49 | 11.3 | 0.45 | 0.22% | 1.10% | 2.615 |
| predicted | 3.51 | 11.3 | 0.45 | 0.22% | 1.10% | 2.627 |

- **α does not change prosody** (F0 std 3.36–3.40, DTW r 0.60–0.61) and barely the guard rails: UTMOSv2 2.611 /
  2.612 / 2.614 on `val` and 2.612 / 2.615 / 2.627 on Freya-100, identical Freya WER. The paper's FID-optimal
  α ≈ 1 and Kyutai's 1.5 are no better than the shipped 2 on these judges.

## Pauses

**Pauses in the training data** (`scripts/pause_stats.py`, 1000 training utterances with an internal mark per group;
119 for the female voice). Each internal mark is located with MAS (v3.1 encoder, as in training). The pause is the
energy-based silence that overlaps the mark and the space after it (± 50 ms; 0 if there is none).

| voice | mark | n | silence mean ± std | median | p90 | ≥ 100 ms | MAS span of mark + space |
|---|---|---|---|---|---|---|---|
| studio (722) | `,` | 1624 | 0.05 ± 0.05 s | 0.05 | 0.12 | 21% | 0.11 s |
| studio (722) | `.` | 1894 | 0.13 ± 0.09 s | 0.11 | 0.26 | 58% | 0.11 s |
| studio (722) | `?` | 83 | 0.10 ± 0.06 s | 0.11 | 0.17 | 59% | 0.15 s |
| male (389) | `,` | 1306 | 0.47 ± 0.21 s | 0.46 | 0.71 | 98% | 0.54 s |
| male (389) | `.` | 178 | 0.85 ± 0.35 s | 0.80 | 1.21 | 99% | 0.87 s |
| female (323) | `,` | 208 | 0.13 ± 0.17 s | 0.05 | 0.40 | 40% | 0.27 s |
| female (323) | `.` | 72 | 0.22 ± 0.33 s | 0.14 | 0.53 | 60% | 0.35 s |
| corpus without 722 | `,` | 790 | 0.16 ± 0.19 s | 0.09 | 0.45 | 49% | 0.23 s |
| corpus without 722 | `.` | 1043 | 0.23 ± 0.33 s | 0.12 | 0.60 | 55% | 0.30 s |
| corpus without 722 | `?` | 74 | 0.24 ± 0.33 s | 0.14 | 0.48 | 61% | 0.32 s |

`!` is too rare (≤ 4 per group) to measure.

- **Pause length is a property of the voice, not of the mark.** The studio voice reads fluently: a full stop inside
  an utterance gets 0.13 s of silence, a comma 0.05 s. The male voice pauses 0.85 s at a full stop. One fixed 0.15 s
  cannot fit both.
- **A generated sentence already carries edge silence.** Leading + trailing silence of single generated sentences
  (T 0.3, α 2, `vocos-ft`, 200 sentences per voice): studio 0.08 + 0.08 = **0.16 s**, male 0.09 + 0.09 = 0.18 s,
  female 0.02 + 0.05 = 0.07 s. The released joins add 0.15 s on top, so the studio voice's sentence gap is about
  0.31 s: **2.3× the 0.13 s of its recordings.**

**The policy** (`PausePolicy`, opt-in: `Synthesizer(...)(text, pause=PausePolicy.for_voice(722))`, CLI
`synthesize --pause-policy punct [--pause-jitter 1]`). The gap after a sentence targets the voice's measured pause at
its final mark (the mean, or with `jitter` j a draw of mean + j·std·N(0, 1)), and the inserted silence is
`max(0, gap − edge silence)`. A sentence cut without a mark (a long one split at a space) uses the comma entry; marks
not measured for a voice use the full stop's. For the studio voice this inserts **nothing**: its edge silence alone
already exceeds its typical pause. The default (`pause=0.15`) is unchanged.

**Results** (studio `val`, 100 utterances, 98 with two or more sentences; pauses ≥ 100 ms between the first and last
speech frame):

| system | pauses/utt | pause s | syl/s | dur ratio | F0 std | DTW F0 r | CER | WER | CER* | WER* | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recording | 1.39 | 0.139 | 6.22 | – | 3.68 | – | 0.88% | 2.06% | 0.30% | 1.44% | 3.093 | – |
| predicted | 1.58 | 0.316 | 5.90 | 1.055 | 3.36 | 0.606 | 0.50% | 2.21% | 0.49% | 2.09% | 2.614 | 0.931 |
| pause-punct | 1.53 | 0.170 | 6.01 | 1.035 | 3.36 | 0.604 | 0.40% | 1.98% | 0.38% | 1.86% | 2.628 | 0.931 |
| pause-punct-j1 | 1.53 | 0.185 | 6.00 | 1.037 | 3.36 | 0.601 | 0.40% | 1.98% | 0.38% | 1.86% | 2.616 | 0.931 |
| onepass | 0.45 | 0.115 | 6.10 | 1.019 | 3.19 | 0.610 | 0.96% | 2.32% | 0.37% | 1.71% | 2.674 | 0.928 |
| oracle-dur | 1.27 | 0.137 | 6.22 | 1.000 | 3.15 | 0.615 | 1.14% | 2.84% | 0.56% | 2.24% | 2.644 | 0.935 |

- **The released joins make the studio voice's sentence pauses 2.3× too long** (0.32 s against 0.14 s) and stretch
  the utterances by 5.5%. The policy (nothing inserted for this voice) brings the pauses to 0.17 s and the length
  ratio to 1.035 (0.185 s with jitter 1); CER 0.50% → 0.40%, UTMOSv2 2.614 → 2.628.
- **One pass (no split) has pauses of the right length but too few of them** (0.45 per utterance against 1.39):
  the duration predictor gives every mark 0.10 ± 0.01 s, mostly under the 100 ms that counts as a pause. Oracle
  durations restore both (1.27 pauses of 0.14 s).

## Paragraph mode

For 98 of the 100 studio texts (two or more sentences), `predicted` (split + 0.15 s) against `onepass` (the whole
text in one pass, in the training distribution: the training utterances are up to 16 s and 98% of the studio
voice's have several sentences):

| system | F0 std | range | DTW F0 r | F0 RMSE | pauses/utt | pause s | syl/s | dur ratio | CER | WER | CER* | WER* | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recording | 3.67 | 11.9 | – | – | 1.40 | 0.140 | 6.22 | – | 0.90% | 2.09% | 0.30% | 1.47% | 3.096 | – |
| copy | 3.69 | 11.8 | 0.835 | 2.02 | 1.47 | 0.140 | 6.22 | 1.000 | 0.91% | 2.25% | 0.32% | 1.62% | 2.742 | 0.969 |
| predicted | 3.36 | 10.8 | 0.609 | 3.07 | 1.61 | 0.316 | 5.89 | 1.056 | 0.51% | 2.25% | 0.50% | 2.13% | 2.616 | 0.931 |
| pause-punct | 3.37 | 10.8 | 0.608 | 3.08 | 1.56 | 0.170 | 6.01 | 1.036 | 0.41% | 2.02% | 0.39% | 1.89% | 2.630 | 0.931 |
| onepass | 3.19 | 10.2 | 0.613 | 2.99 | 0.46 | 0.115 | 6.10 | 1.020 | 0.97% | 2.36% | 0.38% | 1.74% | 2.677 | 0.928 |
| oracle-both | 3.79 | 12.1 | 0.805 | 2.24 | 1.41 | 0.141 | 6.22 | 0.999 | 1.19% | 2.67% | 0.60% | 2.05% | 2.759 | 0.938 |

- **One pass gives the right sentence gaps but a narrower melody** (F0 std 3.19 against 3.36 when split, 3.67 in the
  recordings). Sentence by sentence, every sentence restarts its own declination, which widens the F0 distribution
  but not the contour match (DTW r 0.61 either way).
- **UTMOSv2 prefers one pass** (2.677 against 2.616 split, 2.630 with the pause policy) at the same intelligibility
  (CER* 0.38% against 0.50%). Paragraph mode is in the training distribution and costs nothing here; its weak spot
  is the near-constant pause at every mark (0.46 pauses ≥ 100 ms per utterance against 1.40).

## Attention window on long texts

The DiT is trained on 256-frame crops (128 tokens of 2 frames) but samples whole utterances (up to ~750 tokens).
`@win<N>` restricts every frame token's attention to ± N tokens (registers stay global). The 30 longest of the 100
studio texts (`runs/pe_studio_val_long30`):

| system | F0 std | DTW F0 r | F0 RMSE | render r | CER | WER | CER* | WER* | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recording | 3.55 | – | – | 0.890 | 2.25% | 4.05% | 0.48% | 2.23% | 3.099 | – |
| predicted | 3.30 | 0.610 | 2.99 | – | 0.71% | 3.49% | 0.67% | 3.17% | 2.615 | 0.932 |
| predicted@win64 | 3.27 | 0.597 | 3.01 | – | 0.69% | 3.26% | 0.66% | 2.94% | 2.636 | 0.933 |
| onepass | 3.15 | 0.596 | 2.97 | 0.856 | 2.36% | 4.39% | 0.59% | 2.59% | 2.690 | 0.930 |
| onepass@win64 | 3.13 | 0.599 | 2.94 | 0.864 | 2.36% | 5.17% | 0.56% | 3.17% | 2.616 | 0.932 |
| onepass@win32 | 3.07 | 0.613 | 2.88 | 0.873 | 0.60% | 3.49% | 0.56% | 3.17% | 2.649 | 0.931 |
| oracle-both | 3.73 | 0.786 | 2.28 | 0.813 | 2.51% | 4.39% | 0.75% | 2.59% | 2.746 | 0.939 |
| oracle-both@win64 | 3.66 | 0.803 | 2.17 | 0.826 | 2.42% | 4.27% | 0.66% | 2.47% | 2.702 | 0.941 |

- **The window does not help.** With ground-truth prosody, ± 64 tokens (the training receptive field) moves the
  contour slightly closer to the recording (r 0.786 → 0.803, RMSE 2.28 → 2.17 st); with predicted prosody all
  windows are within noise (r 0.60–0.61). UTMOSv2 drops with the window (one pass 2.690 → 2.616 at ± 64, 2.649 at
  ± 32; oracle 2.746 → 2.702) at the same CER*. Full attention stays.

## Seed diversity

| system | F0 spread st (median) | mean | F0 std CV | length CV | token log-dur std |
|---|---:|---:|---:|---:|---:|
| onepass | 0.317 | 0.589 | 0.052 | 0.0000 | 0.0000 |
| predicted | 0.323 | 0.596 | 0.048 | 0.0000 | – |

- **Seeds change only the fine F0 detail, never the tune or the timing.** Durations are identical across seeds
  (length CV 0, token log-duration std 0); the contours of 8 seeds differ by a median of 0.32 st per frame (after
  alignment), the size of the micro-variation. This is the baseline a stochastic prosody predictor (#39) must raise.

## Second voice (male, 389)

The male voice has only two held-out utterances, so these are 100 of its **training** utterances
(`--split train`, `runs/pe_male_train`): in-sample for every part of the model, an optimistic setting for the
predictors.

| system | F0 std | range | DTW F0 r | F0 RMSE | pauses/utt | pause s | syl/s | dur ratio | CER | WER | UTMOSv2 | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recording | 4.37 | 14.2 | – | – | 4.56 | 0.495 | 3.93 | – | 0.75% | 3.46% | 3.248 | – |
| copy | 4.08 | 13.0 | 0.818 | 2.55 | 4.64 | 0.491 | 3.93 | 0.999 | – | – | – | – |
| predicted | 3.25 | 10.2 | 0.521 | 3.83 | 3.28 | 0.302 | 4.30 | 0.916 | 0.75% | 4.28% | 2.306 | 0.756 |
| onepass | 3.23 | 10.1 | 0.526 | 3.81 | 3.21 | 0.313 | 4.29 | 0.917 | 0.74% | 4.22% | 2.330 | 0.755 |
| oracle-dur | 3.22 | 10.1 | 0.530 | 3.79 | 4.32 | 0.451 | 4.09 | 0.961 | – | – | – | – |
| oracle-pitch | 4.19 | 13.2 | 0.751 | 2.99 | 3.22 | 0.320 | 4.30 | 0.916 | – | – | – | – |
| oracle-both | 4.14 | 13.2 | 0.783 | 2.76 | 4.36 | 0.455 | 4.08 | 0.963 | 4.13% | 9.68% | 2.515 | 0.780 |
| pitch-gain-1.4 | 3.94 | 12.4 | 0.534 | 4.02 | 3.21 | 0.319 | 4.30 | 0.916 | 0.76% | 4.42% | 2.495 | 0.757 |
| pause-punct | 3.25 | 10.2 | 0.521 | 3.83 | 3.28 | 0.342 | 4.24 | 0.931 | 0.70% | 4.01% | 2.302 | 0.756 |
| pause-punct-j1 | 3.24 | 10.2 | 0.521 | 3.83 | 3.26 | 0.341 | 4.24 | 0.930 | – | – | – | – |

Token level, `onepass`:

| system | dur r | dur flat | letter CV | pitch r | pitch flat | pitch MAE st |
|---|---:|---:|---:|---:|---:|---:|
| onepass | 0.578 | 0.512 | 0.323 | 0.580 | 0.633 | 2.83 |

- **The same picture, stronger.** Even on its training texts, the token pitch is 37% flatter than the ground truth
  (r 0.58) and the letter durations 49% flatter (r 0.58). F0 std 3.23 against 4.37 in the recordings; ground-truth
  pitch restores 4.19 and lifts the contour correlation from 0.53 to 0.75 (0.78 with both oracles; copy 0.82).
- **This voice pauses long** (4.6 pauses per utterance of 0.50 s); the predicted durations give 3.2 of 0.31 s, the
  oracle durations 4.3 of 0.45 s. Its sentence joins get 0.67 s inserted under the policy; few of these texts have
  several sentences, so the averages move little.
- **The oracle is only as good as the alignment.** 19 of the 100 oracle renditions lose or garble words (CER 4.1%,
  against 0.74% with predicted durations and 2 such utterances): on this voice's podcast-style data the MAS durations
  are sometimes wrong (in one, a whole phrase of the transcript is missing from the audio; in another Whisper loops
  on a long silence). Its oracle intelligibility is not a fair number; the studio voice's oracle rows are within
  0.25% CER* of the predicted ones.
- **UTMOSv2 rewards the wider melody on this voice too**: pitch gain ×1.4 gives 2.495 against 2.330 (oracle 2.515).
  Speaker similarity is low for every generated row (0.76; 0.93 for the studio voice) and higher with oracle
  prosody (0.78).

## Reproduction checks

- **Freya-100** (`--texts freyavoice/freya-tr-eval --num 100`, band-matched to 8 kHz): `predicted` gives WER 1.10%,
  CER 0.22%, UTMOSv2 2.627, the known v3.1 + `vocos-ft` row of [EXPERIMENTS.md](EXPERIMENTS.md#summary-what-worked-what-did-not).
  `onepass` gives identical numbers (single sentences: the override path reproduces the default path bit for bit,
  also checked in `tests/test_prosody.py`).
- **Prosody-40**: on the first 40 Freya texts, `predicted` has F0 std 3.46, range 11.2, movement 0.66 and
  micro-variation 0.45: the v3.1 + Vocos row of [EXPERIMENTS.md §5](EXPERIMENTS.md#5-robotic-prosody-diagnosis-and-research)
  (3.46 / 11.2 / 0.66 / 0.45).

| system | F0 std | range | kurt | micro | reversals/s | render r | render flat | CER | WER | UTMOSv2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| predicted | 3.51 | 11.3 | 0.76 | 0.45 | 11.4 | – | – | 0.22% | 1.10% | 2.627 |
| onepass | 3.51 | 11.3 | 0.76 | 0.45 | 11.4 | 0.913 | 1.135 | 0.22% | 1.10% | 2.627 |
| pitch-gain-1.2 | 3.85 | 12.4 | 0.62 | 0.48 | 11.4 | 0.902 | 1.053 | 0.18% | 0.88% | 2.645 |
| pitch-gain-1.4 | 4.16 | 13.5 | 0.30 | 0.50 | 11.1 | 0.905 | 0.989 | 0.19% | 0.99% | 2.676 |

## Commands

```bash
# the studio table (100 val utterances; recording, copy, predicted, one-pass and oracle rows, judges, diversity)
drifting-tts prosody --model runs/release/drifting_tts_v3.1.pt --vocoder vocos-ft --out runs/pe_studio_val \
  --systems recording copy predicted onepass oracle-dur oracle-pitch oracle-both pitch-gain-1.4 pause-punct \
  --diversity 8 --diversity-num 10 --diversity-systems onepass predicted
# a voice without held-out utterances (in-sample), the longest texts with an attention window, Freya texts
drifting-tts prosody --split train --speaker 389 --out runs/pe_male_train
drifting-tts prosody --longest 30 --systems recording onepass onepass@win64 --out runs/pe_studio_val_long30
drifting-tts prosody --texts freyavoice/freya-tr-eval --num 100 --systems predicted pitch-gain-1.2 predicted@cfg1.5
# pauses at internal punctuation in the training data and the generated sentences' edge silence
python scripts/pause_stats.py --model runs/release/drifting_tts_v3.1.pt --out runs/pe_pauses.json --generated 200
```

Rows are cached per system (`utterances_<system>.jsonl`, wavs under `wav/<system>/`): a second call adds new systems
only, and a call with judges back-fills the scores of rows computed with `--asr none --sv none --mos none`.
`--workers` (≤ 3 on a shared machine) run the harvest F0 on the CPU while the GPU synthesises and judges. In
Python:

```python
from drifting_tts.prosody import prosody_features, paired_metrics, token_targets, token_report, seed_diversity
feats = prosody_features(wav, text=norm_text)          # F0 std / range / skew / kurt / CV / reversals, pauses, rate
pair = paired_metrics(recording, wav)                  # DTW log-F0 r and RMSE, duration ratio
t = token_targets(model, ids, spk, mel, f0)            # predicted and MAS / token-pitch ground truth
rep = token_report(t, frame_rate=93.75, lf0_std=float(model.lf0_stats[1]), scale=duration_scale)
```

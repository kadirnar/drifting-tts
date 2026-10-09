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

SUMMARY

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
UTMOSv2, WavLM-ECAPA similarity to the recording.

**Seed diversity:** K = 8 seeds for 10 texts; the spread of the time-normalised F0 contours (mean over time of the std
across seeds, semitones), the CV of the per-rendition F0 std and of the length, and the std of token log-durations.

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

ORACLE_AUDIO

Against the recording, on the token alignment, and the guard rails (CER / WER: Whisper large-v3, full band, corpus
level; UTMOSv2; SIM: WavLM-ECAPA cosine with the recording):

ORACLE_PAIRED

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

GAIN_TABLE

- **A gain restores the global statistics, not the contour.** g = 1.4 matches the recording's F0 std and range (3.78
  / 12.2 against 3.68 / 11.9), but the contour correlation barely moves (0.61 → 0.63) and the RMSE grows (3.01 → 3.16
  st): the predictable part of the melody gets louder, the missing (unpredictable) part is still missing.

**Durations** (`dur-gain-g`: letter log-durations scaled around their mean, total length kept; `dur-mix-0.5`: the
geometric mean of predicted and ground-truth durations):

DUR_TABLE

- **A duration gain makes the letters vary more (CV 0.30 → 0.39 at g = 1.4) without making them more right** (r 0.72
  either way). The audio metrics do not move. Half the oracle (r 0.91) does not change intonation either.

**Guidance** (`predicted@cfg<α>`, T 0.3, the released split pipeline):

CFG_TABLE

- **α does not change prosody** (F0 std 3.36–3.40); it trades intelligibility against naturalness only through the
  spectral detail.

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

PAUSE_TABLE

- **The released joins make the studio voice's sentence pauses 2.3× too long** (0.32 s against 0.14 s) and stretch
  the utterances by 5.5%. The policy (nothing inserted for this voice) brings the pauses to 0.17 s, the length
  ratio to 1.035; with jitter 0.19 s.
- **One pass (no split) has pauses of the right length but too few of them** (0.45 per utterance against 1.39):
  the duration predictor gives every mark 0.10 ± 0.01 s, mostly under the 100 ms that counts as a pause. Oracle
  durations restore both (1.27 pauses of 0.14 s).

## Paragraph mode

For 98 of the 100 studio texts (two or more sentences), `predicted` (split + 0.15 s) against `onepass` (the whole
text in one pass, in the training distribution: the training utterances are up to 16 s and 98% of the studio
voice's have several sentences):

PARAGRAPH_TABLE

- **One pass gives the right sentence gaps but a narrower melody** (F0 std 3.19 against 3.37 when split, 3.67 in the
  recordings). Sentence by sentence, every sentence restarts its own declination, which widens the F0 distribution
  but not the contour match (DTW r 0.61 either way).

## Attention window on long texts

The DiT is trained on 256-frame crops (128 tokens of 2 frames) but samples whole utterances (up to ~750 tokens).
`@win<N>` restricts every frame token's attention to ± N tokens (registers stay global). The 30 longest of the 100
studio texts (`runs/pe_studio_val_long30`):

LONG30_TABLE

- **The window changes little.** With ground-truth prosody, ± 64 tokens (the training receptive field) moves the
  contour slightly closer to the recording (r 0.786 → 0.803, RMSE 2.28 → 2.17 st); with predicted prosody all
  windows are within noise (r 0.60–0.61).

## Seed diversity

DIVERSITY_TABLE

- **Seeds change only the fine F0 detail, never the tune or the timing.** Durations are identical across seeds
  (length CV 0, token log-duration std 0); the contours of 8 seeds differ by a median of SPREAD st per frame (after
  alignment), the size of the micro-variation. This is the baseline a stochastic prosody predictor (#39) must raise.

## Second voice (male, 389)

The male voice has only two held-out utterances, so these are 100 of its **training** utterances
(`--split train`, `runs/pe_male_train`): in-sample for every part of the model, an optimistic setting for the
predictors.

MALE_TABLE

Token level, `onepass`:

MALE_TOKENS

- **The same picture, stronger.** Even on its training texts, the token pitch is 37% flatter than the ground truth
  (r 0.58) and the letter durations 49% flatter (r 0.58). F0 std 3.23 against 4.37 in the recordings; ground-truth
  pitch restores 4.19 and lifts the contour correlation from 0.53 to 0.75 (0.78 with both oracles; copy 0.82).
- **This voice pauses long** (4.6 pauses per utterance of 0.50 s); the predicted durations give 3.2 of 0.31 s, the
  oracle durations 4.3 of 0.45 s. Its sentence joins get 0.67 s inserted under the policy; few of these texts have
  several sentences, so the averages move little.

## Reproduction checks

- **Freya-100** (`--texts freyavoice/freya-tr-eval --num 100`, band-matched to 8 kHz): `predicted` gives WER 1.10%,
  CER 0.22%, UTMOSv2 2.627, the known v3.1 + `vocos-ft` row of [EXPERIMENTS.md](EXPERIMENTS.md#summary-what-worked-what-did-not).
  `onepass` gives identical numbers (single sentences: the override path reproduces the default path bit for bit,
  also checked in `tests/test_prosody.py`).
- **Prosody-40**: on the first 40 Freya texts, `predicted` has F0 std 3.46, range 11.2, movement 0.66 and
  micro-variation 0.45: the v3.1 + Vocos row of [EXPERIMENTS.md §5](EXPERIMENTS.md#5-robotic-prosody-diagnosis-and-research)
  (3.46 / 11.2 / 0.66 / 0.45).

FREYA_TABLE

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

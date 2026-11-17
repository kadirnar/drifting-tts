# Changing ARDrift's objective: pilots on the English model (#97, #98, #101, #99)

Four issues propose changes to how ARDrift is trained: a drifting kernel that sees a window (#97), more than one
positive per drift problem (#98), training on the model's own history (#101) and guidance (#99). Each is a
fine-tune of the released English model against the same control, on the same data, scored the same way. This page
holds the protocol and one table for all of them; [HEAD_REALISM.md](HEAD_REALISM.md) has the diagnosis they start
from.

- [Protocol](#protocol)
- [Results](#results)
- [A kernel over a window (#97)](#a-kernel-over-a-window-97)
- [Training on the model's own history (#101)](#training-on-the-models-own-history-101)
- [Positives from a teacher (#98)](#positives-from-a-teacher-98)
- [Guidance (#99)](#guidance-99)
- [Commands](#commands)

## Protocol

- **Start:** `Vyvo/drifting-tts-en` (`ardrift_en.pt`, 200k steps). The released file does not hold its learned
  temperature; the runs start at 0.06 ([ENGLISH.md](ENGLISH.md#voices-from-a-prompt-95)).
- **An arm:** 5,000 steps of 16 utterances at 1e-4 on the streamed corpus (45–55 minutes on an A10), the same seed
  and shard order for every arm. The duration predictor keeps its Poisson re-fit (`loss.duration=0`), so every arm
  speaks with the released model's durations. The control is 5,000 steps of the unchanged objective.
- **Audio** (the 100 panel sentences, T 0.5, the released vocoder, each arm against the control sentence by
  sentence): WER (Whisper large-v3), the seven judges of `scripts/judge_panel.py`, F0 spread and pauses.
- **Mels** (what the judges do not hear, [HEAD_REALISM.md](HEAD_REALISM.md)):
  - the classifier two-sample test against the recordings: free-running windows of 4 and 16 frames at T 0.5, and
    one teacher-forced patch at T 1 (0.5: cannot be told apart);
  - the head's spread over its error by temporal frequency inside the patch (1: calibrated);
  - the gap between teacher-forced and free-running mels: distance to the recording and the energy on the
    recording's quietest frames (`scripts/gta_mismatch.py`).
- 5,000 steps is a pilot: it says which way an objective moves the model, not where a full run would end.

## Results

Every arm against the control (the judges: the difference over the same 100 sentences, `*` outside its 95%
interval; C2ST: accuracy of a classifier that tells the model's mel windows from the recordings', lower is better;
spread-skill: the head's spread over its error, slow to fast inside the patch, 1 is calibrated).

| arm | WER | UTMOSv2 | UTMOS22 | SCOREQ | NISQA | Audiobox PQ | C2ST free, 4 / 16 frames | one patch | spread-skill | pause, free-running |
|---|---|---|---|---|---|---|---|---|---|---|
| the released model | 2.37% | 3.381 | 4.131 | 4.251 | 4.633 | 7.510 | 0.88 / 0.94 | 0.79 | 1.02 0.97 0.88 0.80 | +1.56 dB |
| **control** | 2.17% | 3.357 | 4.116 | 4.246 | 4.658 | 7.513 | 0.85 / 0.92 | 0.78 | 1.02 0.97 0.89 0.80 | +1.34 dB |
| energy score (#97) | 2.17% | −0.003 | −0.003 | +0.002 | −0.014 | −0.006 | **0.82 / 0.84** | **0.71** | 1.05 1.00 0.93 0.84 | +1.66 dB |
| energy score + window maps (#97) | 2.30% | −0.012 | +0.004 | +0.002 | +0.008 | −0.022 | 0.82 / 0.91 | **0.54** | 1.04 1.01 0.94 0.87 | +1.75 dB |
| field + window maps, field (#97) | 2.10% | −0.000 | −0.000 | −0.006 | −0.030 | −0.027 | 0.83 / 0.91 | 0.62 | 1.01 0.99 0.92 0.85 | +1.88 dB |
| noise level as an input (#101) | 2.44% | +0.025 | +0.018 | +0.000 | −0.030 | −0.025* | 0.85 / 0.92 | 0.74 | 1.02 0.97 0.89 0.80 | +1.47 dB |
| rollouts of 4–16 patches (#101) | 2.30% | +0.015 | +0.019 | −0.000 | −0.040* | +0.001 | 0.85 / 0.91 | 0.74 | 1.03 0.98 0.89 0.80 | +1.61 dB |
| one-step own history (#101) | 2.51% | −0.015 | +0.007 | +0.006 | −0.029 | −0.025* | 0.85 / 0.91 | 0.77 | 1.06 0.98 0.89 0.80 | +1.39 dB |
| teacher A: the model's own head (#98) | 2.03% | +0.047 | +0.024 | +0.007 | −0.024 | +0.004 | 0.88 / 0.95 | 0.79 | 1.02 0.96 0.87 0.78 | +1.49 dB |
| teacher B: a flow-matching head, 32 steps (#98) | 2.03% | −0.098* | −0.137* | −0.156* | −0.168* | −0.082* | 0.96 / 0.99 | 0.89 | 0.88 0.86 0.79 0.74 | +1.06 dB |
| text dropout 0.15, scale 1 (#99) | 2.37% | −0.019 | −0.014 | −0.110* | −0.033 | −0.036* | 0.85 / 0.92 | 0.77 | 1.02 0.98 0.90 0.81 | +1.56 dB |

- **No arm moves a judge by more than its interval allows in the good direction.** With the vocoder fine-tuned on
  the model's mels, the released model is already at the recordings' level on all seven
  ([HEAD_REALISM.md](HEAD_REALISM.md#a-panel-of-judges-where-the-audible-gap-is)); what the arms change is what the
  distribution tests see.

## A kernel over a window (#97)

ARDrift's kernel sees one patch at a time: the whole patch, each frame, each frame's harmonic detail. The head-only
experiments ([HEAD_REALISM.md](HEAD_REALISM.md#fine-tuning-the-head-alone)) found the head calibrated overall and
too smooth in the fast part of the patch, the energy score able to calibrate it, and the feature maps of a mel
masked autoencoder over a window (three recorded patches, then the generated one) able to remove what a classifier
sees in one patch. Those runs trained the head on frozen states; here the whole model is trained.

- `drift.loss: energy`: the energy score in place of the drifting field, on the same maps.
- `drift.window`: for 256 of the rows and 16 of their samples, the window through the frozen 2-D Mel-MAE
  (`runs/mae16`: 4.2 M parameters, 16-frame windows); its five stages' maps over the generated patch are five more
  kernel maps, with the energy score or with the field (`drift.window.loss`). Gradients reach the generated patch
  only.

What a classifier still tells from the recordings (accuracy; 0.5: nothing), by what it is shown:

| arm | one teacher-forced patch: whole / harmonic detail / envelope / frame change | 4 free-running frames: whole / detail | 16 free-running frames: whole / detail | spread-skill, slow to fast |
|---|---|---|---|---|
| the released model | 0.79 / 0.86 / 0.57 / 0.88 | 0.88 / 0.89 | 0.94 / 0.95 | 1.02 0.97 0.88 0.80 |
| control | 0.78 / 0.85 / 0.63 / 0.88 | 0.85 / 0.88 | 0.92 / 0.92 | 1.02 0.97 0.89 0.80 |
| energy score | 0.71 / 0.84 / 0.53 / 0.87 | 0.82 / 0.86 | 0.84 / 0.92 | 1.05 1.00 0.93 0.84 |
| energy score + window maps | **0.54** / 0.80 / 0.52 / 0.87 | 0.82 / 0.84 | 0.91 / 0.91 | 1.04 1.01 0.94 0.87 |
| field + window maps with the field | 0.62 / **0.73** / 0.51 / 0.87 | 0.83 / 0.86 | 0.91 / 0.92 | 1.01 0.99 0.92 0.85 |

- **The window maps do to the whole model what they did to the head alone**: one teacher-forced patch is told
  from the recorded one 54% of the time instead of 78%, and the head's spread in the fastest band rises from 0.80
  to 0.87 of its error. 5,000 steps of 16 utterances, the backbone and the text encoder training with the head.
- **The field works with them too.** On the head alone no new map helped while the field was Kyutai's; with the
  window maps and the whole model the field gets 0.62 on the whole patch and the best harmonic detail of all
  (0.73). The energy score is not required; it calibrates a little further (0.87 against 0.85).
- **Nothing a judge hears changes**: all seven inside their intervals for the three arms, word errors 2.1–2.3%.
  With the fine-tuned vocoder the released model is already at the recordings' level on these judges.
- **What the window does not reach.** The frame-to-frame-change view stays at 0.87–0.88 in every arm, as it did
  for every head. Free-running, 4 frames move from 0.85 to 0.82–0.83 and 16 frames stay at 0.91; the energy score
  alone has the lowest 16-frame number (0.84), which the window maps do not keep. The quietest frames get louder
  free-running (+1.8 dB against +1.3 dB).
- **Cost**: 256 rows × 16 samples through the MAE take 5.6 GB more and a quarter of the speed (1.4–1.6 it/s
  against 1.9).
- **Step 1 of #97**, the patch boundaries, was answered before: free-running, the change across a boundary is 1.01
  of the change inside a patch (the recordings: 1.00).
- Not done: the 20–40k steps of the issue, the Turkish model, a longer window (28 frames; the MAE here was trained
  on 16), an encoder that predicts WavLM features, and a listening test. The vocoder was fine-tuned on the released
  model's mels, not on these arms'.

## Training on the model's own history (#101)

ARDrift is trained on the recorded history plus noise of a random level up to 0.3, and generates on its own output
without noise. The backbone is not told the level.

**Step 1, no training: noise on the history at inference.** `generate(history_noise=)` adds Gaussian noise to what
the backbone reads of the patches so far (the patches themselves stay clean, and their own noise is drawn as
without it). If the missing noise were the mismatch, the model would get closer to the recordings with it. The
released model, 200 held-out utterances under the recordings' alignment, T 0.5 (`scripts/gta_mismatch.py`):

| noise on the free-running history | distance to the recording | energy on its quietest frames | frame change across / inside a patch |
|---|---|---|---|
| teacher-forced (the training's regime) | 2.31 dB | +0.75 dB | 1.25 |
| free-running, none (inference) | 4.38 dB | +1.56 dB | 1.01 |
| 0.075 | 4.40 dB | +1.61 dB | 1.02 |
| 0.15 (the mean training level) | 4.42 dB | +1.66 dB | 1.04 |
| 0.3 | 4.51 dB | +1.74 dB | 1.06 |

- **Noise does not help**: every measure moves away from the recordings as the noise grows. On audio (the 100
  panel sentences) noise of 0.15 changes nothing a judge hears: WER 2.30% against 2.37%, no judge outside its
  interval.
- **The model is robust to the level it was not told.** The mismatch between training and inference is not the
  noise; what is left is that the history is the model's own.
- **On this corpus the gap is small**: +0.8 dB on the recording's quietest frames teacher-forced, +1.6 dB
  free-running. The Turkish podcast voices, whose recordings are noisy, had +2.5 and +8.8 dB
  ([AR_DRIFT.md](AR_DRIFT.md#pauses-of-the-podcast-voices-73)).

**Steps 2–4: three fine-tunes against the control** (5,000 steps each; 200 held-out utterances under the
recordings' alignment for the gap, the 100 panel sentences for the audio):

| arm | distance to the recording, free-running | quietest frames: teacher-forced / free-running | C2ST free, 4 / 16 frames | WER | judges against the control |
|---|---|---|---|---|---|
| the released model | 4.38 dB | +0.75 / +1.56 dB | 0.88 / 0.94 | 2.37% | – |
| control | 4.34 dB | +0.77 / +1.34 dB | 0.85 / 0.92 | 2.17% | – |
| `model.ar.noise_level`: the backbone is told the level | 4.38 dB | +0.74 / +1.47 dB | 0.85 / 0.92 | 2.44% | 6 inside their intervals, Audiobox PQ −0.03 |
| `drift.rollout`: 4–16 free-running patches in half of the batches | 4.37 dB | +0.80 / +1.61 dB | 0.85 / 0.91 | 2.30% | 6 inside, NISQA −0.04 |
| `drift.self_history: 0.5`: patches one step after the recording | 4.33 dB | +0.71 / +1.39 dB | 0.85 / 0.91 | 2.51% | 6 inside, Audiobox PQ −0.03 |

(The distances have a standard error of 0.07 dB, the free-running pause energies of 0.25 dB; the word error rates
are 32 to 37 errors of 1,477 words.)

- **None of the three moves the gap.** The free-running distance to the recording stays at 4.33–4.38 dB and the
  free-running pause energy at +1.4 to +1.6 dB; the differences between the arms are inside the measurement.
  Nothing a judge hears changes either.
- **Most of the "gap" is not an error.** Two teacher-forced draws of one utterance are 1.6 dB apart, a
  teacher-forced draw is 2.3 dB from the recording, and a free-running one 4.4 dB: on its own history the model
  reads the sentence its own way, which is another reading and not a worse one. The measurable error is the energy
  on the quietest frames, and on this clean corpus that is under 1 dB.
- **Rollouts cost a quarter of the speed** (1.3 it/s against 1.9 with the evaluation beside it) and bring nothing
  here that the one-step history does not.
- **Where it could matter.** On the Turkish podcast voices the one-step history took the free-running pause energy
  from +8.8 to +6.5 dB and levelled off. `drift.rollout` is the next thing to try there; it could not be tried
  where this was run (no Turkish data).

## Positives from a teacher (#98)

ARDrift estimates every drifting field from one data sample, the recorded patch. `drift.teacher` adds positives
drawn by a frozen model for the same step: its own state of the clean recorded history, then its head. The student
draws 32 samples against 32 positives (31 from the teacher and the recording).

**Teacher A, self-distillation**: the released model's own one-step head.

**Teacher B, a multi-step head**: a flow-matching head on the frozen backbone's states (`FlowHead`: the drifting
head's body with the time as an input, a rectified-flow loss; `scripts/head_finetune.py --loss flow`, head only,
12 it/s). What it is worth as a teacher, on 4,096 held-out states of speech (spread over error by temporal
frequency inside the patch, 1 = calibrated; `outputs` of the same check for both heads):

| head | mean of the four frames | trend | mid | alternation | sample spread | error of the samples' mean |
|---|---|---|---|---|---|---|
| the one-step drifting head | 1.02 | 0.98 | 0.88 | 0.79 | 0.112 | 0.0166 |
| flow head, 8,000 steps, 8 Euler steps | 0.77 | 0.76 | 0.76 | 0.90 | 0.100 | 0.0209 |
| flow head, 30,000 steps, 8 Euler steps | 0.79 | 0.75 | 0.68 | 0.65 | 0.091 | 0.0194 |
| flow head, 30,000 steps, 16 Euler steps | 0.85 | 0.84 | 0.80 | 0.80 | 0.103 | 0.0195 |
| flow head, 30,000 steps, 32 Euler steps | 0.88 | 0.89 | 0.88 | 0.93 | 0.111 | 0.0196 |

- **The flow head is evenly calibrated, the drifting head is not.** With enough Euler steps its spread is the same
  share of its error in every temporal frequency (0.88–0.93), where the drifting head falls from 1.02 to 0.79
  towards the fast one. This is the shape the one-positive field cannot give
  ([HEAD_REALISM.md](HEAD_REALISM.md#the-mechanism-one-positive-unequal-directions)).
- **Its mean is worse** (0.0196 against 0.0166) after 30,000 steps on 6 h of states, and it needs 32 steps: at 8 it
  is narrower than the drifting head everywhere. The first arm with it (8,000 steps, 8 Euler steps) was stopped for
  that reason.

**The two arms** (5,000 steps, 31 teacher positives and the recording against 32 samples; teacher B sampled with
32 Euler steps, which makes a step 2.5 times as slow):

| arm | WER | UTMOSv2 | UTMOS22 | SCOREQ | C2ST free, 4 / 16 frames | spread-skill | error of the samples' mean in training |
|---|---|---|---|---|---|---|---|
| control | 2.17% | 3.357 | 4.116 | 4.246 | 0.85 / 0.92 | 1.02 0.97 0.89 0.80 | 0.0177 |
| teacher A | 2.03% | 3.404 (+0.047) | 4.140 (+0.024) | 4.253 (+0.007) | 0.88 / 0.95 | 1.02 0.96 0.87 0.78 | 0.0176 |
| teacher B | 2.03% | 3.259 (−0.098*) | 3.978 (−0.137*) | 4.090 (−0.156*) | 0.96 / 0.99 | 0.88 0.86 0.79 0.74 | 0.0202 |

- **Teacher A is what the issue expected**: a self-distilled teacher adds no information. Five of seven judges
  rise, none outside its interval (UTMOSv2 +0.05, the size of Kyutai's gain from the same idea), two word errors
  fewer of 1,477. The learned temperature falls from 0.077 to 0.028: with 32 positives the field is less noisy
  and a sharper kernel is affordable. The distribution tests go back to the teacher's values (0.88 / 0.95, the
  released model's 0.88 / 0.94), and the fast band stays under-dispersed (0.78).
- **Teacher B makes the model worse**: six of seven judges fall outside their intervals, and a classifier tells
  its mels from the recordings 96–99% of the time. The student takes on the teacher's mean (its error rises from
  0.0177 to 0.0202, the flow head's own 0.020) and comes out narrower than the teacher in every band.
- **A teacher has to be better than the student, and this one is not.** A flow-matching head trained for 30,000
  steps on 6 h of frozen states has the better *shape* of spread and the worse mean. Kyutai's teacher was a full
  model trained on their data. A flow head worth distilling from would need the backbone's states of the whole
  corpus, not of 6 h.
- Not done: the comparison at 20,000 steps, a teacher B trained on more than 6 h of states, fewer positives
  balanced against fewer samples.

## Guidance (#99)

Guidance is the main quality lever of Pocket TTS (1.03% WER at a scale of 1.5 against 1.43% without). In ARDrift it
was trained once, in a fine-tune, and did not help. The code now has what the issue asks for: text dropout
(`drift.cond_dropout`), voice dropout (`drift.voice_dropout`: no prompt, no reference, the row of a voice that is
not in the table), a null branch without the text, the voice or both (`generate(guide=)`), and the guided state
distilled into the backbone (`drift.distill`).

**Text guidance.** The arm trained with text dropout 0.15, the 100 panel sentences at four scales (judges against
its own scale 1):

| scale | WER | UTMOSv2 | UTMOS22 | SCOREQ | NISQA | Audiobox PQ | F0 std | silence |
|---|---|---|---|---|---|---|---|---|
| control (no dropout), 1.0 | 2.17% | 3.357 | 4.116 | 4.246 | 4.658 | 7.513 | 3.58 st | 35% |
| 1.0 | 2.37% | 3.338 | 4.101 | 4.136 | 4.624 | 7.477 | 3.68 st | 40% |
| 1.25 | 7.85% | 2.891 | 3.109 | 3.198 | 4.065 | 6.971 | 3.85 st | 34% |
| 1.5 | 25.05% | 1.881 | 1.750 | 1.672 | 2.465 | 5.237 | 4.86 st | 24% |
| 1.75 | 44.75% | 1.421 | 1.418 | 1.206 | 1.714 | 4.588 | 5.41 st | 21% |

- **Every scale above 1 destroys the speech**: seven judges lower at each step, word errors three, ten and twenty
  times as many. Nothing like Kyutai's sweep, where 1.5 is the best scale.
- **The dropout alone costs a little**: at scale 1 the arm is 0.11 below the control on SCOREQ and 0.04 on
  Audiobox PQ (both outside their intervals), and its pauses are longer (0.84 s against 0.73 s).

**Why**, teacher-forced on 60 held-out utterances, where no error can accumulate: the head's mean patch from the
guided state `s_null + g (s − s_null)` against the recording.

| scale | 0 (the null state) | 0.5 | **1.0** | 1.1 | 1.25 | 1.5 | 1.75 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| error of the mean patch | 0.0389 | 0.0214 | **0.0132** | 0.0139 | 0.0163 | 0.0242 | 0.0361 | 0.0513 |
| size of the residual over the prior mean (the recordings': 0.081) | 0.051 | 0.060 | 0.076 | 0.080 | 0.088 | 0.103 | 0.121 | 0.140 |

- **The error is smallest at 1 and grows on both sides**, already in the regime the head is trained in. At 1.5 the
  mean patch is 1.8 times as far from the recording, at 2 almost four times.
- **Guidance scales the residual, not the text.** ARDrift's head adds a residual to the aligned prior mean, and the
  prior mean is where the text is. The null branch still gets that mean, so the difference between the two states
  is what knowing the text adds to the *residual*; extrapolating it makes the residual larger than the recordings'
  (0.103 at 1.5 against 0.081) and the patch wrong. In Pocket TTS the text reaches the output only through the
  state, and its extrapolation is more text.
- **What #99 expected** ("the prior mean still carries the text in the null branch... the gain, if any, would be
  in naturalness") holds in its first half only: text guidance has no gain to distil.

**Voice guidance works.** The model with a reference path ([ENGLISH.md](ENGLISH.md#a-reference-from-another-utterance-96))
fine-tuned for 4,000 more steps with voice dropout 0.15 and text dropout 0.1; then guidance with a null branch
without the voice (the row of a voice that is not in the table and the null tokens in place of the reference's;
the text stays). The 36 unseen recorded speakers, the voice from the reference alone:

| scale | SIM-o | SIM with a recording the model was not given | WER | F0 std |
|---|---|---|---|---|
| 1.0 | 0.438 | 0.387 | 2.79% | 3.19 st |
| 1.5 | 0.491 | 0.422 | 2.60% | 3.33 st |
| 2.0 | 0.524 | 0.443 | 2.79% | 3.32 st |
| 2.5 | 0.548 | 0.457 | 2.99% | 3.38 st |
| **3.0** | **0.563** | **0.466** | 3.04% | 3.39 st |
| the prompt, no guidance | 0.546 | 0.464 | 3.24% | 3.03 st |
| prompt + reference, no guidance | 0.554 | 0.464 | 3.24% | 3.01 st |

- **Every step of the scale adds voice**: +0.035, +0.056, +0.070 and +0.079 on the voice alone (the last: 95%
  +0.065 to +0.092, higher on 87% of the items), for a quarter of a point of WER at most. The curve had not turned
  at 3.
- **At 3 the reference alone is the prompt's equal**: the same similarity with an unheard recording (0.466 against
  0.464, the difference inside its interval), a higher SIM-o (+0.017), fewer word errors and a wider F0. A
  reference need not be contiguous with the output and does not carry its recording's conditions into it.
- **Why the voice and not the text.** The voice reaches the output only through the state (the tokens and the
  speaker row), as the text does in Pocket TTS; the text reaches ARDrift's output through the prior mean as well.
- **It does little for a prompt** (prompt + reference at 1.5: +0.003): the prompt is history, and the null branch
  reads it too.
- **Text guidance breaks this model as well**: at 1.5 the reference alone falls to 0.311 with 18.8% WER, with or
  without the voice in the null branch.
- **The dropouts cost nothing**: without guidance the model is where it was before them (0.546 / 0.438 / 0.554 for
  prompt / reference / both).
- Not done: scales above 3, guidance with a prompt in the null branch removed, and the distillation of voice
  guidance into the backbone (`drift.distill` takes voices of the speaker table; a reference would have to come
  with the batch).

**Distillation** (`drift.distill`: the backbone alone is trained towards the frozen teacher's guided state, mean
squared error; the text encoder and the head are not touched). 3,000 steps towards scale 1.5 close 82% of the
distance between the plain and the guided state. In one pass the student then speaks as its teacher does in two:
WER 21.5% against 25.1%, UTMOSv2 2.09 against 1.88, 3.1 pauses per utterance against 3.2, F0 spread 4.6 against
4.9 semitones (the control: 2.2%, 3.36, 2.3, 3.6). The mechanism works; what it was given to distil is damage.

## Commands

```bash
# an arm: the released model, 5,000 steps, the control's settings plus the arm's own
drifting-tts train-ar --config configs/ar_tts_en_10k.yaml --workdir runs/arm train.init_from=ardrift_en.pt \
    train.batch_size=16 train.batch_frames=16000 train.lr=1e-4 train.steps=5000 train.warmup=300 \
    drift.max_rows=768 drift.tau_init=0.06 loss.duration=0 <the arm>
#   #97   drift.loss=energy                                            # the energy score
#         drift.window.mae=runs/mae16/mae_ema.pt drift.window.loss=energy|field [drift.loss=energy]
#   #101  model.ar.noise_level=true | drift.rollout.prob=0.5 "drift.rollout.steps=[4,16]" | drift.self_history=0.5
#   #98   drift.teacher.path=ardrift_en.pt drift.teacher.positives=31 [drift.teacher.head=flow/head.pt
#         drift.teacher.steps=32]
#   #99   drift.cond_dropout=0.15; then drift.distill.teacher=<that model> drift.distill.cfg_scale=1.5
drifting-tts train-mae --config configs/mae2d.yaml --workdir runs/mae16 data.root=data/en_ft \
    model.speaker_head=false "model.mask_patch=[10,4]" train.crop_frames=16 train.steps=10000    # the window's MAE
python scripts/head_finetune.py --model ardrift_en.pt --data data/en_ft --out flow --loss flow --steps 30000 \
    --samples 8 --input-noise 0                                                      # teacher B's flow head
# the tests of an arm
python scripts/judge_panel.py --systems control=wav/control arm=wav/arm --baseline control --out outputs/panel
python scripts/c2st_mel.py --model runs/arm/model_ema.pt --data data/en_val --mode free --temperature 0.5 \
    --windows 4 16 --views full detail envelope delta --active -0.5
python scripts/c2st_mel.py --model runs/arm/model_ema.pt --data data/en_val --mode gta --temperature 1.0 \
    --windows 4 --stride 4 --views full detail envelope delta --active -0.5
python scripts/drift_calibration.py --model runs/arm/model_ema.pt --data data/en_val --active -0.5 --no-free
python scripts/gta_mismatch.py --model runs/arm/model_ema.pt --data data/en_val --num 200 --steps 0 \
    [--history-noise 0.15]
```

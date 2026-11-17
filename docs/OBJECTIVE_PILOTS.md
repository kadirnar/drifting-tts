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
WINDOW_ROWS
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

WINDOW_RESULTS

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
  in naturalness") holds in its first half only: there is no gain to distil.

**Distillation** (`drift.distill`: the backbone alone is trained towards the frozen teacher's guided state, mean
squared error; the text encoder and the head are not touched). 3,000 steps towards scale 1.5 close 82% of the
distance between the plain and the guided state. In one pass the student then speaks as its teacher does in two:
WER 21.5% against 25.1%, UTMOSv2 2.09 against 1.88, 3.1 pauses per utterance against 3.2, F0 spread 4.6 against
4.9 semitones (the control: 2.2%, 3.36, 2.3, 3.6). The mechanism works; what it was given to distil is damage.

## Commands

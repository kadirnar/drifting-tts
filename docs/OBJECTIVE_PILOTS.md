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

*(filled as the arms finish)*

## A kernel over a window (#97)

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

## Guidance (#99)

## Commands

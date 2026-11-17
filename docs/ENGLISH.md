# ARDrift in English: 10,000 hours of synthetic speech

ARDrift ([docs/AR_DRIFT.md](AR_DRIFT.md)) trained in English from scratch, with an English character frontend, on 10,000
hours of synthetic speech too large for the disk, so streamed from the Hub while training.

- [Data](#data)
- [What changed in the code](#what-changed-in-the-code)
- [Results](#results)
- [Limits](#limits)
- [Commands](#commands)

## Data

[VoiceHub/echo-clones-10kh-en](https://huggingface.co/datasets/VoiceHub/echo-clones-10kh-en): the first 6,949 parquet
shards, unchanged, of [SynDataLab-EN/echo-clones-4m-en](https://huggingface.co/datasets/SynDataLab-EN/echo-clones-4m-en):

| | |
|---|---|
| hours / utterances | 10,000.7 h / 3,474,250 (1.33 TB) |
| speech | synthetic, generated with [EchoTTS (jordand/echo-tts-base)](https://huggingface.co/jordand/echo-tts-base) for 4,000 reference speakers, 44.1 kHz |
| text | conversational English, 24-420 characters (fillers, em dashes, few digits or acronyms) |
| license | Apache-2.0 as declared by the source; the generating model is CC-BY-NC-SA-4.0 |

The source's shards are not uniform: the first ones hold mid-length utterances (median 8.3 s in shards 0-69), later
ones many short ones (median 5.6 s in shard 3,000).

**Held out:** the val and dev splits (200 + 300 utterances) come from source shard 07,960, which is not among the
6,949 shards: texts never trained on, read by the training speakers (every speaker occurs throughout the corpus).

## What changed in the code

- **English frontend** (`drifting_tts/text_en.py`): the 26 letters, the apostrophe (`we'll`, `don't`: it changes the
  pronunciation) and the Turkish frontend's punctuation, with `<pad>` / `<blank>` at the same ids. Normalisation
  writes out numbers, years (`1999` → *nineteen ninety nine*), decades, ordinals, money, percentages, clock times,
  common abbreviations and all-caps acronyms letter by letter (`TV` → *tee vee*, while emphasis like `SO` stays a word).
  English spelling is far from phonemic; with this much data the text encoder learns the letters' sounds (no G2P).
- **A language switch**: `model.text.language` sets the text encoder's vocabulary and a prepared dataset's
  `stats.json` records its language (`prepare --language`); the Synthesizer, `evaluate` and Whisper follow them. A
  model or dataset without a language is Turkish, as before. `benchmark` and `prosody` take the language from the
  checkpoint (Whisper, the normaliser of both texts, the syllable count of the speaking rate, the symbol table of the
  token controls) and `score` from the dataset's `stats.json` (`--language` overrides it), so the English data can be
  scored and filtered (#86). `benchmark` has no English default text set: pass `--texts` (a `.txt` / `.jsonl` file,
  or `<HF dataset id>:<file>`).
- **`prepare`**: `--speaker-column`, `--val-glob` (held-out files form the val split, the rest of them dev), and
  `hf://` paths read without a local copy.
- **Streaming** (`drifting_tts/stream.py`, `data.stream`): 10,000 h of log-mels would take ~675 GB. Each DataLoader
  worker downloads every n-th shard of a shuffled order (the next one while it reads the current one), turns each row
  into a log-mel as `prepare` does and yields length-bucketed batches; at most two shards per worker are on disk. One
  core makes mels ~1,400x faster than real time, so the network decides: 12 workers deliver ~2,165 s of audio per
  second, more than the training uses. The mel statistics, the speaker table and the val split come from the
  prepared pilot data.
- `evaluate --asr-device cpu`: Whisper beside a training run that holds the GPU.

## Results

Val 100 (the first 100 held-out utterances): Whisper large-v3 in English (for the 10k-h run on the CPU, int8),
UTMOSv2, WavLM-ECAPA similarity to the recording's copy-synthesis; Vocos v2, noise temperature 0.5, no guidance. The
copy-synthesis row is the recording's mel through Vocos v2: the ceiling of this vocoder.

**Pilot** (`configs/ar_tts_en.yaml`: 87.7 h, source shards 0-69; the Turkish model's size with a wider text encoder:
12.8 + 42.5 + 18.0 M; 30k steps, 7.2 it/s):

| step | WER | CER | UTMOSv2 | SIM |
|---|---|---|---|---|
| 10k | 22.82% | 12.61% | 2.742 | 0.190 |
| 20k | 4.74% | 2.39% | 2.912 | 0.329 |
| 30k | 3.32% | 1.52% | 2.957 | 0.422 |
| copy-synthesis | 2.57% | 1.12% | 2.926 | – |

**10,000 h** (`configs/ar_tts_en_10k.yaml`: 36.2 + 121.2 + 20.5 M, batches of up to 48 utterances, 200k steps, about
one pass over the data, 13 h on one RTX 5090):

| step | WER | CER | UTMOSv2 | SIM |
|---|---|---|---|---|
| 20k | 3.99% | 2.04% | 2.917 | 0.394 |
| 40k | 2.51% | 1.22% | 2.950 | 0.591 |
| 60k | 2.51% | 1.26% | 3.014 | 0.653 |
| 80k | 2.30% | 1.03% | 2.907 | 0.695 |
| 100k | 2.17% | 0.94% | 2.945 | 0.722 |
| 120k | 2.51% | 1.07% | 2.990 | 0.740 |
| 140k | 2.10% | 0.98% | 2.904 | 0.757 |
| 160k | 2.57% | 1.07% | 2.998 | 0.771 |
| 180k | 2.10% | 0.92% | 2.949 | 0.770 |
| 200k | 2.17% | 0.90% | 2.994 | 0.774 |
| copy-synthesis | 2.44% | 1.09% | 2.926 | – |

**All 500 held-out utterances** (val 200 + dev 300, 15,302 words; Whisper on the GPU, fp16), to tell the last
snapshots apart:

| step | WER | CER | UTMOSv2 | SIM |
|---|---|---|---|---|
| 140k | 1.50% | 0.65% | 2.916 | 0.783 |
| 180k | **1.39%** | **0.62%** | 2.921 | 0.800 |
| **200k** (the model in the test Space) | 1.42% | 0.64% | **2.930** | **0.803** |
| copy-synthesis | 1.57% | 0.73% | 2.907 | – |

180k and 200k differ by 5 words; 200k is ahead on similarity and UTMOSv2. The test Space holds the 200k weights in fp16 (`export_tts(half=True)`: 356
MB, a Space stores at most 1 GB): on val 100 the same WER / CER as fp32 (2.23% / 0.90%), UTMOSv2 3.002 against 2.994.

- **As intelligible as the recordings from 40k steps on**, with the vocoder they go through, and slightly more on
  all 500 utterances (1.42% against 1.57%: the source's own slips are not learned); the 100 utterances hold ~1,500
  words, so the snapshots after 40k differ there by a few words.
- **The voices are what more data buys**: similarity to each speaker rises at every snapshot, 0.39 → 0.77, where the
  pilot, with ~9 utterances per speaker, stopped at 0.42.
- **UTMOSv2 at the recordings' level** (2.90-3.01 against 2.926): the synthetic source is the ceiling.

## Vocoder: BigVGAN-v2 fine-tuned on the model's mels (#70)

Vocos v2, the vocoder of the tables above, was trained on Turkish speech. On the English recordings' mels it loses
0.33 UTMOSv2 (val 100: 2.926 against the recordings' 3.259). NVIDIA's BigVGAN-v2 is transparent on those mels
(3.296), but on the model's mels it falls to 2.680, against Vocos v2's 2.997. Both vocoders were fine-tuned on the
200k model's own mels (GTA: 80% of the batches are teacher-forced generated mels at noise temperature 0.5, the rest
recorded mels), 40k steps each, side by side on one RTX 5090. The data is the 37,115 utterances (105 h train) of the
Mimi decoder fine-tune ([MIMI.md](MIMI.md#fine-tuning-mimis-decoder)): the same prepared audio, as BigVGAN log-mels
(`data/en_mel_ft`).

- **BigVGAN-v2** (`configs/vocoder_bigvgan_en_gta.yaml`): the recipe of `bigvgan-v2-ft`, i.e. NVIDIA's generator,
  discriminators and AdamW moments, batch 4, lr 1.35e-5; 1.9-2.4 it/s, 5.4 h.
- **Vocos v2** (`configs/vocoder_vocos_en_gta.yaml`): Vocos v2's own recipe, from the published generator
  (`train.init_from: vocos-v2`: an exported vocoder gives the generator only). Its discriminators were not
  published, so fresh ones train alone for the first 2k steps.

Val 100, the model's output through each snapshot (Whisper on the CPU, as the tables above; Vocos v2: 2.17% / 2.994):

| step | BigVGAN-v2 WER | UTMOSv2 | Vocos v2 WER | UTMOSv2 |
|---|---|---|---|---|
| 5k | 2.10% | 3.165 | 2.30% | 2.927 |
| 10k | 2.10% | 3.165 | 2.03% | 2.949 |
| 15k | 2.17% | 3.207 | 2.10% | 2.985 |
| 20k | 2.03% | 3.163 | 2.10% | 3.001 |
| 25k | 2.03% | 3.197 | 2.10% | 2.981 |
| 30k | 2.10% | 3.217 | 1.90% | 2.998 |
| 35k | 2.17% | 3.178 | 2.03% | 3.025 |
| 40k | 2.37% | **3.272** | 1.90% | 3.043 |

The two best BigVGAN snapshots on all 500 held-out utterances (Whisper on the GPU, as the 500-utterance table
above); copy-synthesis is the recordings' mels through the same vocoder:

| vocoder | WER | CER | UTMOSv2 | copy-synthesis WER | UTMOSv2 |
|---|---|---|---|---|---|
| Vocos v2 | 1.42% | 0.64% | 2.930 | 1.57% | 2.907 |
| BigVGAN-v2 GTA 30k | 1.38% | 0.62% | 3.177 | 1.57% | 3.276 |
| **BigVGAN-v2 GTA 40k** | **1.35%** | **0.60%** | **3.245** | 1.57% | 3.294 |

Against Vocos v2, sentence by sentence (the same mels): UTMOSv2 +0.315 ± 0.025 (95%), higher on 436 of 500; 10
word errors fewer. Speaker similarity is left out: its reference is the copy-synthesis through the same vocoder.

Where the noise is, on val 100 (DNSMOS P.835: SIG speech, BAK background, OVRL overall; noise floor: the 10th
percentile of the 20 ms frame energies minus the 95th, lower is quieter):

| | UTMOSv2 | SIG | BAK | OVRL | noise floor |
|---|---|---|---|---|---|
| recording | 3.259 | 3.53 | 4.12 | 3.29 | −57.3 dB |
| copy-synthesis, Vocos v2 | 2.926 | 3.49 | 4.11 | 3.25 | −60.8 dB |
| copy-synthesis, BigVGAN-v2 GTA 40k | 3.314 | 3.55 | 4.13 | 3.31 | −57.5 dB |
| model, Vocos v2 | 2.997 | 3.48 | 4.07 | 3.21 | −52.2 dB |
| model, BigVGAN-v2 GTA 40k | 3.280 | 3.52 | 4.07 | 3.26 | −50.9 dB |

- **The vocoder gap is closed**: the model's output through the fine-tuned BigVGAN scores close to the recordings
  through it (3.245 against 3.294 on 500), and higher than the recordings through Vocos v2.
- **The model's higher noise floor is shorter pauses, not noise** (see [Pauses](#pauses-the-duration-predictor-re-fitted-to-the-mean-72)):
  the percentile floor rises when an utterance has less silence. Under the recordings' durations the model's pauses
  are as quiet as theirs.
- **Vocos v2 gains little** (+0.05 at 40k, its copy-synthesis 2.93 → 3.10): too small a network, or the fresh
  discriminators.
- **Cost**: 112 M parameters against 13.5 M. On the RTX 5090 the whole pipeline's real-time factor goes from 0.14 to
  0.16.
- **Where**: a private English Space of its own offers it as the default, next to Vocos v2 (`space_en/`,
  `scripts/deploy_english_space.sh`; fp16 weights, 225 MB). The test Space has no room left: a Space stores at most
  1 GB, every stored file version included. Its English tab keeps Vocos v2. The vocoder is not on the Hub and not in
  the vocoder registry.

## GTA mels that match inference (#90)

`finetune-vocoder` draws ARDrift's mels teacher-forced: one pass, every patch from the state after the *recorded*
previous patches (`ARDriftTTS.gta`). At inference every patch follows the model's own patches. Under the recording's
alignment both can be generated frame for frame with the recording, so the difference the vocoder never trains on can
be measured (`scripts/gta_mismatch.py`; the released model, T 0.5, held-out utterances).

**The mels** (200 utterances; dB of log-mel magnitude):

| | recording | teacher-forced | free-running |
|---|---|---|---|
| mean distance to the recording | – | 2.31 dB | 4.38 dB |
| frame-to-frame change inside a patch | 4.96 dB | 4.72 dB | 4.68 dB |
| ... across a patch boundary | 4.92 dB | **5.89 dB** | 4.70 dB |
| energy of the recording's quietest 10% of frames | – | +0.75 dB | +1.56 dB |

Two teacher-forced draws of the same utterance are 1.58 dB apart (the head's own noise). The distances do not grow
along the utterance (by quarter: 5.1 / 4.4 / 3.9 / 4.3 dB free-running), so the model's error does not accumulate.

- **Teacher-forced patches do not join.** Every patch is drawn after the recorded history, independently of the
  generated patch before it, so the mel steps at every patch boundary: 25% more change there than inside a patch.
  A recording has no such grid, and neither has the free-running mel, whose patches continue each other.
- **A classifier tells the two apart** (`c2st_mel.py`'s test; two teacher-forced draws: 0.50): 0.67 from 4 frames,
  0.86 from 16, 0.92 from 64. On one-patch windows it is 0.58 on the patch grid and 0.81 half a patch off it: the
  tell is where the patches join.
- **The free-running mel is a different reading of the sentence**: twice as far from the recording, which makes the
  recording a noisier target for it.

**Through the vocoders** (val 100, the judges of [HEAD_REALISM.md](HEAD_REALISM.md#a-panel-of-judges-where-the-audible-gap-is),
both mels under the recording's alignment):

| vocoder | mel | UTMOSv2 | UTMOS22 | SCOREQ | NISQA | DNSMOS OVRL | SQUIM | Audiobox PQ |
|---|---|---|---|---|---|---|---|---|
| – | recording | 3.27 | 4.20 | 4.28 | 4.63 | 3.29 | 4.33 | 7.49 |
| NVIDIA BigVGAN-v2, as released | teacher-forced | 2.63 | 2.95 | 3.10 | 4.38 | 2.94 | 4.08 | 6.54 |
| NVIDIA BigVGAN-v2, as released | free-running | 2.83 | 3.37 | 3.51 | 4.44 | 3.13 | 4.16 | 6.89 |
| BigVGAN-v2 GTA 40k (teacher-forced GTA) | teacher-forced | 3.29 | 4.05 | 4.17 | 4.59 | 3.29 | 4.30 | 7.44 |
| BigVGAN-v2 GTA 40k (teacher-forced GTA) | free-running | 3.32 | 4.07 | 4.21 | 4.60 | 3.32 | 4.32 | 7.51 |

- **Teacher-forced mels are the harder input, not the easier one.** An unadapted vocoder renders the steps at the
  patch boundaries: 0.42 UTMOS22 and 0.41 SCOREQ below the free-running mels of the same sentences (all seven judges
  significant).
- **The fine-tuned vocoder has learned to repair them**, and decodes the free-running mels it never saw slightly
  better than its own training mels (all seven judges higher, three significantly).

**Does training on free-running mels help?** `train.gta_free_prob` draws that share of the generated batches
free-running (the recording's alignment, the model's own history; guidance is allowed there). Three fine-tunes of
NVIDIA's generator with the released recipe, the same seed and the same data (3,194 utterances of up to 16 s, 6.0 h,
1,266 speakers, from eight shards of the corpus), on an A10. The system as it is used (predicted durations,
free-running, T 0.5; val 100; WER by Whisper large-v3):

| vocoder | steps | UTMOSv2 | UTMOS22 | SCOREQ | NISQA | DNSMOS OVRL | SQUIM | Audiobox PQ | WER |
|---|---|---|---|---|---|---|---|---|---|
| NVIDIA BigVGAN-v2, as released | 0 | 2.91 | 3.45 | 3.61 | 4.46 | 3.12 | 4.17 | 6.95 | 2.10% |
| teacher-forced GTA (`gta_free_prob: 0`) | 2,500 | 3.28 | 4.10 | 4.20 | 4.57 | 3.28 | 4.30 | 7.46 | 2.10% |
| | 5,000 | 3.22 | 4.09 | 4.25 | 4.56 | 3.27 | 4.29 | 7.45 | 2.17% |
| half free-running (`gta_free_prob: 0.5`) | 2,500 | 3.25 | 4.04 | 4.17 | 4.55 | 3.25 | 4.33 | 7.43 | 2.17% |
| | 5,000 | 3.34 | 4.07 | 4.23 | 4.58 | 3.26 | 4.30 | 7.45 | 2.17% |
| all free-running (`gta_free_prob: 1`) | 2,500 | 3.26 | 4.01 | 4.12 | 4.52 | 3.24 | 4.33 | 7.41 | 2.37% |
| released: teacher-forced GTA, 105 h | 40,000 | 3.38 | 4.13 | 4.25 | 4.63 | 3.29 | 4.32 | 7.51 | 2.37% |
| recording | – | 3.27 | 4.20 | 4.28 | 4.63 | 3.29 | 4.33 | 7.49 | 2.44% |

- **No, and the more of them the lower.** After 2,500 steps the system falls with the share of free-running
  batches on five of seven judges: UTMOS22 4.10 / 4.04 / 4.01 and SCOREQ 4.20 / 4.17 / 4.12 for none, half and all
  of them (four judges significantly lower for half, the same four for all; the third run was stopped there). After
  5,000 steps half is mixed (three judges higher; UTMOSv2 +0.13 and SCOREQ -0.03 are the significant ones).
- **What differs between the two vocoders is not how they decode the model's mels.** They differ by the same
  amounts on copy-synthesis of *recorded* mels (UTMOSv2 +0.16, SCOREQ -0.03): system minus copy-synthesis is within
  0.03 on every judge and no interval excludes 0. And each decodes the free-running mel better than the
  teacher-forced one by the same margin (half free-running against teacher-forced vocoder: UTMOS22 +0.10 / +0.12,
  SCOREQ +0.10 / +0.09, Audiobox PQ +0.11 / +0.10): the interaction is inside its interval on all seven judges.
  There is one run per arm, and UTMOSv2 alone moves by 0.06 to 0.10 between two snapshots of one run.
- **Why it need not help.** The teacher-forced mel carries the model's texture *and* the boundary steps, and it is
  the cleaner pair: 2.3 dB from the recording that is the vocoder's target, against 4.4 dB (multi-scale mel loss on
  those batches 1.09 against 1.44). A free-running batch also costs one backbone step per patch: 1.15 it/s against
  1.64.
- **A short fine-tune gives most of the vocoder's gain.** 2,500 steps on 6 h (25 minutes on an A10) recover 91-95%
  of what the released 40k-step vocoder gains over NVIDIA's on UTMOS22, SCOREQ, SQUIM and Audiobox PQ (DNSMOS 89%,
  UTMOSv2 78%, NISQA 64%). What the long fine-tune still adds is mostly the repair of the boundary steps, which
  inference never meets: at 5,000 steps the teacher-forced mel decodes 0.12 UTMOS22 below the free-running one, for
  the 40k vocoder 0.03.

`gta_free_prob` stays 0. `gta_temperature: auto` / `gta_cfg: auto` take the operating point stored in the TTS
checkpoint, so that a vocoder is not fine-tuned at another temperature than the model is used at.

## Pauses: the duration predictor re-fitted to the mean (#72)

#72 planned a discriminator against the pause noise of the table above. The measurement came first (val 100 or 200,
BigVGAN-v2 GTA 40k, T 0.5, mel-domain frame energies), and the noise is not there:

- **Under the recordings' own durations (MAS), the model's pauses are as quiet as the recordings'.** Mel noise floor:
  −59.3 dB teacher-forced and −59.0 dB free-running, against −59.0 dB.
- **With its predicted durations, an utterance holds less silence, not louder silence.**
  - The floor rises to −53.3 dB.
  - 18% of the frames are more than 40 dB below the loudest, against 34% in the recordings.
  - The 20 quietest frames are at −72.3 dB, against −74.0 dB.
  - A percentile floor measures how much of an utterance is silent.
- **The missing silence is at punctuation.** UTMOSv2 does not hear it: the model scores 3.316, its copy-synthesis
  3.314.
- **Nothing for a discriminator to learn**, so #72 changed course.

**The cause.** The duration predictor regresses log-durations (`duration_loss`), so it learns the geometric mean. A
comma is read with no pause by one speaker and a long one by the next, and the geometric mean of such a spread is a
fraction of its mean. Letters vary little, so for them the two means agree.

**The fix.** `drifting-tts train-durations` trains only the duration predictor (0.54 M parameters, the rest frozen)
on the model's own MAS durations. It uses a Poisson loss, minimised at the mean, on the duration minus half a frame:
synthesis rounds every duration up, which adds half a frame on average. Without that offset the ~150 tokens of a
sentence came out 0.8 s long. 4,000 steps on `data/en_pilot`, 3 minutes on the RTX 5090. It was fitted before
`--edges` existed (as `--edges mean`); its sentence edges match the recordings' (lead-in 0.031 s, final 0.105 s
against 0.102 s), unlike the Turkish female voice's ([AR_DRIFT.md](AR_DRIFT.md#durations-re-fitted-to-the-mean)).

Seconds per val utterance (200), by where the tokens are:

| | punctuation | word gaps | letters | edges | total |
|---|---|---|---|---|---|
| recording (MAS) | 1.19 | 0.29 | 4.29 | 0.13 | 5.90 |
| regressor (log-durations) | 0.45 | 0.32 | 4.13 | 0.13 | 5.04 |
| **re-fitted (Poisson)** | **1.09** | **0.29** | **4.15** | 0.14 | **5.66** |

On the 500 held-out utterances (BigVGAN-v2 GTA 40k, as above):

| durations | WER | CER | UTMOSv2 | similarity |
|---|---|---|---|---|
| regressor | 1.35% | 0.60% | 3.245 | 0.824 |
| **re-fitted** | 1.39% | 0.61% | 3.236 | **0.846** |

- **Intelligibility and naturalness are unchanged**: 212 word errors against 207 of 15,302 words. UTMOSv2 differs by
  −0.008 sentence by sentence (95%: −0.034 to +0.017).
- **Closer to the speaker**: similarity +0.022 (95%: +0.019 to +0.025), higher on 398 of 500 sentences. Both rows use
  the same vocoder, so the copy-synthesis reference is the same.
- **Where**: the private English Space offers both and defaults to the re-fitted one. It holds only the predictor
  (`durations_en.pt`, 2.2 MB, `drifting_tts.publish.export_durations`), since the Space stores at most 1 GB.

## Sampled durations and pauses (#100)

ARDrift samples its intonation, but its durations are one number per token (`ceil(exp(logw))`, re-fitted to the
mean in #72): every seed reads a sentence with the same rhythm, and every pause is about as long as the next. The
prosody sampler of the Turkish DiT ([PROSODY_MODEL.md](PROSODY_MODEL.md)) needed a token-pitch predictor, which
ARDrift does not have.

**A duration-only sampler.** `ProsodyNet` with `net.duration_only` (set automatically for a model without token
pitch) holds the pitch channel at zero: the same 8.2 M network, feature maps and drifting loss, on the frozen text
encoder's states, the speaker embedding and the regressor's log-durations. `prosody-cache` stores the MAS durations
alone for such a model. `configs/prosody_drift_ar_en.yaml`: the 3,194 utterances of the vocoder comparison above
(6.0 h, 1,266 speakers), so the sampler is pooled over speakers with two or three utterances each and has one
duration factor for all of them (`calibrate.speakers: all`). 10,000 steps of 16 utterances (2.4 it/s on an A10);
the held-out CRPS stops falling at 6,000.

**Tokens** (dev 300: utterances of 13 s with several sentences; 8 seeds; `scripts/eval_prosody_tokens.py`; a spread
is the within-utterance standard deviation relative to the recordings'):

| durations | letter spread | gap spread | log-duration CRPS | W1 log-duration | W1 gaps | gaps of 0.15 s or more | length | log length between seeds |
|---|---|---|---|---|---|---|---|---|
| recordings | 1 | 1 | 0 | 0 | 0 | 5.36 | 1 | – |
| regressor (re-fitted, #72) | 0.83 | 0.92 | 0.456 | 0.204 | 0.088 | 6.03 | 0.95 | 0 |
| sampler, T 1 | 0.99 | 0.99 | 0.282 | 0.024 | 0.033 | 4.99 | 0.98 | 0.13 |
| sampler, T 0.5 | 0.94 | 0.97 | 0.306 | 0.045 | 0.068 | 4.26 | 0.92 | 0.08 |

The gaps between words in seconds: mean (standard deviation; 10th to 90th percentile):

| durations | between words | at a comma | at a sentence end |
|---|---|---|---|
| recordings | 0.071 (0.067) | 0.34 (0.43; 0.06 to 0.83) | 1.11 (1.44; 0.10 to 2.51) |
| regressor | 0.067 (0.028) | 0.28 (0.18; 0.13 to 0.48) | 0.85 (0.58; 0.27 to 1.67) |
| sampler, T 1 | 0.068 (0.043) | 0.32 (0.46; 0.06 to 0.77) | 0.97 (1.05; 0.11 to 2.36) |
| sampler, T 0.5 | 0.064 (0.034) | 0.27 (0.35; 0.06 to 0.70) | 0.92 (0.82; 0.12 to 2.08) |

- **The regressor reads every comma alike**, 0.13 to 0.48 s, where the recordings go from no pause (0.06 s) to
  0.83 s. The sampler has both ends, and the recordings' letter rhythm (0.99 against 0.83).
- **It is the better forecast, not only the more varied one**: the CRPS, which a deterministic forecast pays as its
  absolute error, falls from 0.456 to 0.282.

**Audio** (val 100, the released vocoder, model T 0.5; a pause is a silent run of 0.1 s or more inside the
utterance; two seeds per sentence; WER by Whisper large-v3; the judges of
[HEAD_REALISM.md](HEAD_REALISM.md#a-panel-of-judges-where-the-audible-gap-is), `*`: the paired 95% interval against
the regressor excludes 0):

| durations | pauses per utterance | pause, s (std) | length, s | length between seeds | WER | UTMOSv2 | UTMOS22 | SCOREQ | NISQA | Audiobox PQ |
|---|---|---|---|---|---|---|---|---|---|---|
| recording | 1.81 | 1.05 (1.03) | 5.61 | – | 2.44% | 3.27 | 4.20 | 4.28 | 4.63 | 7.49 |
| regressor (released) | 2.49 | 0.69 (0.68) | 5.35 | 0% | 2.37% | 3.38 | 4.13 | 4.25 | 4.63 | 7.51 |
| **pauses sampled, T 0.5** (the checkpoint's default) | **1.83** | **1.08 (0.96)** | **5.58** | 12% | 2.30% | 3.40 | 4.12 | 4.23 | 4.62 | 7.50 |
| pauses sampled, T 1 | 1.89 | 1.23 (1.33) | 6.00 | 19% | 2.71% | 3.24* | 4.08* | 4.15* | 4.57* | 7.48 |
| all sampled: letters 0.3, pauses 0.5 | 1.72 | 1.14 (0.95) | 5.55 | 12% | 2.51% | 3.35 | 4.12 | 4.22 | 4.61 | 7.50 |
| all sampled, T 1 | 1.83 | 1.27 (1.33) | 6.15 | 20% | 4.94% | 3.24* | 4.01* | 4.06* | 4.59 | 7.45* |

DNSMOS (3.29 to 3.33) and SQUIM (4.32 to 4.37) do not separate the rows. All rows use the factor 1.046 measured at
T 0.5, which makes the T 1 rows 7 to 10% longer than the recordings.

- **Sampled pauses at T 0.5 give the recordings' pauses.** 1.83 pauses of 1.08 s per utterance against the
  recordings' 1.81 of 1.05 s; the regressor makes 2.49 of 0.69 s. One word error fewer than the regressor (34
  against 35 of 1,477; 95% of the difference: -7 to +4), no judge lower. Two seeds of a sentence differ by 0.3
  pauses, 0.67 s of silence and 12% in length; the regressor's are the same.
- **On all 500 held-out utterances** (the long ones of dev included; pauses of 0.15 s or more): 3.07 pauses and
  2.57 s of silence per utterance against the recordings' 3.03 and 2.82 s (regressor 3.87 and 2.22 s), and 233 word
  errors against the regressor's 208 of 15,302 (WER 1.52% against 1.36%; +25, 95%: +1 to +49; worse on 48
  utterances, better on 28). In the 24 utterances that lose most, 15 of the 38 extra errors are Whisper's other
  spelling of the same words (`gonna` / *going to*, `kinda` / *kind of*), 17 a filler or short word dropped or
  added next to a pause (`um`, `uh`, `and`, `i`), and 6 another word (`core` / *cool*, `onto` / *under*).
- **Sampled letters at full temperature cost words**, as they did for the Turkish DiT's low-data voices
  ([PROSODY_MODEL.md](PROSODY_MODEL.md)): 73 errors against 35 (+38; 95%: +20 to +57; worse on 26 sentences, better
  on 3), with dropped and doubled words, and 0.12 UTMOS22 and 0.19 SCOREQ. At noise temperature 0.3, the plan of
  #100, they cost nothing measurable on these 100 sentences (37 errors, no judge lower), but on the 500 held-out
  utterances 66 word errors (WER 1.79% against 1.36%; 95%: +35 to +99), against 25 for the pauses alone. So the
  checkpoint's default keeps the model's own letters.
- **Pauses at T 1 are too long for the judges, not for the data.** They follow the corpus (sentence ends of up to
  2.5 s), but the predicted MOS falls with the silence itself: cutting every pause to 0.15 s moves the
  *recordings* by +0.19 SCOREQ, +0.04 UTMOS22 and -0.08 Audiobox PQ, and the regressor's audio alike.
- **No judge rewards the rhythm.** The panel separates one vocoder from another by 0.7 UTMOS22; the recordings'
  pauses in place of the regressor's move it by 0.02. These predictors hear the signal, not the timing: what this
  section gains is measured against the recordings (pauses, spreads, CRPS) and between seeds, not by a predicted
  MOS.

**In the pipeline.** The checkpoint stores its operating point (`calibrate.temperature`, `calibrate.durations:
pauses`, the factor), so `Synthesizer(model, prosody="runs/prosody_ar_en/prosody_ema.pt")` samples the pauses at
0.5 and keeps the letters; `prosody_durations="sampled"`, `prosody_temperature` and
`prosody_duration_temperature` override it. ARDrift's streaming path draws the same durations as one pass (it
ignored a prosody predictor before).

**Found on the way.** The helpers that tell letters from the gaps between words (`boundary_tokens`, `word_index`,
`floor_letters`, `pause_durations`) looked up the space and the punctuation in the Turkish symbol table for every
language; the English frontend has other ids for them. They take the model's language now.

## Voices from a prompt (#95)

A voice was a row of two speaker tables: a new one took a fine-tune of 30-40k steps. Kyutai's Pocket TTS, whose
recipe ARDrift follows, has no speaker encoder: the reference audio is a prompt in front of the sequence. The same
here, on mel patches.

**Training** (`model.ar.prompt`, `drift.prompt`; `train_ar.cut_prompts`). An utterance is cut where a word starts,
between 2 and 6 s and at least 1.5 s before its end; the model's own alignment gives the frame. The frames before
the cut, less up to 0.5 s of their end, are the prompt: history that the backbone reads without a text condition.
The tokens and frames after the cut are the utterance that is trained, and the loss is taken there. Kyutai's
loader on why the cut is at a word: "Without a boundary to cut at, the prompt would be a window of the target itself
and the model would learn to copy it instead of reading the text." For half of the prompted utterances the speaker
row is a new one, "the voice is in the prompt", and the text encoder and the duration predictor get a vector pooled
from the prompt's patches instead of a table row. In this corpus 60% of the utterances of a batch get a prompt
(`prob: 0.85`; the others are too short or keep the old path) and 30% lose their speaker row.

**Inference.** `tts(text, ref_audio="voice.wav")`: the first 6 s of the recording, prepared as the training data
(silence trimmed, loudness normalised), no transcript. The prompt stays in front of every unit of a call,
`tts.reference(...)` prepares a voice once, and streaming draws the same audio.

**A pilot.** The released model fine-tuned for 8,000 steps of 16 utterances at 1e-4, streamed from the corpus
(`configs/ar_tts_en_prompt.yaml`; 1.5 it/s on an A10, 90 minutes). The released checkpoint does not hold its learned
temperature: tau alone, optimised on its loss with the model frozen, settles at 0.056-0.065, so the run starts at
0.06 (it ends at 0.075).

**Speakers never heard** (LibriTTS-R test-clean, 24 kHz recordings: 36 speakers, one prompt recording of 5-9 s and
the texts of three other utterances each, 108 items; `scripts/clone_eval.py`). SIM-o: the cosine between the
WavLM-ECAPA embeddings of the generated audio and of the prompt recording; WER by Whisper large-v3:

| system | SIM-o | 95% | WER |
|---|---|---|---|
| the speaker's own recording of the target | 0.656 | 0.636-0.675 | 2.35% |
| another speaker's recording | 0.153 | 0.131-0.175 | – |
| released model, the nearest training speaker (of 1,000 rows, by embedding) | 0.308 | 0.290-0.326 | 3.73% |
| ... which continues the prompt under its transcript (no training) | 0.415 | 0.395-0.434 | 6.72% |
| prompt, 2,000 steps | 0.381 | 0.357-0.404 | 4.85% |
| prompt, 4,000 steps | 0.425 | 0.402-0.446 | 5.05% |
| prompt, 6,000 steps | 0.427 | 0.405-0.448 | 4.90% |
| **prompt, 8,000 steps** | **0.435** | 0.414-0.454 | 3.77% |
| ... the first 3 s of the recording | 0.402 | 0.380-0.421 | 4.36% |
| ... the whole recording (5-9 s) | 0.442 | 0.422-0.461 | 3.43% |

**Voices of the corpus** (40 held-out recordings of training speakers as prompts, three held-out texts each, 120
items; "by its row" is the speaker ID, what the model did before):

| system | SIM-o | 95% | WER |
|---|---|---|---|
| released model, by its row | 0.768 | 0.754-0.782 | 1.35% |
| fine-tuned, by its row | 0.741 | 0.725-0.756 | 1.64% |
| prompt, 2,000 / 4,000 / 6,000 steps | 0.645 / 0.667 / 0.673 | | 1.93% / 2.09% / 2.63% |
| **prompt, 8,000 steps** | **0.680** | 0.659-0.699 | 1.76% |

- **The prompt carries the voice.** For the corpus's voices a recording of 6 s gives 0.68 without their row, against
  0.74 with it: 92% of what the speaker table holds, after 8,000 steps, at the same WER (+3 word errors of 2,437).
- **On speakers never heard it is the best of the systems here and far from the recordings**: 0.435 against 0.656
  for the speaker's own recording and 0.153 for another speaker's. It is 0.13 above the nearest training speaker
  (higher on 89% of the items) at the same WER, and 0.02 above that speaker continuing the prompt under its
  transcript (95%: +0.002 to +0.039), which costs 60 word errors more of 2,040 and needs the transcript.
- **Longer prompts help**: +0.04 from 3 s to the whole recording (higher on 80% of the items), and fewer word
  errors. Between speakers the similarity goes from 0.17 to 0.61 (median 0.44).
- **The curve is flat after 4,000 steps on unseen voices** (0.425, 0.427, 0.435) and still rising slowly on the
  corpus's. More steps of this recipe on this data will not close the gap: the model has heard 4,000 synthetic
  voices of one generator and no recording (#104; recorded speech in the stream, below, adds 0.10). #95 cites 0.66 for F5-TTS on LibriSpeech-PC, a system trained
  on recorded speech; the test set here is another, so only the rows of one table compare.
- **What the pilot costs the old path**: the speaker rows lose 0.03 similarity (0.768 to 0.741) and make 7 word
  errors more of 2,437 (inside its interval). The duration predictor is trained with the log loss again, which the release
  re-fitted to the mean (#72): `train-durations` has to follow a fine-tune, and it does not know prompts yet.

**Recorded speech in the stream (#104).** The gap on unseen voices is the data's: the model had heard 4,000
synthetic voices and no recording. `data.stream.extra` mixes another corpus into the stream; with `speakers:
prompt` its speakers have no row of the table and every one of its utterances is trained through a prompt (cut
whenever it can be; too short, it is the prompted row without a prompt). `configs/ar_tts_en_prompt_recorded.yaml`:
LibriTTS-R train-clean (CC BY 4.0; 245 h, 1,151 speakers, none of the test's) as a quarter of the shards, half of
the utterances, for 6,000 more steps from the pilot above (1.9 it/s, 53 minutes):

| model | unseen recorded speakers: SIM-o | WER | corpus voices, prompt | by its row |
|---|---|---|---|---|
| the speaker's own recording | 0.656 | 2.35% | – | – |
| synthetic corpus only, 8,000 steps | 0.435 | 3.77% | 0.680 | 0.741 |
| **+ recorded speech, 6,000 steps** | **0.532** | 3.77% | 0.678 | 0.731 |

- **Recorded speech is what was missing**: +0.097 on speakers never heard (95%: +0.082 to +0.113; higher on 88% of
  the items) with the same word errors, and nothing lost on the corpus's own voices (prompt -0.002, row -0.009).
  The worst speaker goes from 0.17 to 0.37, the median from 0.44 to 0.54.
- **0.12 is left to the speaker's own recording** (0.532 against 0.656), after 53 minutes on 245 h. The test's
  vocoder was fine-tuned on the model's mels of synthetic voices only, and neither the long run of #95 (40-100k
  steps) nor more recorded speakers (LibriTTS-R has another 500 h) were tried.

Not done: the per-voice pause tables and the prosody sampler fall back to the pooled defaults for a prompted
voice; the benchmark has no `--ref-audio` yet; voice guidance is in [OBJECTIVE_PILOTS.md](OBJECTIVE_PILOTS.md#guidance-99).

### A reference from another utterance (#96)

A prompt is the beginning of the recording that the model continues: it carries the recording's conditions and has
to stand directly in front of the output. The second path gives the model the voice alone: a crop of **another**
utterance of the speaker, no transcript, in front of the sequence.

- **Data.** A shard of the 10,000 h corpus holds one or two utterances per speaker, so every loader worker keeps
  one utterance of each speaker it has read and pairs across shards (`drift.reference`: a crop of 3–6 s, never the
  target; a prepared dataset pairs within itself). Recorded speech (#104) comes in speaker order and pairs within
  a shard.
- **Training.** Prompts and references train together: a prompt for 40% of the utterances that can be cut, a
  reference for 85% of those that have one, the speaker row dropped for half of each. A voice without a row always
  has one of the two.
- **Inference.** `tts.reference("voice.wav", mode=...)`: `prefix` (the first 6 s as a prompt), `reference` (the
  recording through the reference path) or `both`.

**The first design did not train.** `kind: resampler`: four transformer layers over the reference's patches and 32
learned queries (a Perceiver resampler, the conditioning encoder of [XTTS](https://arxiv.org/abs/2406.04904)), whose
tokens stand in front of the backbone's sequence. Two fine-tunes of 6,000 steps from the model with recorded
speech, on the 36 unseen recorded speakers (SIM-o with the prompt recording):

| run | prompt | reference alone | both |
|---|---|---|---|
| control: the prompted model, 6,000 more steps without a reference path | 0.538 | – | – |
| resampler; the reference alone trains on a tenth of the utterances | 0.546 | 0.234 | 0.546 |
| resampler; the reference alone trains on 45% of the utterances | 0.546 | 0.224 | 0.542 |
| another speaker's recording (the floor) | 0.153 | | |

- **The reference tokens carried almost no voice**, with a tenth or with nearly half of the utterances depending on
  them alone. (In the first run a voice without a speaker row was cut into prompt and target whenever it could be,
  so recorded speech never met a reference without a prompt; that was fixed, and it was not the cause.)
- **Why.** Training is teacher-forced: after a few patches the backbone reads the voice from the utterance's own
  recorded history. Only the first fraction of a second needs the reference, so little gradient reaches tokens
  that start as noise. A prompt has no such problem: it *is* history.
- **More prompt training has stopped helping**: 0.532 → 0.538 with 6,000 more steps.

**The second design starts where the prompt is.** `kind: patches` (the layout of VoxCPM2's REF tokens): the
reference's own mel patches stand in front of the sequence. Every token is what a prompt's position looks like to
the backbone (its input projection, no text, the row of a voice that is not in the table), plus a segment embedding
and the correction of two transformer layers whose output starts at zero. The text encoder's vector is the
prompt's pooled one, corrected the same way. Shorter references and voices without one are padded in front with a
learned null token, so nothing needs a mask.

Before any training, on six of the test's speakers, the reference alone already gives the prompt's similarity (0.52
against 0.52). After 6,000 steps (`configs/ar_tts_en_reference.yaml`, 2.1 it/s on an A10, 48 minutes; references of
3–6 s), on the 36 unseen recorded speakers, with a reference of 6 s:

| voice given as | SIM-o (the prompt recording) | SIM with a recording the model was not given | WER |
|---|---|---|---|
| the speaker's own recording of the sentence | 0.656 | 0.650 | 2.35% |
| control: the prompted model, 6,000 more steps | 0.538 | – | 3.73% |
| prompt (6 s of the recording) | 0.546 | 0.462 | 3.63% |
| reference alone (the same recording) | 0.431 | 0.387 | 2.60% |
| prompt + reference (the same recording) | 0.555 | 0.468 | 3.43% |
| reference alone, another recording of the speaker | 0.373 | 0.381 | 2.50% |
| **prompt + reference from another recording** | 0.553 | **0.483** | 3.73% |

SIM-o is measured against the recording the prompt was cut from, and a prompt is continued: a part of its 0.546 is
that recording's conditions. The third column compares with a recording of the speaker that the model was never
given (`scripts/clone_eval.py`): the voice alone.

- **A second recording adds to the prompt.** On the voice alone, the prompt and a reference from another recording
  are 0.021 above the prompt (95%: +0.012 to +0.031, higher on 67% of the items); with the same recording through
  both paths the gain is 0.006 and inside its interval. VoxCPM2 reports the same order: reference alone,
  continuation alone, both.
- **The reference alone carries the voice now** (0.43 on SIM-o, where the resampler gave 0.23), but less of it than
  the prompt: 0.075 lower on the voice alone. It makes a quarter fewer word errors (2.5–2.6% against 3.6%): nothing
  has to be continued.
- **0.084 of the prompt's SIM-o is the recording, not the voice** (0.546 against 0.462 with an unheard recording);
  the speaker's own recording loses 0.006 by the same change.
- **A 10 s reference** (longer than in training) gives the same picture: reference alone 0.443, both 0.559
  (+0.013 over the prompt, 95% +0.006 to +0.019).
- **F0 spread is the guard**: 3.0–3.1 semitones for every path, the recordings' 4.2. No path flattens it further.
- **0.17 is left to the speaker's own recording** on the voice alone (0.483 against 0.650).

On the corpus's own 40 voices (120 sentences):

| model | by its row | prompt | reference alone | both |
|---|---|---|---|---|
| control: the prompted model, 6,000 more steps | 0.731 (WER 1.60%) | 0.685 (1.85%) | – | – |
| resampler (34 of the 40 voices) | 0.708 (1.49%) | 0.665 (2.21%) | 0.401 (1.70%) | 0.670 (2.00%) |
| **patch reference** | 0.722 (1.03%) | 0.688 (1.35%) | **0.667** (1.40%) | **0.701** (1.31%) |

- **On voices of the training corpus the reference alone is almost the prompt** (0.667 against 0.688), and both
  together are again 0.013 above the prompt. The speaker rows keep their voices (0.722 against 0.731, inside the
  interval), with fewer word errors in every column.

`runs/ar_en_reference3` is this model: `Synthesizer(model, "cuda", vocoder=...)(text, ref_audio="voice.wav")` uses
both paths; `tts.reference("voice.wav", mode="reference")` the reference alone.

**With voice guidance the reference alone passes the prompt** (#99,
[OBJECTIVE_PILOTS.md](OBJECTIVE_PILOTS.md#guidance-99)): 4,000 more steps with voice dropout, then guidance with
a null branch without the voice. At a scale of 3: SIM-o 0.563 and 0.466 with an unheard recording (the prompt:
0.546 and 0.464); at 5: 0.593 and 0.485, WER 3.1%. `runs/ar_en_voice` is that model; `tts.guidance = "voice"`
and `cfg_scale` select it.

Not done: the long run (the prompt path stopped rising at 0.54 here; the reference path was still new at 6,000
steps), references longer than the 6 s it was trained on, a reference through a noisy channel (where a prompt would
copy the noise and a reference need not), and the speaker-verification and CTC losses through the head that #96
lists as a later step.
## ARDrift v2: the text as a prefix (#103)

ARDrift is a Grad-TTS front end with an autoregressive back end: monotonic alignment, a duration predictor, a prior
mean, and a head that draws a residual over that mean. Its rhythm is fixed before the first patch is drawn, which is
why a separate duration sampler had to be added (#100). Pocket TTS, CALM, DiTAR and VoxCPM give the text as tokens
and let the autoregressive model decide the timing and the end of the utterance.

**The model** (`model.kind: ar_prefix`, `drifting_tts/models/ar_prefix.py`; `train-ar` trains it):

| | aligned ARDrift | v2 |
|---|---|---|
| text | character encoder, monotonic alignment, the aligned states as a condition of every step | the characters as embeddings in front of the sequence |
| timing | duration predictor (and the sampler of #100) | the model's own inside the utterance; with the clock (below) the tempo is given and the utterance ends with its text, without it the end head ends it |
| what the head draws | a residual over the aligned prior mean | the patch itself |
| voice | a row of the speaker table, a prompt (#95), a reference encoder (#96) | a row, or a crop of another utterance of the speaker in front of the text |
| losses | drifting, prior, duration | drifting, end of utterance (binary cross-entropy, weight 0.1) |

- **The sequence is Kyutai's**: `[voice] [text] [audio]`, one causal transformer over all of it.
- **Batches are padded in front**, so the audio starts at the same position in every row. Rotary positions are
  relative: a row computed alone and in a batch gives the same states (tested).
- **The voice prompt needs no cut.** The aligned model cuts an utterance at a word, which takes an alignment; here
  the prompt is the loader's crop of *another* utterance of the speaker (#96's pairs).
- **A bound on the length**: 0.35 s per character, should the end head not fire.
- **Not there yet**: guidance at inference (the text dropout that trains its null branch is), the fast generator
  with CUDA graphs, streaming, the `Synthesizer`'s units and pauses. `model.synthesize` speaks one utterance.

### The pilot: what makes a prefix model start to read

Three arms on an A10, each from the released model's backbone and head (the character and segment embeddings and
the end head are new), batches of 12,000 frames (about 12 utterances; the aligned model's: 48), learning rate
2e-4. Two measures on held-out sentences:

- **WER** of 24 sentences, free-running (Whisper large-v3);
- **text use**: teacher-forced on 24 to 40 utterances, how much larger the error of the head's mean patch is when
  the text is another utterance's (the same noise for both; 0%: the states do not depend on the text).
  `text_gap / null_steps` in the training log is the same thing on the utterances that lost their text.

| arm | steps | text use (95%) | first eight patches | error of the mean patch, held out | WER, to the length bound | output / recording, length |
|---|---|---|---|---|---|---|
| plain prefix | 5,000 | about 0% | | 0.052 | 98% | 4.6 |
| | 10,000 | +2.4% (+0.3 to +4.4) | +11% | 0.050 | 112% | 4.3 |
| pre-net (256 wide, dropout 0.5) | 5,000 | +1.2% (+0.6 to +1.8) | +2% | 0.116 | 97% | 4.6 |
| | 7,500 | +2.4% (+1.4 to +3.5) | +4% | | | |
| **the text on the clock** | 2,500 | +0.9% (+0.2 to +1.5) | +3% | 0.061 | 97% | 4.6 |
| | 5,000 | +3.2% (+1.8 to +4.8) | +8% | 0.052 | 95% | 4.6 |
| | 7,500 | +8.0% (+5.7 to +10.4) | +16% | 0.049 | 150% | 4.1 |
| | 10,000 | **+13.7%** (+10.5 to +17.1) | +28% | 0.047 | 110% | 4.3 |
| | 12,500 | +26.9% (+21.0 to +32.6) | +40% | 0.045 | 123% | 3.6 |
| | 15,000 | +45.3% (+36.5 to +54.2) | +51% | 0.042 | 114% | 4.2 |
| | 17,500 | +60.8% (+49.0 to +73.6) | +60% | 0.040 | 82% | 4.1 |
| | 20,000 | **+69.5%** (+56.6 to +83.2) | +63% | 0.039 | 88% | 4.3 |

The WER of this table is the training loop's: 24 sentences at temperature 0.5 up to the bound of 0.35 s per
character, which every output reached (the last column), so most of it is what was said after the text.

- **A plain prefix model is an audio model for its first 10,000 steps.** Its states do not depend on the text at
  5,000 steps and barely at 10,000, and what it reads first is the beginning of the utterance. Its end head reads
  the end from the audio (it fires at the right step with another utterance's text as often as with its own), so
  free-running at temperature 0.5 there is no end: every output runs into the length bound.
- **A pre-net** on the history (`model.ar.prenet`; Tacotron 2: "essential for learning attention"; MELLE keeps it,
  dropout 0.5, at inference too) is a quarter faster: at 7,500 steps where the plain model is at 10,000, with twice
  the error of the mean patch, since half of the history is dropped. A fresh pre-net needs the temperature of a
  run from scratch (`drift.tau_init=1.0`): at the fine-tuning value 0.06 the head's spread collapsed (0.04 against
  0.2) for the first 1,000 steps.
- **The clock** (`model.ar.clock`) is the lever: at 10,000 steps the text counts 5.7 times as much as in the plain
  model, and it is the only arm whose error falls below the plain model's (0.047 against 0.050). The text is no
  longer a prefix to search by content: character `j` stands at rotary position `(j + 0.5) * rate - lead` and
  audio step `t` at `t`, so a character is near the steps that say it. It is the diagonal prior of
  attention-based TTS (DC-TTS's guided attention, the beta-binomial prior of RAD-TTS and T5-TTS), given as
  positions. The price: the utterance's tempo (`rate`, patches per character) is an input at inference, the
  corpus's mean (1.57) by default; the rhythm inside the utterance stays the model's.
- **The clock model begins to speak its text.** Its utterances end where the text ends on the clock (`generate`
  stops after `characters × rate` patches; the end head fires by itself in 3 of 100). 100 held-out sentences at
  20,000 steps, at the corpus's mean tempo: **WER 67.6%** (95% 63.1 to 71.8; 66.2% at every recording's own
  tempo); 8 sentences at 15,000 and 17,500 steps: 88% and 83%.
  - The first 3.5 words of a sentence are right on average (of 14.8); 2 sentences in 100 entirely, 8 with fewer
    than a fifth of the words wrong, 24 with fewer than half. Sentences of up to eight words: 53%.
  - It reads from the start of the utterance onwards, and what it loses the text at is mostly a pause: after
    "um", "so", "oh" and a comma the rest is often not speech.
  - Teacher-forced it is further than free-running: the text counts for 70% of the error there.
- **The plain and the pre-net models do not speak the text** in the steps they had (10,000 and 7,500), at
  temperature 0.5 or at 1.0 ([below](#the-continuation-20000-more-steps-and-the-temperature)).
- **The scale of the published models**: MELLE (mel frames, the text as a prefix, no alignment) trains 400,000
  updates of 480,000 frames on 16 GPUs with 50,000 h; CALM's TTS has 88,000 h. 20,000 steps of 12,000 frames are
  0.13% of MELLE's frames, and 2.5% of the utterances the aligned model saw (200,000 steps of 48).

### The continuation: 20,000 more steps, and the temperature

The clock arm went on for 20,000 more steps (40,000 in all; the same batch and learning rate), and its 100
held-out sentences were then read at four noise temperatures. Every free-running number above is at 0.5, the
aligned model's temperature.

| steps | temperature | WER (95%) | CER | quiet share of the output | words transcribed, of the text's | of them the text's | sentences right / under 20% / under 50% WER | UTMOSv2 |
|---|---|---|---|---|---|---|---|---|
| 20,000 | 0.5 | 67.6% (63.1 to 71.8) | 64.1% | 66% | 42% | 79% | 2 / 8 / 24 | 2.78 |
| | 1.0 | 70.9% (66.5 to 75.6) | 51.1% | 39% | 100% | 45% | 0 / 1 / 18 | 2.95 |
| 40,000 | 0.5 | 55.1% (49.4 to 60.8) | 55.1% | 64% | 48% | 94% | 7 / 17 / 49 | 2.84 |
| | 0.75 | 44.1% (38.7 to 49.8) | 40.7% | 55% | 67% | 87% | 7 / 23 / 60 | 3.01 |
| | **1.0** | **35.5%** (31.0 to 39.9) | 29.1% | 44% | 86% | 81% | 9 / 30 / 72 | 3.09 |
| | 1.25 | 43.4% (38.5 to 48.4) | 31.4% | 36% | 103% | 68% | 7 / 25 / 60 | 2.98 |
| recordings | | 2.4% | | 36% | 100% | 98% | 78 / 96 / 100 | 3.27 |
| the aligned model (released) | 0.5 | 2.1% | | 36% | 100% | 98% | 82 / 95 / 100 | 2.91 |

Quiet share: 20 ms frames more than 35 dB under the utterance's loudest one. "Of them the text's": the transcribed
words that are the sentence's words in the sentence's order. UTMOSv2: one crop per utterance.

- **At 0.5 the model falls silent, it does not misread.** At 40,000 steps 94% of what it says is the sentence,
  and it says 48% of it: in 66 sentences of 100 the transcript is the beginning of the sentence and nothing
  else, and 64% of the output is quiet (the recordings: 36%). The aligned model can halve the noise because the
  words are in its prior mean; here the head's draw decides whether the next patch speaks.
- **At the temperature the head is trained with, 1.0, the WER is 35.5%** (19.6 points under 0.5 on the same
  sentences, 95% 12.7 to 26.3): 86% of the words are said, 81% of those right, 72 sentences of 100 under 50% WER
  and 30 under 20%. 0.75 and 1.25 are both worse (+8.6 and +7.9 points): 1.25 says every word and misreads a
  third.
- **The second 20,000 steps halve the WER at 1.0**: 70.9% to 35.5% (35.4 points, 95% 29.3 to 41.6). At 20,000
  steps the temperature only trades silence for wrong words: 67.6% and 70.9% (+3.3 points, 95% -2.3 to +9.4).
- Teacher-forced, text use is +97.0% (95% +90.5 to +103.6, 200 utterances; +69.5% at 20,000 steps), and the
  error of the mean patch on held-out utterances is 0.036 (0.039).
- **The WER in the training loop was at 0.5** (24 sentences, cut at the clock's length): 60.4, 52.4, 54.2 and
  53.5% at 25, 30, 35 and 40 thousand steps. It is flat over the last 10,000 steps, and at 0.5 it measures how
  soon the model falls silent; `configs/ar_prefix_en.yaml` evaluates at 1.0 now.
- The end head fires by itself in 21 utterances of 100 at 1.0 (7 at 0.5); the others end on the clock.
- **The other two arms do not read at 1.0 either** (24 sentences): the transcripts hold 8% (plain, 10,000 steps)
  and 6% (pre-net, 7,500) of the text's words at 0.5, and 10% and 10% at 1.0. The clock model on the same
  sentences: 38% at 0.5 and 47% at 1.0 at 20,000 steps, 73% at 1.0 at 40,000. At 0.5 their outputs are 85% and
  94% quiet; at 1.0 they speak other words (34% and 52% quiet), 1.9 and 1.5 times as many as the text has (WER
  201% and 160%). The plain model's utterances end by themselves at 1.0: 22 of 24, at 0.90 of the recording's
  length (median); at 0.5, 3 of 24.
- At every recording's own tempo the WER is 40.3% at 1.0 (+4.8 points over the mean tempo, 95% -0.2 to +10.0)
  and 53.2% at 0.5 (-1.9, -6.6 to +2.5): the recording's tempo is not better than the corpus's mean.
- UTMOSv2 does not pay for the temperature: 3.09 at 1.0 against 2.84 at 0.5 (the aligned model 2.91, the
  recordings 3.27), on outputs of which 44% is quiet.

The commands. `scripts/prefix_eval.py` makes the rows of the table and the text use; 20,000 steps take 3 h 20 min
on an A10 (16.5 GB):

    drifting-tts train-ar --config configs/ar_prefix_en.yaml --workdir runs/ar_prefix_en \
        train.init_from=checkpoints/en/ardrift_en.pt train.batch_size=16 train.batch_frames=12000 \
        train.lr=0.0002 train.steps=20000 train.warmup=1000 train.save_last=false \
        drift.max_rows=768 drift.tau_init=0.06 drift.reference.max_seconds=4.0
    python scripts/prefix_eval.py render --model runs/ar_prefix_en/model_ema.pt \
        --vocoder checkpoints/en/bigvgan_en.pt --data data/train --out outputs/prefix/t1.0
    python scripts/prefix_eval.py score --data data/train --mos t1.0=outputs/prefix/t1.0
    python scripts/prefix_eval.py text-use --model runs/ar_prefix_en/model_ema.pt --data data/train --num 200

`train.init_from` is the released aligned model for the first 20,000 steps and the run's own `model_ema.pt` for a
continuation (`train.warmup=300`); `render --temperature 0.5` and `--own-tempo` make the other rows.

**What the pilot says about #103's decision.** A prefix model with a drifting head learns to read on one A10
without an aligner, a duration predictor or a prior mean, **if the text stands on the clock**; as a plain prefix
it does not within what a pilot can afford. At 40,000 small steps it says 86% of the words of a sentence and 81%
of those right (WER 35.5% at temperature 1.0), far from the aligned model's 2.1%, and its WER halved over the last
20,000 steps, so the aligned model stays the released one and v2 is a long run to decide on:

- the recipe: `configs/ar_prefix_en.yaml` (the clock is its default now), the aligned model's batch of 48 and
  200,000 steps on the machine that trained it, `text_gap / null_steps` and the in-loop WER at temperature 1.0
  as the signs to watch;
- what the clock leaves of the duration model is **one number per utterance**, the tempo. The pilot speaks at the
  corpus's mean (1.57 patches per character); a model for release needs that number from the text and the voice
  (a regression or a sampler on the prefix), or from the user as a speed control;
- the end of the utterance comes from the clock in the pilot; the end head learned the audio's trailing silence
  and fires on real recordings only.

Not done: the long run; the tempo predictor; the clock and the pre-net together; `fit: global` (one rate for every
utterance in training); guidance, the fast generator, streaming and the `Synthesizer` for this model; Turkish.

## Release

[Vyvo/drifting-tts-en](https://huggingface.co/Vyvo/drifting-tts-en) holds:
- `ardrift_en.pt`: the 200k model with the re-fitted durations, fp32;
- `bigvgan_en.pt`: the BigVGAN-v2 GTA 40k vocoder, fp32;
- `voices.json`: the eight voices, by speaker ID;
- a model card.

The license is CC BY-NC-SA 4.0, since the training speech was generated by a CC-BY-NC-SA-4.0 model.
`scripts/prepare_english_release.py` stages the files and checks them:
- it exports the checkpoints and checks them against their sources (the same weights, the same mel at a fixed seed);
- it scans them for local paths and for the data root's names.

The public demo [Vyvo/drifting-tts-en-demo](https://huggingface.co/spaces/Vyvo/drifting-tts-en-demo) (`space_en_demo/`,
`scripts/deploy_english_demo.sh`) downloads the weights from the model repo. The private English Space stays the place
to compare vocoders and durations.

## Limits

- **Synthetic data**: the model learns EchoTTS's voices and its artefacts; a recorded corpus would be needed to go past
  them.
- **License**: the generating model's CC-BY-NC-SA-4.0 may restrict commercial use of a model trained on this data.
- **Speakers by ID only** (4,000 table rows), no cloning from a reference recording.
- **The test set is from the same synthetic corpus**: texts never seen, voices seen. There is no comparison with
  other English TTS systems on a common benchmark yet.

## Commands

```bash
# pilot data (also the mel statistics, speaker table and val split of the 10k-h run)
drifting-tts prepare --parquet-glob "hf://datasets/SynDataLab-EN/echo-clones-4m-en/data/train-000[0-6][0-9].parquet" \
    --val-glob "hf://datasets/SynDataLab-EN/echo-clones-4m-en/data/train-07960.parquet" --val-size 200 \
    --out data/en_pilot --backend bigvgan --language en --speaker-column speaker_id --workers 24
drifting-tts train-ar --config configs/ar_tts_en.yaml --workdir runs/ar_en_pilot
drifting-tts train-ar --config configs/ar_tts_en_10k.yaml --workdir runs/ar_en_10k     # streamed from the Hub
drifting-tts evaluate --model runs/ar_en_10k/model_ema.pt --split val --num 100 --temperature 0.5 --vocoder vocos-v2
# vocoder (#70): data/en_mel_ft = data/en_mimi_ft's utterances and audio.bin, with BigVGAN log-mels of that audio
drifting-tts finetune-vocoder --config configs/vocoder_bigvgan_en_gta.yaml --workdir runs/en_bigvgan_gta
drifting-tts evaluate --model runs/ar_en_10k/model_ema.pt --split val --num 100 --temperature 0.5 \
    --vocoder runs/en_bigvgan_gta/bigvgan_ft_40000.pt
# GTA mels (#90): teacher-forced against free-running mels under the recording's alignment; a fine-tune that
# draws half of its generated batches free-running; the judges on wav folders of the same sentences
python scripts/gta_mismatch.py --model runs/en_dur/model_poisson.pt --data data/en_pilot --split val --num 200 \
    --temperature 0.5 --out outputs/diag/gta_mismatch_T0.5.json
drifting-tts finetune-vocoder --config configs/vocoder_bigvgan_en_gta.yaml --workdir runs/en_bigvgan_free \
    train.gta_free_prob=0.5 train.steps=5000
python scripts/judge_panel.py --systems recording=wav/recording tf=wav/tf free=wav/free --baseline tf \
    --out outputs/judge/panel
# an external text set (one sentence per line), band-matched as the Turkish benchmark; data scores in English
drifting-tts benchmark --model runs/ar_en_10k/model_ema.pt --vocoder vocos-v2 --texts texts_en.txt --speaker 42
drifting-tts score --data data/en_pilot --scorers asr,rate,mos
# pauses (#72): the duration predictor re-fitted to the mean duration, the rest of the model unchanged
drifting-tts train-durations --model runs/ar_en_10k/model_ema.pt --out runs/en_dur/model_poisson.pt
# sampled pauses (#100): a duration-only drift sampler on the frozen text encoder (a data root with train and dev
# splits); the checkpoint stores its operating point: the pauses sampled at 0.5, the letters the model's own
drifting-tts prosody-cache --model runs/en_dur/model_poisson.pt --data data/en_mel_ft --out runs/pm_cache/targets_en.pt
drifting-tts train-prosody --config configs/prosody_drift_ar_en.yaml --workdir runs/prosody_ar_en
python scripts/eval_prosody_tokens.py --cache runs/pm_cache/targets_en.pt --tts runs/en_dur/model_poisson.pt \
    --prosody sampler=runs/prosody_ar_en/prosody_ema.pt --temperatures 1.0 0.5 --splits dev --out outputs/prosody_ar_en
# voices from a prompt (#95): the released model fine-tuned; SIM-o and WER on a manifest of prompts and targets
drifting-tts train-ar --config configs/ar_tts_en_prompt.yaml --workdir runs/ar_en_prompt
drifting-tts train-ar --config configs/ar_tts_en_prompt_recorded.yaml --workdir runs/ar_en_prompt_rec  # + LibriTTS-R
python scripts/clone_eval.py --model runs/ar_en_prompt/model_ema.pt --vocoder runs/en_bigvgan_gta/bigvgan_ft_40000.pt \
    --manifest data/clone_eval/libritts_r/manifest.json --systems recording other prompt --out outputs/clone/libritts_r
drifting-tts synthesize --model runs/ar_en_prompt/model_ema.pt --ref-audio voice.wav --text "Hello there."
# release: stage, verify and scan, then upload; the public demo downloads from the model repo
python scripts/prepare_english_release.py --model runs/en_dur/model_poisson.pt \
    --vocoder runs/en_bigvgan_gta/bigvgan_ft_40000.pt --data data/en_pilot --out runs/en_release
hf upload Vyvo/drifting-tts-en runs/en_release .
scripts/deploy_english_demo.sh Vyvo/drifting-tts-en-demo
```

```python
tts = Synthesizer("runs/ar_en_10k/model_ema.pt", "cuda", vocoder="vocos-v2")
wav, _ = tts("Hello! It's 7:45, and the 90s called.", speaker="spk_0042", temperature=0.5)
varied = tts.variant(prosody="runs/prosody_ar_en/prosody_ema.pt")   # sampled pauses (#100): another seed, another rhythm
clone = Synthesizer("runs/ar_en_prompt/model_ema.pt", "cuda", vocoder="vocos-v2")   # trained with prompts (#95)
wav, _ = clone("Hello! It's 7:45, and the 90s called.", ref_audio="voice.wav")    # the voice of a recording
```

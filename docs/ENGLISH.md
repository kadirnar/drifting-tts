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
```

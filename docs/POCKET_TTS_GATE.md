# Pocket TTS gate: an autoregressive Turkish Pocket TTS on our protocols (#42)

Issue #42 asks whether Kyutai's Pocket TTS recipe (an autoregressive transformer over continuous Mimi latents with a
one-step head) is worth adopting for Turkish. Before training anything, this page measures the cheapest evidence:

1. a public Turkish Pocket TTS (`kaanhgunay/pocket-tts-tr`) on **Freya-100** with our judges, next to v3.1, with the
   prosody statistics of [PROSODY.md](PROSODY.md) and the CPU speed;
2. the **Mimi codec gate**: held-out recordings encoded and decoded by the codec that model uses, next to the
   recordings and `vocos-ft` copy synthesis.

Nothing was trained or fine-tuned. Recording-derived audio stayed on the machine.

- [Summary](#summary)
- [Setup](#setup)
- [Freya-100](#freya-100)
- [Prosody](#prosody)
- [Speed and size](#speed-and-size)
- [Mimi codec gate](#mimi-codec-gate)
- [Licences](#licences)
- [What this means for #42](#what-this-means-for-42)
- [Caveats and what was not verified](#caveats-and-what-was-not-verified)
- [Reproduce](#reproduce)

## Summary

- **Intelligibility: on par.** pocket-tts-tr: WER 1.54% / 1.43% (8 kHz band) with the two voice prompts, against
  1.10% for v3.1. The paired difference is +0.44 pp [−0.56, +1.52]: not significant.
- **UTMOSv2: pocket-tts-tr is ahead, and its codec explains it.** It scores 2.96 against 2.63 for v3.1 (+0.34
  [+0.27, +0.40]) and 2.71 with the stochastic prosody predictor. On the same studio recordings, though, the codec
  alone scores 3.26 after a Mimi round trip, against 2.74 for `vocos-ft` copy synthesis and 3.09 for the recordings.
  Measured against its own decoder's score on those recordings, v3.1 loses 0.11 and pocket-tts-tr 0.29 (different
  texts, so only a rough comparison). v3.1 through BigVGAN-v2-ft already scores 2.934 on Freya-100
  ([EXPERIMENTS.md](EXPERIMENTS.md#summary-what-worked-what-did-not)). The lead is the decoder's, not the
  autoregressive prosody's. DNSMOS OVRL is level (3.38 / 3.30 against 3.31).
- **Intonation: no wider pitch when it imitates our voice.** Prompted with a 5 s clip of our studio voice, its F0
  std is 3.41 st and its range 10.9 st, against 3.43 / 11.1 for v3.1 and 3.83 / 12.1 for the prosody predictor.
  Pitch movement and micro-variation are the same. With an expressive English prompt it reaches 3.89 / 12.8. The
  prompt sets the pitch range and the tempo, not the autoregressive backbone.
- **What it does differently:**
  - *Phrase breaks.* It pauses inside 61 of the 89 sentences without a comma (1.03 pauses per sentence, 0.31 s
    each). v3.1 never pauses there, and the prosody predictor pauses 0.14 times per sentence (0.13 s each). That is
    twice as often and twice as long as the studio voice it imitates: its recordings pause 0.12 times per second of
    speech, 0.14 s each, even counting the pauses between sentences.
  - *Sentence-final F0 depends on the type.* With the `alba` prompt it ends statements 5.8 st below the median and
    questions 3.4 st below. v3.1 ends every type at −6.5 st, and the prosody predictor at −6.1 st.
  - *Slower speech* with the studio prompt: 5.78 against 6.07 syllables per second.
- **Speaker and bandwidth.** Cloning our 5 s clip gives SIM 0.68 to the studio voice, against 0.895 for v3.1. Its
  output stops at **8 kHz**, consistent with its 16 kHz training corpus; v3.1 reaches 11.5 kHz. Every judge here
  works at 16 kHz and cannot see the missing band.
- **Cost.** 336 M parameters (316 M of them the flow LM) against 81 M (v3.1 plus `vocos-ft`). On the CPU it runs at
  RTF 0.72, about 14× slower than v3.1 on the same CPU (0.052). Its first audio comes after 0.29 s.
- **Mimi codec gate: fine for the studio voice, lossy on real multi-speaker speech.**
  - *Studio voice:* WER 2.17% against 2.09% for the recordings (+0.07 pp, not significant), UTMOSv2 3.26 against
    3.09, SIM 0.907.
  - *Base corpus:* WER 8.66% against 6.63% (+2.0 pp [+1.0, +3.2]), CER +0.8 pp, SIM 0.844. That is the worst
    intelligibility and speaker similarity of every codec measured on Resynthesis-100: BigVGAN-v2-ft 6.77% / 0.966,
    DAC-VAE 6.90% / 0.983, VoxCPM2 7.10% / 0.962.
  - It flattens F0 by about 4% (std 3.62 → 3.46 st on the studio voice).
- **Licence.** The weights are CC BY 4.0, but the training corpus obliges anyone who trains on it to publish open
  weights, and the rights of its recordings are not cleared ([Licences](#licences)).

## Setup

**The community model.** [`kaanhgunay/pocket-tts-tr`](https://huggingface.co/kaanhgunay/pocket-tts-tr) at commit
`e5aa490` (2026-09-02, "v0.1 base"; the repository's SHA-256 sums verified):

| | |
|---|---|
| architecture | Pocket TTS **24-layer teacher** (d 1024): flow LM 316 M parameters + Mimi 20 M; fine-tuned from Kyutai's English 24-layer model (`english_2026-04_24l`, `reset_text_embedding`) |
| head | **LSD** (Lagrangian self-distillation), not drifting (`flow: {type: lsd}` in its training config); 1 decode step at inference |
| text | Turkish SentencePiece BPE, 4,000 tokens; given the original cased, punctuated Freya text (Freya-100 has no digits) |
| training | 112k steps, batch 16 × 4 on one RTX 5090, constant LR 2e-4, text and voice dropout 0.2 |
| training data (card) | "primarily" a public 2,724 h Turkish audiobook read-speech corpus (16 kHz audio); see [Licences](#licences) |
| inference | `pocket-tts` 3.3.0 defaults: temperature 0.3 (a variance: noise std 0.55), EOS threshold −4, 1–3 + 2 frames after EOS. The package has **no latent CFG**; the model's training samples used CFG 2.0 |

The card's `@v0.1-base` revision does not exist on the Hub, so its `config.yaml` fails to load. The run used the same
file with both `@v0.1-base` replaced by the commit hash.

**The codec.** The model's Mimi has 32-d continuous latents at 12.5 Hz, is causal, and runs 24 kHz in and out. Its
decoder side (decoder, decoder transformer, upsampler, latent projection) is bit-identical to Kyutai's public English
model (`kyutai/pocket-tts-without-voice-cloning`). Its encoder could not be compared: Kyutai's copy is in a gated
repository.

**Voice prompts.** Pocket TTS clones a voice prompt; its training used prompts of up to 5 s. Neither prompt comes from
our private data:

- **studio**: a 5.4 s clip synthesised by v3.1 + `vocos-ft` (studio voice, T 0.3, α 2, seed 0) from a neutral
  sentence that is not in Freya-100. It asks for our voice, so timbre and similarity are comparable. It also hands
  the model v3.1's prosody and Vocos' artefacts to imitate.
- **alba**: the first 5 s of Kyutai's `alba-mackenna/casual.wav` (CC BY 4.0, an English-speaking female voice
  actor). This is the voice `pocket-tts` uses by default with community configs. It is cross-lingual and expressive.

**Runtime.** A separate venv: `pocket-tts` 3.3.0 from PyPI, torch 2.14.1+cpu, Python 3.12. It ran on the CPU (AMD
EPYC 9135, shared with other jobs at a load of 4–7 out of a 7.7-core quota), with the package's single intra-op thread
for the backbone plus its Mimi decoder thread. One sample per sentence, `torch.manual_seed(sentence index)`;
re-running gives bit-identical audio.

**Our rows** use the same 100 texts, studio voice 722, T 0.3, α 2, seed = sentence index and `vocos-ft`:
- v3.1 as released;
- v3.1 with the stochastic prosody predictor of #39 (`runs/pm_drift_final/prosody_ema.pt`, prosody temperature 0.5,
  sampled durations).

Their audio is the Freya-100 output of the prosody-model work (`runs/pm_freya100/{v31,final_T0.5}`).

**Judges** (the repo's code, applied to the saved 24 kHz wavs):
- **Freya-100** replicates `drifting-tts benchmark`:
  - Whisper large-v3 (Turkish, beam 5, one utterance at a time) on audio band-matched to 8 kHz, and on the full
    band. Both texts go through the repo's normaliser without punctuation. WER and CER are corpus-level, with 95%
    bootstrap intervals.
  - UTMOSv2 and DNSMOS P.835 on the full band.
  - SIM: WavLM-ECAPA cosine with the centroid of the 100 studio `val` recordings.
- **The codec gate** replicates `scripts/resynthesis_benchmark.score`: batched Whisper (batch 16), SIM with each
  utterance's own recording, UTMOSv2, DNSMOS and bandwidth.
- **Prosody:** `drifting_tts.prosody.prosody_features` (harvest F0, semitones), with silent frames removed (see
  [Prosody](#prosody)).

**Reproduction checks:**

| row | this page | reference |
|---|---|---|
| v3.1 on Freya-100: WER / CER / UTMOSv2 / DNSMOS OVRL | 1.10% / 0.22% / 2.627 / 3.307 | the known row (1.10% / 0.22% / 2.627 / 3.31) |
| base `val` recordings (Resynthesis-100): UTMOSv2 / DNSMOS OVRL | 2.953 / 3.268 | identical in [EXPERIMENTS.md §4](EXPERIMENTS.md#4-audio-vae-latent-spaces) |
| base `val` recordings: WER / CER | 6.63% / 2.19% | 6.56% / 2.18% (not traced; the audio is the same, as UTMOSv2 and DNSMOS show) |
| studio `val`: recording WER (full band), copy-synthesis SIM / UTMOSv2 | 2.06%, 0.969 / 2.737 | 2.06%, 0.969 / 2.738 ([PROSODY.md](PROSODY.md#oracle-prosody-ab-studio-voice)) |
| studio `val` recordings, prosody (raw F0) | 3.68 / 11.9 / 0.228 / 0.63 / 0.42 / 10.9 / 1.39 / 6.22 | identical in PROSODY.md |

## Freya-100

| system | WER 8 kHz [95% CI] | CER 8 kHz | WER full band | UTMOSv2 | DNSMOS OVRL | SIG | BAK | SIM to studio voice | bandwidth | length |
|---|---|---|---|---|---|---|---|---|---|---|
| v3.1 + `vocos-ft` (released pipeline) | 1.10% [0.44, 1.89] | 0.22% | 0.55% | 2.627 | 3.307 | 3.534 | 4.129 | 0.895 | 11.5 kHz | 3.99 s |
| v3.1 + stochastic prosody predictor (T 0.5) | 0.99% [0.33, 1.73] | 0.22% | 0.55% | 2.712 | 3.315 | 3.540 | 4.131 | 0.894 | 11.5 kHz | 3.94 s |
| pocket-tts-tr, studio-voice prompt | 1.54% [0.74, 2.46] | 0.32% | 0.99% | **2.964** | 3.377 | 3.599 | 4.181 | 0.680 | **8.4 kHz** | 4.69 s |
| pocket-tts-tr, `alba` prompt | 1.43% [0.55, 2.49] | 0.30% | 1.32% | 2.948 | 3.300 | 3.538 | 4.121 | 0.359 | **8.0 kHz** | 4.20 s |

SIM scale: a studio recording against the centroid of the other 99 scores 0.953. Paired differences, pocket-tts-tr
(studio prompt) minus v3.1: WER +0.44 pp [−0.56, +1.52], CER +0.10 pp [−0.08, +0.28], UTMOSv2 +0.34 [+0.27, +0.40].
Shared errors: the time "üç buçuğa", transcribed "3.30'a", costs every system a word, and "tek kelimeyle" heard as
"tek kelime ile" costs both pocket-tts-tr rows and the prosody-predictor row. pocket-tts-tr files carry 0.31 s of leading and
0.34 s of trailing silence (v3.1: 0.09 / 0.08 s), which accounts for part of the length difference.

## Prosody

`prosody_features` on harvest F0, overall means. Harvest marks low-level tonal noise in near-silent stretches as
voiced, at spurious pitches up to 10–17 st above the median. In pocket-tts-tr's leading and trailing silence and its
pauses (about −62 dB) that is 8.6–8.8% of its "voiced" frames, against 2.5% for v3.1. The main columns therefore zero F0 on the frames that `silent_frames`
marks as silent (35 dB below the loud level). The `raw` columns are `prosody_features` unchanged.

| system | texts | F0 std | range | F0 CV | move | micro | reversals/s | pauses/utt | pause s | syl/s | raw F0 std | raw range | frames dropped |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.1 + `vocos-ft` (released pipeline) | Freya-100 | 3.43 | 11.1 | 0.202 | 0.64 | 0.44 | 11.1 | 0.00 | – | 6.07 | 3.51 | 11.3 | 2.5% |
| v3.1 + stochastic prosody predictor (T 0.5) | Freya-100 | 3.83 | 12.1 | 0.227 | 0.69 | 0.49 | 10.9 | 0.14 | 0.13 | 6.21 | 3.89 | 12.3 | 3.2% |
| pocket-tts-tr, studio-voice prompt | Freya-100 | 3.41 | 10.9 | 0.212 | 0.63 | 0.45 | 10.4 | 1.03 | 0.31 | 5.78 | 3.90 | 12.9 | 8.6% |
| pocket-tts-tr, `alba` prompt | Freya-100 | 3.89 | 12.8 | 0.234 | 0.63 | 0.47 | 9.9 | 0.59 | 0.20 | 6.53 | 4.15 | 13.7 | 8.8% |
| recording | studio `val` | 3.62 | 11.7 | 0.223 | 0.61 | 0.42 | 10.6 | 1.39 | 0.14 | 6.22 | 3.68 | 11.9 | 2.3% |
| copy synthesis (`vocos-ft`) | studio `val` | 3.64 | 11.7 | 0.224 | 0.67 | 0.48 | 11.3 | 1.46 | 0.14 | 6.22 | 3.69 | 11.8 | 2.6% |
| Mimi round trip | studio `val` | 3.46 | 11.3 | 0.213 | 0.59 | 0.39 | 10.7 | 1.82 | 0.14 | 6.23 | 3.55 | 11.5 | 3.8% |
| recording | base `val` | 4.20 | 13.4 | 0.251 | 0.60 | 0.44 | 9.7 | 2.25 | 0.26 | 5.58 | 4.45 | 14.3 | 6.6% |
| Mimi round trip | base `val` | 4.04 | 13.0 | 0.244 | 0.58 | 0.42 | 9.1 | 2.66 | 0.27 | 5.60 | 4.33 | 13.8 | 8.8% |

By sentence type (Freya-100: 54 statements, 30 questions, 16 exclamations). "Final F0" is the median F0 of the last
25 voiced frames (silent frames removed) against the utterance median, in semitones. It is a coarse measure.

| system | F0 std: . / ? / ! | final F0 (st): . / ? / ! | pauses per sentence: . / ? / ! |
|---|---|---|---|
| v3.1 + `vocos-ft` | 3.40 / 3.41 / 3.57 | −6.5 / −6.6 / −6.5 | 0.00 / 0.00 / 0.00 |
| v3.1 + prosody predictor (T 0.5) | 3.87 / 3.79 / 3.77 | −6.1 / −6.1 / −6.2 | 0.17 / 0.07 / 0.19 |
| pocket-tts-tr, studio prompt | 3.44 / 3.33 / 3.44 | −3.8 / −2.9 / −4.6 | 1.15 / 0.77 / 1.12 |
| pocket-tts-tr, `alba` prompt | 4.04 / 3.65 / 3.86 | −5.8 / −3.4 / −3.7 | 0.81 / 0.23 / 0.50 |

- **Pitch spread follows the prompt.** With our voice as the prompt, pocket-tts-tr has v3.1's spread (3.41 vs 3.43
  st). The prosody predictor of #39 has more (3.83). Movement and micro-variation are the same in all generated rows.
  pocket-tts-tr reverses its pitch slope less often (10.4 / 9.9 per second against 11.1; recordings 10.6), so its
  contours are smoother.
- **Pauses are phrase breaks, not artefacts.** 79 of 103 lie between 25% and 75% of the speech span, with a median
  of 2.1 s of speech before and 1.8 s after. All 11 sentences with a comma get one, and so do 61 of the 89 without.
  The studio recordings pause 0.12 times per second of speech (0.14 s each), and that count includes the pauses
  between their sentences. pocket-tts-tr prompted with that voice pauses 0.25 times per second (0.31 s each) inside
  single sentences.
- **Final contour.** v3.1 and the predictor end every sentence type at the same −6 to −6.5 st, while pocket-tts-tr
  lowers questions less than statements. Turkish polar questions mostly end low as well (the peak sits on the syllable
  before *mI*), so this coarse measure cannot say which contour is right; listening can.
- **Mimi flattens F0 slightly:** −0.16 st of std on both sets, and less micro-variation (0.39 vs 0.42). Its higher
  pause count comes from the level-based detector, since a codec does not add pauses. Mimi lowers the noise floor
  (DNSMOS BAK 4.11 vs 4.06), so more short gaps fall 35 dB below the loud level.

## Speed and size

| system | parameters | device | RTF | first audio |
|---|---|---|---|---|
| pocket-tts-tr (studio prompt) | 336 M (flow LM 316 M + Mimi 20 M) | CPU, 1 backbone thread + Mimi decoder thread | 0.715 (median 0.714) | 0.29 s (p90 0.31 s), streamed per 80 ms frame |
| pocket-tts-tr (`alba` prompt) | | same | 0.741 | 0.29 s |
| v3.1 + `vocos-ft` | 67.7 M + 13.5 M | same CPU, 2 threads, whole sentence | 0.052 | the whole sentence: ≈ 0.21 s for the mean 4 s |
| v3.1 + `vocos-ft` | | RTX 5090, `fast=True`, streaming | – | 12–14 ms ([RESULTS.md](RESULTS.md#latency-and-size)) |

These were measured on a shared CPU (Freya-100, 4.7 s of audio per sentence for pocket-tts-tr); an idle machine would
be faster for both systems. Kyutai's released students have 6 backbone layers instead of this teacher's 24; the
community model has no distilled student yet.

## Mimi codec gate

Recordings encoded to Mimi latents (12.5 Hz × 32 dims) and decoded, on the CPU in fp32. The output lag is 0 samples
(±1 on a few base files), and the output is cut to the input length. Encode and decode RTF are about 0.1 each
(1 thread). The rows compare against the recording and against `vocos-ft` on the dataset's BigVGAN mel of the same
recording.

**Studio voice, the 100 `val` utterances** (12.1 s mean):

| system | WER 8 kHz [95% CI] | CER 8 kHz | WER full band [95% CI] | CER full band | UTMOSv2 | DNSMOS OVRL | SIG | BAK | SIM to recording | bandwidth |
|---|---|---|---|---|---|---|---|---|---|---|
| recording | 2.09% [1.05, 3.64] | 0.94% | 2.06% [1.03, 3.56] | 0.89% | 3.093 | 3.329 | 3.606 | 4.060 | – | 11.5 kHz |
| BigVGAN mel → `vocos-ft` (copy synthesis) | 2.21% [1.16, 3.76] | 0.95% | 2.17% [1.12, 3.69] | 0.91% | 2.737 | 3.315 | 3.567 | 4.096 | 0.969 | 11.5 kHz |
| Mimi encode → decode | 2.17% [1.12, 3.74] | 0.95% | 2.28% [1.24, 3.84] | 0.97% | **3.258** | 3.369 | 3.620 | 4.108 | 0.907 | 11.2 kHz |

**Base corpus, the first 100 `val` utterances** (Resynthesis-100, many speakers, 7.2 s mean):

| system | WER 8 kHz [95% CI] | CER 8 kHz | WER full band [95% CI] | CER full band | UTMOSv2 | DNSMOS OVRL | SIG | BAK | SIM to recording | bandwidth |
|---|---|---|---|---|---|---|---|---|---|---|
| recording | 6.63% [4.88, 8.64] | 2.19% | 6.29% [4.51, 8.27] | 2.07% | 2.953 | 3.268 | 3.569 | 4.015 | – | 11.9 kHz |
| BigVGAN mel → `vocos-ft` (copy synthesis) | 7.85% [5.86, 9.97] | 2.52% | 7.24% [5.23, 9.51] | 2.34% | 2.181 | 3.210 | 3.492 | 4.037 | 0.890 | 12.0 kHz |
| Mimi encode → decode | **8.66%** [6.73, 10.84] | 2.99% | 8.59% [6.62, 10.87] | 2.80% | 2.921 | 3.335 | 3.595 | 4.099 | **0.844** | 11.2 kHz |

Paired, Mimi minus recording: on the studio voice, WER +0.07 pp [−0.19, +0.34] and UTMOSv2 +0.17 [+0.12, +0.21]. On
the base corpus, WER +2.03 pp [+0.97, +3.15], CER +0.79 pp [+0.34, +1.31] and UTMOSv2 −0.03 [−0.10, +0.04].
`vocos-ft` was fine-tuned on the studio voice's GTA mels, which is why its base-corpus row is weak (UTMOSv2 2.18).
That row only shows our vocoder's own limit off its voice.

On the same base recordings ([EXPERIMENTS.md §4](EXPERIMENTS.md#4-audio-vae-latent-spaces)):

| codec | WER 8 kHz | SIM | frames |
|---|---|---|---|
| BigVGAN-v2-ft (mel) | 6.77% | 0.966 | 93.75 Hz × 100 |
| DAC-VAE | 6.90% | 0.983 | 25 Hz × 128 |
| VoxCPM2 | 7.10% | 0.962 | 25 Hz × 64 |
| VoxCPM1.5 | 7.71% | 0.966 | 25 Hz × 64 |
| **Mimi (Pocket TTS)** | **8.66%** | **0.844** | **12.5 Hz × 32** |

**Verdict.** Mimi is adequate for Turkish on a clean single voice. On the studio voice it is transparent for
intelligibility and scores above the recordings on UTMOSv2 and DNSMOS. It is not a Turkish-specific problem but a
rate problem: at 400 numbers per second it is the most compressed latent tested. On real, varied speech it costs
2 pp of WER and much of the speaker identity (SIM 0.84), and it caps the band at about 11 kHz.

## Licences

- **`pocket-tts` code:** MIT (package metadata).
- **Kyutai's Pocket TTS weights:** CC BY 4.0. `kyutai/pocket-tts` is gated; its copy without voice cloning,
  `kyutai/pocket-tts-without-voice-cloning`, is public.
- **`alba` prompt:** CC BY 4.0 (`kyutai/tts-voices`).
- **`kaanhgunay/pocket-tts-tr`:** CC BY 4.0 per its card, "derived from Kyutai Pocket TTS".
- **Its training corpus**, `serdarcaglar/turkish-tts-audiobooks`: gated with automatic approval, licence "other:
  open-weights-with-attribution". These terms are read from the Hub metadata at commit `744ea6a`:
  - cite the corpus;
  - **publish the weights of any model trained on it** under a licence that allows commercial use and
    redistribution, no later than its first use outside the team;
  - **closed-weight or API-only models are not permitted**;
  - these obligations pass on to whoever uses a derived version;
  - **the rights of the source recordings are not cleared**: the user takes the legal responsibility and deletes
    copies on a rightsholder's request.
- **Consequence for #42:** the CC BY 4.0 on the model card does not lift the corpus terms for anyone who trains on
  that corpus. Fine-tuning or distilling from `pocket-tts-tr` inherits at least the attribution duty and the
  uncleared-rights risk. Kyutai's English teacher (CC BY 4.0) is the clean starting point.

## What this means for #42

- **The evidence does not justify changing the architecture for intonation.** Conditioned on our voice, the
  autoregressive teacher has v3.1's pitch spread, and the stochastic prosody predictor already exceeds it.
- **It differs in two places, and neither needs an autoregressive model:**
  - phrase breaks: sampled pauses inside sentences (#39, with the word-level context of #40);
  - sentence-type-dependent endings: the prosody predictor still gives every type the same final fall (#39, #40).
- **Its UTMOSv2 lead belongs to the decoder.** Mimi resynthesis beats `vocos-ft` copy synthesis by 0.52 on the same
  recordings, more than the whole 0.34 gap, and v3.1 with BigVGAN-v2-ft (2.934) nearly matches pocket-tts-tr
  (2.964). Relative to its own decoder's ceiling, pocket-tts-tr loses more than v3.1 (0.29 against 0.11). The
  measured naturalness gap is the vocoder's (the Vocos work, PR #47).
- **The codec gate (the first checkbox of #42) passes for the studio voice and is marginal for multi-speaker use:**
  +2 pp WER, SIM 0.84, about 11 kHz bandwidth.
- **Data and compute (the second checkbox).** This community model is the existence proof: 112k steps at batch 64
  on one RTX 5090, with 2.7k hours of public Turkish speech. Kyutai's README asks for at least 100 h, and 1000+ h for
  a strong model.
- **The price of the recipe:** a 14× slower CPU path, a 0.29 s first audio, and a 6-layer distillation still needed
  for Kyutai's advertised speed.
- **If #42 goes ahead:** start from Kyutai's English teacher rather than from `pocket-tts-tr` ([Licences](#licences)),
  train at 24 kHz to keep the band above 8 kHz, and judge with a listening test. UTMOSv2 is nearly blind to prosody
  and works at 16 kHz.
- **Next cheap step:** a blind listening A/B of pocket-tts-tr against v3.1 + the prosody predictor (naturalness,
  phrasing, question contours), before any training. Samples of all four rows on 12 Freya sentences are kept locally
  in `runs/pg_samples` (generated audio only).

## Caveats and what was not verified

- **One community checkpoint** (v0.1, 112k steps), run without CFG (`pocket-tts` has none), at one temperature
  (0.3), as a teacher without distillation. A better checkpoint or CFG could do better.
- **The voice prompt decides the speaker similarity and the prosodic style.** The studio prompt is our own synthetic
  voice, so it carries v3.1's prosody and Vocos' artefacts. `alba` is an English-speaking female voice actor. A
  natural Turkish prompt from our studio recordings was not used (private data), and it might give pocket-tts-tr
  wider intonation.
- **The judges.** UTMOSv2 and DNSMOS are trained on English, work at 16 kHz (so they cannot see pocket-tts-tr's
  8 kHz band limit) and are nearly blind to prosody. There was no listening test.
- **Prosody numbers** depend on gating harvest's F0 by level (both versions are shown). The final-F0 measure is
  coarse. Freya-100 sentences are single sentences, so no behaviour between sentences was measured.
- **Speed** was measured on a shared CPU.
- **Not checked:**
  - text overlap between Freya-TR-Eval and the audiobook corpus (the corpus is gated and was not downloaded);
  - Mimi's encoder weights against Kyutai's (gated; the decoder side is identical);
  - the card's "beginning-of-generation artefact" (the leading 0.31 s is silence by level, but nobody listened).
- **Our own audio** for the two v3.1 rows was generated by the prosody-model work. It reproduces the known v3.1 row,
  and was not regenerated here.

## Reproduce

Scripts: `scripts/pocket_tts_gate/`. The two that import `pocket_tts` run in their own venv; the rest run in the repo's
environment. Local outputs: `runs/pg_freya_{studio,alba}` (generated audio and `gen.jsonl`), `runs/pg_mimi`
(recording-derived, never published), `runs/pg_judges`, `runs/pg_prosody`, and listening samples in `runs/pg_samples`.

```bash
# 1. isolated environment for the third-party code (CPU torch)
uv venv runs/pocket_venv --python 3.12
VIRTUAL_ENV=runs/pocket_venv uv pip install pocket-tts soundfile --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
# the card's config with its missing tag replaced by the commit
python -c "from huggingface_hub import hf_hub_download as d; print(d('kaanhgunay/pocket-tts-tr', 'config.yaml', \
  revision='e5aa490d9aa6075cf047e4286fb57cecb99aa1ed'))" | xargs sed 's/@v0.1-base/@e5aa490d9aa6075cf047e4286fb57cecb99aa1ed/' \
  > runs/pocket_tts_tr.yaml

# 2. voice prompts: our studio voice (v3.1 + vocos-ft), and the first 5 s of kyutai/tts-voices alba-mackenna/casual.wav
drifting-tts synthesize --model runs/release/drifting_tts_v3.1.pt --vocoder vocos-ft --speaker studio --cfg 2 \
  --temperature 0.3 --seed 0 --text "Bugün hava oldukça güzel ve insanlar parkta yürüyüş yapıyor, çocuklar da bahçede oynuyor." \
  --out runs/pg_prompts/studio_v31_prompt.wav

# 3. Freya-100 with pocket-tts-tr (CPU)
OMP_NUM_THREADS=2 runs/pocket_venv/bin/python scripts/pocket_tts_gate/gen_pocket.py --config runs/pocket_tts_tr.yaml \
  --voice runs/pg_prompts/studio_v31_prompt.wav --num 100 --out runs/pg_freya_studio   # and alba_casual_5s.wav

# 4. codec gate: recordings + vocos-ft copy synthesis, then the Mimi round trip
python scripts/pocket_tts_gate/dump_recordings.py --data data/train --out runs/pg_mimi
runs/pocket_venv/bin/python scripts/pocket_tts_gate/mimi_roundtrip.py --config runs/pocket_tts_tr.yaml --root runs/pg_mimi --threads 1

# 5. judges (sets.json: Freya systems -> wav lists + the studio recordings as the SIM centroid; gate sets -> recording /
#    vocos_ft / mimi) and prosody (prosody_spec.json: one entry per wav set); formats in the scripts' docstrings
python scripts/pocket_tts_gate/judge.py --sets sets.json --out runs/pg_judges
python scripts/pocket_tts_gate/prosody_stats.py --spec prosody_spec.json --out runs/pg_prosody/prosody.json --workers 3
```

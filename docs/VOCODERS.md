# Vocoders

The model writes BigVGAN-style log-mels (24 kHz, 100 bands, n_fft 1024, hop 256, fmax 12 kHz, uncentred frames,
natural log floored at 1e-5). Any vocoder trained on this mel can turn them into audio without retraining.
`Synthesizer(model, vocoder=<name>)`, `drifting-tts synthesize / benchmark --vocoder <name>` and
`scripts/bench_ttfa.py --vocoder <name>` take any name below, or a checkpoint path.

| name | vocoder | weights |
|---|---|---|
| `bigvgan-v2-ft` | BigVGAN-v2 fine-tuned on this model's mels (the release default) | `Vyvo/drifting-tts-tr`, `bigvgan_v2_ft.pt` |
| `bigvgan-v2` | NVIDIA BigVGAN-v2, 24 kHz, 100 bands, 256x | `nvidia/bigvgan_v2_24khz_100band_256x` |
| `bigvgan-v1` | NVIDIA BigVGAN (v1), 24 kHz, 100 bands | `nvidia/bigvgan_24khz_100band` |
| `bigvgan-base` | NVIDIA BigVGAN-base (v1), 24 kHz, 100 bands | `nvidia/bigvgan_base_24khz_100band` |
| `bigvgan-base-ft` | BigVGAN-base fine-tuned on this model's mels (14 M parameters) | `Vyvo/drifting-tts-tr`, `bigvgan_base_ft.pt` |
| `vocos-ft` | Vocos fine-tuned on this model's mels (13.5 M parameters) | `Vyvo/drifting-tts-tr`, `vocos_ft.pt` |
| `vocos-v2` | `vocos-ft` trained further with the second recipe ([below](#training-vocos-further)): the vocoder of release v3.2 (`vocos-ft2` is an alias) | `Vyvo/drifting-tts-tr`, `vocos_v2.pt` |
| `vocos-v2-balanced` | `vocos-v2` trained 20k steps further on speaker-balanced batches ([below](#speaker-balance)): the demo's vocoder for the male and female voices (with v3.2, Freya-495 UTMOSv2 male 2.896 → 2.944, female 2.722 → 2.749) | `Vyvo/drifting-tts-tr`, `vocos_v2_balanced.pt` |
| `griffin-lim` | mel filterbank inverted by non-negative least squares, then 64 iterations of fast Griffin-Lim | none |
| `vocos` | `charactr/vocos-mel-24khz`, for models trained on Vocos's own mels | `charactr/vocos-mel-24khz` |
| `revox` | Minori Live — [Revox Vocoder 1.0](https://huggingface.co/minori-live/revox-vocoder-1) (PC-NSF-Vocos, 48 kHz, 4.5 M parameters) on converted mels, with F0 from the Griffin-Lim audio (`revox:<F0 source>[:dio\|harvest]`). **CC BY-NC-SA 4.0: non-commercial use only.** [Below](#revox-vocoder-10-non-commercial) | `minori-live/revox-vocoder-1`, `vocoder.onnx`, downloaded at runtime |

- **Checkpoints.** A path is recognised by its keys. `{"generator", "repo", "hparams"}` is a BigVGAN-family
  fine-tune (`drifting-tts finetune-vocoder`), built with the code of its NVIDIA `repo`. `{"vocos", "init", "mel":
  "bigvgan", "head_padding": "same"}` is a Vocos fine-tuned on BigVGAN-style mels: with a `same` ISTFT head, frame
  *i* is centred on sample *i* · 256 + 128, as in BigVGAN, and T frames give T · 256 samples.
  `{"decoder", "backend", "target_rate"}` is a fine-tuned VAE decoder (`decoder_ft.pt`, `vocoder.arch:
  vae_decoder`) for a model trained on VAE latents: see [LATENTS.md](LATENTS.md#fine-tuning-the-voxcpm2-decoder-on-generated-latents-27)
  (VoxCPM2) and [LATENTS.md](LATENTS.md#fine-tuning-the-dac-vae-decoder-on-generated-latents-32) (DAC-VAE).
- **The NVIDIA repos** ship the same `bigvgan.py`, and their configs use the same mel as ours (v1: `fmax: 12000`;
  v2: `fmax: null`, i.e. 12 kHz). All three load through `build_bigvgan(repo=...)`. The fused anti-aliased activation
  kernel (`cuda_kernel=True`) is the same snake-beta kernel in v1, v1-base and v2. On real mels its output is within
  62–68 dB SNR of the PyTorch path.
- **Streaming.** Each entry has its own window context (`Vocoder.context`, below). Griffin-Lim cannot stream exactly,
  so `stream()` yields each of its sentences in one piece. With `fast=True`, the fixed-size windows of every neural
  vocoder run as CUDA graphs, and a vocoder whose capture fails falls back to eager windows.

## Comparison on Freya-TR-Eval

All rows vocode the same v3.1 mels of the 495 sentences (protocol below).

- **Fine-tuning matters most.** The fine-tuned BigVGAN-v2 is the most natural vocoder: UTMOSv2 2.94 and DNSMOS
  OVRL 3.32. The same network with NVIDIA's weights is the least natural BigVGAN or Vocos on these mels: UTMOSv2
  2.40 and DNSMOS OVRL 2.81.
- **`bigvgan-base-ft` is the small vocoder to use.** Fine-tuning lifts BigVGAN-base from UTMOSv2 2.79 to 2.91 and
  DNSMOS OVRL from 3.18 to 3.34, the highest OVRL of all rows. That is within 0.03 UTMOSv2 of `bigvgan-v2-ft` with
  one eighth of the parameters, half the vocoder time, first audio after 7.8 ms instead of 12.4 ms, and 16 frames of
  streaming context instead of 32.
- **`vocos-ft` is the fastest.** Its ISTFT head makes the vocoder 20× faster than BigVGAN-v2 (RTF 0.0003), and the
  first audio comes after 4.9 ms, most of it the acoustic model. It is less natural than the BigVGAN fine-tunes
  (UTMOSv2 2.63, DNSMOS OVRL 3.30), with a slightly higher WER (1.56%).
- **WER does not rank vocoders.** Every row lies between 1.20% and 1.71% WER, and the intervals overlap. Even
  Griffin-Lim reaches 1.20%: Whisper on 8 kHz band-matched audio ignores phase artefacts. UTMOSv2 and DNSMOS are what
  separate the vocoders.
- **Griffin-Lim** is intelligible, but it sounds clearly synthetic (UTMOSv2 1.77, DNSMOS P.808 3.47). It is the
  floor that needs no weights.
- **`revox`** (Minori Live's Revox Vocoder 1.0, non-commercial) is the least natural neural vocoder here: UTMOSv2
  2.27 and DNSMOS OVRL 3.10. It needs frame-level F0 and cannot stream; see [below](#revox-vocoder-10-non-commercial).
- **Context.** BigVGAN-v2 needs 28 frames of context for the streamed audio to equal whole-sentence vocoding, v1 needs
  24 and v1-base 16; fine-tuning does not change this. The Vocos backbone sees 3 + 8 × 3 frames, plus 2 for the
  overlap of its ISTFT, so 29 frames are exact; `tests/test_vocoder_registry.py` checks this in float64. `vocos-ft`
  passes 55 dB at 24 frames and levels off at 81–82 dB from 28 frames on.

All rows were measured on an idle GPU, except the `revox` speeds (†). Those were measured on a GPU and CPU shared with
training jobs. Under the same load, `bigvgan-v2-ft` measured RTF 0.0118 and TTFA 32.1 / 39.5 ms, and `griffin-lim`
0.0085 and 50.6 / 40.6 ms: 2–3× their idle figures.

<!-- vocoders:begin -->
| vocoder | parameters | WER | CER | UTMOSv2 | DNSMOS OVRL | DNSMOS P.808 | vocoder RTF | TTFA short | TTFA long |
|---|---|---|---|---|---|---|---|---|---|
| `bigvgan-v2-ft` | 112.4 M | 1.23% [0.82, 1.68] | 0.24% [0.16, 0.33] | 2.935 | 3.324 | 3.960 | 0.0058 | 12.4 ms | 14.0 ms |
| `bigvgan-v2` | 112.4 M | 1.20% [0.81, 1.65] | 0.24% [0.16, 0.34] | 2.398 | 2.810 | 3.779 | 0.0058 | 12.4 ms | 14.0 ms |
| `bigvgan-v1` | 112.4 M | 1.36% [0.91, 1.84] | 0.26% [0.17, 0.36] | 2.756 | 3.169 | 3.962 | 0.0058 | 12.2 ms | 13.7 ms |
| `bigvgan-base-ft` | 14.0 M | 1.33% [0.90, 1.80] | 0.26% [0.17, 0.36] | 2.906 | 3.335 | 3.959 | 0.0031 | 7.8 ms | 9.4 ms |
| `bigvgan-base` | 14.0 M | 1.41% [0.95, 1.91] | 0.25% [0.17, 0.35] | 2.793 | 3.176 | 3.976 | 0.0032 | 7.8 ms | 9.4 ms |
| `vocos-ft` | 13.5 M | 1.56% [1.13, 2.05] | 0.29% [0.21, 0.38] | 2.627 | 3.302 | 3.887 | 0.0003 | 4.9 ms | 6.4 ms |
| `griffin-lim` | 0 | 1.20% [0.82, 1.63] | 0.24% [0.16, 0.34] | 1.772 | 3.127 | 3.465 | 0.0038 | 14.8 ms | 17.3 ms |
| `revox` | 4.5 M | 1.61% [1.17, 2.09] | 0.29% [0.21, 0.40] | 2.270 | 3.100 | 3.813 | 0.0265 † | 84.5 ms † | 177.7 ms † |
| `revox:griffin-lim:harvest` | 4.5 M | 1.71% [1.24, 2.25] | 0.32% [0.22, 0.42] | 2.263 | 3.094 | 3.806 | 0.1769 † | 288.0 ms † | 1035.5 ms † |
| `revox:bigvgan-v2-ft` | 4.5 M | 1.51% [1.06, 1.98] | 0.28% [0.20, 0.37] | 2.260 | 3.094 | 3.815 | 0.0452 † | 121.4 ms † | 288.2 ms † |
| `revox:none` | 4.5 M | 1.51% [1.05, 2.02] | 0.29% [0.19, 0.39] | 2.044 | 2.044 | 3.564 | 0.0108 † | 58.6 ms † | 83.6 ms † |

Streamed against whole-sentence audio, SNR in dB (full fp32), by context in frames on each side of a window:

| vocoder | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 40 | 48 | 64 | > 55 dB from | > 90 dB from | used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bigvgan-v2-ft` | 3.9 | 6.6 | 10.4 | 17.3 | 25.7 | 43.3 | 82.2 | 97.6 | 98.4 | 98.9 | 98.9 | 98.1 | 24 | 28 | **32** |
| `bigvgan-v2` | 6.4 | 11.2 | 16.2 | 22.8 | 34.0 | 48.4 | 72.9 | 96.0 | 95.8 | 96.6 | 96.3 | 96.1 | 24 | 28 | **32** |
| `bigvgan-v1` | 5.5 | 8.8 | 16.4 | 24.1 | 38.0 | 58.7 | 94.2 | 99.3 | 100.1 | 100.2 | 99.5 | 100.0 | 20 | 24 | **24** |
| `bigvgan-base-ft` | 5.8 | 10.3 | 21.6 | 74.0 | 92.5 | 92.4 | 92.6 | 92.6 | 92.6 | 92.5 | 92.4 | 92.4 | 12 | 16 | **16** |
| `bigvgan-base` | 6.2 | 10.2 | 21.7 | 77.2 | 93.8 | 93.5 | 93.8 | 93.5 | 93.7 | 93.6 | 93.8 | 93.9 | 12 | 16 | **16** |
| `vocos-ft` | 2.4 | 5.0 | 8.3 | 13.9 | 20.9 | 35.6 | 62.2 | 82.4 | 80.9 | 81.1 | 81.7 | 81.3 | 24 | – | **32** |

| vocoder | kind | streaming | BigVGAN CUDA kernel | CUDA graphs (`fast=True`) |
|---|---|---|---|---|
| `bigvgan-v2-ft` | bigvgan | 32 frames of context | yes | yes |
| `bigvgan-v2` | bigvgan | 32 frames of context | yes | yes |
| `bigvgan-v1` | bigvgan | 24 frames of context | yes | yes |
| `bigvgan-base-ft` | bigvgan | 16 frames of context | yes | yes |
| `bigvgan-base` | bigvgan | 16 frames of context | yes | yes |
| `vocos-ft` | vocos | 32 frames of context | – | yes |
| `griffin-lim` | griffin-lim | one piece per sentence | – | no (eager) |
| `revox` (all F0 sources) | revox | one piece per sentence | – | no (eager) |
<!-- vocoders:end -->

## Training Vocos further

`vocos-ft` is the fastest vocoder but the least natural fine-tune. Its network stays as it is: the backbone, the
ISTFT head and the 32 frames of streaming context. Only its training changes. The second recipe is
`configs/vocoder_vocos_v2.yaml`, and its result is the registry entry `vocos-v2` (`vocos_v2.pt` on the Hub), the
vocoder of release v3.2.

### Diagnosis: pitch and periodicity

`scripts/vocoder_quality.py` adds pitch and periodicity measures to the judges. The recorded mels of the 100 `val`
utterances of the studio voice (speaker 722) are vocoded and compared with the recordings:

| system | UTMOSv2 | DNSMOS OVRL | VDE | GPE | F0 error | periodicity RMSE | periodicity bias | F0 micro-variation |
|---|---|---|---|---|---|---|---|---|
| recording | 3.093 | 3.329 | – | – | – | – | – | 0.420 st |
| `bigvgan-v2-ft` | 3.174 | 3.358 | 5.1% | 2.64% | 51.9 cents | 0.048 | +0.008 | 0.415 st |
| `vocos-ft` | 2.738 | 3.315 | 6.3% | 5.17% | 88.2 cents | 0.120 | −0.030 | 0.491 st |

- **The gap is in pitch and periodicity.** DNSMOS barely separates the two vocoders, but the pitch measures do.
  - `vocos-ft` misses the recording's F0 by 88 cents RMS, against 52 for BigVGAN-v2-ft.
  - It makes twice as many gross pitch errors.
  - Its periodicity error is 2.5 times BigVGAN-v2-ft's.
- **Rough voiced frames.** Voiced frames are less periodic than in the recording (−0.03), and the F0 jitters more
  from frame to frame (0.49 st against 0.42). This matches the "buzzy" texture heard in listening. Revisiting Vocos
  (arXiv 2607.24323) traces the same problem to the phase that the network predicts.
- **The same on the TTS mels.** On Freya-100, `vocos-ft`'s F0 is 93 cents RMS away from BigVGAN-v2-ft's on the same
  mels, its periodicity RMSE against it is 0.118, and its F0 micro-variation is 0.450 st against 0.395.

### Pilots

Every pilot starts from `vocos-ft`: step 40k of its run, with its generator, discriminators and AdamW states
(`train.init_from`). The data is the same as for `vocos-ft`: v3.1 mels at T = 0.3 under the ground-truth alignment
in 80% of the batches, recorded mels in the rest. Batches hold 16 × 16,384 samples unless stated. P1–P3 decay
their learning rates with a 25k-step cosine and were compared at 10k steps, where the rate is still 69% of its peak.

| pilot | discriminators | losses | learning rate |
|---|---|---|---|
| P0 | `vocos-ft`'s fresh MPD + MRD, hinge | 45 × single-scale log-mel, MRD × 0.1, FM × 1 (the first recipe, continued) | 1e-4, constant |
| P1 | NVIDIA's released BigVGAN-v2 MPD + CQT-D with their AdamW moments, instead of Vocos's | BigVGAN-v2's: LSGAN, FM × 2, 15 × 7-scale log10-mel | G and D 5e-5, cosine; batch 8 |
| P2 | `vocos-ft`'s MPD + MRD | 15 × 7-scale log10-mel, MRD × 1, FM × 2 | G 5e-5, D 1e-4, cosine |
| P3 | as P2 | P2 + 20 × instantaneous-frequency loss | as P2 |

Freya-100 (v3.1 mels, `studio` voice, T = 0.3, α = 2). The last two columns compare each output with
BigVGAN-v2-ft's on the same mels:

| vocoder | WER | CER | UTMOSv2 | DNSMOS OVRL | DNSMOS P.808 | F0 micro-variation | F0 vs BigVGAN-v2-ft | periodicity RMSE vs BigVGAN-v2-ft |
|---|---|---|---|---|---|---|---|---|
| `bigvgan-v2-ft` | 0.66% [0.22, 1.22] | 0.14% | 2.934 | 3.335 | 3.992 | 0.395 st | – | – |
| `vocos-ft` (40k) | 1.10% [0.44, 1.89] | 0.22% | 2.627 | 3.309 | 3.930 | 0.450 st | 92.9 cents | 0.118 |
| P0, 45k | 0.88% | 0.18% | 2.618 | 3.295 | 3.928 | 0.433 st | 95.4 cents | 0.119 |
| P1, 10k | 0.88% [0.33, 1.54] | 0.18% | 2.840 | 3.325 | 4.028 | 0.433 st | 97.4 cents | 0.114 |
| P2, 10k | 0.66% [0.22, 1.21] | 0.14% | 2.917 | 3.348 | 4.067 | 0.408 st | 92.7 cents | 0.111 |
| P3, 10k | 0.77% [0.22, 1.35] | 0.16% | **2.922** | **3.348** | **4.078** | **0.399 st** | **91.8 cents** | **0.111** |

Copy-synthesis of the 100 studio `val` utterances. The recorded mels are as in the diagnosis; GTA mels are the
acoustic model's mels of the same utterances under their ground-truth alignment and pitch. Both are compared with
the recordings:

| vocoder | mels | UTMOSv2 | F0 error | GPE | periodicity RMSE | periodicity bias | F0 micro-variation |
|---|---|---|---|---|---|---|---|
| `bigvgan-v2-ft` | recorded | 3.174 | 51.9 cents | 2.64% | 0.048 | +0.008 | 0.415 st |
| `vocos-ft` | recorded | 2.738 | 88.2 cents | 5.17% | 0.120 | −0.030 | 0.491 st |
| P0, 45k | recorded | 2.712 | 89.4 cents | 5.30% | 0.119 | −0.025 | 0.475 st |
| P1, 10k | recorded | 2.878 | 93.9 cents | 5.36% | 0.114 | −0.012 | 0.454 st |
| P2, 10k | recorded | 2.986 | 89.4 cents | 5.03% | 0.111 | −0.016 | 0.457 st |
| P3, 10k | recorded | 2.971 | 85.0 cents | 4.88% | 0.108 | −0.015 | 0.453 st |
| `bigvgan-v2-ft` | GTA | 2.940 | 84.4 cents | 6.69% | 0.128 | −0.019 | 0.477 st |
| `vocos-ft` | GTA | 2.764 | 94.3 cents | 5.72% | 0.145 | −0.039 | 0.503 st |
| P1, 10k | GTA | 2.890 | 95.3 cents | 5.79% | 0.139 | −0.021 | 0.458 st |
| P2, 10k | GTA | 2.986 | 92.1 cents | 5.59% | 0.136 | −0.024 | 0.471 st |
| P3, 10k | GTA | **3.044** | **88.7 cents** | **5.36%** | **0.136** | −0.024 | 0.463 st |

- **More steps of the first recipe do nothing (P0).** At 45k it is within noise of `vocos-ft` on every measure.
- **The loss balance is the main lever (P2).** The ingredients are MRD weight 1, feature matching × 2, the
  multi-scale mel at 15 instead of 45 × single-scale, and a decaying rate. Together they lift Freya-100 UTMOSv2
  from 2.63 to 2.92, as close to BigVGAN-v2-ft (2.93) as noise allows, already at 5k steps. DNSMOS OVRL (3.35) and
  P.808 (4.07) now exceed BigVGAN-v2-ft's. WER is 0.66%, against 1.10%.
- **P2 also helps pitch, though less.** Periodicity RMSE falls from 0.120 to 0.111, the periodicity bias halves, and
  the F0 micro-variation falls from 0.450 to 0.408 st on Freya-100. The F0 error stays at ~89 cents.
- **The instantaneous-frequency loss helps pitch (P3).** It compares the frame-to-frame phase advance of each STFT
  bin with the recording's, weighted by the recording's magnitude (`train.iaf_loss_coeff`, `phase_derivative_loss`),
  and a constant delay of the output does not change it.
  - On top of P2 it lowers the F0 error, on both recorded mels (85.0 against 89.4 cents) and GTA mels (88.7
    against 92.1), and the gross pitch errors.
  - It brings the Freya-100 F0 micro-variation to 0.399 st (BigVGAN-v2-ft: 0.395), at no cost in UTMOSv2.
  - On GTA mels, P3 is the most natural vocoder by UTMOSv2 (3.04, against 2.94 for BigVGAN-v2-ft). Its F0 error is
    4.3 cents above BigVGAN-v2-ft's, and its periodicity RMSE 0.007 above.
- **NVIDIA's discriminators instead of Vocos's (P1) are worse and costly.** P1 reached UTMOSv2 2.84 at 10k, with no
  better pitch than P2. The CQT-D makes it about 3× slower per sample (1.3–1.8 it/s at batch 8 on a shared GPU), and
  it needs 12 GB at batch 8 (17 GB at batch 16). It was stopped at 12k.
- **What remains.** On recorded mels, Vocos's F0 error (85 cents) and periodicity RMSE (0.108) are still about
  twice BigVGAN-v2-ft's (52 cents, 0.048). The voiced frames stay slightly less periodic than the recording's.
- **Level.** P3's output is 0.8–1.2 dB quieter than the recording in every band (`vocoder_quality.py`'s band bias).
  `vocos-ft` was 0.5–0.7 dB louder above 1 kHz, and BigVGAN-v2-ft stays within ±0.7 dB.
- **Snapshots fluctuate.** P1's micro-variation was 0.388 st at 5k and 0.433 at 10k. Pick snapshots on several
  measures, not one.

**Streaming is unchanged.** The network is the same, so `tests/test_vocoder_registry.py` (29 frames exact) still
holds. Measured on the 24 longest Freya sentences in full fp32, P3 at 10k gives 87.1 dB at 28 frames of context and
87.5 dB at 32, against 82.4 and 80.9 dB for `vocos-ft`.

**Cost.** The second recipe runs at ~4.5 it/s with 6.5 GB on an RTX 5090, as fast as the first. On the shared GPU,
the pilots took 0.6–1.5 h per 10k steps.

### The long run

`configs/vocoder_vocos_v2.yaml` (the P3 recipe) runs 160k steps from `vocos-ft`, 200k steps in total. The cosine
decays the rates to 10% of their peaks, and a snapshot is kept every 5k steps. The snapshot to publish is chosen on
Freya-100 (UTMOSv2, DNSMOS, WER) and copy-synthesis (F0 error, periodicity), because the last step is not always the
best. The final snapshot (160k steps) is the vocoder of release v3.2, published as `vocos_v2.pt`
(`scripts/prepare_release.py` keeps only what `load_vocoder` reads).

### Speaker balance

Vocos v2 reached BigVGAN-v2-ft on the studio voice, but not on the voices with little data. Its training set (the
data filters of `tts_v3.yaml`: 49,926 utterances of 642 speakers) lists the studio voice twice, so the studio voice
fills 39% of the draws. The `male` voice (speaker 389, 59 min) gets 0.8% and the `female` voice (323, 19 min) 0.3%.

**Where Vocos v2 falls short.** In copy-synthesis (recorded mels, UTMOSv2), Vocos v2 is 0.03 below the recording on
the studio voice, but 0.18–0.24 below on the other speakers, where BigVGAN-v2-ft stays at the recording's level. The
two voices have only 2 and 1 held-out utterances, so their sets come from the `train` split:

- `train`: the first 50 training utterances of the speaker, which the vocoder has seen.
- `unseen`: the first 50 (389) or all 14 (323) `train` utterances that the fine-tune's data filters drop (speaking
  rate, ASR mismatch, bandwidth below 10 kHz), which no vocoder fine-tune has seen.
- `others`: the first 200 `val` utterances of the other speakers (156 speakers).

On the TTS mels (Freya-100, v3.1, T = 0.3, α = 2), Vocos v2 is at BigVGAN-v2-ft's level for `studio` (3.015 against
2.934) and `male` (2.809 against 2.797), and 0.135 below it for `female` (2.661 against 2.796).

**The recipe.** `train.speaker_balance` (opt-in) draws each epoch's utterances with replacement. Chosen speakers get
fixed shares of the draws, and the other speakers split the rest in proportion to their amount of audio raised to a
temperature. `configs/vocoder_vocos_v2_balance.yaml` gives the studio voice 30%, `male` 8% and `female` 6%. The
other 639 speakers share 56% at temperature 0.5, so the smallest of them are drawn up to 10 times as often as before.

- It continues Vocos v2's `last.pt` (160k steps: generator, MPD, MRD and both AdamW states) for 30k steps with the
  same losses. The rates restart at 2e-5 (generator) and 4e-5 (discriminators), against 5e-6 and 1e-5 at the end of
  Vocos v2, and decay to 10% with a cosine. The generator warms up over 500 steps.
- The control is the same continuation without the balance, stopped at 20k.
- Each `female` training utterance is drawn ~22 times as often as in Vocos v2 (`male`: ~10 times), about 140 times
  in 20k steps. The generated mels change with every draw (fresh noise at T = 0.3), and the random 64-frame windows
  move.

Freya-100 UTMOSv2 by voice (v3.1 mels; same mels for every row):

| vocoder | studio | male | female |
|---|---|---|---|
| `bigvgan-v2-ft` | 2.934 | 2.797 | 2.796 |
| Vocos v2 (160k) | 3.015 | 2.809 | 2.661 |
| balanced, 5k | 2.993 | 2.851 | 2.699 |
| balanced, 10k | 2.960 | 2.834 | 2.660 |
| balanced, 15k | 3.013 | 2.882 | 2.698 |
| **balanced, 20k** | **3.019** | **2.860** | **2.714** |
| balanced, 25k | 3.028 | 2.861 | 2.697 |
| balanced, 30k | 3.013 | 2.835 | 2.715 |
| control, 5k | 3.013 | 2.797 | 2.666 |
| control, 10k | 3.010 | 2.776 | 2.707 |
| control, 15k | 2.992 | 2.768 | 2.682 |
| control, 20k | 2.974 | 2.733 | 2.719 |

Copy-synthesis UTMOSv2 by speaker:

| system | studio (100 `val`) | 389 `train` (50) | 389 `unseen` (50) | 323 `train` (50) | 323 `unseen` (14) | others (200 `val`) |
|---|---|---|---|---|---|---|
| recording | 3.093 | 3.217 | 3.285 | 3.062 | 2.902 | 2.897 |
| `bigvgan-v2-ft` | 3.174 | 3.312 | 3.343 | 3.039 | 2.912 | 2.886 |
| Vocos v2 (160k) | 3.064 | 3.002 | 3.043 | 2.819 | 2.721 | 2.684 |
| balanced, 15k | 3.029 | 3.123 | 3.176 | 2.905 | 2.808 | 2.732 |
| balanced, 20k | 3.046 | 3.046 | 3.122 | 2.881 | 2.771 | 2.712 |
| control, 15k | 3.040 | 2.974 | 2.999 | 2.797 | 2.675 | 2.682 |
| control, 20k | 3.025 | 2.967 | 2.980 | 2.844 | 2.717 | 2.672 |

Freya-495 with Whisper (v3.1 mels, T = 0.3, α = 2; WER with 95% bootstrap intervals). The last two rows are the
paired UTMOSv2 differences of the balanced 20k snapshot, with 95% bootstrap intervals over the 495 sentences:

| vocoder | studio WER | studio UTMOSv2 | male WER | male UTMOSv2 | female WER | female UTMOSv2 |
|---|---|---|---|---|---|---|
| `bigvgan-v2-ft` | 1.23% [0.82, 1.68] | 2.935 | 1.74% [1.24, 2.30] | 2.814 | 3.02% [2.38, 3.71] | 2.752 |
| Vocos v2 (160k) | 1.18% [0.79, 1.62] | 3.017 | 1.48% [1.05, 1.95] | 2.809 | 3.27% [2.59, 4.03] | 2.638 |
| **balanced, 20k** | 1.33% [0.93, 1.79] | **3.004** | 1.61% [1.15, 2.11] | **2.876** | 3.22% [2.55, 3.95] | **2.688** |
| control, 20k | 1.38% [0.94, 1.86] | 3.001 | 1.43% [1.01, 1.91] | 2.759 | 3.35% [2.64, 4.07] | 2.671 |
| balanced − Vocos v2 | | −0.014 [−0.028, +0.001] | | +0.067 [+0.049, +0.086] | | +0.050 [+0.036, +0.065] |
| balanced − control | | +0.003 [−0.014, +0.019] | | +0.116 [+0.097, +0.135] | | +0.018 [+0.001, +0.034] |

- **The balance is what lifts the `male` voice.** Every balanced snapshot is 0.03–0.07 above Vocos v2 on `male`
  Freya-100, while the control drifts down by 0.01–0.08. On Freya-495 the balanced 20k snapshot gains 0.067 and
  passes BigVGAN-v2-ft (2.876 against 2.814), 0.116 above the control. On recorded mels it gains 0.03–0.13 on
  speaker 389, as much on the `unseen` utterances as on the `train` ones, so it is not memorisation.
- **The `female` voice gains mostly from the continuation.** On Freya-495 the balanced 20k snapshot gains 0.050,
  which closes 44% of the gap to BigVGAN-v2-ft (0.114 → 0.064). The control gains 0.032, so the balance adds 0.018.
  On Freya-100 both runs gain about 0.05 at 20k. On recorded mels of speaker 323, the balanced run is 0.04–0.13
  above the control at 15k and 20k, and within ±0.06 of it at 5k and 10k.
- **The studio voice: −0.014 UTMOSv2 and 6 more word errors.** On Freya-100 the balanced snapshots stay within 0.03 of Vocos v2
  from 15k on (20k: +0.004); on Freya-495 the 20k one loses 0.014 and stays 0.07 above BigVGAN-v2-ft. Its WER rises
  from 1.18% to 1.33%: 6 more word errors of 3,911, in 5 sentences, one-consonant slips such as *fırından* →
  *sırından* (BigVGAN-v2-ft: 1.23%). The control loses as much (0.016, 1.38%), so the continuation causes it, not
  the balance.
- **WER and DNSMOS.** For `male` and `female`, WER moves within its intervals. DNSMOS OVRL changes by less than
  0.015 on every voice.
- **Other speakers gain a little.** On the 200 `val` utterances of the other speakers, the balanced snapshots gain
  up to 0.05 (20k: +0.03), the control at most 0.02 (20k: −0.01). BigVGAN-v2-ft is still 0.17 higher.
- **Pitch is untouched.** In copy-synthesis at 15k and 20k, the F0 error moves by at most 2.2 cents and the
  periodicity RMSE by at most 0.003 (the control's grows by up to 0.010 on speaker 389). Against the recording,
  Vocos stays at 79–93 cents and 0.08–0.15 on every speaker, against 52–62 cents and 0.05 for BigVGAN-v2-ft. The
  balance does not address that part of the gap.
- **Snapshots fluctuate by about ±0.05 per voice.** The balanced run lost 0.055 on the studio voice at 10k and was
  back at 15k. The snapshot is therefore chosen on all voices: on Freya-100 the 20k one has the best worst-voice gain
  (+0.051 on `male`, +0.053 on `female`), with the studio voice at +0.004.

**Streaming is unchanged.** On the 24 longest Freya sentences in full fp32, the balanced 20k snapshot gives 88.2 dB
at 32 frames of context, against 86.3 dB for Vocos v2: both at the fp32 noise floor.

**Cost.** The 30k steps took 2 h 19 min (3.4–4.4 it/s on the shared GPU), with 6.7 GB.

## Revox Vocoder 1.0 (non-commercial)

[Revox Vocoder 1.0](https://huggingface.co/minori-live/revox-vocoder-1) by **Minori Live** is a 48 kHz PC-NSF-Vocos:
a Vocos backbone with a neural source-filter excitation, 4.46 M parameters, conditioned on a log-mel and on
frame-level F0. **Its weights are licensed CC BY-NC-SA 4.0: non-commercial use only.**

- The weights are not part of this repository or of `Vyvo/drifting-tts-tr`. `revox` downloads `vocoder.onnx` from
  the original repo, at a pinned revision, when it is loaded. Only ONNX Runtime executes it (`pip install
  "drifting-tts[revox]"`).
- Audio made with it falls under the same non-commercial, share-alike terms. Credit it as "Minori Live — Revox
  Vocoder 1.0" with a link to its repo.

**Verdict: not worth it for this model.**

- **Quality.** On the TTS mels, `revox` scores UTMOSv2 2.27, below every BigVGAN and Vocos row; only Griffin-Lim is
  lower. Its DNSMOS OVRL (3.10) is below even Griffin-Lim's (3.13).
- **Not the conversion.** On real recordings, Revox stays 0.63 UTMOSv2 below `bigvgan-v2-ft` even with ideal
  inputs: its own mel of the recording and the recording's F0. The mel conversion and the F0 source cost little on
  top of that.
- **Cost.** It needs an F0 step (WORLD on Griffin-Lim audio), runs on the CPU and cannot stream. Its license
  excludes commercial use.
- **Its strengths do not apply here.** It offers 48 kHz output and explicit pitch control, but our mels stop at
  12 kHz and the model has no frame-level pitch to give.
- The registry entry stays as an opt-in experiment.

### From our mel to Revox's

- **The two mels.** Revox takes the 128-band Slaney log-mel of 48 kHz audio: n_fft 2048, hop 480, frames centred at
  *k* · 10 ms. Ours is 24 kHz, 100 bands, n_fft 1024, hop 256, with uncentred frames.
- **Same bins, twice the magnitude.** Both STFTs have 23.4375 Hz bins and a 42.7 ms periodic Hann window: the
  2048-point window, sampled every other sample, is the 1024-point one. So for audio band-limited to 12 kHz and
  upsampled ×2, the 48 kHz magnitude is twice the 24 kHz one in the 513 shared bins.
  - On the held-out recordings upsampled with torchaudio's default resampler, the least-squares gain is 2.000 up to
    8 kHz. Above that the resampler rolls off: 1.95 at 8–10 kHz and 1.63 at 10–11 kHz.
  - With a sharp Kaiser resampler (`resample_sharp`), the gain on white noise is 2.000 up to 11 kHz.
- **The conversion** (`RevoxLogMel.convert`):
  1. `griffin-lim`'s NNLS inverts our filterbank to the linear magnitude.
  2. The magnitude is doubled and mapped by Revox's filters; the bins above 12 kHz are empty.
  3. Linear interpolation moves it from frames centred at (*i* + ½) · 256 samples (93.75 Hz) to 100 Hz frames.
  4. Then ln(max(·, 1e-5)).
- **Interpolation domain.** Interpolating in the linear domain commutes with the filterbank. Log-domain
  interpolation (a geometric mean) has the same error, but its bias is negative. Cubic interpolation would cut the
  interpolation error from 0.56 to 0.46 dB, which is small next to the NNLS error.
- **Output rate.** The output is 48 kHz, resampled to 24 kHz for the pipeline. The mel carries nothing above 12 kHz,
  and Revox puts only 3 · 10⁻⁷ of its energy there. Everything downstream (`Synthesizer`, streaming, the web demo,
  the judges) runs at 24 kHz. `Revox.generate` returns the 48 kHz waveform.
- **Streaming.** The graph resets its source phase on every call, so `revox` vocodes each sentence whole
  (`context=None`), like `griffin-lim`.

The converted mel is checked against the Revox mel of the same recordings upsampled to 48 kHz: 100 held-out
recordings, the 105 bands below 11.5 kHz, and the cells within 60 dB of each utterance's peak.

| conversion | mean abs. error | bias | 95th percentile |
|---|---|---|---|
| NNLS + linear interpolation (used) | 0.91 dB | +0.10 dB | 2.85 dB |
| NNLS + log interpolation | 0.92 dB | −0.17 dB | 2.70 dB |
| NNLS alone (on our frames, no time resampling) | 0.65 dB | −0.05 dB | 2.13 dB |
| exact magnitude + linear interpolation | 0.56 dB | +0.14 dB | 1.92 dB |
| exact magnitude + log interpolation | 0.56 dB | −0.12 dB | 1.74 dB |

In copy-synthesis (below), the conversion costs Revox 0.07 UTMOSv2 (2.29 → 2.23) and leaves WER and DNSMOS
unchanged.

### F0

The model predicts pitch per token only, so `revox` takes the frame-level F0 from elsewhere. `revox:<F0
source>[:dio|harvest]` picks the source and the WORLD method:

- `griffin-lim` (the default): WORLD on the Griffin-Lim audio of the same mel. It needs no weights and reuses the
  NNLS magnitude.
- A registry vocoder, e.g. `revox:bigvgan-v2-ft`: WORLD on that vocoder's audio. This is a cascade of two vocoders.
- `none`: every frame is pitch-invalid, so Revox has to read the pitch from the mel.

WORLD's unvoiced frames are passed as reliable unvoiced decisions (`pitch_valid` is true everywhere).

**Voicing is what matters.** Revox cannot voice a frame without F0. With `none`, only 35% of its output frames are
voiced (82% in the recordings), and it sounds hoarse: UTMOSv2 1.94 and DNSMOS OVRL 2.00 in copy-synthesis.

- WORLD dio with its default voicing threshold (`allowed_range` 0.1) voices only 65% of the frames of the
  Griffin-Lim audio of TTS mels, against 92% for harvest. On TTS mels that cost 0.14 UTMOSv2 against harvest.
- dio with `allowed_range` 0.2, the `revox` default, voices 86% of the frames. It matches harvest on 80 TTS
  sentences (UTMOSv2 2.25 vs 2.24) at a twentieth of its cost.
- Filling short unvoiced gaps instead helps less (2.14).

F0 sources against the recording, on the 100 held-out recordings. The reference is the 77% of 10 ms frames where
WORLD harvest and dio agree on the recording (82% of them voiced); harvest alone extends voicing into pauses.

| F0 source (as `revox` runs it) | VDE | GPE | fine error |
|---|---|---|---|
| Griffin-Lim audio, dio (`revox`) | 4.2% | 0.90% | 45 cents |
| Griffin-Lim audio, harvest | 4.5% | 0.93% | 35 cents |
| BigVGAN-v2-ft audio, dio | 5.6% | 1.15% | 43 cents |
| BigVGAN-v2-ft audio, harvest | 5.7% | 0.56% | 34 cents |

VDE: voicing decision error; GPE: gross pitch error, more than 20% off on frames voiced in both; fine error: RMS of
the remaining frames. Griffin-Lim audio is as good an F0 source as BigVGAN-v2-ft's, so the cascade buys nothing.

### Copy-synthesis on held-out recordings

These are the 100 recordings and the judges of `scripts/resynthesis_benchmark.py` (docs/LATENTS.md). The recording
and BigVGAN-v2-ft rows reproduce that table. "Output F0" is WORLD harvest on the output against the reference above.

| system | mel | F0 | WER 8 kHz [95% CI] | CER 8 kHz | UTMOSv2 | DNSMOS OVRL | speaker sim. | output F0: VDE / GPE / cents |
|---|---|---|---|---|---|---|---|---|
| recording | – | – | 6.56% [4.80, 8.58] | 2.18% | 2.953 | 3.268 | – | – |
| `bigvgan-v2-ft` | ours | – | 6.77% [4.95, 8.89] | 2.24% | **2.922** | **3.305** | **0.966** | 5.7% / 0.56% / 34 |
| Revox | its own, from the recording | recording (harvest) | 7.98% [5.96, 10.21] | 2.44% | 2.294 | 2.962 | 0.884 | 7.8% / 0.14% / 21 |
| Revox | converted | recording (harvest) | 8.19% [6.14, 10.50] | 2.57% | 2.227 | 2.966 | 0.871 | 8.0% / 0.13% / 21 |
| `revox` | converted | Griffin-Lim audio, dio | 7.92% [5.96, 10.07] | 2.53% | 2.252 | 3.009 | 0.873 | 7.3% / 1.34% / 44 |
| `revox:griffin-lim:harvest` | converted | Griffin-Lim audio, harvest | 8.53% [6.34, 11.11] | 2.56% | 2.213 | 2.973 | 0.866 | 9.2% / 1.16% / 39 |
| `revox:bigvgan-v2-ft` | converted | BigVGAN-v2-ft audio, dio | 8.32% [6.32, 10.48] | 2.68% | 2.254 | 2.960 | 0.871 | 9.3% / 1.34% / 42 |
| `revox:bigvgan-v2-ft:harvest` | converted | BigVGAN-v2-ft audio, harvest | 8.05% [6.02, 10.26] | 2.69% | 2.245 | 2.966 | 0.864 | 10.1% / 0.85% / 38 |
| `revox:none` | converted | none | 8.39% [6.45, 10.61] | 2.48% | 1.940 | 1.997 | 0.818 | 51.0% / 19.07% / 138 |

- **Revox is below BigVGAN-v2-ft even with ideal inputs.** Given its own mel of the recording and the recording's F0,
  it scores UTMOSv2 2.29 against 2.92, DNSMOS OVRL 2.96 against 3.31 and speaker similarity 0.884 against 0.966. Its
  WER is 7.98% against 6.77%.
- **It follows the F0 it is given** more closely than BigVGAN reproduces the recording's pitch: GPE 0.14% and 21
  cents, against 0.56% and 34 cents.
- **Among WORLD sources, the F0 barely matters on recordings.** UTMOSv2 stays between 2.21 and 2.25, and WER stays
  within the intervals. Its absence matters a lot.
- **No hallucinated band.** Revox puts 3 · 10⁻⁷ of its output energy above 12 kHz on converted mels, and 3 · 10⁻⁶ on
  its own mel of the recordings. The recordings have 2 · 10⁻¹⁰ there.

### On the TTS mels

These are the 495 Freya-TR-Eval sentences of the comparison above. The `revox` rows are in its table.

| vocoder | WER | UTMOSv2 | DNSMOS OVRL | DNSMOS P.808 |
|---|---|---|---|---|
| `bigvgan-v2-ft` | 1.23% [0.82, 1.68] | **2.935** | 3.324 | 3.960 |
| `bigvgan-v2` (NVIDIA's weights) | 1.20% [0.81, 1.65] | 2.398 | 2.810 | 3.779 |
| `revox` (Griffin-Lim audio, dio) | 1.61% [1.17, 2.09] | 2.270 | 3.100 | 3.813 |
| `revox:griffin-lim:harvest` | 1.71% [1.24, 2.25] | 2.263 | 3.094 | 3.806 |
| `revox:bigvgan-v2-ft` | 1.51% [1.06, 1.98] | 2.260 | 3.094 | 3.815 |
| `revox` with dio's default voicing | 1.43% | 2.120 | 2.978 | 3.803 |
| `revox:none` | 1.51% [1.05, 2.02] | 2.044 | 2.044 | 3.564 |
| `griffin-lim` | 1.20% [0.82, 1.63] | 1.772 | 3.127 | 3.465 |

- **Last among the neural vocoders.** `revox` sits between Griffin-Lim and every BigVGAN or Vocos row. It is 0.67
  UTMOSv2 below `bigvgan-v2-ft` and 0.13 below NVIDIA's stock BigVGAN-v2. Its DNSMOS OVRL (3.10) is below
  Griffin-Lim's.
- **The F0 source does not matter, but voicing does.** Griffin-Lim audio with dio or harvest and BigVGAN-v2-ft audio
  are within 0.01 UTMOSv2. dio's default voicing threshold costs 0.15, and no F0 costs 0.23 plus 1.06 in DNSMOS
  OVRL.
- **WER** is 1.5–1.7%, slightly above most other rows, but the intervals overlap.

### Speed

Each stage was timed on the first 20 held-out recordings (134 s of audio): RTX 5090, 3 CPU threads, under the
shared load of the † speeds above. BigVGAN-v2-ft in eager PyTorch measured 0.0229 in the same run.

| stage | runs on | RTF |
|---|---|---|
| NNLS inversion of our mel | GPU | 0.0017 |
| mel conversion (×2, filters, interpolation) | GPU | 0.0006 |
| F0: Griffin-Lim audio | GPU | 0.0033 |
| F0: WORLD dio | CPU, 1 thread | 0.0079 |
| F0: WORLD harvest (instead of dio) | CPU, 1 thread | 0.1523 |
| F0: BigVGAN-v2-ft audio (instead of Griffin-Lim) | GPU | 0.0229 |
| Revox network | ONNX Runtime, CPU, 3 threads | 0.0063 |
| ISTFT | GPU | 0.0017 |
| resampling 48 → 24 kHz | GPU | 0.0013 |
| **`revox`, whole call** | | **0.0195** |
| `revox:none`, whole call | | 0.0093 |

- **The F0 step costs more than the network.** Griffin-Lim plus dio take an RTF of 0.011, against 0.0063 for the
  network. Harvest alone takes 8× the whole `revox` call.
- **The network runs on the CPU.** ONNX Runtime's CUDA provider was 2–7× slower than the CPU on this GPU for 1–10 s
  inputs; that was onnxruntime-gpu 1.22, the last CUDA 12 build, since 1.30 needs CUDA 13. `onnx2torch` cannot
  convert the graph.
- **Slower than BigVGAN overall.** The network alone (0.0063 on 3 CPU threads) is faster than BigVGAN-v2-ft under
  the same load. With the conversion and the F0 step, though, `revox` is about 2× slower than `bigvgan-v2-ft`: RTF
  0.0265 against 0.0118 in the TTS table.
- **No streaming.** The first audio waits for the whole sentence: 84.5 ms for the short sentence and 177.7 ms for
  the long one (†), against 32.1 / 39.5 ms for `bigvgan-v2-ft` under the same load.

### Where it falls short

- **Not the plumbing.** On 20 recordings with the oracle inputs, UTMOSv2 does not improve with a different input
  level: 2.24, 2.27, 2.17 and 1.93 at −6, 0, +6 and +12 dB. It does not improve with other voicing masks either
  (2.24–2.27).
- **Not the alignment.** The output's lag against the recording is ±5 ms, with mixed signs, as for any vocoder that
  makes its own phase.
- **The empty band above 12 kHz.** Revox expects full-band 48 kHz mels. Filling the empty band by extrapolating the
  6–11 kHz slope gains 0.07 UTMOSv2, which is small next to the gap.
- **Where the output differs from the input.** The mel of Revox's output deviates from the input mel about twice as
  much as BigVGAN-v2-ft's does. The gap is largest below 1 kHz (the harmonics), in pauses (the noise floor) and at
  the top of our band. Mean absolute dB, 30 recordings, both measured with Revox's 48 kHz front end:

  | | voiced frames | unvoiced frames | silent frames | < 1 kHz | 8–11.5 kHz bias |
  |---|---|---|---|---|---|
  | Revox (oracle inputs) | 2.87 | 3.13 | 3.97 | 3.53 | −2.18 dB |
  | BigVGAN-v2-ft | 1.61 | 1.65 | 1.78 | 1.67 | −0.01 dB |

- **Domain.** BigVGAN-v2-ft was fine-tuned on this model's mels; Revox was not. Its README does not state its
  training data. It cites singing-voice tools (NSF-HiFiGAN from OpenVPI's SingingVocoders, OpenTune) and targets
  pitch-controlled resynthesis.

## Protocol

- **Same mels for every vocoder.** `scripts/compare_vocoders.py` generates the v3.1 mels of the 495 Freya-TR-Eval
  sentences once, as `drifting-tts benchmark` does: seed = sentence index, `studio` voice, T = 0.3, α = 2. Texts are
  split into sentences as in `Synthesizer`, and the vocoded sentences are joined with 0.15 s pauses. The rows differ
  only in the vocoder.
- **Quality.**
  - WER / CER: Whisper large-v3 (Turkish, beam 5) on audio band-matched to 8 kHz, with `benchmark`'s normaliser.
    Both are corpus-level, with 95% bootstrap intervals.
  - UTMOSv2 on the full band.
  - DNSMOS P.835 OVRL and P.808 (`drifting_tts.score.DnsMos`).
  - The vocoders run in PyTorch, without the BigVGAN CUDA kernel.
- **Speed.** `Synthesizer(fast=True, cuda_kernel=True)` on an RTX 5090 with PyTorch 2.11.
  - Vocoder RTF: vocoder time over every sentence of the set (whole sentences, batch 1), divided by their duration.
  - TTFA: `stream()`, from the text to the first audio on the host. Each value is the median of 50 runs of the
    short and the long sentence of `scripts/bench_ttfa.py`, after warm-up.
- **Streaming context.** `stream_vocoder` vocodes the first 32 frames, then windows with `context` frames on each
  side.
  - The sweep streams the 24 longest sentences with 64-frame windows (four times as many seams as the default 256)
    and compares the result with whole-sentence vocoding.
  - It runs in full fp32 (TF32 off), so the cut context is the only difference.
  - Above 55 dB SNR the pieces count as matching, and above 90 dB they are equal up to the fp32 noise floor. Each
    entry uses the smallest multiple of 8 frames above 90 dB.
  - In normal use, TF32 convolutions (PyTorch's default in cuDNN) limit the agreement of any two BigVGAN runs on
    different window lengths to about 52 dB, whatever the context.

- **Revox.**
  - The TTS rows come from `compare_vocoders.py` like every other row. The judges get its 24 kHz output.
  - `scripts/revox_benchmark.py` runs on the first 100 recordings of the `val` split, the set of
    `resynthesis_benchmark.py`. It does the conversion check, the F0 sources, the copy-synthesis rows (scored by
    `resynthesis_benchmark.score`, with Revox's 48 kHz output resampled by the judges) and the per-stage speed.

```bash
python scripts/compare_vocoders.py --model drifting_tts_v3.1.pt --table docs/VOCODERS.md \
    --extra vocos-ft=runs/vocos_bigvgan/vocos_ft.pt bigvgan-base-ft=runs/bigvgan_base_ft/bigvgan_ft.pt
python scripts/compare_vocoders.py --model drifting_tts_v3.1.pt --table docs/VOCODERS.md \
    --vocoders revox revox:griffin-lim:harvest revox:bigvgan-v2-ft revox:none
python scripts/revox_benchmark.py --data data/train --out outputs/revox
```

- **Training Vocos further.** `scripts/vocoder_quality.py` reuses `compare_vocoders.py`'s cached Freya mels and
  judges, and adds the pitch measures.
  - F0: WORLD harvest, 10 ms frames, 60–500 Hz.
  - F0 error: VDE, GPE (more than 20% off), and the RMS in cents of the other frames voiced in both.
  - Periodicity: 1 minus YIN's minimum cumulative mean normalised difference over 2–16.7 ms lags, per 10 ms frame.
    The RMSE counts frames within 40 dB of the utterance's peak; the bias counts the voiced ones.
  - F0 micro-variation: as in [EXPERIMENTS.md §5](EXPERIMENTS.md#5-robotic-prosody-diagnosis-and-research).
  - Copy-synthesis uses the first 100 `val` utterances of speaker 722: `studio` for the recorded mels, `studio_gta`
    for the acoustic model's (T = 0.3, α = 2, ground-truth durations and pitch, seed = utterance index).
  - Other copy sets: `others` (every speaker but 722) and `spk<ID>` (one speaker) from `val`; `@<split>` takes them
    from another split (`spk389@train`), and `@unseen` from the `train` utterances that the fine-tune of
    `--train-config` leaves out. Freya runs of other voices take `--speaker male` or `--speaker female`.
  - Speaker balance: the paired intervals bootstrap the per-sentence UTMOSv2 differences of two vocoders on the same
    100 Freya sentences (`freya100_<name>.jsonl`, also written without ASR).

```bash
python scripts/vocoder_quality.py --model drifting_tts_v3.1.pt --data data/train \
    --mels outputs/vocoders/mels_495_studio_T0.3_cfg2.pt --out outputs/vocoder_quality \
    --vocoders bigvgan-v2-ft --extra vocos-ft=runs/vocos_bigvgan/vocos_ft.pt --copy-sets studio studio_gta
drifting-tts finetune-vocoder --config configs/vocoder_vocos_v2.yaml --workdir runs/vocos_v2 \
    tts.path=drifting_tts_v3.1.pt train.init_from=runs/vocos_bigvgan/last.pt
drifting-tts finetune-vocoder --config configs/vocoder_vocos_v2_balance.yaml --workdir runs/vocos_v2_balance \
    tts.path=drifting_tts_v3.1.pt train.init_from=runs/vocos_v2/last.pt
python scripts/vocoder_quality.py --model drifting_tts_v3.1.pt --data data/train --phases copy --copy-num 50 \
    --train-config runs/vocos_v2_balance/config.yaml --copy-sets spk389@train spk389@unseen spk323@train \
    spk323@unseen --vocoders bigvgan-v2-ft --extra balanced=runs/vocos_v2_balance/vocos_ft_20000.pt
python scripts/vocoder_quality.py --model drifting_tts_v3.1.pt --phases freya --speaker female --no-asr \
    --vocoders --extra balanced=runs/vocos_v2_balance/vocos_ft_20000.pt --out outputs/vocoder_quality_female
```

# drifting-tts: one-step Turkish text-to-speech

[![Demo](https://img.shields.io/badge/🤗%20Demo-Space-yellow)](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-demo)
[![WebGPU](https://img.shields.io/badge/🤗%20WebGPU-in%20your%20browser-orange)](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-webgpu)
[![Model](https://img.shields.io/badge/🤗%20Model-Vyvo%2Fdrifting--tts--tr-blue)](https://huggingface.co/Vyvo/drifting-tts-tr)
[![Paper](https://img.shields.io/badge/arXiv-2602.04770-b31b1b)](https://arxiv.org/abs/2602.04770)

A Turkish TTS model that turns text into speech with **a single network pass**, with no diffusion or flow steps.
It is trained with *[Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770)* (Deng et al., 2026), using
the learned-temperature recipe from [Kyutai's Pocket TTS](https://kyutai.org/blog/2026-09-28-pocket-tts-drifting/).

| | |
|---|---|
| **Quality** | [Freya-TR-Eval](https://huggingface.co/datasets/freyavoice/freya-tr-eval) WER **1.33%** (v3.2, studio voice; v3.1 1.23%; Piper 3.76%, MMS-TTS 6.26% under the same protocol) · UTMOSv2 **3.02** (v3.1 2.94) |
| **Intonation** (v3.2) | the pitch is **sampled** by a small model trained with drifting: pitch spread 3.53 semitones on held-out sentences (recordings 3.68, v3.1 3.21) |
| **Speed** (RTX 5090) | first audio after **6–7 ms** for any sentence length (v3.2, streaming, CUDA graphs; v3.1: 12–14 ms), 200–600× faster than real time |
| **Size** | 67.7 M acoustic model + 8.1 M prosody model + 13.5 M Vocos vocoder (v3.1: 67.7 M + 112 M BigVGAN-v2) |
| **Voices** | `studio` (default), `male` and `female` |

**New in v3.2:** v3.1's acoustic weights with three changes. The intonation (the pitch of every character) is
sampled by a small prosody model trained with drifting, so each seed reads a sentence with another natural tune; the
durations, i.e. the rhythm, stay v3.1's. A retrained Vocos vocoder (Vocos v2, 13.5 M) replaces BigVGAN-v2, and the
pauses between sentences follow the punctuation. The studio voice is about as intelligible as in v3.1 and scores
higher on UTMOSv2; the male and female voices lose some intelligibility (Freya WER 2.28% and 3.99%, against 1.74%
and 3.02%) ([results](docs/RESULTS.md#v32-sampled-intonation-vocos-v2-punctuation-pauses)). v3.1 stays available
unchanged.

## Listen

| text | studio | male | female |
|---|---|---|---|
| *Merhaba! Bu ses, tek bir ağ değerlendirmesiyle üretildi; difüzyon adımı yok.* | [wav](docs/samples/v31_studio_1.wav) | [wav](docs/samples/v31_male_1.wav) | [wav](docs/samples/v31_female_1.wav) |
| *İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı.* | [wav](docs/samples/v31_studio_2.wav) | [wav](docs/samples/v31_male_2.wav) | [wav](docs/samples/v31_female_2.wav) |
| *Toplantı yarın saat 14:30'da, 2. katta; lütfen geç kalmayın.* | [wav](docs/samples/v31_studio_3.wav) | [wav](docs/samples/v31_male_3.wav) | [wav](docs/samples/v31_female_3.wav) |
| *Prof. Dr. Ayşe Yılmaz, 250 TL'lik bağışın tamamının öğrencilere ayrılacağını söyledi.* | [wav](docs/samples/v31_studio_4.wav) | [wav](docs/samples/v31_male_4.wav) | [wav](docs/samples/v31_female_4.wav) |

Or try any text in the **[online demo](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-demo)**, or run the model
**[in your browser with WebGPU](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-webgpu)**.

**Experimental:** the same model trained in the latent space of an audio VAE, with the VAE decoder fine-tuned on
its latents: [DAC-VAE](https://huggingface.co/Vyvo/drifting-tts-tr-dacvae) (WER 1.32%, UTMOSv2 2.71) and
[VoxCPM2](https://huggingface.co/Vyvo/drifting-tts-tr-voxcpm2) (1.87%, 2.53), against 0.66% and 2.93 for v3.1 on the
same 100 Freya-TR-Eval sentences ([docs/LATENTS.md](docs/LATENTS.md)).
[This Space](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-compare) plays all three side by side.

## Quick start

```bash
pip install "drifting-tts[bigvgan] @ git+https://github.com/kadirnar/drifting-tts"   # needs a CUDA build of PyTorch
```

```python
import soundfile as sf
from drifting_tts.synthesize import Synthesizer

tts = Synthesizer.from_pretrained("v3.2", "cuda")   # model, prosody predictor and Vocos v2 from Vyvo/drifting-tts-tr

wav, info = tts("Merhaba, bu cümle tek adımda üretildi. Nasıl buldunuz?", speaker="studio", cfg_scale=2.0,
                temperature=0.3, seed=0)
sf.write("merhaba.wav", wav.numpy(), 24000)
```

- `speaker`: `"studio"` (default, the clearest voice), `"male"` or `"female"`. Any of the model's 723 speaker
  IDs also works, e.g. `speaker=17`; [docs/SPEAKERS.md](docs/SPEAKERS.md) scores every one of them.
- `seed`: v3.2 samples the intonation, so another seed gives another tune for the same text (with the same rhythm).
- `temperature`: the noise level of the acoustic model. 0.3 sounds clearest; higher values give more variety.
- `prosody_temperature` (v3.2, default 0.5, the tested setting): how freely the pitch is sampled; higher is more
  varied.
- `prosody_durations="sampled"` (opt-in): the prosody model also samples the durations, so the rhythm and the pauses
  inside a sentence vary too. It costs intelligibility on new text for the voices with little data (Freya WER male
  5.78%, female 11.28%); use it only with the studio voice (1.89%) and a prosody temperature of at most 0.5
  ([details](docs/RESULTS.md#v32-sampled-intonation-vocos-v2-punctuation-pauses)).
- `cfg_scale`: the guidance strength, learned during training, so it costs nothing at inference.
- `pause`: the silence between sentences: `"punct"` (v3.2: by the final punctuation, measured per voice) or seconds
  (v3.1: 0.15).
- **v3.1**, as released: `Synthesizer.from_pretrained("v3.1", "cuda")` (BigVGAN-v2-ft, deterministic pitch and
  durations, 0.15 s pauses). Keyword arguments override a release's parts, e.g.
  `from_pretrained("v3.2", vocoder="bigvgan-v2-ft")`; `tts.variant(...)` gives a second pipeline that shares the
  acoustic model. The explicit form still works:
  `Synthesizer(hf_hub_download("Vyvo/drifting-tts-tr", "drifting_tts_v3.1.pt"), "cuda", vocoder="bigvgan-v2-ft")`.
- `vocoder`: `"vocos-v2"` (v3.2) and `"bigvgan-v2-ft"` (v3.1) are fine-tuned on this model's mels (downloaded from
  the Hub), as are `"vocos-ft"` and `"bigvgan-base-ft"`. Other names: the stock NVIDIA `"bigvgan-v2"`,
  `"bigvgan-v1"` and `"bigvgan-base"` (14 M), or the weight-free `"griffin-lim"`; a checkpoint path also works.
  [docs/VOCODERS.md](docs/VOCODERS.md) compares them.
- `prosody`: `"drift"` (v3.2's predictor), `None` (the model's deterministic pitch regressor) or a checkpoint.
- Numbers, dates, times, units, currencies and common abbreviations are read out in Turkish automatically.

The same from the command line (`--release` downloads the parts; `--model`, `--vocoder`, `--prosody` and `--pause`
override them):

```bash
drifting-tts synthesize --release v3.2 --speaker female --cfg 2 --temperature 0.3 \
    --text "Merhaba, nasılsınız?" --out merhaba.wav
```

**Lowest latency: streaming.** `fast=True` runs the acoustic model, the prosody predictor included, as CUDA graphs,
with the same output. `stream()` yields the audio in pieces, and on an RTX 5090 the first piece (0.34 s) is ready
after about 6–7 ms (v3.1 with BigVGAN-v2: 12–14 ms). The vocoder streams in overlapping windows whose pieces join into the whole-sentence audio;
Freya WER and CER are unchanged ([details](docs/RESULTS.md#latency-and-size)):

```python
tts = Synthesizer.from_pretrained("v3.2", "cuda", fast=True)
for piece in tts.stream("Merhaba! Bu ses parça parça, bekletmeden geliyor.", speaker="studio",
                        cfg_scale=2.0, temperature=0.3):
    play(piece)   # float32 tensor at 24 kHz
```

## In the browser (WebGPU)

The model (v3.1's pipeline: the v3.2 prosody predictor and Vocos v2 are not exported to ONNX yet) also runs entirely
client-side with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/): the
**[WebGPU demo](https://huggingface.co/spaces/Vyvo/drifting-tts-tr-webgpu)** downloads about 385 MB once, and after
that the text never leaves the device. On an RTX 5090 in Chrome the first audio arrives after about 0.1 s, and
speech is generated 12–22× faster than real time ([details](docs/RESULTS.md#in-the-browser-webgpu)).

- **Graphs:** `scripts/export_onnx.py` writes the three ONNX graphs (text encoder, generator, vocoder) and checks each
  against PyTorch. They are published under [`onnx/`](https://huggingface.co/Vyvo/drifting-tts-tr/tree/main/onnx) in
  the model repo.
- **Page:** [`web/`](web/) holds the page, the pipeline (`tts.js`) and a JavaScript port of the Turkish text frontend
  (`text.js`).

## On a Mac (MLX)

`drifting_tts.mlx` runs the model with [MLX](https://github.com/ml-explore/mlx) on Apple silicon. It needs neither
PyTorch nor the rest of the training stack:

```bash
pip install mlx huggingface_hub numpy
pip install --no-deps "drifting-tts @ git+https://github.com/kadirnar/drifting-tts"
python -m drifting_tts.mlx --stream --text "Merhaba, nasılsınız?" --speaker studio --out merhaba.wav
```

```python
from drifting_tts.mlx import Synthesizer, write_wav
import mlx.core as mx

mx.set_cache_limit(256 * 1024 * 1024)      # process-wide unused allocator cache; not total memory
tts = Synthesizer.from_pretrained()        # downloads mlx/ from Vyvo/drifting-tts-tr (about 500 MB, once)
wav, info = tts("Merhaba, bu cümle MLX ile üretildi.", speaker="studio")
write_wav("merhaba.wav", wav)
```

For low latency, consume `tts.stream(text)` as it yields `(audio, info)` chunks. The first chunk contains 256 ms of
host-ready audio; later chunks grow to reduce repeated vocoder work. `tts(text)` still waits for the whole waveform.
See [streaming, vocoders, memory settings and reproducible benchmarks](docs/MLX.md).

- **Settings:** the same voices and options as the PyTorch API. The defaults are the recommended T = 0.3 and α = 2.
  The MLX port runs v3.1's pipeline: the v3.2 prosody predictor and Vocos v2 are not ported yet.
- **Vocoders:** BigVGAN-v2 by default; `Synthesizer.from_pretrained(vocoder="bigvgan-base-ft")` or `"vocos-ft"`
  (CLI `--vocoder`) uses a 14 M-parameter vocoder instead of 112 M ([quality](docs/VOCODERS.md),
  [MLX parity](docs/MLX.md#vocoders)). Only the chosen vocoder is downloaded.
- **Weights:** `python -m drifting_tts.mlx.convert` converts the PyTorch checkpoints. The result is published under
  [`mlx/`](https://huggingface.co/Vyvo/drifting-tts-tr/tree/main/mlx).
- **Mac validation:** tested on Apple M2 Pro (16 GB), macOS 26.5.2, MLX 0.32.3. With BigVGAN-v2, warm streaming
  TTFA is **84–97 ms** across short/long sentences and a paragraph ([methodology and results](docs/RESULTS.md#mlx)).
  The first request and optional `--compile` need separate measurement; these are not model-loading times. The small
  vocoders have not been timed on a Mac yet.

## On iPhone and native macOS

The native Swift/MLX engine, iPhone app, installation instructions and device benchmarks are maintained in
[**drifting-tts-swift**](https://github.com/kadirnar/drifting-tts-swift).

## Benchmark: Freya-TR-Eval

[Freya-TR-Eval](https://huggingface.co/datasets/freyavoice/freya-tr-eval) has 495 everyday Turkish sentences that this
model never saw. The protocol is the one in the FreyaTTS report: audio downsampled to 8 kHz, transcribed by
Whisper large-v3, and both texts normalised the same way. Lower is better.

| system | parameters | WER | CER |
|---|---|---|---|
| **drifting-tts v3.2, studio voice** | 68 M + 8 M prosody + 14 M vocoder | **1.33%** | **0.27%** |
| drifting-tts v3.2, male voice | 68 M + 8 M prosody + 14 M vocoder | 2.28% | 0.45% |
| drifting-tts v3.2, female voice | 68 M + 8 M prosody + 14 M vocoder | 3.99% | 0.97% |
| drifting-tts v3.1, studio voice | 68 M + 112 M vocoder | 1.23% | 0.24% |
| drifting-tts v3.1, male voice | 68 M + 112 M vocoder | 1.74% | 0.38% |
| drifting-tts v3.1, female voice | 68 M + 112 M vocoder | 3.02% | 0.70% |
| Piper (tr, dfki) | 16 M | 3.76% (report: 4.4%) | 0.83% (1.1%) |
| MMS-TTS (tr) | 36 M | 6.26% (report: 6.8%) | 1.43% (1.7%) |
| FreyaTTS | 183 M | report: 8.0% | report: 3.0% |
| XTTS-v2 | 470 M | report: 11.1% | report: 3.9% |

- Piper and MMS-TTS were re-run in this repository's harness. Their scores are close to those in the FreyaTTS
  report, so the numbers are comparable.
- "report" values are copied from the FreyaTTS report (arXiv 2607.09530, Table 2).
- Reproduce with `drifting-tts benchmark --model drifting_tts_v3.2.pt --vocoder vocos-v2 --prosody drift
  --prosody-durations regressor --pause punct --speaker studio` (v3.2; v3.1: `--model drifting_tts_v3.1.pt --vocoder
  bigvgan-v2-ft`), or all rows with `scripts/eval_release.sh`.

## How it works

```
text ─► Turkish normaliser ─► text encoder: durations ─► prosody model: pitch ─► DriftDiT (1 pass) ─► mel ─► Vocos ─► audio
```

1. A text encoder reads the text and predicts how long each character lasts. A small prosody model, also trained
   with drifting, samples the pitch of each character (v3.1: a deterministic regressor), and the text is laid out
   over time.
2. The **DriftDiT** generator turns random noise plus that layout into a mel spectrogram in one forward pass.
3. Training uses a **drifting field**: generated samples are pulled toward real recordings and pushed away from each
   other, so the model's output distribution moves toward the data distribution. Similarity is measured in the
   features of a frozen mel autoencoder, with a learned kernel temperature.
4. A vocoder fine-tuned on the model's own spectrograms (Vocos v2 in v3.2, BigVGAN-v2 in v3.1) turns the mel into a
   24 kHz waveform.

## Train it yourself

Any Hugging Face parquet dataset with `audio` and `text` columns works (`speaker` is optional).

```bash
pip install -e ".[dev,eval,bigvgan,score]"
drifting-tts prepare --dataset <hf-dataset-id> --out data/train --backend bigvgan --f0 --save-audio --dev-size 200
drifting-tts train-mae --config configs/mae2d.yaml --workdir runs/mae2d train.steps=60000  # feature encoder
drifting-tts score --data data/train                                                       # data quality scores
drifting-tts train --config configs/tts_v3.yaml --workdir runs/tts_v3                     # the TTS model
drifting-tts calibrate-durations --model runs/tts_v3/model_ema.pt --temperature 0.3
drifting-tts finetune-vocoder --config configs/vocoder_bigvgan.yaml --workdir runs/vocoder_v3 \
    tts.path=runs/tts_v3/model_ema.pt train.steps=15000
```

On one RTX 5090 the TTS model trains in 13.6 h (150k steps) and the vocoder fine-tunes in about 1 h. New voices can
be added later by fine-tuning ([docs/TRAINING.md](docs/TRAINING.md#adding-a-voice)).

## Documentation

| | |
|---|---|
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | **experiment log**: every experiment since v3.1, what worked and what did not, pitfalls, open work |
| [docs/RESULTS.md](docs/RESULTS.md) | benchmark details, voices, latency and parameter counts |
| [docs/SPEAKERS.md](docs/SPEAKERS.md) | WER, CER, DNSMOS, UTMOSv2, pitch and speaking rate of all 723 speaker IDs |
| [docs/VOCODERS.md](docs/VOCODERS.md) | the vocoder registry and a comparison on the same mels: quality, speed, streaming |
| [docs/TRAINING.md](docs/TRAINING.md) | the training recipe, the evidence behind each choice, adding a voice |
| [docs/DESIGN.md](docs/DESIGN.md) | how the drifting method maps to TTS, deviations from the paper, related work |
| [docs/EVALUATION.md](docs/EVALUATION.md) | evaluation judges, the benchmark command, data scoring and filtering |
| [docs/PROSODY.md](docs/PROSODY.md) | prosody metrics against recordings (`drifting-tts prosody`), oracle prosody, pitch gain, pause policy |
| [docs/PROSODY_MODEL.md](docs/PROSODY_MODEL.md) | the stochastic prosody predictor of v3.2 (drifting), against regression and flow matching |
| [docs/LATENTS.md](docs/LATENTS.md) | audio-VAE latent spaces (DAC-VAE, VoxCPM): the backends, their resynthesis ceiling, the latent TTS pilots and the VoxCPM2 and DAC-VAE decoder GTA fine-tunes |
| [space/](space/) | the Gradio demo (`scripts/deploy_space.sh` deploys it) |
| [drifting_tts/mlx/](drifting_tts/mlx/) | MLX inference for Apple silicon (`python -m drifting_tts.mlx`) |
| [drifting-tts-swift](https://github.com/kadirnar/drifting-tts-swift) | native Swift MLX engine and iPhone app (separate repository) |
| [web/](web/) | the WebGPU demo and the ONNX pipeline in JavaScript (`scripts/deploy_webgpu_space.sh` deploys it) |
| [scripts/bench_ttfa.py](scripts/bench_ttfa.py) | latency benchmark |
| [scripts/eval_release.sh](scripts/eval_release.sh) | the evaluation of a release against v3.1 (Freya, prosody); `scripts/prepare_release.py` stages its files |
| [scripts/compare_vocoders.py](scripts/compare_vocoders.py) | vocoder comparison ([docs/VOCODERS.md](docs/VOCODERS.md)) |

## Limitations

- **Voices:** three built-in voices; no voice cloning from a reference recording.
- **Prosody:** v3.2 samples the intonation with the recordings' spread, but the rhythm still comes from a
  deterministic duration predictor, and each sentence is generated without the context of its neighbours. Sampling
  the durations too (opt-in) costs intelligibility for the male and female voices.
- **Intelligibility:** v3.2's male and female voices are less intelligible than v3.1's (Freya WER 2.28% and 3.99%
  against 1.74% and 3.02%); `from_pretrained("v3.1")` remains available.
- **Naturalness:** measured only with automatic scores (UTMOSv2, DNSMOS, F0 statistics), not by listeners.

## Citation and license

```bibtex
@article{deng2026drifting,
  title   = {Generative Modeling via Drifting},
  author  = {Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal = {arXiv preprint arXiv:2602.04770},
  year    = {2026}
}
```

The code is MIT. The vocoders are fine-tuned from [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)
(MIT; v3.2) and [NVIDIA BigVGAN-v2](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x) (MIT; v3.1).

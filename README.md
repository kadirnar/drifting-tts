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
| **Quality** | [Freya-TR-Eval](https://huggingface.co/datasets/freyavoice/freya-tr-eval) WER **1.23%** (Piper 3.76%, MMS-TTS 6.26% under the same protocol) · UTMOSv2 2.94 |
| **Speed** (RTX 5090) | first audio after **12–14 ms** for any sentence length (streaming, CUDA graphs), 50–100× faster than real time |
| **Size** | 67.7 M acoustic model + 112 M BigVGAN-v2 vocoder |
| **Voices** | `studio` (default), `male` and `female` |

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
from huggingface_hub import hf_hub_download
from drifting_tts.synthesize import Synthesizer

repo = "Vyvo/drifting-tts-tr"
tts = Synthesizer(hf_hub_download(repo, "drifting_tts_v3.1.pt"), "cuda", vocoder="bigvgan-v2-ft")

wav, info = tts("Merhaba, bu cümle tek adımda üretildi.", speaker="studio", cfg_scale=2.0, temperature=0.3)
sf.write("merhaba.wav", wav.numpy(), 24000)
```

- `speaker`: `"studio"` (default, the clearest voice), `"male"` or `"female"`. Any of the model's 723 speaker
  IDs also works, e.g. `speaker=17`; [docs/SPEAKERS.md](docs/SPEAKERS.md) scores every one of them.
- `temperature`: the noise level. 0.3 sounds clearest; higher values give more variety.
- `cfg_scale`: the guidance strength, learned during training, so it costs nothing at inference.
- `vocoder`: `"bigvgan-v2-ft"` is the BigVGAN-v2 fine-tuned on this model's mels (downloaded from the Hub). Other
  names: the stock NVIDIA `"bigvgan-v2"`, `"bigvgan-v1"` and `"bigvgan-base"` (14 M), or the weight-free
  `"griffin-lim"`; a checkpoint path also works. [docs/VOCODERS.md](docs/VOCODERS.md) compares them.
- Numbers, dates, times, units, currencies and common abbreviations are read out in Turkish automatically.

The same from the command line:

```bash
hf download Vyvo/drifting-tts-tr --local-dir .
drifting-tts synthesize --model drifting_tts_v3.1.pt --vocoder bigvgan_v2_ft.pt --speaker female --cfg 2 \
    --text "Merhaba, nasılsınız?" --out merhaba.wav
```

**Lowest latency: streaming.** `fast=True` runs the acoustic model as CUDA graphs, with the same output. `stream()`
yields the audio in pieces, and on an RTX 5090 the first piece (0.34 s) is ready after about 12 ms. The vocoder streams
in overlapping windows whose pieces join into the whole-sentence audio; Freya WER and CER are unchanged
([details](docs/RESULTS.md#latency-and-size)):

```python
tts = Synthesizer(hf_hub_download(repo, "drifting_tts_v3.1.pt"), "cuda", vocoder="bigvgan-v2-ft",
                  cuda_kernel=True, fast=True)
for piece in tts.stream("Merhaba! Bu ses parça parça, bekletmeden geliyor.", speaker="studio",
                        cfg_scale=2.0, temperature=0.3):
    play(piece)   # float32 tensor at 24 kHz
```

## In the browser (WebGPU)

The model also runs entirely client-side with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/): the
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
| **drifting-tts v3.1, studio voice** | 68 M + 112 M vocoder | **1.23%** | **0.24%** |
| drifting-tts v3.1, male voice | 68 M + 112 M vocoder | 1.74% | 0.38% |
| drifting-tts v3.1, female voice | 68 M + 112 M vocoder | 3.02% | 0.70% |
| Piper (tr, dfki) | 16 M | 3.76% (report: 4.4%) | 0.83% (1.1%) |
| MMS-TTS (tr) | 36 M | 6.26% (report: 6.8%) | 1.43% (1.7%) |
| FreyaTTS | 183 M | report: 8.0% | report: 3.0% |
| XTTS-v2 | 470 M | report: 11.1% | report: 3.9% |

- Piper and MMS-TTS were re-run in this repository's harness. Their scores are close to those in the FreyaTTS
  report, so the numbers are comparable.
- "report" values are copied from the FreyaTTS report (arXiv 2607.09530, Table 2).
- Reproduce with `drifting-tts benchmark --model drifting_tts_v3.1.pt --vocoder bigvgan_v2_ft.pt --speaker studio`.

## How it works

```
text ─► Turkish normaliser ─► text encoder ─► durations + pitch ─► DriftDiT (1 pass) ─► mel ─► BigVGAN-v2 ─► audio
```

1. A text encoder predicts how long each character lasts and its pitch, then lays the text out over time.
2. The **DriftDiT** generator turns random noise plus that layout into a mel spectrogram in one forward pass.
3. Training uses a **drifting field**: generated samples are pulled toward real recordings and pushed away from each
   other, so the model's output distribution moves toward the data distribution. Similarity is measured in the
   features of a frozen mel autoencoder, with a learned kernel temperature.
4. BigVGAN-v2, fine-tuned on the model's own spectrograms, turns the mel into a 24 kHz waveform.

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
| [docs/RESULTS.md](docs/RESULTS.md) | benchmark details, voices, latency and parameter counts |
| [docs/SPEAKERS.md](docs/SPEAKERS.md) | WER, CER, DNSMOS, UTMOSv2, pitch and speaking rate of all 723 speaker IDs |
| [docs/VOCODERS.md](docs/VOCODERS.md) | the vocoder registry and a comparison on the same mels: quality, speed, streaming |
| [docs/TRAINING.md](docs/TRAINING.md) | the training recipe, the evidence behind each choice, adding a voice |
| [docs/DESIGN.md](docs/DESIGN.md) | how the drifting method maps to TTS, deviations from the paper, related work |
| [docs/EVALUATION.md](docs/EVALUATION.md) | evaluation judges, the benchmark command, data scoring and filtering |
| [docs/LATENTS.md](docs/LATENTS.md) | audio-VAE latent spaces (DAC-VAE, VoxCPM): the backends, their resynthesis ceiling, the latent TTS pilots and the VoxCPM2 and DAC-VAE decoder GTA fine-tunes |
| [space/](space/) | the Gradio demo (`scripts/deploy_space.sh` deploys it) |
| [drifting_tts/mlx/](drifting_tts/mlx/) | MLX inference for Apple silicon (`python -m drifting_tts.mlx`) |
| [drifting-tts-swift](https://github.com/kadirnar/drifting-tts-swift) | native Swift MLX engine and iPhone app (separate repository) |
| [web/](web/) | the WebGPU demo and the ONNX pipeline in JavaScript (`scripts/deploy_webgpu_space.sh` deploys it) |
| [scripts/bench_ttfa.py](scripts/bench_ttfa.py) | latency benchmark |
| [scripts/compare_vocoders.py](scripts/compare_vocoders.py) | vocoder comparison ([docs/VOCODERS.md](docs/VOCODERS.md)) |

## Limitations

- **Voices:** three built-in voices; no voice cloning from a reference recording.
- **Prosody:** durations and pitch come from simple predictors. Intonation is natural but flatter than in real speech.
- **Naturalness:** measured only with an automatic score (UTMOSv2), not by listeners.

## Citation and license

```bibtex
@article{deng2026drifting,
  title   = {Generative Modeling via Drifting},
  author  = {Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal = {arXiv preprint arXiv:2602.04770},
  year    = {2026}
}
```

The code is MIT. The vocoder is fine-tuned from
[NVIDIA BigVGAN-v2](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x) (MIT).

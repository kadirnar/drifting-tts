"""Drifting TTS demo: one-step Turkish text to speech.

text -> Turkish normaliser -> text encoder (durations, pitch) -> ONE DriftDiT pass -> BigVGAN-v2 fine-tuned on the
model's own mels. The acoustic model was trained with a drifting objective (Deng et al., arXiv 2602.04770) using
Kyutai's learned-temperature field. Code: https://github.com/kadirnar/drifting-tts. The weights are downloaded from
https://huggingface.co/Vyvo/drifting-tts-tr; `scripts/deploy_space.sh` copies this app and the package into the Space.
"""

import time

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer
from drifting_tts.text import normalize

try:
    import spaces

    gpu = spaces.GPU(duration=30)
except ImportError:  # running locally
    gpu = lambda f: f  # noqa: E731

MODEL_REPO = "Vyvo/drifting-tts-tr"
synth = Synthesizer(hf_hub_download(MODEL_REPO, "drifting_tts_v3.1.pt"), "cuda",
                    vocoder=hf_hub_download(MODEL_REPO, "bigvgan_v2_ft.pt"))
VOICES = {"Studio male voice (recommended)": "studio", "Male voice": "male", "Female voice": "female"}
MAX_CHARS = 600

EXAMPLES = [
    ["Merhaba! Bu ses, tek bir ağ değerlendirmesiyle üretildi; difüzyon adımı yok."],
    ["İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı."],
    ["Yapay zekâ modelleri her geçen gün daha hızlı ve daha verimli hâle geliyor."],
    ["Toplantı yarın saat 14:30'da, 2. katta; lütfen geç kalmayın."],
    ["Prof. Dr. Ayşe Yılmaz, 250 TL'lik bağışın tamamının öğrencilere ayrılacağını söyledi."],
    ["Bir varmış, bir yokmuş; evvel zaman içinde, kalbur saman içinde, küçük bir köyde yaşlı bir masalcı yaşarmış."],
]


@gpu
def generate(text, voice, temperature, guidance, rate, seed):
    text = " ".join((text or "").split())
    if not text:
        raise gr.Error("Please enter some Turkish text.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Please keep the text under {MAX_CHARS} characters.")
    if not normalize(text):
        raise gr.Error("Nothing to read after normalisation.")
    t0 = time.time()
    wav, info = synth(text, speaker=VOICES[voice], cfg_scale=float(guidance), temperature=float(temperature),
                      length_scale=1.0 / float(rate), seed=int(seed))
    took = time.time() - t0
    audio = (SAMPLE_RATE, np.clip(wav.float().numpy(), -1, 1))
    stats = (f"{info['seconds']:.1f} s of audio in {took:.2f} s, one generator pass per sentence. "
             f"Normalised text: *{normalize(text)}*")
    return audio, stats


with gr.Blocks(title="Drifting TTS: one-step Turkish TTS") as demo:
    gr.Markdown(
        "# Drifting TTS: one-step Turkish text to speech\n"
        "A single network evaluation turns text into a mel spectrogram; there are no diffusion steps. The model was "
        "trained with a **drifting** objective ([Deng et al., 2026](https://arxiv.org/abs/2602.04770)) using Kyutai's "
        "learned-temperature field. The vocoder is BigVGAN-v2, fine-tuned on the model's "
        "own mels. On the Freya-TR-Eval benchmark the studio voice reaches 1.23% WER. "
        "[Code](https://github.com/kadirnar/drifting-tts) · [Model](https://huggingface.co/Vyvo/drifting-tts-tr)"
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="Turkish text", lines=4, max_length=MAX_CHARS,
                              value=EXAMPLES[0][0], placeholder="Türkçe bir metin yazın...")
            voice = gr.Dropdown(list(VOICES), value=next(iter(VOICES)), label="Voice")
            with gr.Accordion("Settings", open=False):
                temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Noise temperature",
                                        info="Lower: clearer and more stable; higher: more varied (0.3 is best)")
                guidance = gr.Slider(1.0, 4.0, value=2.0, step=0.25, label="Guidance scale α",
                                     info="Classifier-free guidance learned at training time (free at inference)")
                rate = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Speaking rate")
                seed = gr.Number(value=0, precision=0, label="Seed")
            button = gr.Button("Synthesise", variant="primary")
        with gr.Column(scale=2):
            audio = gr.Audio(label="Speech (24 kHz)", type="numpy", autoplay=True)
            stats = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[text])
    button.click(generate, [text, voice, temperature, guidance, rate, seed], [audio, stats])

if __name__ == "__main__":
    demo.queue(max_size=20).launch()

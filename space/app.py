"""Drifting TTS demo: one-step Turkish text to speech, release v3.2 by default and v3.1 for comparison.

v3.2: text -> Turkish normaliser -> text encoder (durations) -> stochastic prosody predictor (token pitch sampled,
trained with the drifting objective) -> ONE DriftDiT pass -> Vocos v2, sentences joined by punctuation-aware pauses.
v3.1: the same acoustic weights with the deterministic pitch regressor, BigVGAN-v2-ft and 0.15 s pauses.
Code: https://github.com/kadirnar/drifting-tts. The weights are downloaded from https://huggingface.co/Vyvo/drifting-tts-tr
(``Synthesizer.from_pretrained``); `scripts/deploy_space.sh` copies this app and the package into the Space.
"""

import os
import time

import gradio as gr
import numpy as np

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer
from drifting_tts.text import normalize

try:
    import spaces

    gpu = spaces.GPU(duration=30)
except ImportError:  # running locally
    gpu = lambda f: f  # noqa: E731

DEVICE = os.environ.get("DEMO_DEVICE", "cuda")
v32 = Synthesizer.from_pretrained("v3.2", DEVICE)  # vocos-v2, prosody="drift" (pitch only), pause="punct"
v31 = v32.variant(vocoder="bigvgan-v2-ft", prosody=None, pause=0.15)  # same acoustic model, loaded once
# opt-in: the prosody model samples the durations (rhythm, pauses inside sentences) too; studio voice only, since the
# voices with little data then slip on words (Freya-495 WER male 5.78%, female 11.28%)
v32_rhythm = v32.variant(prosody=v32.prosody, prosody_durations="sampled")
RHYTHM = "v3.2 + sampled rhythm (experimental, studio voice only)"
RELEASES = {
    "v3.2 (new): sampled intonation, Vocos v2, punctuation pauses": v32,
    RHYTHM: v32_rhythm,
    "v3.1: deterministic intonation, BigVGAN-v2": v31,
}
VOICES = {"Studio male voice (recommended)": "studio", "Male voice": "male", "Female voice": "female"}
MAX_CHARS = 600

EXAMPLES = [
    ["Merhaba! Bu ses, tek bir ağ değerlendirmesiyle üretildi; difüzyon adımı yok."],
    ["İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı."],
    ["Yarın akşam ne yapıyorsun? Belki sinemaya gideriz, ne dersin?"],
    ["Toplantı yarın saat 14:30'da, 2. katta; lütfen geç kalmayın."],
    ["Prof. Dr. Ayşe Yılmaz, 250 TL'lik bağışın tamamının öğrencilere ayrılacağını söyledi."],
    ["Bir varmış, bir yokmuş; evvel zaman içinde, kalbur saman içinde, küçük bir köyde yaşlı bir masalcı yaşarmış."],
]

RESULTS_MD = """
**Freya-TR-Eval** (495 everyday sentences never seen in training; Whisper large-v3 on 8 kHz audio, UTMOSv2 on the
full band; noise temperature 0.3, α = 2). WER / UTMOSv2:

| voice | v3.2: sampled intonation + Vocos v2 (13.5 M) | v3.1: BigVGAN-v2-ft (112 M) |
|---|---|---|
| studio | 1.33% / 3.02 | 1.23% / 2.94 |
| male | 2.28% / 2.90 | 1.74% / 2.81 |
| female | 3.99% / 2.72 | 3.02% / 2.75 |

v3.2 samples only the intonation: the rhythm (durations) stays v3.1's. On 100 held-out sentences of the studio
voice its pitch spread is 3.53 semitones against 3.68 in the recordings and 3.21 for v3.1, so
the intonation is less flat; each seed gives another plausible tune. The male and female voices are somewhat less
intelligible than in v3.1.

**v3.2 + sampled rhythm** (experimental) lets the prosody model sample the durations as well: a livelier rhythm with
more pauses inside sentences (held-out studio texts: 2.35 pauses per utterance, against 1.39 in the recordings and
1.54 for v3.2). Studio voice, Freya-495: WER 1.89% / UTMOSv2 3.03 at prosody temperature 0.5, 1.46% / 3.04 at 0.3.
The other voices have too little data for it, so with them this option falls back to v3.2.
"""


@gpu
def generate(text, release, voice, temperature, guidance, prosody_temperature, rate, seed):
    text = " ".join((text or "").split())
    if not text:
        raise gr.Error("Please enter some Turkish text.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Please keep the text under {MAX_CHARS} characters.")
    if not normalize(text):
        raise gr.Error("Nothing to read after normalisation.")
    synth, note = RELEASES[release], ""
    if release == RHYTHM and VOICES[voice] != "studio":
        synth, note = v32, " Sampled rhythm is for the studio voice only: this voice used v3.2 (sampled intonation)."
    t0 = time.time()
    wav, info = synth(text, speaker=VOICES[voice], cfg_scale=float(guidance), temperature=float(temperature),
                      length_scale=1.0 / float(rate), seed=int(seed),
                      prosody_temperature=float(prosody_temperature) if synth.prosody is not None else None)
    took = time.time() - t0
    audio = (SAMPLE_RATE, np.clip(wav.float().numpy(), -1, 1))
    stats = (f"{info['seconds']:.1f} s of audio in {took:.2f} s, one generator pass per sentence. "
             f"Normalised text: *{normalize(text)}*{note}")
    return audio, stats


with gr.Blocks(title="Drifting TTS: one-step Turkish TTS") as demo:
    gr.Markdown(
        "# Drifting TTS: one-step Turkish text to speech\n"
        "A single network evaluation turns text into a mel spectrogram; there are no diffusion steps. The model was "
        "trained with a **drifting** objective ([Deng et al., 2026](https://arxiv.org/abs/2602.04770)) using Kyutai's "
        "learned-temperature field. **New in v3.2:** a small prosody model, also trained with drifting, samples the "
        "pitch of every sound, so the intonation is no longer the same flat average every time (change the seed to "
        "hear another tune; the rhythm stays v3.1's); a retrained Vocos vocoder (13.5 M) and pauses that follow the "
        "punctuation. v3.1 is one click away for comparison. "
        "[Code](https://github.com/kadirnar/drifting-tts) · [Model](https://huggingface.co/Vyvo/drifting-tts-tr)"
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="Turkish text", lines=4, max_length=MAX_CHARS,
                              value=EXAMPLES[0][0], placeholder="Türkçe bir metin yazın...")
            release = gr.Radio(list(RELEASES), value=next(iter(RELEASES)), label="Model")
            voice = gr.Dropdown(list(VOICES), value=next(iter(VOICES)), label="Voice")
            with gr.Accordion("Settings", open=False):
                temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Noise temperature",
                                        info="Lower: clearer and more stable; higher: more varied (0.3 is best)")
                guidance = gr.Slider(1.0, 4.0, value=2.0, step=0.25, label="Guidance scale α",
                                     info="Classifier-free guidance learned at training time (free at inference)")
                prosody_temperature = gr.Slider(
                    0.0, 1.0, value=v32.prosody_temperature, step=0.05, label="Prosody temperature (v3.2)",
                    info="How freely the pitch (and, with sampled rhythm, the durations) is sampled: 0.5 is the "
                         "tested setting; higher is more varied, 0 is the predictor's most typical reading")
                rate = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Speaking rate")
                seed = gr.Number(value=0, precision=0, label="Seed")
            button = gr.Button("Synthesise", variant="primary")
        with gr.Column(scale=2):
            audio = gr.Audio(label="Speech (24 kHz)", type="numpy", autoplay=True)
            stats = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[text])
    gr.Markdown(RESULTS_MD)
    button.click(generate, [text, release, voice, temperature, guidance, prosody_temperature, rate, seed],
                 [audio, stats])

if __name__ == "__main__":
    demo.queue(max_size=20).launch()

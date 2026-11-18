"""Drifting TTS English pilots (private test): ARDrift v2, the text as a prefix without an aligner (#103), and the
aligned model with a voice from a recording and guidance on the voice (#96, #99). The weights are in the Space
itself (fp16) or are downloaded from the pilots' model repo (``engine.py``; a private repo then needs the Space
secret ``HF_TOKEN``); ``scripts/deploy_pilots_space.sh`` copies this app, the package and the weights into the
Space. Not a release.
"""

import os

import gradio as gr
import torch
from engine import MAX_CHARS_CLONE, MAX_CHARS_V2, MODES, REPO, Pilots, default_device

from drifting_tts.audio import SAMPLE_RATE

try:
    import spaces

    gpu = spaces.GPU(duration=60)
except ImportError:  # running locally
    gpu = lambda f: f  # noqa: E731

DEVICE = default_device()
if DEVICE == "cpu":  # a CPU Space: its vCPUs, not the host's cores
    torch.set_num_threads(int(os.environ.get("DEMO_THREADS", "2")))
pilots = Pilots(DEVICE)

V2_EXAMPLES = [
    ["Wait, so I can just not go? Seriously?"],
    ["Oh man, the rain is just coming down sideways."],
    ["Well, you know how sometimes grown ups just run out of juice."],
    ["Honestly, I didn't expect the weather to turn this quickly, but here we are, stuck inside again."],
]
CLONE_EXAMPLES = [
    ["The meeting moved to Thursday, so we have two more days. Is that all right?"],
    ["Hello! It's 7:45, and the 90s called. They want their dial-up modem back."],
    ["I spent $12.50 on coffee this week, which is honestly about 20% of my budget."],
]
SLOW = ("\n\n**This Space runs on a CPU: a sentence takes about half a minute.**" if DEVICE == "cpu" else "")

V2_MD = """
The text is a prefix of the sequence, `[voice] [text] [audio]`, through one causal transformer: no aligner, no
duration predictor. The model decides the timing itself. It is a **pilot of 40,000 small steps**, not a model to
speak with: about one word in three is wrong or missing.

| steps | temperature | WER, 100 held-out sentences | words said | of them the text's |
|---|---|---|---|---|
| 20,000 | 1.0 | 70.9% | 100% | 45% |
| 40,000 | 0.5 | 55.1% | 48% | 94% |
| 40,000 | **1.0** | **35.5%** | 86% | 81% |
| the released aligned model | 0.5 | 2.1% | 100% | 98% |

- **Temperature 1.0** is the measured setting. At 0.5 the model falls silent after the first words.
- Short sentences work best. The utterance ends where the text ends on the model's clock; "speaking rate" moves
  that clock.
"""
CLONE_MD = """
The released aligned model, fine-tuned to take a voice from a recording: the recording's own patches stand in front
of the sequence as reference tokens, and guidance pushes towards that voice. The first six seconds of the recording
are used; no transcript is needed.

36 recorded speakers the model never saw (similarity with the reference recording / with a recording the model was
not given / WER):

| voice given as | SIM-o | SIM, unheard | WER |
|---|---|---|---|
| the speaker's own recording | 0.656 | 0.650 | 2.35% |
| reference, guidance 1 | 0.438 | 0.387 | 2.79% |
| reference, guidance 3 | 0.563 | 0.466 | 3.04% |
| reference, guidance 5 | 0.593 | 0.485 | 3.09% |
| prompt (the recording continued), no guidance | 0.546 | 0.464 | 3.24% |

- **reference**: the voice alone, the recording is not continued. **prompt**: the model continues the recording,
  which also copies its room and microphone; it runs without guidance. **prompt + reference**: the two together.
- Use a clean recording of one speaker, and only a voice you have the right to use.
"""


@gpu
def speak(text, voice, steps, temperature, speed, seed):
    try:
        wav, info = pilots.speak_v2(text, voice, steps, temperature, speed, seed)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    return (SAMPLE_RATE, wav), info


@gpu
def clone(text, recording, mode, scale, temperature, seed):
    try:
        wav, info = pilots.clone(text, recording, mode, scale, temperature, seed)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    return (SAMPLE_RATE, wav), info


with gr.Blocks(title="Drifting TTS: English pilots") as demo:
    gr.Markdown(
        "# Drifting TTS: English pilots\n"
        "Two research pilots of [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts), each a short "
        f"fine-tune of the released English model. Weights: [{REPO}](https://huggingface.co/{REPO}). The released "
        "model has its own demo: [Vyvo/drifting-tts-en-demo](https://huggingface.co/spaces/Vyvo/drifting-tts-en-demo)."
        + SLOW
    )
    with gr.Tab("ARDrift v2: the text as a prefix"):
        with gr.Row():
            with gr.Column(scale=3):
                v2_text = gr.Textbox(label="English text (one sentence)", lines=3, max_length=MAX_CHARS_V2,
                                     value=V2_EXAMPLES[0][0])
                v2_voice = gr.Dropdown(list(pilots.voices), value=next(iter(pilots.voices)), label="Voice")
                v2_steps = gr.Radio(list(pilots.v2), value=next(iter(pilots.v2)), label="Checkpoint")
                with gr.Accordion("Settings", open=False):
                    v2_temperature = gr.Slider(0.3, 1.3, value=1.0, step=0.05, label="Noise temperature",
                                               info="1.0 is the measured setting")
                    v2_speed = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Speaking rate")
                    v2_seed = gr.Number(value=0, precision=0, label="Seed")
                v2_button = gr.Button("Speak", variant="primary")
            with gr.Column(scale=2):
                v2_audio = gr.Audio(label="Speech (24 kHz)", type="numpy", autoplay=True)
                v2_info = gr.Markdown()
        gr.Examples(V2_EXAMPLES, inputs=[v2_text])
        gr.Markdown(V2_MD)
        v2_button.click(speak, [v2_text, v2_voice, v2_steps, v2_temperature, v2_speed, v2_seed], [v2_audio, v2_info],
                        api_name="speak_v2")
    with gr.Tab("A voice from a recording"):
        with gr.Row():
            with gr.Column(scale=3):
                c_text = gr.Textbox(label="English text", lines=3, max_length=MAX_CHARS_CLONE,
                                    value=CLONE_EXAMPLES[0][0])
                c_recording = gr.Audio(label="The voice: a recording of 3 to 10 seconds", type="filepath",
                                       sources=["upload", "microphone"])
                c_mode = gr.Radio(list(MODES), value=next(iter(MODES)), label="Voice given as")
                with gr.Accordion("Settings", open=False):
                    c_scale = gr.Slider(1.0, 5.0, value=3.0, step=0.5, label="Guidance on the voice",
                                        info="1: none. Higher: closer to the recording's voice")
                    c_temperature = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Noise temperature",
                                              info="0.5 is the measured setting")
                    c_seed = gr.Number(value=0, precision=0, label="Seed")
                c_button = gr.Button("Speak", variant="primary")
            with gr.Column(scale=2):
                c_audio = gr.Audio(label="Speech (24 kHz)", type="numpy", autoplay=True)
                c_info = gr.Markdown()
        gr.Examples(CLONE_EXAMPLES, inputs=[c_text])
        gr.Markdown(CLONE_MD)
        c_button.click(clone, [c_text, c_recording, c_mode, c_scale, c_temperature, c_seed], [c_audio, c_info],
                       api_name="clone")
    gr.Markdown("**License.** CC BY-NC-SA 4.0, for non-commercial use: the training speech was generated with a "
                "CC-BY-NC-SA-4.0 model.")

if __name__ == "__main__":
    demo.queue(max_size=20).launch()

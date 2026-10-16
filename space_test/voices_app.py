"""Drifting TTS, new voices (the "Yeni sesler" tab of the private test Space, app.py): ARDrift (docs/AR_DRIFT.md)
fine-tuned with six voices learned from 16 kHz recordings, and the mel bandwidth extension (drifting_tts/bwe.py) that
fills their mels above ~8 kHz. Runs on its own too (``python voices_app.py``).

Files beside this app (``scripts/deploy_test_space.sh``): ``ardrift_voices.pt`` (the model), ``bwe.pt`` (MelBWE),
``voices.json`` (each voice's label, speaker ID, whether its recordings are band-limited and its measured results)
and, optionally:
- ``durations_tr.pt``: the duration predictor re-fitted to the mean duration (``train-durations``), offered next to the
  model's own predictor and its per-voice tempo factors;
- ``ardrift_voices_restored.pt``: the same model fine-tuned on the voices' recordings restored with Sidon
  (``restore-audio``, #71), offered first next to it. Its voices learned from full-band audio, so it takes no
  bandwidth extension; ``voices.json`` may carry its results as ``results_restored``.
"""

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.bwe import MelBWE, extend_vocoder
from drifting_tts.synthesize import Synthesizer
from drifting_tts.text import normalize
from drifting_tts.train_durations import duration_choices, use_durations

try:
    import spaces

    gpu = spaces.GPU(duration=40)
except ImportError:  # running locally
    gpu = lambda f: f  # noqa: E731

HERE = Path(__file__).parent
DEVICE = os.environ.get("DEMO_DEVICE", "cuda")
CFG = json.loads((HERE / "voices.json").read_text())
VOICES = {v["label"]: v for v in CFG["voices"]}
synth = Synthesizer(HERE / "ardrift_voices.pt", DEVICE, vocoder="vocos-v2", pause="punct")
bwe = MelBWE.load(HERE / "bwe.pt", DEVICE)
wide = {c: synth.variant(vocoder=extend_vocoder(synth.vocoder, bwe, c))
        for c in {v["cutoff_hz"] for v in CFG["voices"] if v.get("band_limited")}}
RESTORED = HERE / "ardrift_voices_restored.pt"
MODELS = {"Onarılmış kayıtlarla (yeni)": Synthesizer(RESTORED, DEVICE, vocoder="vocos-v2", pause="punct"),
          "Orijinal kayıtlarla": synth} if RESTORED.is_file() else {}  # label -> model; the first is the default
DURATIONS = {}  # label -> durations; the synthesizers above share the one model, its durations are swapped
if (HERE / "durations_tr.pt").is_file():  # first: the default
    DURATIONS = duration_choices(synth.model, HERE / "durations_tr.pt", "Yeni: ortalama süreler",
                                 "Eski: modelin kendi süreleri ve ses başına hız katsayıları")
MAX_CHARS = 600

EXAMPLES = [
    ["Merhaba! Bugün sizlere çok ilginç bir konudan bahsetmek istiyorum."],
    ["İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı."],
    ["Yarın akşam ne yapıyorsun? Belki sinemaya gideriz, ne dersin?"],
    ["Toplantı yarın saat 14:30'da, 2. katta; lütfen geç kalmayın."],
    ["Bir varmış, bir yokmuş; evvel zaman içinde, kalbur saman içinde, küçük bir köyde yaşlı bir masalcı yaşarmış."],
]


def results_md() -> str:
    rows = ["| ses | kayıt | WER | CER | UTMOSv2 | SIM |", "|---|---|---|---|---|---|"]
    for v in CFG["voices"]:
        band = "16 kHz" if v.get("band_limited") else "24 kHz"
        runs = [(band, v.get("results", {}))]
        if "results_restored" in v:  # the model trained on the restored recordings, under the one above
            runs = [(f"{band} (eski model)", runs[0][1]), ("yeni model", v["results_restored"])]
        for kind, r in runs:
            rows.append(f"| {v['label']} | {kind} | " + " | ".join(
                f"{r[k]:.2f}%" if k in ("wer", "cer") and k in r else f"{r[k]:.3f}" if k in r else "–"
                for k in ("wer", "cer", "utmos", "sim")) + " |")
    return CFG.get("results_intro", "") + "\n\n" + "\n".join(rows) + "\n\n" + CFG.get("results_note", "")


@gpu
def generate(text, voice, trained, extend, durations, temperature, rate, seed):
    text = " ".join((text or "").split())
    if not text:
        raise gr.Error("Lütfen Türkçe bir metin girin.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Metin en fazla {MAX_CHARS} karakter olabilir.")
    if not normalize(text):
        raise gr.Error("Normalleştirmeden sonra okunacak bir şey kalmadı.")
    v = VOICES[voice]
    restored = bool(MODELS) and MODELS[trained] is not synth  # full-band voices: no bandwidth extension
    if DURATIONS and not restored:
        use_durations(synth.model, DURATIONS[durations])
    s = MODELS[trained] if restored else wide[v["cutoff_hz"]] if extend and v.get("band_limited") else synth
    t0 = time.time()
    wav, info = s(text, speaker=int(v["id"]), cfg_scale=1.0, temperature=float(temperature),
                  length_scale=1.0 / float(rate), seed=int(seed))
    took = time.time() - t0
    band = ("" if restored or not v.get("band_limited") else " Bant genişletme: açık." if s is not synth
            else " Bant genişletme: kapalı (16 kHz bant).")
    stats = f"{info['seconds']:.1f} s ses, {took:.2f} s'de üretildi.{band} Normalleştirilmiş metin: *{normalize(text)}*"
    return (SAMPLE_RATE, np.clip(wav.float().numpy(), -1, 1)), stats


with gr.Blocks(title="Drifting TTS: yeni sesler (test)") as demo:
    gr.Markdown(
        "# Drifting TTS: yeni sesler (özel test)\n"
        "ARDrift, kendi otoregresif modelimiz: nedensel bir transformer meli 43 ms'lik parçalar halinde üretir, tek "
        "adımlı bir drifting başlığı her parçayı örnekler. Yeni sesler 16 kHz kayıtlardan öğrenildi; **bant "
        "genişletme** onların 8 kHz üstünü doldurur (kapatıp karşılaştırabilirsiniz)."
        + (" **Onarılmış kayıtlarla** eğitilen model, aynı seslerin Sidon ile gürültüden arındırılıp tam banda "
           "genişletilmiş kayıtlarından öğrendi; bant genişletme gerektirmez." if MODELS else "")
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="Türkçe metin", lines=4, max_length=MAX_CHARS, value=EXAMPLES[0][0])
            voice = gr.Dropdown(list(VOICES), value=next(iter(VOICES)), label="Ses")
            trained = gr.Radio(list(MODELS) or ["-"], value=next(iter(MODELS), "-"), label="Model",
                               info="Yeni seslerin hangi kayıtlarından öğrenildiği", visible=bool(MODELS))
            extend = gr.Checkbox(value=True, label="Bant genişletme (16 kHz sesler için)")
            durations = gr.Radio(list(DURATIONS) or ["-"], value=next(iter(DURATIONS), "-"), label="Duraklamalar",
                                 info="Yeni: virgül ve kelime aralarında kayıtlardaki kadar durur"
                                 + (" (onarılmış kayıtlarla eğitilen model hep yenileri kullanır)" if MODELS else ""),
                                 visible=bool(DURATIONS))
            with gr.Accordion("Ayarlar", open=False):
                temperature = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Gürültü sıcaklığı",
                                        info="ARDrift'te sonucu çok az değiştirir (0.5 ölçülen ayar)")
                rate = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Konuşma hızı")
                seed = gr.Number(value=0, precision=0, label="Seed")
            button = gr.Button("Seslendir", variant="primary")
        with gr.Column(scale=2):
            audio = gr.Audio(label="Konuşma (24 kHz)", type="numpy", autoplay=True)
            stats = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[text])
    gr.Markdown(results_md())
    button.click(generate, [text, voice, trained, extend, durations, temperature, rate, seed], [audio, stats])

if __name__ == "__main__":
    demo.queue(max_size=20).launch()

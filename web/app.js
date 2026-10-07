import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { DriftingTTS, SAMPLE_RATE, toWav } from "./tts.js";

const params = new URLSearchParams(location.search);
// pinned to a revision: the browser cache is keyed by URL, so a model update must change the URL
const MODELS = params.get("models") ??
  "https://huggingface.co/Vyvo/drifting-tts-tr/resolve/501190557dec33d0127187b1f40cb3c698c27a35/onnx";
const EXAMPLES = [
  "Merhaba, nasılsınız? Bugün hava çok güzel.",
  "İstanbul'dan Ankara'ya giden hızlı tren saat 09.15'te kalkıyor.",
  "Ürünün fiyatı 1.250 TL, kargo ücreti ise 39,90 TL'dir.",
  "Bu model, metni tek bir ağ geçişinde konuşmaya dönüştürüyor.",
];

const $ = (id) => document.getElementById(id);
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let tts = null;
let audioCtx = null;

for (const id of ["temperature", "cfg", "speed"]) {
  const show = () => ($(`${id}Out`).textContent = Number($(id).value).toFixed(2));
  $(id).addEventListener("input", show);
  show();
}
for (const text of EXAMPLES) {
  const b = document.createElement("button");
  b.textContent = text.length > 38 ? `${text.slice(0, 36)}…` : text;
  b.title = text;
  b.onclick = () => ($("text").value = text);
  $("examples").append(b);
}

function setStatus(text, fraction = null) {
  $("status").textContent = text;
  if (fraction !== null) $("bar").style.width = `${Math.round(100 * fraction)}%`;
}

function warn(text) {
  $("warn").hidden = false;
  $("warn").textContent = text;
}

async function init() {
  try {
    const mb = (b) => (b / 1e6).toFixed(0);
    tts = await DriftingTTS.load(ort, MODELS, {
      onProgress: (got, total) => setStatus(`Downloading the model… ${mb(got)} / ${mb(total)} MB`, total ? got / total : 0),
    });
    $("backend").textContent = tts.provider === "webgpu" ? "WebGPU" : "WASM (CPU)";
    if (tts.provider !== "webgpu") {
      warn("WebGPU is not available in this browser, so the model runs on the CPU and is much slower. " +
           "Use a recent Chrome or Edge on a desktop for real-time speed.");
    }
    setStatus("Preparing the GPU (first run compiles the kernels)…", 1);
    const t0 = performance.now();
    await tts.synthesize("Merhaba.", { voice: "studio" });
    setStatus(`Ready (warm-up took ${((performance.now() - t0) / 1000).toFixed(1)} s).`, 1);
    $("progress").hidden = true;
    $("speak").disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("Failed to load the model.");
    warn(String(e?.message ?? e));
  }
}

async function speak() {
  const text = $("text").value.trim();
  if (!text || !tts) return;
  $("speak").disabled = true;
  audioCtx ??= new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioCtx.resume();
  const opts = {
    voice: $("voice").value,
    temperature: Number($("temperature").value),
    cfgScale: Number($("cfg").value),
    lengthScale: 1 / Number($("speed").value),
    seed: Number($("seed").value) || 0,
  };
  const pause = new Float32Array(Math.round(0.15 * SAMPLE_RATE));
  const parts = [];
  let playAt = 0, firstMs = null, acoustic = 0, vocoder = 0;
  const t0 = performance.now();
  setStatus("Generating…");
  try {
    for await (const r of tts.stream(text, opts)) {
      if (firstMs === null) firstMs = performance.now() - t0;
      acoustic += r.acousticMs;
      vocoder += r.vocoderMs;
      const chunk = r.index > 0 && r.piece === 0 ? concat([pause, r.audio]) : r.audio;
      parts.push(chunk);
      // stream: play every piece as soon as it is ready, back to back
      const buf = audioCtx.createBuffer(1, chunk.length, SAMPLE_RATE);
      buf.copyToChannel(chunk, 0);
      const src = audioCtx.createBufferSource();
      src.buffer = buf;
      src.connect(audioCtx.destination);
      playAt = Math.max(playAt, audioCtx.currentTime + 0.02);
      src.start(playAt);
      playAt += buf.duration;
      setStatus(`Generating… sentence ${r.index + 1} / ${r.count}`);
    }
    const audio = concat(parts);
    const total = performance.now() - t0;
    const url = URL.createObjectURL(toWav(audio));
    $("audio").src = url;
    $("download").href = url;
    $("result").hidden = false;
    const seconds = audio.length / SAMPLE_RATE;
    $("stats").textContent =
      `${seconds.toFixed(1)} s of audio · first audio after ${firstMs.toFixed(0)} ms · total ${total.toFixed(0)} ms ` +
      `(acoustic model ${acoustic.toFixed(0)} ms, vocoder ${vocoder.toFixed(0)} ms) · ` +
      `${(seconds / (total / 1000)).toFixed(1)}× real time on ${tts.provider === "webgpu" ? "WebGPU" : "CPU"}`;
    setStatus("Ready.");
  } catch (e) {
    console.error(e);
    setStatus("Generation failed.");
    warn(String(e?.message ?? e));
  } finally {
    $("speak").disabled = false;
  }
}

function concat(arrays) {
  const out = new Float32Array(arrays.reduce((s, a) => s + a.length, 0));
  let off = 0;
  for (const a of arrays) {
    out.set(a, off);
    off += a.length;
  }
  return out;
}

$("speak").addEventListener("click", speak);
init();

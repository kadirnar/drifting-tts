// Drifting TTS in the browser: the three ONNX graphs of scripts/export_onnx.py glued together.
//
//   text -> normalize -> ids -> text_encoder -> repeat each token ceil(exp(logw) * scale) times
//        -> generator (one pass, noise * temperature) -> log-mel -> vocoder -> 24 kHz audio
//
// Works with onnxruntime-web (WebGPU, falling back to WASM) and onnxruntime-node; pass the `ort` module in.

import { normalize, splitSentences, textToIds } from "./text.js";

export const SAMPLE_RATE = 24000;
const GRAPHS = ["text_encoder", "generator", "vocoder"];

// Seeded uniform PRNG (mulberry32) and standard normals (Box-Muller): reproducible noise for a given seed.
export function makeRandom(seed = 0) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    const u = 1 - uniform(), v = uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  return { uniform, normal, int: (n) => Math.floor(uniform() * n) };
}

// Hard alignment of the encoder output: column i of `cond` [C, N] repeated durations[i] times -> [C, T].
export function expandTokens(cond, channels, n, durations) {
  const T = Math.max(1, durations.reduce((s, d) => s + d, 0));
  const out = new Float32Array(channels * T);
  for (let c = 0; c < channels; c++) {
    let t = c * T;
    for (let i = 0; i < n; i++) {
      const v = cond[c * n + i];
      for (let k = 0; k < durations[i]; k++) out[t++] = v;
    }
  }
  return { frames: out, T };
}

// ceil(exp(logw) * scale) in float32, as durations_to_alignment does in PyTorch.
export function durations(logw, scale) {
  const s = Math.fround(scale);
  return Array.from(logw, (l) => Math.max(0, Math.ceil(Math.fround(Math.fround(Math.exp(l)) * s))));
}

// Download with progress, keeping a copy in the Cache API so later visits load offline.
export async function fetchCached(url, onProgress = () => {}, cacheName = "drifting-tts-v1") {
  let cache = null;
  try {
    cache = await caches.open(cacheName);
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      onProgress(buf.byteLength, buf.byteLength);
      return buf;
    }
  } catch {
    cache = null; // no Cache API (private window, file://): download every time
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  try {
    await cache?.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
  } catch {
    // quota exceeded: keep working without the cache
  }
  return buf.buffer;
}

// requestAdapter() can return null on the first call while the GPU process starts: ask twice.
async function hasWebGPU() {
  if (typeof navigator === "undefined" || !navigator.gpu) return false;
  for (let i = 0; i < 2; i++) {
    if (await navigator.gpu.requestAdapter().catch(() => null)) return true;
  }
  return false;
}

export class DriftingTTS {
  constructor(ort, sessions, config, provider) {
    this.ort = ort;
    this.sessions = sessions;
    this.config = config;
    this.provider = provider;
  }

  // `load(url)` reads `url`/config.json, then the graphs it lists under "web" (fp16-weight generator and vocoder);
  // `suffix` ("" or "_fp16") picks one variant for all three instead. `onProgress(loaded, total, file)` reports the
  // download.
  static async load(ort, baseUrl, { suffix = null, providers = null, onProgress = () => {}, fetchBytes = null } = {}) {
    const get = fetchBytes ?? fetchCached;
    const config = JSON.parse(new TextDecoder().decode(await get(`${baseUrl}/config.json`)));
    const names = GRAPHS.map((g) => (suffix === null && config.web?.[g]) || `${g}${suffix ?? "_fp16"}.onnx`);
    const sizes = names.map((n) => config.files?.[n] ?? 0);
    const total = sizes.reduce((s, x) => s + x, 0);
    const loaded = names.map(() => 0);
    const buffers = await Promise.all(names.map((n, i) => get(`${baseUrl}/${n}`, (got, size) => {
      loaded[i] = got;
      onProgress(loaded.reduce((s, x) => s + x, 0), total || size, n);
    })));
    const eps = providers ?? ((await hasWebGPU()) ? ["webgpu", "wasm"] : ["wasm"]);
    const sessions = {};
    for (let i = 0; i < GRAPHS.length; i++) {
      sessions[GRAPHS[i]] = await ort.InferenceSession.create(new Uint8Array(buffers[i]), {
        executionProviders: eps,
        graphOptimizationLevel: "all",
      });
      buffers[i] = null;
    }
    return new DriftingTTS(ort, sessions, config, eps[0]);
  }

  voiceId(voice) {
    const v = this.config.voices[voice];
    if (v === undefined) throw new Error(`unknown voice ${voice}`);
    return v;
  }

  tensor(type, data, dims) {
    return new this.ort.Tensor(type, data, dims);
  }

  // One sentence (already normalised) -> Float32Array audio. `random` is shared across the sentences of a text.
  async sentence(text, speaker, { temperature, cfgScale, lengthScale, random, noise = null }) {
    const cfg = this.config;
    const ids = textToIds(text, { normalized: true });
    const spk = this.tensor("int64", BigInt64Array.from([BigInt(speaker)]), [1]);
    const t0 = performance.now();
    const enc = await this.sessions.text_encoder.run({
      text: this.tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
      speaker: spk,
    });
    const C = enc.cond.dims[1], N = enc.cond.dims[2];
    const scale = (cfg.duration_scales[String(speaker)] ?? cfg.duration_scale) * lengthScale;
    const { frames, T } = expandTokens(enc.cond.data, C, N, durations(enc.logw.data, scale));
    const zData = new Float32Array(cfg.n_mels * T);
    const labels = new BigInt64Array(cfg.noise_coords);
    if (noise) {
      noise(zData, labels, T); // tests: inject the noise of a reference run
    } else {
      for (let i = 0; i < zData.length; i++) zData[i] = random.normal();
      for (let i = 0; i < labels.length; i++) labels[i] = BigInt(random.int(cfg.noise_classes));
    }
    for (let i = 0; i < zData.length; i++) zData[i] *= temperature;
    const gen = await this.sessions.generator.run({
      z: this.tensor("float32", zData, [1, cfg.n_mels, T]),
      cond: this.tensor("float32", frames, [1, C, T]),
      speaker: spk,
      cfg_scale: this.tensor("float32", Float32Array.from([cfgScale]), [1]),
      noise_labels: this.tensor("int64", labels, [1, cfg.noise_coords]),
    });
    const t1 = performance.now();
    const voc = await this.sessions.vocoder.run({ mel: gen.mel });
    const audio = voc.audio.data instanceof Float32Array ? voc.audio.data : Float32Array.from(voc.audio.data);
    return { audio, acousticMs: t1 - t0, vocoderMs: performance.now() - t1, frames: T };
  }

  // Sentence by sentence: yields each sentence's audio as soon as it is ready (streaming playback).
  async *stream(text, { voice = this.config.default_voice, temperature = this.config.temperature ?? 0.3,
                       cfgScale = 2.0, lengthScale = 1.0, seed = 0 } = {}) {
    const speaker = typeof voice === "number" ? voice : this.voiceId(voice);
    const random = makeRandom(seed);
    const sentences = splitSentences(normalize(text));
    for (let i = 0; i < sentences.length; i++) {
      const r = await this.sentence(sentences[i], speaker, { temperature, cfgScale, lengthScale, random });
      yield { ...r, index: i, count: sentences.length, text: sentences[i] };
    }
  }

  // Whole text -> one Float32Array, sentences joined with `pause` seconds of silence.
  async synthesize(text, opts = {}) {
    const pause = new Float32Array(Math.round((opts.pause ?? 0.15) * SAMPLE_RATE));
    const parts = [];
    const stats = { acousticMs: 0, vocoderMs: 0, firstAudioMs: null };
    const t0 = performance.now();
    for await (const r of this.stream(text, opts)) {
      if (stats.firstAudioMs === null) stats.firstAudioMs = performance.now() - t0;
      stats.acousticMs += r.acousticMs;
      stats.vocoderMs += r.vocoderMs;
      if (parts.length) parts.push(pause);
      parts.push(r.audio);
    }
    const audio = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
    let off = 0;
    for (const p of parts) {
      audio.set(p, off);
      off += p.length;
    }
    return { audio, ...stats, totalMs: performance.now() - t0, seconds: audio.length / SAMPLE_RATE };
  }
}

// 16-bit PCM WAV for download / the <audio> element.
export function toWav(audio, sampleRate = SAMPLE_RATE) {
  const buf = new ArrayBuffer(44 + audio.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + audio.length * 2, true);
  str(8, "WAVEfmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, audio.length * 2, true);
  for (let i = 0; i < audio.length; i++) {
    const s = Math.max(-1, Math.min(1, audio[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

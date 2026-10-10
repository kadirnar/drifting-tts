# Latency: one request, and 64 / 128 / 256 requests at once

How long a listener waits for the first audio of the released **v3.2** on one RTX 5090, alone and when many
requests arrive together. All numbers below were measured on 10 October 2026 on an otherwise idle GPU.

| requests at the same instant | first audio, one after another (FIFO) | first audio, batched |
|---|---|---|
| 1 | **5.8–7.4 ms** (p99 ≤ 7.5 ms) | 13 ms (the batched passes run without CUDA graphs) |
| 64 | median 413 ms, last 813 ms | **71 ms** for every request |
| 128 | median 825 ms, last 1.64 s | **150 ms** for every request |
| 256 | median 1.66 s, last 3.19 s | **304 ms** for every request |

- **One request:** the published v3.2 row reproduces (5.8 / 7.4 / 7.0 ms for a short sentence, a long sentence and a
  paragraph). Over two rounds of 100 runs, p99 was at most 0.7 ms above the median.
- **Many requests:** with the single-request fast path, requests queue. Request *k* starts speaking after about
  7 ms + 12.4 ms × *k*, and the GPU generates 420–460 s of audio per second whatever the load.
- **Batched serving:** one batched pass gives all 256 requests their first piece after 304 ms. All of their audio is
  done after 0.69 s, which is 1,950 s of audio per second, 4.6 times FIFO.
- **No gaps:** in every configuration, each later piece arrived while the earlier audio was still playing. At 256
  batched requests, at least 249 ms of audio was still buffered when the next piece arrived.

## Setup

**System under test:** `Synthesizer.from_pretrained("v3.2", "cuda", fast=True)`. This loads v3.1's acoustic
weights, the `drift` prosody predictor that samples the token pitch, Vocos v2 and punctuation pauses. The Hub files
are `drifting_tts_v3.2.pt`, `prosody_drift_v3.2.pt` and `vocos_v2.pt`, with sha256 7940508e…, 2d3733e7… and
a9e99783… (identical to the staged `runs/rel_publish/` files). Settings:

- studio voice, T = 0.3, α = 2;
- fp32 as shipped: PyTorch defaults, so fp32 matmuls, with cuDNN free to pick TF32 convolution kernels;
- pause 0 between a paragraph's sentences, so the seconds of audio count speech only.

The reference row is v3.1 + BigVGAN-v2-ft with its CUDA kernel (`from_pretrained("v3.1", cuda_kernel=True)`).

**Machine:**

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB, driver 580.173.02 (CUDA 13.0) |
| software | CUDA runtime 12.8, cuDNN 9.19, PyTorch 2.11.0+cu128, Python 3.12.14 |
| CPU | AMD EPYC 9135, container quota 7.68 CPUs, `OMP_NUM_THREADS=2` (2 torch threads) |

Before each benchmark script, `nvidia-smi` showed no other process on the GPU, and the other jobs on the machine kept
off it during the runs. `bench_concurrency.py` also records the GPU processes it sees at its start: none.

## Definitions

- **TTFA (time to first audio):** the time from the call with the input text to the first audio piece on the host.
  Text normalisation and sentence splitting are included. `Synthesizer.stream` returns CPU tensors, so the copy
  from the GPU is included too. A piece is float32 PCM at 24 kHz, and the first one is 0.34 s long (32 mel frames).
  The clock starts after `torch.cuda.synchronize()` on an idle GPU. Not included: network, audio encoding,
  playback.
- **Single request:** an idle GPU, the model warm (one cold call, then 10 warm-up paragraphs), `fast=True` streaming
  (CUDA graphs over length buckets, streaming vocoder windows), 100 runs per input with seeds 0–99
  (`scripts/bench_ttfa.py`). The inputs are a short sentence (1.3 s of audio), a long sentence (6.0 s) and a
  4-sentence paragraph (17.4 s). **RTF** is the median total time divided by the audio duration.
- **N concurrent requests:** all N requests arrive at the same instant *t0*. Each request's TTFA runs from *t0* to
  its own first piece on the host, so queueing is included (`scripts/bench_concurrency.py`).
  - **Request mix:** Freya-TR-Eval sentences in order. Every fourth request is a paragraph of three consecutive
    sentences; the others are single sentences. For N = 256 that is 192 sentences and 64 paragraphs: 385 sentences
    and 1,340 s of audio, 5.2 s per request on average. One Freya item has two sentences.
  - **Seeds:** request *i* uses seed *i* under every strategy, so the audio is the same.
  - **Runs:** each (strategy, N) runs once to warm up, then 5 timed runs. TTFA percentiles pool the 5 × N requests.
    The other columns are the median of the 5 runs.
  - **all done:** when the last request's last piece arrived.
  - **audio/s:** seconds of audio delivered per second of wall time, up to *all done*.
  - **buffered:** a client plays each request from its first piece onward. This is the least audio it still had
    buffered when a later piece arrived, over all requests. Below 0 the client would wait for audio (a stall).
  - **peak GB:** `torch.cuda.max_memory_allocated()`. It includes the 0.68 GB that stays resident: weights and CUDA
    graphs.
- **Strategies:**
  - **(a) FIFO:** one request after another, each streamed to its end with the single-request fast path
    (`Synthesizer(fast=True).stream`), as a naive server does.
  - **(b) batched:** all N requests in one batch (`drifting_tts.batched.stream_batched`), in rounds. The first round
    runs one padded pass of the text encoder, prosody predictor and DiT over the first sentence of every request. It
    then vocodes the first window of every request as one batch, so every request gets its first piece after that
    round. Each later round vocodes the next window of every unfinished request as one batch (256 frames with
    32 frames of context on each side). A request whose sentence has ended joins the round's acoustic batch with its
    next sentence. Batched passes run eagerly: there are no CUDA graphs for these shapes.
  - **(c) micro-batches:** groups of 16 or 32 requests in arrival order. Each group is streamed to its end as in (b)
    before the next group starts.

## One request

| system | short sentence p50 / p90 / p99 | long sentence | 4-sentence paragraph | RTF short / long / paragraph |
|---|---|---|---|---|
| v3.1 + BigVGAN-v2-ft (reference) | 12.2 / 12.3 / 12.3 ms | 13.7 / 13.8 / 14.0 ms | 13.5 / 13.5 / 13.6 ms | 0.0184 / 0.0098 / 0.0101 |
| **v3.2** | **5.8 / 5.8 / 5.8 ms** | **7.4 / 7.4 / 7.5 ms** | **7.0 / 7.1 / 7.5 ms** | **0.0050 / 0.0017 / 0.0020** |
| v3.2, sampled rhythm (the demo's option) | 7.0 / 7.0 / 7.3 ms | 8.9 / 8.9 / 9.0 ms | 8.3 / 8.3 / 8.4 ms | 0.0060 / 0.0020 / 0.0024 |
| v3.2 without CUDA graphs (`stream`, eager) | 14.9 / 15.0 / 18.5 ms | 15.2 / 15.3 / 15.6 ms | 15.2 / 15.3 / 15.6 ms | 0.0120 / 0.0030 / 0.0039 |

- **Reproduced:** the v3.2 and v3.1 rows match the published ones ([RESULTS.md](RESULTS.md#latency-and-size): 5.8 /
  7.4 / 7.1 ms and 12.3 / 13.8 / 13.5 ms).
  - A second round of 100 runs agrees within 0.1 ms on every median (v3.2: 5.8 / 7.4 / 7.1 ms).
  - So does a third round after merging main 64b8d94: 5.8 / 7.4 / 7.0 ms.
- **First call after loading:** 28–35 ms. Loading takes 2.3 s, including the capture of the CUDA graphs. Peak
  memory is 0.69 GB (v3.1: 1.09 GB).
- **Sampled rhythm:** `prosody_durations="sampled", prosody_duration_temperature=0.3`. The durations get their own
  row in the prosody predictor's batch (the letters at temperature 0.3, the pauses at the pitch's temperature),
  which costs 1.2–1.5 ms of TTFA.
- **Over the request mix, one at a time:** 100 Freya requests of the mix with the fast path give TTFA 7.1 / 7.3 /
  7.4 ms (p50 / p90 / p99, max 7.5 ms), RTF 0.0022. Freya sentences are long sentences, about 3.5 s of audio
  each. v3.1 + BigVGAN-v2-ft gives 13.6 / 13.8 / 13.9 ms, RTF 0.0115.

## N requests at once

v3.2, fp32 as shipped. Times in ms: TTFA percentiles, then *all done*.

| N | strategy | TTFA p50 | p90 | p99 | max | all done | audio/s | buffered (ms) | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| 64 | (a) FIFO | 413 | 744 | 812 | 813 | 830 | 463 | 340 | 0.69 |
| 64 | (c) micro-batches of 16 | 151 | 272 | 273 | 273 | 327 | 1,175 | 337 | 0.81 |
| 64 | (c) micro-batches of 32 | 96 | 153 | 153 | 153 | 226 | 1,702 | 333 | 0.93 |
| 64 | (b) **batched** | **71** | 72 | 72 | **72** | 181 | **2,119** | 321 | 1.18 |
| 128 | (a) FIFO | 825 | 1,484 | 1,632 | 1,641 | 1,659 | 459 | 340 | 0.69 |
| 128 | (c) micro-batches of 16 | 312 | 584 | 586 | 586 | 631 | 1,209 | 337 | 0.81 |
| 128 | (c) micro-batches of 32 | 209 | 389 | 390 | 390 | 450 | 1,693 | 333 | 0.93 |
| 128 | (b) **batched** | **150** | 151 | 151 | **151** | 361 | **2,110** | 296 | 1.69 |
| 256 | (a) FIFO | 1,661 | 2,865 | 3,159 | 3,188 | 3,187 | 421 | 339 | 0.69 |
| 256 | (c) micro-batches of 16 | 620 | 1,056 | 1,144 | 1,144 | 1,187 | 1,129 | 337 | 0.81 |
| 256 | (c) micro-batches of 32 | 440 | 778 | 780 | 780 | 859 | 1,560 | 332 | 0.93 |
| 256 | (b) **batched** | **304** | 305 | 305 | **305** | 687 | **1,952** | 249 | 2.70 |

- **Reproducibility:** two more complete runs of this table (before the merge with main) agree within 1–2% on every
  entry. For example, batched at N = 256 gave 304.7 / 304.7 / 304.3 ms, and FIFO's last first audio gave 3,183 /
  3,180 / 3,188 ms.
- **Stalls:** none in any run. The column *buffered* stays at or above 249 ms.
- **Batched times are flat:** p50 = max within 1 ms, because every request gets its first piece in the same round.

**Reference: v3.1 + BigVGAN-v2-ft, FIFO.** Mostly because of its vocoder, it delivers 4.8 times less audio per
second than v3.2:

| N | TTFA p50 | p90 | p99 | max | all done | audio/s | peak GB |
|---|---|---|---|---|---|---|---|
| 64 | 2,003 | 3,596 | 3,925 | 3,926 | 4,028 | 95 | 1.09 |
| 128 | 3,984 | 7,173 | 7,897 | 7,942 | 8,051 | 95 | 1.09 |
| 256 | 8,559 | 14,198 | 15,717 | 15,837 | 15,946 | 84 | 1.09 |

**Option: TF32 matmuls (`--tf32`).** This applies to the fast path's graphs and the batched passes, and it changes
the output slightly:

| N | strategy | TTFA p50 | max | all done | audio/s |
|---|---|---|---|---|---|
| 64 | micro-batches of 32 | 77 | 123 | 179 | 2,144 |
| 64 | batched | 57 | 57 | 143 | 2,686 |
| 128 | micro-batches of 32 | 171 | 325 | 370 | 2,060 |
| 128 | batched | 119 | 119 | 286 | 2,663 |
| 256 | micro-batches of 32 | 364 | 658 | 724 | 1,851 |
| 256 | batched | 244 | 244 | 545 | 2,458 |

## What dominates the time

- **FIFO is a queue.** Each request occupies the GPU for 12.4 ms on average: each of its sentences goes through the
  acoustic model in CUDA graphs, then every vocoder window is computed. The slope of TTFA against arrival position
  is 12.45 ms per request.
  - Request 0 starts after 7.2 ms and request 255 after about 3.2 s.
  - The GPU runs batch-1 work all the time, so throughput stays at 420–460 s of audio per second whatever N is.
- **Batched first round at N = 256 (304 ms),** synchronised between the stages:

  | stage | time |
  |---|---|
  | frontend: normalisation, sentence split (CPU) | 10 ms |
  | acoustic model | 278 ms |
  | first vocoder windows (256 × 64 frames) | 15 ms |
  | copy to the host | 0.3 ms |

  The acoustic model's 278 ms break down as follows:
  - DiT: 218 ms;
  - text encoder + prosody predictor: 48 ms;
  - per-request noise draws: 11 ms. Each request draws from its own generator in the single-request order, so a
    seed gives the same audio as alone, up to float rounding.

  The round costs about 1.1–1.2 ms per request (N = 64: 71 ms; 128: 150 ms). Since it grows linearly with N, it is
  limited by computation, not by kernel launches: at N = 256 the DiT runs about 70k patch tokens through 60 M
  parameters.
- **Padding:** each batch is padded to its longest first sentence. At N = 256 the longest is 516 frames and the
  mean is 327, so 37% of the DiT's frames are padding (26% at N = 64). Bucketing by length would cut that; it is not
  implemented.
- **TF32:** TF32 matmuls save only 20% (304 → 244 ms). Half precision was not tested.
- **The rest of the audio:** each later round vocodes one 320-frame window per request. At N = 256 the next piece
  arrived at most 92 ms after the first, while the 0.34 s first piece was still playing (at least 249 ms of audio
  still buffered).
  - Paragraphs add an acoustic batch for their next sentences in the round where their first sentence ends.
  - If each round grows linearly with N, rounds would outlast the first piece somewhere near N ≈ 900. That is an
    extrapolation from these runs; it was not measured.
- **Micro-batches:** a group starts speaking after every group before it has been streamed to its end, plus its own
  first round. At N = 256, groups of 32 take about 107 ms each, so the last group starts speaking after 780 ms. Smaller groups lose the batching
  gain: groups of 16 give 1,130 s of audio per second, against 1,560 for 32 and 1,950 for one batch.
- **Batched at N = 1** costs 13 ms instead of 7 ms, because the batched passes run without CUDA graphs. A server
  should use the fast path for a lone request.

## Are the batched outputs the same?

`--check 32` runs the first 32 requests of the mix both ways with the same seeds:

- through `stream_batched`, as one batch;
- one by one through the fast path.

For scale, the same comparison was made between the eager single-request path and the fast path, which the docs
call the same output.

| | batched vs fast path | eager vs fast path (reference) |
|---|---|---|
| mel frames equal | 32 / 32 requests | 32 / 32 |
| mel SNR (first sentence), min / median | 72.1 / 79.5 dB | 68.5 / 99.6 dB |
| audio samples equal | 32 / 32 | 32 / 32 |
| audio log-mel distance, median / max | 0.02 / 0.05 dB | 0.01 / 0.04 dB |
| worst 0.1 s of each request, median / max | 0.18 / 1.79 dB | 0.05 / 0.96 dB |
| waveform SNR, min / median | 21.1 / 37.3 dB | 28.7 / 48.1 dB |

- **Every request matches.** It has the same durations and the same number of samples. The mel agrees to float
  rounding, and the audio's log-mel differs by 0.02 dB on average.
- **Waveform SNR is a poor measure here.** Vocos v2 turns float-rounding differences into waveform differences. A
  random perturbation of the fast path's own mel at 80 dB SNR gave a waveform SNR of 33–52 dB and a log-mel
  distance of 0.01–0.025 dB on 8 of these requests.
- **Local differences come from the vocoder.** One case was examined: request 11, whose waveform SNR was the
  lowest of the first 16 requests.
  - Its batched mel differs from the fast path's by at most 4e-3 in log-mel units.
  - Vocoded whole by the ordinary single-request vocoder, that mel still differs by 2.3 dB over one 0.1 s window.
  - That is the window where random 80 dB perturbations of the fast path's mel also change the audio most.
  - So the vocoder amplifies these differences; the batching only supplies the rounding. The largest worst-window
    value in the table (1.8 dB) belongs to another request, which was not examined.
- **TF32 run:** with TF32 in both paths (`--check 8`), all 8 requests have equal frames and samples, mel SNR
  68–71 dB and a log-mel distance of 0.03–0.07 dB.
- **CPU tests:** `tests/test_batched.py` checks the same on tiny models within 1e-5 for three parts:
  - every row of a padded acoustic batch against `ProsodyPredictor.predict` + `DriftingTTS.synthesize` at batch 1,
    with no prosody predictor, pitch only (v3.2) and sampled durations;
  - every row of the masked Vocos batch against vocoding it alone;
  - every request's pieces from `stream_batched` against `Synthesizer.stream`, paragraph pauses included.

## Memory

`torch.cuda.max_memory_allocated()`. `nvidia-smi` shows more, because it also counts the CUDA context and the
allocator's cache.

| configuration | peak GB |
|---|---|
| resident after loading (weights + CUDA graphs) | 0.68 |
| one request or FIFO at any N | 0.69 |
| micro-batches of 16 / 32 | 0.81 / 0.93 |
| batched N = 64 / 128 / 256 | 1.18 / 1.69 / 2.70 |
| v3.1 + BigVGAN-v2-ft, FIFO | 1.09 |

A batch of 256 adds 2 GB; the largest tensors are the DiT's activations over the padded batch. Extrapolating at
this rate, a few thousand requests would fit in 32 GB, but their first audio would come about 1.2 ms per request
later.

## Caveats

- **Same-instant arrival.** It is the worst case for FIFO and the best case for batching. Requests that arrive over
  time need continuous batching (new requests join the next round), which is not implemented. Their latency would
  lie between the two.
- **Sentence-level streaming.** The first piece waits for the acoustic model over the whole first sentence, so in a
  batch TTFA grows with the longest first sentence. Paragraphs only need their first sentence for the first piece.
- **Eager batched passes.** The batched passes run without CUDA graphs. At N = 1 that costs 13 ms against the fast
  path's 7 ms; in large batches the computation dominates (about 1.1 ms per request).
- **fp32 as shipped.** TF32 matmuls gain 20% in batches. Lower precision was not tried, and it would need a quality
  check.
- **One voice per batch.** `stream_batched` takes one voice. Mixing voices needs per-row speaker IDs and duration
  factors, which is not implemented.
- **This mix only.** Throughput depends on it (5.2 s of audio per request). With v3.2's punctuation pauses
  (`pause="punct"`), the requests would also carry silence that costs no GPU time.
- **Not included:** one process on one GPU; no HTTP, audio encoding or network.

## Reproduce

```bash
# one request (100 runs per input, after warm-up)
python scripts/bench_ttfa.py --release v3.2 --mode fast --out runs/lat_ttfa_v32_fast.json
python scripts/bench_ttfa.py --release v3.2 --mode fast --prosody-durations sampled \
    --prosody-duration-temperature 0.3 --out runs/lat_ttfa_v32_rhythm.json
python scripts/bench_ttfa.py --release v3.1 --mode fast --cuda-kernel --out runs/lat_ttfa_v31_bigvgan.json

# N requests at once (5 timed runs per strategy and N; 100 single requests for N = 1; output check on 32 requests)
python scripts/bench_concurrency.py --release v3.2 --requests 1 64 128 256 \
    --strategy fifo batched microbatch --micro-batch 16 32 --check 32 --out runs/lat_concurrency_v32.json
python scripts/bench_concurrency.py --release v3.2 --tf32 --requests 64 128 256 \
    --strategy batched microbatch --micro-batch 32 --check 8 --out runs/lat_concurrency_v32_tf32.json
python scripts/bench_concurrency.py --release v3.1 --cuda-kernel --requests 1 64 128 256 \
    --strategy fifo --check 0 --out runs/lat_concurrency_v31_fifo.json
```

`--texts` takes another text set (a HF dataset id, `.jsonl` or `.txt`). `--paragraph-every` and
`--paragraph-sentences` change the mix. The JSON output records the environment, the GPU processes seen at the
start, and each request's TTFA from the first timed run. In Python:

```python
from drifting_tts.batched import stream_batched
from drifting_tts.synthesize import Synthesizer

tts = Synthesizer.from_pretrained("v3.2", "cuda", fast=True)
for pieces in stream_batched(tts, texts, speaker="studio", cfg_scale=2.0, temperature=0.3):
    for i, audio in pieces:   # request index, float32 CPU tensor at 24 kHz
        send(i, audio)
```

# Evaluation and data curation

## Evaluation

| what | judge | notes |
|---|---|---|
| CER / WER | Whisper **large-v3** (`faster-whisper`, Turkish, beam 5, temperature 0) | FLEURS-tr WER 6.7%. `--asr large-v3-turbo` is faster. |
| speaker similarity | **WavLM-Large + ECAPA-TDNN** speaker verification (UniSpeech, VoxCeleb1-O EER 0.43%) | The SIM model of Seed-TTS-eval and F5-TTS. Scored against the *recording* of the same sentence (`prepare --save-audio`), else its vocoded version. |
| naturalness | **UTMOSv2** (`--mos utmos22` as an alternative) | Trained on English: compare against the vocoded recordings, not as an absolute MOS. `--mos-repetitions N` averages N random crops per sentence. |
| spectral detail | `--harmonic`: harmonic contrast, global variance and level per band vs. the recordings | Uses the ground-truth alignment and pitch, with no vocoder or judges. Takes seconds. |

- **Intervals and reproducibility:** every metric comes with a 95% bootstrap interval over sentences.
  Sentence *i* always uses seed *i*, and per-sentence results go to `utterances_*.jsonl`.
- **Rows:** `--temperature` and `--cfg` take several values, and `evaluate` writes one row for each
  combination.
- **Unbiased tuning:** tune on a split you do not report.
  - Data prepared with `prepare --dev-size 200` has a disjoint `dev` split. Tune with `--split dev`
    and report on `val`. Utterances in `dev` are excluded from training, so the split must exist
    before training.
  - For a model trained without a dev split, split `val` in two: tune with `--offset 100 --num 100`
    and report with `--num 100`.

```bash
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu128   # match your torch build
uv pip install -e ".[eval]"                                                      # faster-whisper, UTMOSv2, ...
drifting-tts evaluate --model runs/tts/model_ema.pt --harmonic --split dev --temperature 0.3 0.5 0.7 1.0
drifting-tts evaluate --model runs/tts/model_ema.pt --split dev --num 100 --temperature 0.3 0.5 0.7 --cfg 1.0 1.5
```

## External benchmark: Freya-TR-Eval

```bash
drifting-tts benchmark --model runs/tts_v3/model_ema.pt --vocoder runs/vocoder_v3/bigvgan_ft.pt --speaker studio
```

`benchmark` synthesises each sentence of a text set and scores it. The default text set is the 495-sentence
`freyavoice/freya-tr-eval`; `--texts` also takes a `.jsonl` (field `text`) or `.txt` file. The default protocol is
the FreyaTTS report's:
- audio band-matched to 8 kHz before ASR (`--band 0` turns this off);
- Whisper large-v3;
- the same normaliser on both sides.

UTMOSv2 is computed on the full-band audio.

## Data curation

Training data collected in the wild needs cleaning. Wrong transcripts corrupt the alignment and the duration
model, noisy clips get reproduced, and mislabelled speakers make a one-step generator average voices.
`drifting-tts score` rates every utterance of a prepared dataset (it needs `prepare --save-audio`) and
writes `scores.jsonl` next to `index.jsonl`, keyed by line number, so delete it and `spk_emb.npy` after
re-running `prepare`. The run is resumable, and each scorer can be run on its own.

| score | how | catches |
|---|---|---|
| `cer`, `wer`, `hyp` | Whisper large-v3 (faster-whisper, Turkish, beam 5) against the normalised transcript | wrong or truncated transcripts, cross-talk |
| `cps` | letters per second | segmentation errors |
| `mos_sig`, `mos_bak`, `mos_ovrl`, `mos_p808` | DNSMOS P.835 / P.808 (Microsoft's ONNX models, run in torch) | noise, reverb, music |
| `bw_hz` | highest frequency of the long-term spectrum within 50 dB of its peak | narrowband guests, low-bitrate MP3 |
| `spk_sim`, `spk_next` | ECAPA-TDNN cosine to the speaker's centroid / to the closest other speaker's centroid | mislabelled speakers |

```bash
uv pip install -e ".[score]"
drifting-tts score --data data/train                     # --scorers asr,rate,mos,bandwidth,spk
drifting-tts score --data data/train --summary --config configs/tts.yaml --filter max_cer=0.1 --filter min_mos=3
```

The summary prints percentiles of every score, outlier counts, the hours that survive the filters,
per-speaker counts and the closest pairs of speaker centroids. Thresholds then go in the config:

```yaml
data:
  filters:              # applied when scores.jsonl exists
    max_cer: 0.1
    min_mos: 3.0        # DNSMOS OVRL
    min_spk_sim: 0.5
    rate: [8, 20]       # letters / s; any score field works as min_<field>, max_<field> or <field>: [lo, hi]
```

Only the `train` split is filtered, so evaluation stays comparable; training prints how many val
utterances would fail (`apply_to_val: true` filters them too). DNSMOS and ECAPA work at 16 kHz and the
MOS models were trained on English, so use their scores to rank clips, not as absolute quality.

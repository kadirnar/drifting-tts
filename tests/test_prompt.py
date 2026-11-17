"""Voices from a prompt (#95): the cut of an utterance into prompt and target, the training step with speaker-ID
dropout, synthesis from a prompt, and the Synthesizer's ``ref_audio``. Tiny models on the CPU."""

import pytest
import torch

from drifting_tts.config import Config, load_config
from drifting_tts.models.ar import ARDriftTTS
from drifting_tts.text import BLANK_ID, SYMBOL_TO_ID, text_to_ids
from drifting_tts.train_ar import cut_prompts, prepend_prompts, training_step

CFG = {"kind": "ar", "text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
       "ar": {"patch": 4, "lookahead": 2, "dim": 32, "depth": 2, "heads": 2, "head_dim": 32, "head_depth": 2,
              "prompt": True}}
PROMPT = {"prob": 1.0, "id_dropout": 0.5, "min_seconds": 0.1, "max_seconds": 0.45, "min_target": 0.1,
          "max_trim": 0.05}  # frames: a cut between 9 and 42, at least 9 frames left, up to 5 trimmed


def _model(seed: int = 0, prompt: bool = True) -> ARDriftTTS:
    torch.manual_seed(seed)
    cfg = {**CFG, "ar": {**CFG["ar"], "prompt": prompt}}
    m = ARDriftTTS(Config(cfg), num_speakers=3, n_mels=100).eval()
    with torch.no_grad():
        for p in m.parameters():  # zero-initialised output layers: make every path visible
            p.add_(torch.randn_like(p) * 0.05)
    return m


def _batch() -> dict:
    ids = [torch.tensor(text_to_ids(t, normalized=True)) for t in ("ab cd ef gh ij", "kl mn op")]
    text = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True)
    torch.manual_seed(1)
    mel = torch.randn(2, 100, 64)
    mel[1, :, 50:] = 0  # padding, as collate leaves it
    return {"text": text, "text_len": torch.tensor([len(i) for i in ids]), "mel": mel,
            "mel_len": torch.tensor([64, 50]), "spk": torch.tensor([0, 1])}


def test_a_prompted_model_has_one_more_speaker_row_and_a_prompt_encoder():
    m, plain = _model(), _model(prompt=False)
    assert m.spk.num_embeddings == 4 == m.encoder.spk.num_embeddings and m.prompt_row == 3
    assert plain.spk.num_embeddings == 3 and plain.prompt_enc is None
    with pytest.raises(ValueError, match="prompts"):
        _ = plain.prompt_row
    mel = torch.randn(2, 100, 16)
    v = m.prompt_vector(mel, torch.tensor([16, 8]))
    assert v.shape == (2, 8)
    torch.testing.assert_close(v[1], m.prompt_vector(mel[1:, :, :8])[0], atol=1e-6, rtol=1e-5)  # the padding is masked


def test_cut_prompts_cuts_at_a_word_and_keeps_text_and_frames_together():
    m, b = _model(), _batch()
    space = SYMBOL_TO_ID[" "]
    torch.manual_seed(0)
    cut = cut_prompts(m, b["text"], b["text_len"], b["mel"], b["mel_len"], b["spk"], Config(PROMPT), space)
    assert (cut["prompt_len"] > 0).all() and (cut["prompt_len"] % m.patch == 0).all()
    for i in range(2):
        n, t, p = int(cut["text_len"][i]), int(cut["mel_len"][i]), int(cut["prompt_len"][i])
        full = b["text"][i, : int(b["text_len"][i])]
        assert torch.equal(cut["text"][i, :n], full[-n:]) and n < len(full)  # the words after the cut
        assert cut["text"][i, 0] == BLANK_ID and full[-n - 1] == space  # ... which starts where a word starts
        f0 = int(b["mel_len"][i]) - t  # the cut frame
        assert 9 <= f0 <= 42 and t >= 9 and f0 - 5 - m.patch < p <= f0
        assert torch.equal(cut["mel"][i, :, :t], b["mel"][i, :, f0: f0 + t])
        assert torch.equal(cut["prompt"][i, :, :p], b["mel"][i, :, :p])
    none = cut_prompts(m, b["text"], b["text_len"], b["mel"], b["mel_len"], b["spk"], Config({**PROMPT, "prob": 0.0}),
                       space)
    assert not none["prompt_len"].any() and not none["dropped"].any()
    assert torch.equal(none["text"], b["text"]) and torch.equal(none["mel"], b["mel"])
    cond = torch.randn(2, 116, int(cut["mel_len"].max()))
    c, y = prepend_prompts(cond, cut["mel"], cut["mel_len"], cut["prompt"], cut["prompt_len"])
    p, t = int(cut["prompt_len"][0]), int(cut["mel_len"][0])
    assert not c[0, :, :p].any() and torch.equal(c[0, :, p: p + t], cond[0, :, :t])  # no text for the prompt
    assert torch.equal(y[0, :, :p], cut["prompt"][0, :, :p]) and torch.equal(y[0, :, p: p + t], cut["mel"][0, :, :t])


def test_training_step_with_prompts_trains_the_prompt_path():
    m = _model().train()
    over = ["drift.gen_per_cond=4", "drift.max_rows=64", "model.ar.prompt=true"] + [f"drift.prompt.{k}={v}"
                                                                                    for k, v in PROMPT.items()]
    cfg = load_config("configs/ar_tts.yaml", over + ["drift.prompt.id_dropout=1.0"])
    tau = torch.nn.Parameter(torch.tensor(1.0))
    torch.manual_seed(0)
    loss, metrics = training_step(m, _batch(), cfg, "cpu", tau)
    loss.backward()
    assert torch.isfinite(loss) and metrics["prompted"] == 1.0 and metrics["id_dropped"] == 1.0
    assert metrics["rows"] < (64 + 50) // 4  # the loss is taken after the cut only
    assert m.prompt_enc.out.weight.grad.abs().sum() > 0  # the text encoder's voice comes from the prompt
    assert m.spk.weight.grad[m.prompt_row].abs().sum() > 0 and not m.spk.weight.grad[:3].any()
    assert not m.encoder.spk.weight.grad.any()  # no speaker row was read
    keep = load_config("configs/ar_tts.yaml", over + ["drift.prompt.id_dropout=0.0"])  # prompts, the IDs kept
    m.zero_grad()
    torch.manual_seed(0)
    training_step(m, _batch(), keep, "cpu", tau)[0].backward()
    assert m.spk.weight.grad[:2].abs().sum() > 0 and not m.spk.weight.grad[m.prompt_row].any()
    off = load_config("configs/ar_tts.yaml", over[:2])  # a model without prompts: the step as it was
    plain = _model(prompt=False).train()
    assert "prompted" not in training_step(plain, _batch(), off, "cpu", tau)[1]


def test_synthesis_from_a_prompt_continues_it_without_its_text():
    m = _model()
    text = torch.tensor([text_to_ids("ab cd", normalized=True)])
    n, spk = torch.tensor([text.shape[1]]), torch.tensor([0])
    prompt = torch.randn(1, 100, 18)  # 18 frames: the last two are not a whole patch and are dropped
    g = [torch.Generator().manual_seed(3) for _ in range(3)]
    mel, y_len = m.synthesize(text, n, spk, temperature=0.5, generator=g[0], prompt=prompt)
    assert mel.shape[-1] == int(y_len[0]) and torch.isfinite(mel).all()
    assert torch.equal(mel, m.synthesize(text, n, torch.tensor([2]), temperature=0.5, generator=g[1], prompt=prompt)[0])
    other = m.synthesize(text, n, spk, temperature=0.5, generator=g[2], prompt=prompt + 1.0)[0]
    assert other.shape != mel.shape or not torch.allclose(other, mel, atol=1e-3)  # the voice is the prompt's
    # the same as the model's own pieces: the prompted row, the pooled vector, the prompt as a prefix with no text
    row, p = torch.tensor([m.prompt_row]), prompt[..., :16]
    cond, _ = m.aligned_condition(text, n, row, spk_emb=m.prompt_vector(p))
    ref = m.generate(cond, row, 0.5, torch.Generator().manual_seed(3), prefix=p, prefix_cond=torch.zeros(1, 116, 16))
    torch.testing.assert_close(ref, mel, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="prompts"):
        _model(prompt=False).synthesize(text, n, spk, prompt=prompt)


def test_train_ar_with_prompts_from_a_plain_checkpoint_and_ref_audio(tmp_path):
    """Fine-tuning a model trained without prompts: the speaker tables grow by the prompted row; the Synthesizer then
    speaks in the voice of a recording (``ref_audio``), in one pass and streamed."""
    import soundfile as sf

    from drifting_tts.cli import main
    from drifting_tts.synthesize import Reference, Synthesizer
    from tests.test_data import _fake_parquet
    from tests.test_loop_eval import TINY

    _fake_parquet(tmp_path / "x.parquet", n=14)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "0", "--no-trim", "--backend", "bigvgan"])
    base, work = tmp_path / "base", tmp_path / "prompted"
    main(["train-ar", "--workdir", str(base), f"data.root={data}", *TINY, "train.steps=1", "train.save_every=1"])
    prompts = [f"drift.prompt.{k}={v}" for k, v in {**PROMPT, "min_seconds": 0.3, "max_seconds": 1.5}.items()]
    main(["train-ar", "--workdir", str(work), f"data.root={data}", *TINY, "train.steps=2", "train.save_every=2",
          f"train.init_from={base / 'model_ema.pt'}", "model.ar.prompt=true", *prompts])
    synth = Synthesizer(work / "model_ema.pt", "cpu", vocoder="griffin-lim")
    synth.vocoder.model.n_iter = 2
    assert synth.model.prompted and synth.model.spk.num_embeddings == 4  # three speakers and the prompted row
    with torch.no_grad():
        for q in synth.model.parameters():  # two steps leave the output layers near zero
            q.add_(torch.randn_like(q) * 0.05)
    t = torch.arange(24000) / 24000
    sf.write(tmp_path / "a.wav", (0.3 * torch.sin(2 * torch.pi * 220 * t)).numpy(), 24000)
    sf.write(tmp_path / "b.wav", (0.3 * torch.sin(2 * torch.pi * 330 * t) * (1 + t)).numpy(), 24000)
    ref = synth.reference(tmp_path / "a.wav")
    assert isinstance(ref, Reference) and ref.mel.shape[-1] % synth.model.patch == 0 and ref.vector.shape == (1, 8)
    text = "merhaba dünya. nasılsın?"
    a = synth(text, ref_audio=tmp_path / "a.wav", seed=1)[0]
    assert torch.isfinite(a).all() and torch.equal(synth(text, speaker=ref, seed=1)[0], a)
    b = synth(text, ref_audio=tmp_path / "b.wav", seed=1)[0]
    assert a.shape != b.shape or not torch.allclose(a, b, atol=1e-4)  # another recording, another voice
    assert not torch.equal(synth(text, speaker=0, seed=1)[0], a)  # the speaker rows still work
    streamed = torch.cat(list(synth.stream(text, ref_audio=ref, seed=1)))
    torch.testing.assert_close(streamed, a, atol=1e-4, rtol=1e-4)
    two = synth.mels(text, ref_audio=ref, seed=1, paragraph=False)  # the prompt stays in front of every unit
    assert len(two) == 2 and all(torch.isfinite(m).all() for m in two)
    plain = Synthesizer(base / "model_ema.pt", "cpu", vocoder="griffin-lim")
    with pytest.raises(ValueError, match="prompts"):
        plain.reference(tmp_path / "a.wav")

    # scripts/clone_eval.py: a model without prompts continues a recording under its transcript (the baseline)
    from scripts.clone_eval import continue_prompt, summarize

    plain.vocoder.model.n_iter = 2
    mel = plain.reference_mel(tmp_path / "a.wav")
    assert mel.shape[-1] % plain.model.patch == 0
    wav = continue_prompt(plain, mel, "merhaba", "dünya nasılsın", 0, 0.5, seed=1)
    assert wav.numel() > 0 and torch.isfinite(wav).all()
    assert torch.equal(continue_prompt(plain, mel, "merhaba", "dünya nasılsın", 0, 0.5, seed=1), wav)
    r = summarize([{"sim": 0.5, "word_errors": 1, "words": 10}, {"sim": 0.7, "word_errors": 0, "words": 10}])
    assert r["items"] == 2 and abs(r["sim"] - 0.6) < 1e-9 and r["wer"] == 0.05


def test_voices_without_a_speaker_row_are_always_prompted():
    """A corpus streamed with ``speakers: prompt`` (#104): speaker id -1. Such an utterance never reads a speaker
    row: it is cut whenever it can be, and its voice is the prompt's."""
    m = _model().train()
    b = _batch()
    b["spk"] = torch.tensor([-1, 1])
    over = ["drift.gen_per_cond=4", "drift.max_rows=64", "model.ar.prompt=true"] + [f"drift.prompt.{k}={v}"
                                                                                    for k, v in PROMPT.items()]
    cfg = load_config("configs/ar_tts.yaml", over + ["drift.prompt.id_dropout=0.0", "drift.prompt.prob=0.0"])
    tau = torch.nn.Parameter(torch.tensor(1.0))
    torch.manual_seed(0)
    loss, metrics = training_step(m, b, cfg, "cpu", tau)
    loss.backward()
    assert metrics["no_row"] == 0.5 and metrics["prompted"] == 0.5 and metrics["id_dropped"] == 0.5  # only that one
    grad = m.spk.weight.grad
    assert grad[m.prompt_row].abs().sum() > 0 and grad[1].abs().sum() > 0 and not grad[0].any() and not grad[2].any()
    with pytest.raises(ValueError, match="prompts"):
        training_step(_model(prompt=False).train(), b, load_config("configs/ar_tts.yaml", over[:2]), "cpu", tau)


def test_stream_mixes_in_a_corpus_of_prompted_voices(tmp_path):
    from drifting_tts.cli import main
    from drifting_tts.stream import ParquetStream
    from tests.test_text_en import _english_parquet

    _english_parquet(tmp_path / "train-0.parquet", 12)
    _english_parquet(tmp_path / "other-0.parquet", 8, start=50)
    out = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "train-*.parquet"), "--val-size", "0", "--out", str(out),
          "--workers", "1", "--no-trim", "--backend", "bigvgan", "--language", "en", "--speaker-column", "speaker_id"])
    extra = [{"files": [str(tmp_path / "other-0.parquet")], "speaker_column": "speaker_id", "share": 0.5,
              "speakers": "prompt"}]
    stream = ParquetStream([str(tmp_path / "train-0.parquet")], out, speaker_column="speaker_id", min_frames=1,
                           batch_frames=10**6, max_batch=4, trim=False, extra=extra)
    it = iter(stream)
    kinds = set()
    for _ in range(24):
        spk = next(it)["spk"]
        assert (spk < 0).all() or (spk >= 0).all()  # a batch is of one shard
        kinds.add(bool((spk < 0).all()))
    assert kinds == {True, False}  # both corpora came by: the extra one's speakers have no row, whatever their name

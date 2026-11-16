"""Voices from a reference (#96): the reference encoder and its tokens in front of the backbone's sequence, the
training step with a reference from another utterance, the loaders' same-speaker pairs and the Synthesizer's
reference modes. Tiny models on the CPU."""

import random
from collections import OrderedDict

import pytest
import torch

from drifting_tts.config import Config, load_config
from drifting_tts.data import collate, crop_reference, reference_frames
from drifting_tts.models.ar import ARDriftTTS, ARGenerator, patchify, unpatchify
from drifting_tts.text import text_to_ids
from drifting_tts.train_ar import training_step
from tests.test_prompt import PROMPT, _batch

CFG = {"kind": "ar", "text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
       "ar": {"patch": 4, "lookahead": 2, "dim": 32, "depth": 2, "heads": 2, "head_dim": 32, "head_depth": 2,
              "prompt": True, "reference": {"dim": 16, "layers": 1, "heads": 2, "tokens": 3, "query_layers": 1}}}
OVER = ["drift.gen_per_cond=4", "drift.max_rows=64", "model.ar.prompt=true", "model.ar.reference.dim=16",
        "model.ar.reference.layers=1", "model.ar.reference.heads=2", "model.ar.reference.tokens=3",
        "model.ar.reference.query_layers=1"]


def _model(seed: int = 0, prompt: bool = True) -> ARDriftTTS:
    torch.manual_seed(seed)
    m = ARDriftTTS(Config({**CFG, "ar": {**CFG["ar"], "prompt": prompt}}), num_speakers=3, n_mels=100).eval()
    with torch.no_grad():
        for p in m.parameters():  # zero-initialised output layers: make every path visible
            p.add_(torch.randn_like(p) * 0.05)
    return m


def _ref_batch() -> dict:
    b = _batch()
    torch.manual_seed(2)
    b["ref"], b["ref_len"] = torch.randn(2, 100, 24), torch.tensor([24, 0])  # the second utterance has none
    return b


def test_the_reference_encoder_gives_a_fixed_number_of_tokens_and_masks_the_padding():
    m = _model()
    assert m.spk.num_embeddings == 4 and m.prompt_row == 3  # one extra row for prompts and references together
    assert _model(prompt=False).spk.num_embeddings == 4
    ref = torch.randn(2, 100, 24)
    tokens, vec = m.reference_tokens(ref, torch.tensor([24, 12]))
    assert tokens.shape == (2, 3, 32) and vec.shape == (2, 8)
    alone, v = m.reference_tokens(ref[1:, :, :12])  # as many tokens for 12 frames as for 24
    torch.testing.assert_close(tokens[1], alone[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(vec[1], v[0], atol=1e-5, rtol=1e-5)
    none, _ = m.reference_tokens(ref, torch.tensor([24, 0]))  # no reference: the null tokens
    assert torch.equal(none[1], m.ref_enc.null_tokens(1)[0]) and torch.equal(none[0], tokens[0])
    use, _ = m.reference_tokens(ref, torch.tensor([24, 24]), torch.tensor([False, True]))
    assert torch.equal(use[0], m.ref_enc.null_tokens(1)[0]) and torch.isfinite(use).all()
    assert torch.equal(m.reference_tokens(None, batch=2)[0], m.ref_enc.null_tokens(2))
    empty = m.reference_tokens(torch.zeros(2, 100, 1), torch.tensor([0, 0]))  # a batch without any reference
    assert torch.equal(empty[0], m.ref_enc.null_tokens(2)) and empty[1] is None


def test_generation_with_reference_tokens_matches_teacher_forcing_and_the_fast_generator():
    m = _model()
    cond, spk = torch.randn(2, 116, 29), torch.tensor([0, 3])
    tokens, _ = m.reference_tokens(torch.randn(2, 100, 24))
    mel = m.generate(cond, spk, temperature=0.0, tokens=tokens)
    padded = m.pad_frames(cond)
    s = m.teacher_forced(padded, m.pad_frames(mel), spk, tokens=tokens)
    assert s.shape[1] == padded.shape[-1] // 4  # the tokens' own states are not returned
    z = torch.zeros(2 * s.shape[1], m.head.noise_dim)
    again = patchify(padded[:, :100], 4) + m.head(s.flatten(0, 1), z).view(2, s.shape[1], -1)
    torch.testing.assert_close(unpatchify(again, 4, 100)[..., :29], mel, atol=1e-4, rtol=1e-4)
    other = m.generate(cond, spk, temperature=0.0, tokens=m.reference_tokens(torch.randn(2, 100, 24))[0])
    assert not torch.allclose(other, mel, atol=1e-3)  # the backbone reads them
    # without tokens: the null ones, in training and in generation alike
    null = m.generate(cond, spk, temperature=0.0)
    torch.testing.assert_close(null, m.generate(cond, spk, 0.0, tokens=m.ref_enc.null_tokens(2)), atol=1e-6, rtol=1e-6)
    # guidance: the conditional and the null rows share the tokens
    assert torch.isfinite(m.generate(cond, spk, 0.5, torch.Generator().manual_seed(1), cfg_scale=1.5,
                                     tokens=tokens)).all()
    fast = ARGenerator(m, max_steps=16, graphs=False)
    assert fast.lead == 3 and fast.fits(13 * 4) and not fast.fits(14 * 4)  # the tokens take three positions
    prefix, pc = torch.randn(2, 100, 8), torch.zeros(2, 116, 8)
    for kw in ({}, {"cfg_scale": 1.5}, {"prefix": prefix, "prefix_cond": pc}):
        ref = m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5), tokens=tokens, **kw)
        for _ in range(2):  # the second call runs on the first one's stale caches
            out = fast(cond, spk, 0.7, torch.Generator().manual_seed(5), tokens=tokens, **kw)
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fast(cond, spk, 0.0), null, atol=1e-4, rtol=1e-4)
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")
    mc = m.cuda()
    ref = mc.generate(cond.cuda(), spk.cuda(), 0.7, torch.Generator(device="cuda").manual_seed(5), tokens=tokens.cuda())
    g = ARGenerator(mc, max_steps=16)
    for _ in range(2):
        out = g(cond.cuda(), spk.cuda(), 0.7, torch.Generator(device="cuda").manual_seed(5), tokens=tokens.cuda())
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_training_step_with_references_trains_the_reference_path():
    m = _model().train()
    no_prompt = [f"drift.prompt.{k}={v}" for k, v in {**PROMPT, "prob": 0.0}.items()]
    cfg = load_config("configs/ar_tts.yaml", OVER + no_prompt + ["drift.reference.prob=1.0",
                                                                  "drift.reference.id_dropout=1.0"])
    tau = torch.nn.Parameter(torch.tensor(1.0))
    torch.manual_seed(0)
    loss, metrics = training_step(m, _ref_batch(), cfg, "cpu", tau)
    loss.backward()
    assert torch.isfinite(loss) and metrics["referenced"] == 0.5 and metrics["id_dropped"] == 0.5
    assert metrics["prompted"] == 0.0 and metrics["no_voice"] == 0.0
    assert m.ref_enc.out.weight.grad.abs().sum() > 0 and m.ref_enc.vec.weight.grad.abs().sum() > 0
    assert m.ref_enc.null.grad.abs().sum() > 0  # the utterance without a reference has the null tokens
    grad = m.spk.weight.grad  # the referenced utterance reads the prompted row, the other its own
    assert grad[m.prompt_row].abs().sum() > 0 and grad[1].abs().sum() > 0 and not grad[0].any()
    assert m.encoder.spk.weight.grad[1].abs().sum() > 0 and not m.encoder.spk.weight.grad[0].any()
    # the IDs kept: the reference is read beside the speaker's row
    keep = load_config("configs/ar_tts.yaml", OVER + no_prompt + ["drift.reference.prob=1.0",
                                                                   "drift.reference.id_dropout=0.0"])
    m.zero_grad()
    torch.manual_seed(0)
    training_step(m, _ref_batch(), keep, "cpu", tau)[0].backward()
    assert m.spk.weight.grad[0].abs().sum() > 0 and not m.spk.weight.grad[m.prompt_row].any()
    assert m.ref_enc.out.weight.grad.abs().sum() > 0 and not m.ref_enc.vec.weight.grad.any()
    # a voice without a row and without a reference or a prompt: the null voice (voice guidance's other branch)
    b = _ref_batch()
    b["spk"] = torch.tensor([-1, -1])
    m.zero_grad()
    torch.manual_seed(0)
    late = keep.to_dict()  # too short for a prompt (a voice without a row is cut whenever it can be)
    late["drift"]["prompt"].update(min_seconds=5.0, max_seconds=6.0)
    metrics = training_step(m, b, Config(late), "cpu", tau)[1]
    assert metrics["no_row"] == 1.0 and metrics["id_dropped"] == 0.5 and metrics["no_voice"] == 0.5
    assert metrics["prompted"] == 0.0
    # prompts and references together; a batch without references (the held-out objective): the null tokens
    both = load_config("configs/ar_tts.yaml", OVER + [f"drift.prompt.{k}={v}" for k, v in PROMPT.items()]
                       + ["drift.reference.prob=1.0", "drift.prompt.id_dropout=1.0"])
    m.zero_grad()
    torch.manual_seed(0)
    loss, metrics = training_step(m, _ref_batch(), both, "cpu", tau)
    loss.backward()
    assert metrics["prompted"] == 1.0 and metrics["referenced"] == 0.5 and metrics["id_dropped"] == 1.0
    assert m.prompt_enc.out.weight.grad.abs().sum() > 0 and m.ref_enc.vec.weight.grad.abs().sum() > 0
    plain = training_step(m, _batch(), both, "cpu", tau)[1]
    assert plain["referenced"] == 0.0 and torch.isfinite(plain["loss"])


def test_synthesis_from_a_reference_and_from_both():
    m = _model()
    text = torch.tensor([text_to_ids("ab cd", normalized=True)])
    n, spk = torch.tensor([text.shape[1]]), torch.tensor([0])
    ref, prompt = torch.randn(1, 100, 24), torch.randn(1, 100, 16)
    g = [torch.Generator().manual_seed(3) for _ in range(4)]
    mel, y_len = m.synthesize(text, n, spk, temperature=0.5, generator=g[0], reference=ref)
    assert mel.shape[-1] == int(y_len[0]) and torch.isfinite(mel).all()
    assert torch.equal(mel, m.synthesize(text, n, torch.tensor([2]), 1.0, 0.5, generator=g[1], reference=ref)[0])
    other = m.synthesize(text, n, spk, temperature=0.5, generator=g[2], reference=ref + 1.0)[0]
    assert other.shape != mel.shape or not torch.allclose(other, mel, atol=1e-3)  # the voice is the reference's
    # the model's own pieces: the prompted row, the reference's tokens and vector
    row = torch.tensor([m.prompt_row])
    tokens, vec = m.reference_tokens(ref)
    cond, _ = m.aligned_condition(text, n, row, spk_emb=vec)
    torch.testing.assert_close(m.generate(cond, row, 0.5, torch.Generator().manual_seed(3), tokens=tokens), mel,
                               atol=1e-5, rtol=1e-5)
    both = m.synthesize(text, n, spk, temperature=0.5, generator=g[3], prompt=prompt, reference=ref)[0]
    out = m.generate(cond, row, 0.5, torch.Generator().manual_seed(3), prefix=prompt,
                     prefix_cond=torch.zeros(1, 116, 16), tokens=tokens)
    torch.testing.assert_close(out, both, atol=1e-5, rtol=1e-5)


def test_references_are_another_utterance_of_the_speaker():
    assert reference_frames(None) is None and reference_frames({"min_seconds": 1, "max_seconds": 2}) == (93, 187)
    rng = random.Random(0)
    mel = torch.arange(50.0)[None].expand(100, -1)
    for _ in range(20):
        c = crop_reference(mel, 20, 30, rng)
        assert 20 <= c.shape[1] <= 30 and torch.equal(c[0], torch.arange(c.shape[1]) + c[0, 0])
    assert crop_reference(mel[:, :25], 20, 30, random.Random(1)).shape[1] <= 25
    out = collate([{"text": torch.zeros(3, dtype=torch.long), "mel": torch.zeros(100, 8), "spk": 0, "index": 0,
                    "ref": torch.ones(100, 6)},
                   {"text": torch.zeros(2, dtype=torch.long), "mel": torch.zeros(100, 5), "spk": 1, "index": 1,
                    "ref": None}])
    assert out["ref"].shape == (2, 100, 6) and out["ref_len"].tolist() == [6, 0] and not out["ref"][1].any()

    from drifting_tts.stream import ParquetStream

    stream = ParquetStream.__new__(ParquetStream)
    stream.reference, stream.max_voices = (4, 6), 2
    kept: OrderedDict = OrderedDict()

    def items(voices, start=0):  # every utterance's mel holds its own number
        return [{"mel": torch.full((100, 8 if v != "short" else 3), float(start + i)), "voice": (0, v)}
                for i, v in enumerate(voices)]

    first = items(["a", "a", "b", "short", "short"])
    stream._pair(first, kept, random.Random(0))
    assert float(first[0]["ref"][0, 0]) == 1.0 and float(first[1]["ref"][0, 0]) == 0.0  # each other, never itself
    assert 4 <= first[0]["ref"].shape[1] <= 6
    assert first[2]["ref"] is None  # the only utterance of its speaker so far
    assert first[3]["ref"] is None and first[4]["ref"] is None  # too short to be a reference
    assert set(kept) == {(0, "a"), (0, "b")} and kept[(0, "b")].dtype == torch.float16
    second = items(["b", "c"], start=10)
    stream._pair(second, kept, random.Random(0))
    assert float(second[0]["ref"][0, 0]) == 2.0  # the utterance kept from the shard before
    assert second[1]["ref"] is None
    assert set(kept) == {(0, "b"), (0, "c")}  # at most max_voices speakers, the oldest dropped


def test_train_ar_with_references_and_the_synthesizers_modes(tmp_path):
    """Fine-tuning a prompted model with a reference encoder on prepared data (``MelDataset``'s references), then
    the Synthesizer: a recording as a prompt, through the reference encoder, or both."""
    import soundfile as sf

    from drifting_tts.cli import main
    from drifting_tts.data import MelDataset
    from drifting_tts.synthesize import Synthesizer
    from tests.test_data import _fake_parquet
    from tests.test_loop_eval import TINY

    _fake_parquet(tmp_path / "x.parquet", n=14)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "0", "--no-trim", "--backend", "bigvgan"])
    ds = MelDataset(data, "train", min_frames=1, reference=(20, 60))
    seen = 0
    for i in range(len(ds)):
        it = ds[i]
        if it["ref"] is not None:
            seen += 1
            assert 20 <= it["ref"].shape[1] <= 60
            others = [j for j in ds.by_speaker[it["spk"]] if j != i]
            assert any(_contains(ds._mel(j), it["ref"]) for j in others) and others  # another utterance's frames
    assert seen > 0
    prompts = [f"drift.prompt.{k}={v}" for k, v in {**PROMPT, "min_seconds": 0.3, "max_seconds": 1.5}.items()]
    base, work = tmp_path / "base", tmp_path / "ref"
    main(["train-ar", "--workdir", str(base), f"data.root={data}", *TINY, "train.steps=1", "train.save_every=1",
          "model.ar.prompt=true", *prompts])
    main(["train-ar", "--workdir", str(work), f"data.root={data}", *TINY, "train.steps=2", "train.save_every=2",
          f"train.init_from={base / 'model_ema.pt'}", *OVER[2:], *prompts, "drift.reference.min_seconds=0.3",
          "drift.reference.max_seconds=1.0"])
    synth = Synthesizer(work / "model_ema.pt", "cpu", vocoder="griffin-lim")
    synth.vocoder.model.n_iter = 2
    assert synth.model.ref_enc is not None and synth.model.spk.num_embeddings == 4  # no second extra row
    with torch.no_grad():
        for q in synth.model.parameters():
            q.add_(torch.randn_like(q) * 0.05)
    t = torch.arange(36000) / 24000
    sf.write(tmp_path / "a.wav", (0.3 * torch.sin(2 * torch.pi * 220 * t)).numpy(), 24000)
    sf.write(tmp_path / "b.wav", (0.3 * torch.sin(2 * torch.pi * 330 * t) * (1 + t)).numpy(), 24000)
    text = "merhaba dünya. nasılsın?"
    refs = {mode: synth.reference(tmp_path / "a.wav", seconds=0.5, mode=mode) for mode in ("prefix", "reference",
                                                                                           "both")}
    assert refs["prefix"].tokens is None and refs["reference"].mel is None
    assert refs["both"].mel is not None and refs["both"].tokens.shape == (1, 3, 32)
    assert torch.equal(refs["both"].vector, refs["reference"].vector)
    out = {mode: synth(text, ref_audio=r, seed=1)[0] for mode, r in refs.items()}
    assert all(torch.isfinite(w).all() for w in out.values())
    assert not torch.equal(out["prefix"], out["reference"]) and not torch.equal(out["both"], out["reference"])
    assert torch.equal(synth(text, ref_audio=synth.reference(tmp_path / "a.wav", seconds=0.5), seed=1)[0],
                       out["both"])  # the default of a model trained for both
    b = synth(text, ref_audio=synth.reference(tmp_path / "b.wav", mode="reference"), seed=1)[0]
    assert b.shape != out["reference"].shape or not torch.allclose(b, out["reference"], atol=1e-4)
    for mode, r in refs.items():  # streamed: the same audio
        streamed = torch.cat(list(synth.stream(text, ref_audio=r, seed=1)))
        torch.testing.assert_close(streamed, out[mode], atol=1e-4, rtol=1e-4)
    assert torch.isfinite(synth(text, speaker=0, seed=1)[0]).all()  # a speaker row, with the null tokens
    prompted = Synthesizer(base / "model_ema.pt", "cpu", vocoder="griffin-lim")
    with pytest.raises(ValueError, match="trained for prefix"):
        prompted.reference(tmp_path / "a.wav", mode="reference")


def _contains(mel: torch.Tensor, crop: torch.Tensor) -> bool:
    n = crop.shape[1]
    return any(torch.equal(mel[:, s: s + n], crop) for s in range(mel.shape[1] - n + 1))

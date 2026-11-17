"""One codec frame per step (#92): a tied ARDrift model reads and draws one latent frame per step and copies it over
the patch, targets sampled from the codec's posterior, latents stored once (``extract-latents --native``) with the
posterior's standard deviations (``--save-std``), and the gate's one-copy decoding. Tiny models on the CPU."""

import json

import numpy as np
import pytest
import torch

from drifting_tts import latents
from drifting_tts.cli import main
from drifting_tts.config import Config, load_config
from drifting_tts.data import MelDataset, collate
from drifting_tts.models.ar import ARDriftTTS, ARGenerator, ar_features, patchify, unpatchify
from drifting_tts.train_ar import cut_prompts, sample_targets, training_step
from tests.test_data import _fake_parquet
from tests.test_latent_tts import FakeVAE
from tests.test_prompt import PROMPT, _batch

CFG = {"kind": "ar", "text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
       "ar": {"patch": 4, "lookahead": 2, "dim": 32, "depth": 2, "heads": 2, "head_dim": 32, "head_depth": 2,
              "tied": True}}


def _model(seed: int = 0, n_mels: int = 6, **ar) -> ARDriftTTS:
    torch.manual_seed(seed)
    m = ARDriftTTS(Config({**CFG, "ar": {**CFG["ar"], **ar}}), num_speakers=3, n_mels=n_mels).eval()
    with torch.no_grad():
        for p in m.parameters():  # zero-initialised output layers: make every path visible
            p.add_(torch.randn_like(p) * 0.05)
    return m


def _tied(frames: int, batch: int = 2, n: int = 6) -> torch.Tensor:
    """Latents ``[B, n, frames]`` whose every four frames are copies, as ``extract-latents`` stores them."""
    return torch.randn(batch, n, -(-frames // 4)).repeat_interleave(4, -1)[..., :frames]


def test_a_tied_model_reads_and_draws_one_frame_per_step():
    m, plain = _model(), _model(tied=False)
    assert m.x_in.in_features == 6 and m.head.out.out_features == 6 and m.head.noise_dim == 6 and m.bos.shape == (6,)
    assert plain.x_in.in_features == 24 and plain.head.out.out_features == 24
    p = torch.randn(2, 5, 24)
    assert m.squeeze(p).shape == (2, 5, 6) and torch.equal(plain.squeeze(p), p)
    f = torch.randn(2, 5, 6)
    torch.testing.assert_close(m.squeeze(m.spread(f)), f)
    assert torch.equal(unpatchify(m.spread(f), 4, 6)[..., 0], unpatchify(m.spread(f), 4, 6)[..., 3])
    feats = ar_features(torch.randn(3, 2, 6), 1, 6, detail=False)
    assert set(feats) == {"patch"} and feats["patch"].shape == (3, 2, 1, 6)  # one frame: one map
    assert set(ar_features(torch.randn(3, 2, 24), 4, 6, detail=False)) == {"patch", "frame"}


def test_tied_generation_emits_equal_copies_and_matches_teacher_forcing():
    m = _model()
    cond, spk = torch.randn(2, 22, 29), torch.tensor([0, 2])  # 6 latent + 16 text channels; the last patch is padded
    mel = m.generate(cond, spk, temperature=0.7, generator=torch.Generator().manual_seed(1))
    assert mel.shape == (2, 6, 29)
    groups = mel[..., :28].unflatten(-1, (7, 4))
    assert torch.equal(groups, groups[..., :1].expand_as(groups))  # the copies of a frame cannot disagree
    mel = m.generate(cond, spk, temperature=0.0)
    padded = m.pad_frames(cond)
    full = m.pad_frames(mel)
    full[..., 28:] = full[..., 28:29]  # the cut-off copies of the last frame
    s = m.teacher_forced(padded, full, spk)
    z = torch.zeros(2 * s.shape[1], m.head.noise_dim)
    again = m.emit(patchify(padded[:, :6], 4).flatten(0, 1), s.flatten(0, 1), z).view(2, s.shape[1], -1)
    torch.testing.assert_close(unpatchify(again, 4, 6)[..., :29], mel, atol=1e-4, rtol=1e-4)
    gta = m.gta(padded, full, spk, 0.0)
    torch.testing.assert_close(gta[..., :29], mel, atol=1e-4, rtol=1e-4)
    fast = ARGenerator(m, max_steps=16, graphs=False)
    prefix, pc = _tied(8), torch.zeros(2, 22, 8)
    for kw in ({}, {"cfg_scale": 1.5}, {"prefix": prefix, "prefix_cond": pc}):
        ref = m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5), **kw)
        for _ in range(2):
            torch.testing.assert_close(fast(cond, spk, 0.7, torch.Generator().manual_seed(5), **kw), ref, atol=1e-4,
                                       rtol=1e-4)


def test_sampled_targets_share_a_draw_within_a_latent_frame():
    y, y_len = _tied(16), torch.tensor([16, 10])
    y[1, :, 10:] = 0
    torch.manual_seed(0)
    out = sample_targets(y, y_len, 0.1, group=4)
    assert not out[1, :, 10:].any() and not torch.equal(out[0], y[0])  # the padding stays, the frames move
    d = (out - y)[0].unflatten(-1, (4, 4))
    assert torch.equal(d, d[..., :1].expand_as(d)) and 0.05 < float(d.std()) < 0.15
    single = sample_targets(y, y_len, 0.1)[0] - y[0]
    assert not torch.equal(single[:, 0], single[:, 1])  # mel frames: a draw each
    std = torch.zeros_like(y)
    std[0, 0] = 2.0  # the posterior's own spread: one channel of one utterance
    out = sample_targets(y, y_len, "posterior", 4, std)
    assert torch.equal(out[1], y[1]) and torch.equal(out[0, 1:], y[0, 1:]) and float((out - y)[0, 0].std()) > 1.0
    with pytest.raises(ValueError, match="save-std"):
        sample_targets(y, y_len, "posterior", 4)


def test_training_step_of_a_tied_model_with_sampled_targets_and_prompts():
    m = _model(prompt=True).train()
    b = _batch()
    b["mel"] = _tied(64) * torch.tensor([1.0, 1.0])[:, None, None]
    b["mel"][1, :, 50:] = 0
    b["mel_len"] = torch.tensor([64, 48])
    over = ["drift.gen_per_cond=4", "drift.max_rows=64", "drift.detail=false", "model.ar.tied=true"]
    cfg = load_config("configs/ar_tts.yaml", over + ["drift.target_noise=0.1"])
    tau = torch.nn.Parameter(torch.tensor(1.0))
    torch.manual_seed(0)
    loss, metrics = training_step(m, b, cfg, "cpu", tau, detail=True)
    loss.backward()
    assert torch.isfinite(loss) and m.head.out.weight.grad.abs().sum() > 0 and m.x_in.weight.grad.abs().sum() > 0
    assert "loss/patch" in metrics and "loss/frame" not in metrics  # the kernel sees the frame once
    quiet = load_config("configs/ar_tts.yaml", over)
    torch.manual_seed(0)
    a = training_step(m, b, quiet, "cpu", tau)[1]
    torch.manual_seed(0)
    assert float(a["prior"]) != float(training_step(m, b, cfg, "cpu", tau)[1]["prior"])  # the targets were sampled
    # a prompt is cut on the grid of the copies, so that a patch never holds two latent frames
    torch.manual_seed(0)
    cut = cut_prompts(m, b["text"], b["text_len"], b["mel"], b["mel_len"], b["spk"], Config(PROMPT), 1)
    assert (cut["prompt_len"] % 4 == 0).all() and ((b["mel_len"] - cut["mel_len"]) % 4 == 0).all()
    prompts = load_config("configs/ar_tts.yaml", over + ["model.ar.prompt=true"]
                          + [f"drift.prompt.{k}={v}" for k, v in PROMPT.items()])
    torch.manual_seed(0)
    assert torch.isfinite(training_step(m, b, prompts, "cpu", tau)[0])


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("native")
    _fake_parquet(tmp / "x.parquet", n=8)
    main(["prepare", "--parquet-glob", str(tmp / "*.parquet"), "--out", str(tmp / "mel"), "--workers", "1",
          "--val-size", "0", "--no-trim", "--save-audio"])

    class WithPosterior(FakeVAE):
        def posterior(self, wav, sr):
            z = self.encode(wav, sr)
            return z, 0.1 + 0.05 * torch.arange(z.shape[1])[None, :, None].expand_as(z).float()

    original = latents.load_backend
    latents.load_backend = lambda name, device="cpu", **kw: WithPosterior()
    try:
        for name, flags in (("repeated", []), ("native", ["--native", "--save-std"])):
            main(["extract-latents", "--data", str(tmp / "mel"), "--backend", "voxcpm2", "--out", str(tmp / name),
                  "--device", "cpu", *flags])
        latents.load_backend = lambda name, device="cpu", **kw: FakeVAE()
        with pytest.raises(SystemExit, match="no posterior"):
            main(["extract-latents", "--data", str(tmp / "mel"), "--backend", "voxcpm2", "--out", str(tmp / "no"),
                  "--device", "cpu", "--save-std"])
    finally:
        latents.load_backend = original
    return tmp


def test_native_storage_reads_as_the_repeated_one_and_carries_the_posterior(roots):
    rep, nat = (MelDataset(roots / name, "train", min_frames=1) for name in ("repeated", "native"))
    st = json.loads((roots / "native" / "stats.json").read_text())
    assert st["stored_repeat"] == 1 and st["latent_repeat"] == 4 and st["posterior_std"] and st["posterior_rel"] > 0
    other = json.loads((roots / "repeated" / "stats.json").read_text())
    assert other["stored_repeat"] == 4 and not other["posterior_std"] and other["active"] == st["active"]
    assert (roots / "native" / "mels.bin").stat().st_size * 4 == (roots / "repeated" / "mels.bin").stat().st_size
    assert len(rep) == len(nat) and nat.copies == 4 and rep.copies == 1
    for i in range(len(rep)):
        assert torch.equal(rep[i]["mel"], nat[i]["mel"]) and rep.frames(i) == nat.frames(i)
    with pytest.raises(FileNotFoundError, match="save-std"):
        MelDataset(roots / "repeated", "train", min_frames=1, with_std=True)
    ds = MelDataset(roots / "native", "train", min_frames=1, with_std=True)
    it = ds[0]
    assert it["mel_std"].shape == it["mel"].shape and torch.equal(it["mel_std"][:, 0], it["mel_std"][:, 3])
    raw = it["mel_std"] * ds.stats.std  # back in the codec's units: 0.1, 0.15, ... per channel
    act = np.array(st["active"])
    np.testing.assert_allclose(raw[:, 0].numpy()[act], (0.1 + 0.05 * np.arange(6))[act], rtol=2e-3)
    assert act.any() and not it["mel_std"][~torch.from_numpy(act)].any()
    batch = collate([ds[0], ds[1]])
    assert batch["mel_std"].shape == batch["mel"].shape


def test_unused_channels_are_left_out_of_the_normalised_frames():
    from drifting_tts.data import MelStats

    st = MelStats([1.0, -2.0, 0.5], [2.0, 0.5, 0.1], active=[True, False, True])
    x = torch.randn(2, 3, 7)
    z = st.normalize(x)
    assert not z[:, 1].any() and torch.allclose(z[:, 0], (x[:, 0] - 1.0) / 2.0)
    back = st.denormalize(z + 0.3)  # whatever a model writes there, the channel decodes as its mean
    assert torch.allclose(back[:, 1], torch.full((2, 7), -2.0))
    assert torch.allclose(back[:, 2], (z[:, 2] + 0.3) * 0.1 + 0.5)
    again = MelStats.from_dict(st.to_dict())
    assert again.to_dict() == st.to_dict() and st.to_dict()["active"] == [True, False, True]
    assert MelStats([0.0, 1.0], [1.0, 1.0], active=[True, True]).active is None  # all used: nothing to mask
    assert "active" not in MelStats([0.0, 1.0], [1.0, 1.0]).to_dict()
    y = torch.randn(2, 3, 8)
    y[:, 1] = 0  # the unused channel of a batch stays 0 under sampled targets
    out = sample_targets(y, torch.tensor([8, 8]), 0.2)
    assert not out[:, 1].any() and not torch.equal(out[:, 0], y[:, 0])


def test_train_ar_on_native_latents_with_a_tied_model(roots, tmp_path):
    from tests.test_loop_eval import TINY

    work = tmp_path / "run"
    index = roots / "native" / "index.jsonl"  # two utterances as the held-out split of the loop's evaluation
    lines = index.read_text().splitlines()
    index.write_text("\n".join(line.replace('"split": "train"', '"split": "val"') if n < 2 else line
                               for n, line in enumerate(lines)) + "\n")
    quiet = ["train.vocoder=griffin-lim", "train.eval.num=1", "train.eval.asr=none", "train.eval.mos=none",
             "train.eval.f0=false", "train.eval.batches=1", "train.eval.batch_size=2"]
    class Decoder(FakeVAE):  # the loop's audio comes from the codec's decoder
        hop_out, device = 960, "cpu"

        def decode(self, latent):
            return torch.zeros(latent.shape[0], latent.shape[-1] * 960)

    from drifting_tts.latents import vocoder

    original = vocoder.load_backend
    vocoder.load_backend = lambda name, device="cpu", **kw: Decoder()
    try:
        main(["train-ar", "--workdir", str(work), f"data.root={roots / 'native'}", *TINY, "train.steps=2",
              "train.save_every=2", "model.ar.tied=true", "drift.detail=false", "drift.target_noise=posterior",
              "train.sample_every=0", "train.eval_every=2", *quiet])
    finally:
        vocoder.load_backend = original
    assert (work / "eval_history.json").exists()  # the held-out objective draws posterior samples too
    ck = torch.load(work / "model_ema.pt", map_location="cpu", weights_only=False)
    assert ck["ema"]["head.out.weight"].shape[0] == 6 and ck["ema"]["x_in.weight"].shape[1] == 6
    from drifting_tts.train import load_tts

    model, _, _ = load_tts(work / "model_ema.pt")
    assert model.tied
    ds = MelDataset(roots / "native", "train", min_frames=1)
    it = ds[0]
    with torch.no_grad():
        mel, n = model.synthesize(it["text"][None], torch.tensor([it["text"].numel()]), torch.tensor([0]),
                                  temperature=0.5, generator=torch.Generator().manual_seed(0))
    whole = mel[..., : int(n) // 4 * 4].unflatten(-1, (-1, 4))
    assert torch.equal(whole, whole[..., :1].expand_as(whole))

"""The codec gate (#91) on the CPU: its numeric core, and the whole script on a tiny data root with tiny backends,
stub judges and a stub aligner. The real judges, codec weights and MMS aligner are not exercised here."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchaudio

from drifting_tts.audio import HOP_LENGTH, resample
from drifting_tts.cli import main
from drifting_tts.latents import LatentStats
from drifting_tts.latents.dacvae import DACVAE, DACVAEBackend
from drifting_tts.metrics import band_lsd
from scripts import codec_gate as gate
from tests.test_data import _fake_parquet
from tests.test_latents import tiny_dacvae, tiny_voxcpm


def test_band_lsd_sees_a_missing_band():
    torch.manual_seed(0)
    x = 0.1 * torch.randn(48_000)
    assert band_lsd(x, x, 24_000, 8000, 12_000) == 0.0
    assert band_lsd(x, 0.5 * x, 24_000, 8000, 12_000) == pytest.approx(20 * math.log10(2), abs=1e-6)  # a level change
    dull = resample(resample(x, 24_000, 16_000), 16_000, 24_000)  # nothing above 8 kHz
    assert band_lsd(x, dull, 24_000, 8000, 12_000) > 30
    assert band_lsd(x, dull, 24_000, 1000, 4000) < 0.5
    assert band_lsd(x, dull[:40_000], 24_000, 1000, 4000) < 0.5  # the longer one is cut
    silent = torch.zeros(48_000)
    assert band_lsd(silent, silent, 24_000, 8000, 12_000) == 0.0
    with pytest.raises(ValueError, match="empty"):
        band_lsd(x, x, 16_000, 8000, 12_000)


def test_perturbations_work_in_normalised_units():
    torch.manual_seed(0)
    std, mean = torch.tensor([1.0, 3.0, 0.5, 0.0]), torch.tensor([0.0, 2.0, -1.0, 5.0])
    z = torch.randn(4, 4000) * std[:, None] + mean[:, None]
    stats = LatentStats.from_latents([z])
    assert gate.active_channels(stats).tolist() == [True, True, True, False]
    zn = gate.normalize(z, stats)
    assert zn[:3].std(1).tolist() == pytest.approx([1, 1, 1], abs=1e-3) and (zn[3] == 0).all()  # unused channel: 0

    noisy = gate.perturb(z, stats, "noise", 0.2, torch.randn(z.shape))
    assert ((noisy - z).std(1) / stats.std).tolist()[:3] == pytest.approx([0.2] * 3, rel=0.05)
    assert torch.allclose(noisy[3], z[3], atol=1e-4)  # nothing is added to a constant channel
    eps = torch.randn(z.shape)  # the levels share one draw
    torch.testing.assert_close(gate.perturb(z, stats, "noise", 0.4, eps) - z,
                               2 * (gate.perturb(z, stats, "noise", 0.2, eps) - z))

    shrunk = gate.perturb(z, stats, "scale", 0.7)
    torch.testing.assert_close(shrunk.mean(1), z.mean(1), atol=1e-4, rtol=0)
    assert (shrunk.std(1) / z.std(1))[:3].tolist() == pytest.approx([0.7] * 3, abs=1e-4)

    smooth = gate.perturb(z, stats, "smooth")
    assert smooth.shape == z.shape
    torch.testing.assert_close(smooth[:, 1:-1], 0.25 * z[:, :-2] + 0.5 * z[:, 1:-1] + 0.25 * z[:, 2:])
    ramp = torch.arange(10.0)[None]
    torch.testing.assert_close(gate.perturb(ramp, LatentStats.from_latents([ramp]), "smooth")[:, 1:-1], ramp[:, 1:-1])
    assert [c for c, _, _ in gate.perturbations((0.1, 0.5), (0.9,))] == ["noise0.1", "noise0.5", "scale0.9", "smooth"]
    with pytest.raises(ValueError):
        gate.perturb(z, stats, "pitch")


def test_noise_level_that_costs_the_loss():
    levels, scores = (0.1, 0.2, 0.5), (2.95, 2.70, 2.00)  # clean 3.0: losses 0.05, 0.30, 1.00
    assert gate.level_for_loss(levels, scores, 3.0, 0.2) == pytest.approx(0.1 + 0.1 * 0.15 / 0.25)
    assert gate.level_for_loss(levels, scores, 3.0, 0.02) == pytest.approx(0.04)  # on the first segment, from (0, 0)
    assert gate.level_for_loss(levels, scores, 3.0, 1.5) is None  # never reached: "> 0.5"
    assert gate.level_for_loss(levels[::-1], scores[::-1], 3.0, 0.2) == pytest.approx(0.16)  # any order
    assert gate.loss_at(levels, scores, 3.0, 0.2) == pytest.approx(0.30)
    assert gate.loss_at(levels, scores, 3.0, 0.35) == pytest.approx(0.65)
    summary = gate.tolerance_summary({f"noise{v:g}": {"utmosv2": {"mean": s}} for v, s in zip(levels, scores)}, 3.0,
                                     levels)
    assert summary["noise_for_loss"] == pytest.approx(0.16) and summary["loss_at_noise"] == pytest.approx(0.30)
    assert gate.tolerance_summary({}, 3.0, levels) == {} and gate.tolerance_summary({}, None, levels) == {}


def test_span_frames_and_segments():
    spans = [("a", 0.10, 0.13), ("b", 0.13, 0.20), ("c", 0.95, 1.20)]
    assert gate.span_frames(spans, 100, 100.0) == [("a", 10, 13), ("b", 13, 20), ("c", 95, 100)]  # centres inside
    # 12.5 Hz (80 ms frames): a span shorter than a frame takes the frame holding its midpoint
    assert gate.span_frames(spans[:2], 100, 12.5) == [("a", 1, 2), ("b", 2, 3)]
    assert gate.span_frames([("a", 9.0, 9.1)], 100, 100.0) == []  # past the end
    z = torch.arange(100.0).repeat(2, 1)
    emb, labels = gate.segment_embeddings(z, spans, 100.0)
    assert labels == ["a", "b", "c"] and emb[:, 0].tolist() == [11.0, 16.0, 97.0]
    x, y = gate.frame_labels(z, spans, 100.0)
    assert x.shape == (15, 2) and y == ["a"] * 3 + ["b"] * 7 + ["c"] * 5
    assert gate.segment_embeddings(z, [], 100.0)[0].shape == (0, 2)


def test_abx_error():
    torch.manual_seed(0)
    labels = [k % 4 for k in range(200)]
    centres = 10 * torch.randn(4, 8)
    emb = centres[labels] + 0.1 * torch.randn(200, 8)
    groups = [k // 100 for k in range(200)]
    res = gate.abx_error(emb, labels, groups)
    assert res["error"] == 0.0 and res["cells"] == 2 * 4 * 3 and res["triplets"] == 24 * 50
    noise = gate.abx_error(torch.randn(200, 8), labels, groups, per_pair=200)
    assert 0.4 < noise["error"] < 0.6 and noise["ci"][0] < noise["error"] < noise["ci"][1]
    assert gate.abx_error(torch.zeros(200, 8), labels, groups)["error"] == 0.5  # ties count half
    # triplets stay within a group: a speaker offset far larger than the classes' distance does not matter
    shifted = emb + 1000 * torch.tensor(groups)[:, None]
    assert gate.abx_error(shifted, labels, groups)["error"] == 0.0
    assert gate.abx_error(shifted, [(lab + g) % 4 for lab, g in zip(labels, groups)], [0] * 200)["error"] > 0.2
    assert gate.abx_error(emb[:3], [0, 1, 2], [0, 0, 0]) is None  # no label has two segments
    assert gate.abx_error(emb, labels, groups, seed=1) == gate.abx_error(emb, labels, groups, seed=1)


def test_linear_probe():
    torch.manual_seed(0)
    y = [("a", "b", "c")[k % 3] if k % 2 else "a" for k in range(600)]  # "a" is the majority (2/3)
    centres = {"a": torch.tensor([3.0, 0, 0, 0]), "b": torch.tensor([0, 3.0, 0, 0]), "c": torch.tensor([0, 0, 3.0, 0])}
    x = torch.stack([centres[c] for c in y]) + 0.3 * torch.randn(600, 4)
    res = gate.linear_probe(x[:400], y[:400], x[400:], y[400:])
    assert res["accuracy"] > 0.98 and res["baseline"] == pytest.approx(sum(c == "a" for c in y[400:]) / 200)
    assert res["classes"] == 3 and res["train_frames"] == 400 and res["test_frames"] == 200
    noise = gate.linear_probe(torch.randn(400, 4), y[:400], torch.randn(200, 4), y[400:])
    assert abs(noise["accuracy"] - noise["baseline"]) < 0.1
    unseen = gate.linear_probe(x[:400], y[:400], x[400:], ["z"] * 200)  # a label absent from training is an error
    assert unseen["accuracy"] == 0.0
    assert gate.linear_probe(x[:0], [], x, y) is None and gate.linear_probe(x, y, x[:0], []) is None


def test_repeat_deviation():
    torch.manual_seed(0)
    native = torch.randn(6, 50)
    stored = native.repeat_interleave(4, dim=1)
    res = gate.repeat_deviation([stored, stored[:, :102]], 4)  # a trailing partial group is left out
    assert res["positions"] == [0.0] * 4 and res["share"] == 0.0 and res["groups"] == 50 + 25
    iid = gate.repeat_deviation([torch.randn(6, 40_000)], 4)  # independent frames: 3/4 of the variance is within
    assert iid["share"] == pytest.approx(0.75, abs=0.02) and iid["positions"] == pytest.approx([0.75] * 4, abs=0.03)
    off = stored.clone()
    off[:, 3::4] += 1.0  # only the last copy deviates: it is 3/4 from the group mean, the others 1/4
    pos = gate.repeat_deviation([off], 4)["positions"]
    assert pos[3] == pytest.approx(9 * pos[0]) and pos[0] == pytest.approx(pos[1]) == pytest.approx(pos[2])
    assert gate.repeat_deviation([], 4) == {"positions": [], "share": None, "groups": 0}


def test_lag_and_fit():
    torch.manual_seed(0)
    x = torch.randn(2000)
    assert gate.lag(x, torch.cat([torch.zeros(7), x])[:2000], 20) == 7
    assert gate.lag(x, x[5:], 20) == -5
    assert gate.fit(torch.arange(6.0), 4, 1).tolist() == [1, 2, 3, 4]
    assert gate.fit(torch.arange(3.0), 6, -2).tolist() == [0, 0, 0, 1, 2, 0]


def test_sample_metrics():
    torch.manual_seed(0)
    logmel = gate.BigVGANLogMel()
    x = 0.1 * torch.randn(24_000)
    same = gate.sample_metrics(x, 24_000, x, 24_000, logmel, {"snr": lambda r, e: float(len(r))})
    assert same["mel_l1"] == 0.0 and same["lsd_8_12"] == 0.0 and same["lsd_12_20"] is None and same["snr"] == 16_000
    x48 = 0.1 * torch.randn(48_000)
    dull = resample(resample(x48, 48_000, 24_000), 24_000, 48_000)[:48_000]  # nothing above 12 kHz
    m = gate.sample_metrics(x48, 48_000, dull, 48_000, logmel)
    assert m["lsd_12_20"] > 30 and m["lsd_8_12"] < 5 and m["mel_l1"] < 0.05  # 8-12 kHz: only the filter's edge
    low = gate.sample_metrics(x48, 48_000, resample(x48, 48_000, 24_000), 24_000, logmel)  # compared at 24 kHz
    assert low["lsd_12_20"] is None and low["lsd_8_12"] == 0.0 and low["mel_l1"] == 0.0
    failing = gate.sample_metrics(x, 24_000, x, 24_000, logmel, {"pesq": lambda r, e: 1 / 0})
    assert failing["pesq"] is None
    rows = [{"a": 1.0, "b": None}, {"a": 3.0, "b": None, "c": 2.0}]
    assert gate.mean_metrics(rows) == {"a": 2.0, "b": None, "c": 2.0}


def test_verdict_and_tables():
    rec = {"wer": 0.06, "utmosv2": {"mean": 3.0}}
    good = {"input_rate": 16_000, "output_rate": 48_000, "frame_rate": 25.0, "dim": 64,
            "roundtrip": {"24k": {"wer": 0.07, "utmosv2": {"mean": 2.9}, "speaker_sim": {"mean": 0.96},
                                  "mel_l1": 0.5, "lsd_8_12": 4.2, "lsd_12_20": None, "pesq": None},
                          "native": {"wer": 0.07, "utmosv2": {"mean": 2.95}, "lsd_8_12": 3.0, "lsd_12_20": 9.5}},
            "tolerance": {"conditions": {"noise0.2": {"utmosv2": {"mean": 2.8}, "wer": 0.08},
                                         "smooth": {"utmosv2": {"mean": 2.0}, "wer": 0.2}},
                          "noise_for_loss": None, "loss_at_noise": 0.1, "max_noise": 0.5},
            "abx": {"error": 0.123, "ci": [0.1, 0.15]}, "probe": {"accuracy": 0.5, "baseline": 0.1}}
    bad = {"input_rate": 24_000, "output_rate": 24_000, "frame_rate": 12.5, "dim": 32,
           "roundtrip": {"24k": {"utmosv2": {"mean": 2.9}, "mel_l1": 0.4}},
           "tolerance": {"conditions": {"noise0.2": {"utmosv2": {"mean": 2.2}}}, "noise_for_loss": 0.07,
                         "loss_at_noise": 0.7, "max_noise": 0.5}, "abx": None, "probe": None}
    bare = {"input_rate": 24_000, "output_rate": 24_000, "frame_rate": 93.75, "dim": 100,
            "roundtrip": {"24k": {"mel_l1": 0.2, "lsd_8_12": 1.0}}}
    assert gate.verdict(good, rec, None, None) is None  # no thresholds, no verdict
    assert gate.verdict(good, rec, 0.2, 0.3) == "pass" and gate.verdict(good, rec, 0.05, 0.3) == "fail"
    assert gate.verdict(bad, rec, 0.2, 0.3) == "fail" and gate.verdict(bad, rec, 0.2, None) == "pass"
    assert gate.verdict(bare, rec, 0.2, 0.3) is None and gate.verdict(good, {}, 0.2, None) is None  # numbers missing

    results = {"recording": rec, "backends": {"voxcpm2": good, "mimi": bad, "bigvgan": bare}}
    lines = gate.tables(results).splitlines()
    assert "gate" not in lines[0] and lines[2].startswith("| recording | – | – | 6.00% | 3.000 |")
    row = {name: next(line for line in lines if line.startswith(f"| {name} | "))
           for name in ("voxcpm2", "mimi", "bigvgan")}
    assert "| 16 → 48 kHz | 25 Hz | 7.00% | 2.900 | -0.100 | 0.960 | – | – | 0.500 | 4.20 dB |" in row["voxcpm2"]
    assert "| 3.00 dB | 9.50 dB | > 0.5 std | -0.100 | -0.900 | 12.3% | 50.0% (10.0%) |" in row["voxcpm2"]
    assert "| 0.07 std | -0.700 | – | – | – |" in row["mimi"]
    assert row["bigvgan"].count("–") == 15 and len({line.count("|") for line in lines[:6]}) == 1
    assert any(line.startswith("| backend | clean | noise0.2 | smooth |") for line in lines)
    assert "| mimi | 2.900 (–) | 2.200 (–) | – (–) |" in lines

    for r in results["backends"].values():
        r["gate"] = gate.verdict(r, rec, 0.2, 0.3)
    results["repeated_latents"] = {"model": "m.pt", "repeat": 2,
                                   "generated": {"positions": [0.01, 0.02], "share": 0.015},
                                   "recorded": {"positions": [0.0, 0.0], "share": 0.0}}
    text = gate.tables(results)
    assert text.splitlines()[0].endswith("| gate |") and "| pass |" in text and "| fail |" in text
    assert "| generated | 0.0100 | 0.0200 | 0.0150 |" in text and "| recorded | 0.0000 | 0.0000 | 0.0000 |" in text


def test_mel_vocoder_backend(monkeypatch):
    import drifting_tts.vocoder

    class FakeVocoder:
        def __init__(self, spec, device):
            self.mel, self.name = "vocos" if spec == "vocos" else "bigvgan", spec

        def __call__(self, mel):
            frames = mel.shape[-1] - (self.mel == "vocos")
            return 0.1 * torch.sin(0.05 * torch.arange(frames * HOP_LENGTH, dtype=torch.float32))[None]

    monkeypatch.setattr(drifting_tts.vocoder, "load_vocoder", FakeVocoder)
    args = SimpleNamespace(device="cpu", vocoder=None, vocos=None)
    wav = 0.1 * torch.randn(5000)
    for name, spec, frames in (("bigvgan", "bigvgan-v2-ft", 19), ("vocos", "vocos-v2", 19), ("mel:vocos", "vocos", 20)):
        be = gate.build_backend(name, args)
        assert (be.name, be.spec) == (name, spec) and be.num_frames(5000, 24_000) == frames
        assert be.encode(wav, 24_000).shape == (1, 100, frames)
        outs, latents, lags = gate.roundtrip(be, [wav], 24_000)
        assert outs[0].shape == (5000,) and latents[0].shape == (100, frames) and lags == [0]  # no lag search
    assert gate.build_backend("vocos", SimpleNamespace(device="cpu", vocoder=None, vocos="vocos-ft")).spec == "vocos-ft"
    assert gate.build_backend("bigvgan", SimpleNamespace(device="cpu", vocoder="ft.pt", vocos=None)).spec == "ft.pt"
    with pytest.raises(SystemExit, match="unknown backend"):
        gate.build_backend("encodec", args)


def test_mms_aligner_labels_the_original_characters(monkeypatch):
    labels = ("-", "a", "b", "c", "s", "i", "'")

    class Bundle:
        @staticmethod
        def get_labels(star=None):
            return labels

        @staticmethod
        def get_model(with_star=False):
            class Model(torch.nn.Module):
                def forward(self, wav):  # the tokens of "caba si" in turn, 3 frames each, over 20 ms frames
                    ids = torch.tensor([3, 1, 2, 1, 4, 5]).repeat_interleave(3)
                    emission = torch.full((1, len(ids), len(labels)), -20.0)
                    emission[0, torch.arange(len(ids)), ids] = -1e-4
                    return emission, None

            return Model()

    monkeypatch.setattr(torchaudio.pipelines, "MMS_FA", Bundle)
    align = gate.MmsAligner("cpu")
    spans = align(torch.zeros(18 * 320), "Çaba, şı 3!")  # digits, punctuation and spaces are not aligned
    assert [c for c, _, _ in spans] == ["ç", "a", "b", "a", "ş", "ı"]
    assert [a for _, a, _ in spans] == pytest.approx([0.06 * k for k in range(6)])  # 3 frames of 20 ms each
    assert [b for _, _, b in spans] == pytest.approx([0.06 * (k + 1) for k in range(6)])
    assert align(torch.zeros(18 * 320), "... 12") is None
    assert gate.MmsAligner("cpu", min_score=2.0)(torch.zeros(18 * 320), "çaba şı") == []  # all below the score


SPK = torch.nn.functional.normalize(torch.ones(8), dim=0)


class StubJudges:
    """ASR that hears "merhaba dünya", one speaker, and a MOS that falls with the clip's roughness."""

    def __init__(self):
        self.built = []

    def asr(self, language):
        self.built.append(("asr", language))
        return lambda wavs: ["Merhaba, dünya!"] * len(wavs)

    def sv(self):
        self.built.append(("sv",))
        return lambda w: SPK

    def mos(self):
        self.built.append(("mos",))
        return lambda w: 4.0 - float(np.abs(np.diff(w)).mean() / (np.abs(w).mean() + 1e-9))


def stub_aligner(wav16, text):
    letters = [c for c in text if c.isalpha()]
    d = len(wav16) / 16_000 / len(letters)
    return [(c, k * d, (k + 1) * d) for k, c in enumerate(letters)]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gate")
    _fake_parquet(tmp / "x.parquet", n=6)
    main(["prepare", "--parquet-glob", str(tmp / "*.parquet"), "--out", str(tmp / "data"), "--workers", "1",
          "--val-size", "0", "--no-trim", "--save-audio"])
    return tmp


def gate_args(root, out, *extra):
    return gate.parse_args(["--data", str(root / "data"), "--split", "train", "--num", "5", "--out", str(root / out),
                            "--device", "cpu", "--noise", "0.1", "0.2", "--scale", "0.8", "--probe-steps", "20",
                            "--abx-per-pair", "10", *extra])


def test_gate_runs_end_to_end_with_stubs(root, capsys):
    torch.manual_seed(0)
    judges = StubJudges()
    backends = {"dacvae": tiny_dacvae(), "voxcpm2": tiny_voxcpm(True)}
    args = gate_args(root, "out", "--backends", "dacvae", "voxcpm2", "--max-roundtrip-loss", "10", "--max-noise-loss",
                     "10", "--save-wavs", "1")
    res = gate.run(args, judges=judges, aligner=stub_aligner, backends=backends)
    assert res == json.loads((root / "out" / "results.json").read_text())
    assert res["utterances"] == 5 and res["language"] == "tr" and res["hr_rate"] is None
    assert judges.built == [("asr", "tr"), ("sv",), ("mos",)]  # one at a time, Whisper in the data's language
    assert res["recording"]["wer"] == pytest.approx(1 / 3) and "speaker_sim" not in res["recording"]  # 3 words, 1 wrong
    for name, be in backends.items():
        r = res["backends"][name]
        assert (r["input_rate"], r["output_rate"], r["frame_rate"]) == (be.input_rate, be.output_rate, 100.0)
        rt = r["roundtrip"]["24k"]
        assert set(r["roundtrip"]) == {"24k"}  # below 24 kHz: no native-rate round trip
        assert rt["mel_l1"] > 0 and rt["lsd_8_12"] is None  # the tiny codecs run at 0.8 / 1.2 kHz
        assert rt["wer"] == pytest.approx(1 / 3) and rt["speaker_sim"]["mean"] == pytest.approx(1.0)
        tol = r["tolerance"]
        assert list(tol["conditions"]) == ["noise0.1", "noise0.2", "scale0.8", "smooth"]
        assert all("utmosv2" in c and "wer" in c and "speaker_sim" not in c for c in tol["conditions"].values())
        scores = [c["utmosv2"]["mean"] for c in tol["conditions"].values()]
        assert len({rt["utmosv2"]["mean"], *scores}) == 5  # every perturbation changes the decoded audio
        assert tol["max_noise"] == 0.2 and tol["loss_at_noise"] == pytest.approx(rt["utmosv2"]["mean"] - scores[1])
        assert 0 <= r["abx"]["error"] <= 1 and r["abx"]["cells"] > 0 and r["segments"] == 77  # the texts' letters
        assert 0 <= r["probe"]["accuracy"] <= 1 and r["probe"]["test_frames"] > 0
        assert r["gate"] == "pass" and r["active_channels"] <= be.dim
    md = (root / "out" / "results.md").read_text()
    assert "| dacvae | 0.8 → 0.8 kHz | 100 Hz | 33.33% |" in md and "| pass |" in md
    assert "| backend | clean | noise0.1 |" in md
    assert len(list((root / "out" / "wav").glob("*.wav"))) == 2
    assert "audio_hr.bin" not in capsys.readouterr().out  # no note either: these codecs have no rate above 24 kHz


def test_gate_without_judges_and_parts(root):
    args = gate_args(root, "bare", "--backends", "voxcpm2", "--skip", "tolerance", "phonetic")
    res = gate.run(args, backends={"voxcpm2": tiny_voxcpm(True)})
    r = res["backends"]["voxcpm2"]
    assert res["recording"] == {} and "tolerance" not in r and "abx" not in r and "gate" not in r
    assert set(r["roundtrip"]["24k"]) >= {"mel_l1", "lsd_8_12", "lsd_12_20"}
    assert "| voxcpm2 | 0.6 → 1.2 kHz | 100 Hz | – | – |" in (root / "bare" / "results.md").read_text()


def tiny_dacvae_48k() -> DACVAEBackend:
    torch.manual_seed(0)
    model = DACVAE(encoder_dim=4, encoder_rates=[2, 4], latent_dim=16, decoder_dim=48, decoder_rates=[4, 2],
                   codebook_dim=8, wm_rates=[2, 2], wm_latent=8, sample_rate=48_000)
    return DACVAEBackend("cpu", model)


def test_native_rate_round_trip_needs_hires_audio(root, capsys):
    args = gate_args(root, "native", "--backends", "dacvae", "--skip", "tolerance", "phonetic", "--num", "2")
    res = gate.run(args, judges=StubJudges(), backends={"dacvae": tiny_dacvae_48k()})
    assert set(res["backends"]["dacvae"]["roundtrip"]) == {"24k"}
    assert "no native-rate round trip" in capsys.readouterr().out  # a 48 kHz codec, but only 24 kHz recordings

    # the layout of `prepare --save-audio --hires-rate 48000`: audio_hr.bin, hr_offset / hr_samples, stats hr_rate
    data = root / "data"
    audio = np.memmap(data / "audio.bin", dtype=np.float16, mode="r")
    index, offset = [json.loads(line) for line in open(data / "index.jsonl")], 0
    with open(data / "audio_hr.bin", "wb") as f:
        for e in index:
            w = torch.from_numpy(np.array(audio[e["audio_offset"]: e["audio_offset"] + e["audio_samples"]], np.float32))
            hr = resample(w, 24_000, 48_000).numpy().astype(np.float16)
            f.write(hr.tobytes())
            e["hr_offset"], e["hr_samples"] = offset, len(hr)
            offset += len(hr)
    (data / "index.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in index))
    stats = json.loads((data / "stats.json").read_text())
    (data / "stats.json").write_text(json.dumps({**stats, "hr_rate": 48_000}))

    res = gate.run(args, judges=StubJudges(), backends={"dacvae": tiny_dacvae_48k()})
    rt = res["backends"]["dacvae"]["roundtrip"]
    assert res["hr_rate"] == 48_000 and rt["native"]["source_rate"] == 48_000
    assert rt["native"]["lsd_12_20"] > 0 and rt["native"]["lsd_8_12"] > 0 and rt["24k"]["lsd_12_20"] is None
    assert "utmosv2" in rt["native"] and "speaker_sim" in rt["native"] and "wer" in rt["native"]
    skipped = gate.run(gate_args(root, "native2", "--backends", "dacvae", "--skip", "native", "tolerance", "phonetic",
                                 "--num", "2"), backends={"dacvae": tiny_dacvae_48k()})
    assert set(skipped["backends"]["dacvae"]["roundtrip"]) == {"24k"}


def test_repeated_latents_part(root):
    stored = torch.randn(4, 20).repeat_interleave(4, dim=1)
    generated = stored + 0.1 * torch.randn(4, 80)
    args = gate_args(root, "rep", "--backends", "voxcpm2", "--skip", "tolerance", "phonetic", "--model", "m.pt")
    res = gate.run(args, backends={"voxcpm2": tiny_voxcpm(True)}, generate=lambda a: ([generated], [stored], 4))
    rep = res["repeated_latents"]
    assert rep["repeat"] == 4 and rep["recorded"]["share"] == 0.0 and 0 < rep["generated"]["share"] < 0.05
    assert len(rep["generated"]["positions"]) == 4 and "position 3" in (root / "rep" / "results.md").read_text()
    args = gate_args(root, "rep2", "--backends", "voxcpm2", "--skip", "tolerance", "phonetic", "repeat", "--model",
                     "m.pt")
    assert "repeated_latents" not in gate.run(args, backends={"voxcpm2": tiny_voxcpm(True)},
                                              generate=lambda a: 1 / 0)

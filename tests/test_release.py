"""Release plumbing (v3.2): Hub names, local mirror, pauses by name, prosody by name, publishing helpers. CPU, tiny
models, no downloads."""

import argparse
import random
import warnings

import pytest
import torch

from drifting_tts.audio import HOP_LENGTH, SAMPLE_RATE
from drifting_tts.hub import HUB_DIR_ENV, LATEST, PROSODY_MODELS, RELEASES, hub_file, resolve_prosody
from drifting_tts.models.prosody_net import CHECKPOINT_KEYS, ProsodyPredictor, export_checkpoint, tts_fingerprint
from drifting_tts.prosody import PausePolicy
from drifting_tts.publish import export_tts, export_vocos, pickle_strings, private_terms, scan
from drifting_tts.synthesize import Synthesizer, pause_arg, pipeline_args, resolve_pause
from drifting_tts.vocoder import VOCODER_ALIASES, VOCODERS

MODEL = {"text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
         "gen": {"hidden": 32, "depth": 1, "heads": 2, "patch": 2, "mlp_ratio": 2.0, "n_registers": 2,
                 "noise_classes": 4, "noise_coords": 2, "num_steps": 1},
         "pitch": {"enabled": True}}
NET = {"kind": "drift", "d": 32, "layers": 1, "heads": 2, "ffn": 64, "noise_tok": 4, "noise_glob": 4, "out_init": 1.0}
TEXT = "merhaba dünya. nasılsın?"


def _tts_file(path, seed: int = 0, root: str = "/workspace/data/private_corpus") -> str:
    from drifting_tts.config import Config
    from drifting_tts.models.tts import DriftingTTS

    torch.manual_seed(seed)
    tts = DriftingTTS(Config(MODEL), num_speakers=3)
    torch.save({"ema": tts.state_dict(), "config": {"data": {"root": root}, "model": MODEL,
                                                    "mae": {"path": "/workspace/runs/mae2d/mae_ema.pt"}},
                "num_speakers": 3, "stats": {"mean": -5.0, "std": 2.0, "backend": "bigvgan"},
                "duration_scales": {2: 1.1}, "temperature": 0.3, "opt": {"lr": 1.0}, "taus": [0.1]}, path)
    return str(path)


def _prosody_ck(temperature: float = 0.5) -> dict:
    torch.manual_seed(1)
    pred = ProsodyPredictor(NET, cond_dim=16 + 8 + 2)
    return {"ema": pred.net.state_dict(), "stats": pred.stats.state_dict(), "net_cfg": NET, "cond_dim": 26,
            "duration_scales": {"2": 1.05}, "temperature": temperature, "flow_steps": 8, "step": 10,
            "tts": "/workspace/runs/release/x.pt", "config": {"cache": "/workspace/runs/pm_cache/targets.pt"}}


def test_release_table_and_registry_names():
    assert LATEST == "v3.2" and set(RELEASES) >= {"v3.1", "v3.2"}
    v31, v32 = RELEASES["v3.1"], RELEASES["v3.2"]
    assert (v31["vocoder"], v31["prosody"], v31["pause"]) == ("bigvgan-v2-ft", None, 0.15)  # v3.1 as released
    assert (v32["vocoder"], v32["prosody"], v32["pause"]) == ("vocos-v2", "drift", "punct")
    assert v32["vocoder"] in VOCODERS and v32["prosody"] in PROSODY_MODELS
    e = VOCODERS["vocos-v2"]  # the same network as vocos-ft: same mel front end and streaming context
    assert (e.kind, e.mel, e.context, e.hub_file) == ("vocos", "bigvgan", VOCODERS["vocos-ft"].context, "vocos_v2.pt")
    assert VOCODER_ALIASES["vocos-ft2"] == "vocos-v2" and "vocos-ft2" not in VOCODERS
    files = {e.hub_file for e in VOCODERS.values() if e.hub_file} | {f for f, _ in PROSODY_MODELS.values()}
    assert len(files) == len([e for e in VOCODERS.values() if e.hub_file]) + len(PROSODY_MODELS)  # no clashes


def test_hub_dir_mirror_and_prosody_names(tmp_path, monkeypatch):
    (tmp_path / "prosody_drift_v3.2.pt").write_bytes(b"x")
    monkeypatch.setenv(HUB_DIR_ENV, str(tmp_path))
    assert hub_file("prosody_drift_v3.2.pt") == str(tmp_path / "prosody_drift_v3.2.pt")
    assert resolve_prosody("drift") == str(tmp_path / "prosody_drift_v3.2.pt")
    assert resolve_prosody(tmp_path / "prosody_drift_v3.2.pt") == str(tmp_path / "prosody_drift_v3.2.pt")
    with pytest.raises(ValueError, match="drift"):
        resolve_prosody("no-such-predictor")


def test_pause_by_name():
    assert pause_arg("0.3") == 0.3 and pause_arg("punct") == "punct" and pause_arg("punct:1") == "punct:1"
    with pytest.raises(argparse.ArgumentTypeError):
        pause_arg("long")
    assert resolve_pause(0.2, 722) == 0.2
    p = resolve_pause("punct", 389)
    assert isinstance(p, PausePolicy) and p("a.") == PausePolicy.for_voice(389)("a.") and p.jitter == 0
    assert resolve_pause("punct:0.5", 722).jitter == 0.5
    with pytest.raises(ValueError):
        resolve_pause("pause", 722)


def _synth(tmp_path, **kw) -> Synthesizer:
    synth = Synthesizer(_tts_file(tmp_path / "tts.pt"), "cpu", vocoder="griffin-lim", **kw)
    synth.vocoder.model.n_iter = 2
    return synth


def test_synthesizer_pause_option(tmp_path):
    synth = _synth(tmp_path, pause="punct")
    mels = synth.mels(TEXT, speaker=0)
    speech = sum(m.shape[-1] for m in mels) * HOP_LENGTH
    gap = int(PausePolicy.for_voice(0)("merhaba dünya.") * SAMPLE_RATE)  # speaker 0: the corpus policy
    assert gap > 0
    wav, _ = synth(TEXT, speaker=0)
    assert wav.numel() == speech + gap
    assert synth(TEXT, speaker=0, pause=0.1)[0].numel() == speech + int(0.1 * SAMPLE_RATE)  # per call
    assert torch.equal(torch.cat(list(synth.stream(TEXT, speaker=0))), wav)
    assert _synth(tmp_path)(TEXT, speaker=0)[0].numel() == speech + int(0.15 * SAMPLE_RATE)  # default unchanged
    with pytest.raises(ValueError):
        _synth(tmp_path, pause="long")


def test_synthesizer_prosody_by_path_and_set_prosody(tmp_path):
    torch.save(_prosody_ck(), tmp_path / "prosody.pt")
    synth = _synth(tmp_path, prosody=str(tmp_path / "prosody.pt"))
    assert synth.prosody is not None and synth.prosody_temperature == 0.5  # the checkpoint's preferred temperature
    assert synth._speaker(2)[1] == 1.05  # the sampler's own per-voice factor
    a, _ = synth(TEXT, speaker=2, seed=3)
    assert torch.equal(synth(TEXT, speaker=2, seed=3)[0], a) and not torch.equal(synth(TEXT, speaker=2, seed=4)[0], a)
    assert torch.equal(torch.cat(list(synth.stream(TEXT, speaker=2, seed=3))), a)
    synth.set_prosody(None)
    assert synth.prosody is None and synth._speaker(2)[1] == 1.1  # back to the model's regressors and factors
    synth.set_prosody(str(tmp_path / "prosody.pt"), temperature=0.0, durations="regressor")
    assert synth.prosody_temperature == 0.0 and synth._speaker(2)[1] == 1.1
    with pytest.raises(ValueError):
        synth.set_prosody(None, durations="both")


def test_from_pretrained_and_cli_release_from_a_local_mirror(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hub.mkdir()
    _tts_file(hub / "drifting_tts_v3.1.pt")
    _tts_file(hub / "drifting_tts_v3.2.pt")  # v3.1's acoustic weights, published with a sanitised config
    torch.save(_prosody_ck(), hub / "prosody_drift_v3.2.pt")
    monkeypatch.setenv(HUB_DIR_ENV, str(hub))
    synth = Synthesizer.from_pretrained("v3.2", "cpu", vocoder="griffin-lim")  # vocos-v2 needs the Hub
    assert synth.prosody is not None and synth.pause == "punct" and synth.vocoder.name == "griffin-lim"
    assert synth.prosody_durations == "regressor"  # v3.2 samples the token pitch only
    v31 = Synthesizer.from_pretrained("v3.1", "cpu", vocoder="griffin-lim")
    assert v31.prosody is None and v31.pause == 0.15
    with pytest.raises(ValueError, match="v3.2"):
        Synthesizer.from_pretrained("v9", "cpu")

    from drifting_tts.synthesize import add_args

    p = argparse.ArgumentParser()
    add_args(p)
    a = pipeline_args(p.parse_args(["--release", "v3.2", "--vocoder", "griffin-lim"]))
    assert a == {"model_path": str(hub / "drifting_tts_v3.2.pt"), "vocoder": "griffin-lim", "prosody": "drift",
                 "prosody_durations": "regressor", "pause": "punct"}
    a = pipeline_args(p.parse_args(["--release", "v3.2", "--prosody-durations", "sampled"]))
    assert a["prosody_durations"] == "sampled"
    a = pipeline_args(p.parse_args(["--release", "v3.2", "--prosody", "none", "--pause", "0.2"]))
    assert a["prosody"] is None and a["pause"] == 0.2 and a["vocoder"] == "vocos-v2"
    a = pipeline_args(p.parse_args(["--model", "m.pt"]))  # no release: v3.1 behaviour
    assert a == {"model_path": "m.pt", "vocoder": None, "prosody": None, "prosody_durations": "sampled", "pause": 0.15}
    a = pipeline_args(p.parse_args(["--model", "m.pt", "--pause-policy", "punct", "--pause-jitter", "1"]))
    assert a["pause"] == "punct:1.0"


def test_prosody_export_keeps_what_load_needs_and_fingerprints_the_encoder(tmp_path):
    from drifting_tts.train import load_tts

    tts = load_tts(_tts_file(tmp_path / "tts.pt"))[0]
    out = export_checkpoint(_prosody_ck(), tts)
    assert set(out) <= set(CHECKPOINT_KEYS) and "config" not in out and "tts" not in out and "step" not in out
    assert out["duration_scales"] == {2: 1.05} and out["tts_fingerprint"] == tts_fingerprint(tts)
    torch.save(out, tmp_path / "p.pt")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ProsodyPredictor.load(tmp_path / "p.pt", tts=tts)  # same encoder: no warning
    other = load_tts(_tts_file(tmp_path / "tts2.pt", seed=5))[0]
    with pytest.warns(UserWarning, match="another acoustic model"):
        ProsodyPredictor.load(tmp_path / "p.pt", tts=other)


def test_publish_exports_and_scan(tmp_path):
    torch.save(_prosody_ck(), tmp_path / "raw.pt")
    terms = {"names": ["private_corpus", "abc"]}  # terms under 4 characters are ignored
    hits = scan(tmp_path / "raw.pt", terms)
    assert hits["paths"] >= 2  # 'tts' and 'config.cache' hold absolute paths
    raw = torch.load(_tts_file(tmp_path / "tts.pt"), weights_only=False)
    assert scan(tmp_path / "tts.pt", terms)["names"] >= 1  # data.root names the corpus
    clean = export_tts(raw)
    assert "opt" not in clean and "taus" not in clean and clean["config"]["data"]["root"] == "data/train"
    assert clean["config"]["mae"]["path"] == "runs/mae2d/mae_ema.pt" and clean["duration_scales"] == {2: 1.1}
    torch.save(clean, tmp_path / "clean.pt")
    assert scan(tmp_path / "clean.pt", terms) == {"paths": 0, "names": 0}
    assert "ema" in pickle_strings(tmp_path / "clean.pt")
    with pytest.raises(ValueError, match="Hub repo id"):
        export_vocos({"vocos": {}, "init": "/workspace/runs/vocos.yaml"})
    v = export_vocos({"vocos": {"w": torch.ones(2)}, "init": "charactr/vocos-mel-24khz", "mel": "bigvgan",
                      "head_padding": "same", "step": 5, "opt": {}, "config": {"workdir": "/workspace/x"}})
    assert set(v) == {"vocos", "init", "mel", "head_padding", "step"}


def test_private_terms_from_a_data_root(tmp_path):
    import json

    root = tmp_path / "my_corpus"
    root.mkdir()
    (root / "speakers.json").write_text(json.dumps({"spk_alpha": 0, "spk_beta": 1}))
    (root / "index.jsonl").write_text(json.dumps({"show_name": "Some Show", "spk_id": 0}) + "\n")
    t = private_terms(root)
    assert t["speaker names"] == ["spk_alpha", "spk_beta"] and t["source names"] == ["Some Show"]
    assert "my_corpus" in t["data root"]


def test_release_system_of_the_prosody_eval():
    from drifting_tts.prosody_eval import parse_system

    r = parse_system("release")
    assert r.kind == "split" and r.sampled and r.vocoder == "release" and r.jitter == 0.0
    s = parse_system("predicted@voc=bigvgan-v2-ft")
    assert s.kind == "split" and s.vocoder == "bigvgan-v2-ft" and not s.sampled
    assert parse_system("copy@voc=vocos-ft").vocoder == "vocos-ft"
    for bad in ("predicted@voc=/tmp/x.pt", "predicted@voc=nope", "recording@voc=vocos-ft", "copy@cfg2"):
        with pytest.raises(ValueError):
            parse_system(bad)


def test_pause_policy_draws_are_reproducible():
    p = resolve_pause("punct:1", 389)
    assert p("a.", random.Random(3)) == p("a.", random.Random(3))


def test_variant_shares_the_model_and_per_call_prosody_temperature(tmp_path):
    torch.save(_prosody_ck(), tmp_path / "prosody.pt")
    v32 = _synth(tmp_path, prosody=str(tmp_path / "prosody.pt"), pause="punct")
    v31 = v32.variant(prosody=None, pause=0.15)
    assert v31.model is v32.model and v31.vocoder is v32.vocoder  # nothing loaded twice
    assert v31.prosody is None and v31.pause == 0.15 and v32.prosody is not None and v32.pause == "punct"
    gl = v32.variant(vocoder="griffin-lim")
    assert gl.vocoder is not v32.vocoder and gl.prosody is v32.prosody
    a = v32(TEXT, speaker=2, seed=1)[0]
    assert torch.equal(v32(TEXT, speaker=2, seed=1, prosody_temperature=0.5)[0], a)  # 0.5: the stored one
    cold = [v32.mels(TEXT, speaker=2, seed=s, prosody_temperature=0.0) for s in (1, 2)]
    assert all(m1.shape == m2.shape for m1, m2 in zip(*cold))  # zero prosody noise: the same durations per seed
    assert v32.prosody_temperature == 0.5  # per call only


def test_punct_pauses_use_the_prosody_predictors_edge_silence(tmp_path):
    assert PausePolicy.for_voice(722, edge=0.0)("a.") == pytest.approx(PausePolicy.for_voice(722).gaps["."][0])
    ck = _prosody_ck()
    ck["pause_edges"] = {"0": 0.0}  # speaker 0 (corpus policy): no edge silence -> the whole measured gap
    out = export_checkpoint(ck)
    assert out["pause_edges"] == {0: 0.0}
    torch.save(out, tmp_path / "prosody.pt")
    synth = _synth(tmp_path, prosody=str(tmp_path / "prosody.pt"), pause="punct")
    assert synth.prosody.pause_edges == {0: 0.0}
    gap = PausePolicy.for_voice(0).gaps["."][0]
    assert synth._pause(None, 0)("a.") == pytest.approx(gap)
    assert synth._pause(None, 2)("a.") == PausePolicy.for_voice(2)("a.")  # no measured edge: the table's
    synth.set_prosody(synth.prosody, durations="regressor")  # the regressors' durations: the table's edge again
    assert synth._pause(None, 0)("a.") == PausePolicy.for_voice(0)("a.") < gap


def test_duration_temperature_and_rhythm_table_of_the_prosody_checkpoint(tmp_path):
    ck = _prosody_ck()
    ck["duration_temperature"], ck["rhythm"] = 0.0, {"2": 0}  # voice 2 samples speaker 0's rhythm at T 0
    out = export_checkpoint(ck)
    assert out["duration_temperature"] == 0.0 and out["rhythm"] == {2: 0}
    torch.save(out, tmp_path / "prosody.pt")
    synth = _synth(tmp_path, prosody=str(tmp_path / "prosody.pt"))
    assert synth.prosody_duration_temperature == 0.0 and synth.prosody.rhythm == {2: 0}
    rs, edge = synth._rhythm(2)
    assert rs.tolist() == [0] and edge == pytest.approx(1.1 / 1.05)  # the model's factor over the sampler's
    assert synth._rhythm(1) is None and synth._duration_row()
    a, b = (synth.mels(TEXT, speaker=2, seed=s) for s in (1, 2))
    assert [m.shape for m in a] == [m.shape for m in b]  # durations at T 0: the same for every seed
    hot = synth.variant(prosody=synth.prosody, prosody_temperature=0.5, prosody_duration_temperature=1.0)
    assert hot.prosody_duration_temperature == 1.0 and synth.prosody_duration_temperature == 0.0
    pitch_only = synth.variant(prosody=synth.prosody, prosody_durations="regressor")
    assert pitch_only._rhythm(2) is None and not pitch_only._duration_row()
    args = argparse.Namespace(release=None, model=None, prosody=None, pause=None, vocoder=None)
    assert "prosody_durations" in pipeline_args(args)


def test_second_pitch_predictor_and_sentence_features_run_eagerly(tmp_path, monkeypatch):
    """Phase 2 (#40) on set_prosody: a second predictor gives only the token pitch (the durations stay the first one's
    draws), and what the CUDA graphs cannot run (sentence features, a second pitch predictor) falls back to eager
    synthesis; a plain predictor is still captured."""
    import drifting_tts.fast as fast
    from drifting_tts.sentence_features import DIM
    from drifting_tts.text import normalize, text_to_ids

    torch.save(_prosody_ck(), tmp_path / "prosody.pt")
    path = str(tmp_path / "prosody.pt")
    synth = _synth(tmp_path, prosody=path)
    assert synth.prosody_pitch is None
    torch.manual_seed(2)
    sent = ProsodyPredictor({**NET, "sent_dim": DIM, "ctx_pitch_only": True, "pitch_layers": 1}, cond_dim=26).eval()
    two = synth.variant(prosody=path, prosody_pitch=sent)
    fed, synthesize = [], synth.model.synthesize  # what the acoustic model is given (shared by the variant)

    def spy(*a, **kw):
        fed.append((kw["durations"], kw["pitch"]))
        return synthesize(*a, **kw)

    monkeypatch.setattr(synth.model, "synthesize", spy)
    one = "merhaba dünya, nasılsın?"  # one sentence: the next one's draws would follow the second predictor's
    synth.mels(one, speaker=2, seed=3)
    two.mels(one, speaker=2, seed=3)
    (d1, p1), (d2, p2) = fed
    assert torch.equal(d2, d1) and not torch.equal(p2, p1)  # the first predictor's durations, another pitch
    ids = torch.tensor([text_to_ids(normalize(one), normalized=True)])
    n, spk, g = torch.tensor([ids.shape[1]]), torch.tensor([2]), torch.Generator().manual_seed(3)
    two.prosody.predict(two.model, ids, n, spk, two.prosody_temperature, generator=g)
    assert torch.equal(sent.predict(two.model, ids, n, spk, two.prosody_temperature, generator=g)[1], p2)  # drawn next
    assert two.prosody is not sent and two.prosody_pitch is sent and synth.prosody_pitch is None
    assert len(synth.variant(prosody=sent).mels(TEXT, speaker=2)) == 2  # sentence features, eagerly
    with pytest.raises(ValueError, match="pitch predictor"):
        synth.variant(prosody=None, prosody_pitch=sent)

    built = []

    class Graphs:  # stands in for the CUDA graphs on the CPU: records what would be captured
        def __init__(self, model, prosody=None, **kw):
            built.append(prosody)

        def warmup(self):
            pass

    monkeypatch.setattr(fast, "GraphedAcoustic", Graphs)
    synth.fast = True
    synth.set_prosody(path)
    assert isinstance(synth.acoustic, Graphs) and len(built) == 1
    assert fast.graphable(synth.prosody) and not fast.graphable(sent)
    for kw in ({"prosody": sent}, {"prosody": path, "pitch": sent}):
        with pytest.warns(UserWarning, match="eagerly"):
            synth.set_prosody(**kw)
        assert synth.acoustic is None
    assert len(built) == 1

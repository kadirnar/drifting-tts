"""Batched synthesis (drifting_tts.batched): each row of a padded batch equals the single-request path with the same
seed. CPU, tiny models, no downloads."""

import pytest
import torch

from drifting_tts.audio import HOP_LENGTH
from drifting_tts.batched import acoustic_batch, stream_batched, stream_windows, vocode_masked
from drifting_tts.config import Config
from drifting_tts.fast import stream_vocoder
from drifting_tts.models.prosody_net import ProsodyPredictor
from drifting_tts.models.tts import DriftingTTS
from drifting_tts.synthesize import Synthesizer
from drifting_tts.text import text_to_ids
from drifting_tts.vocoder import Vocoder

MODEL = {"text": {"d": 16, "heads": 2, "layers": 2, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
         "gen": {"hidden": 32, "depth": 2, "heads": 2, "patch": 2, "mlp_ratio": 4.0, "n_registers": 4,
                 "noise_classes": 8, "noise_coords": 3, "residual_prior": True, "num_steps": 1},
         "pitch": {"enabled": True}}
NET = {"kind": "drift", "d": 32, "layers": 1, "heads": 2, "ffn": 64, "noise_tok": 4, "noise_glob": 4, "out_init": 1.0}
TEXTS = ["merhaba.", "bu bir deneme cümlesidir, tek adımda üretilir.", "kısa bir cümle daha?",
         "yapay zekâ modelleri her geçen gün daha hızlı hâle geliyor."]


def _jitter(*modules, scale: float = 0.1) -> None:
    """Zero-initialised output layers would hide most of the network."""
    torch.manual_seed(0)
    with torch.no_grad():
        for m in modules:
            for p in m.parameters():
                p.add_(torch.randn_like(p) * scale)


def _models():
    torch.manual_seed(0)
    model = DriftingTTS(Config(MODEL), num_speakers=3).eval()
    pred = ProsodyPredictor(NET, cond_dim=16 + 8 + 2).eval()
    _jitter(model, pred)
    return model, pred


@pytest.mark.parametrize("prosody", [None, "regressor", "sampled"])
def test_acoustic_batch_rows_match_single_requests(prosody):
    model, pred = _models()
    pred = None if prosody is None else pred
    ids = [text_to_ids(t) for t in TEXTS]
    spk, scale, pt = 2, 1.3, 0.7
    mel, lens = acoustic_batch(model, ids, spk, 1.5, 0.5, scale, [torch.Generator().manual_seed(s) for s in range(4)],
                               prosody=pred, prosody_temperature=pt, prosody_durations=prosody or "sampled")
    assert mel.shape[0] == 4 and len(set(lens.tolist())) > 1  # different lengths: the padding matters
    for b, x in enumerate(ids):
        g = torch.Generator().manual_seed(b)
        text, text_len, s = torch.tensor([x]), torch.tensor([len(x)]), torch.tensor([spk])
        durations = pitch = None
        if pred is not None:  # Synthesizer._mel: the prosody predictor first, from the same generator
            durations, pitch = pred.predict(model, text, text_len, s, pt, scale, generator=g)
            durations = None if prosody == "regressor" else durations
        ref, ref_len = model.synthesize(text, text_len, s, cfg_scale=1.5, temperature=0.5, length_scale=scale,
                                        generator=g, durations=durations, pitch=pitch)
        t = int(lens[b])
        assert t == int(ref_len[0])
        torch.testing.assert_close(mel[b: b + 1, :, :t], ref, rtol=0, atol=1e-5)
        assert not mel[b, :, t:].any()  # zero after the row's length


def _tiny_vocos(seed: int = 0) -> Vocoder:
    from vocos.feature_extractors import MelSpectrogramFeatures
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone
    from vocos.pretrained import Vocos

    torch.manual_seed(seed)
    model = Vocos(MelSpectrogramFeatures(), VocosBackbone(100, 16, 32, 2), ISTFTHead(16, 1024, HOP_LENGTH)).eval()
    return Vocoder.wrap(model, "vocos", "cpu", "bigvgan", "tiny", context=8)


def test_vocode_masked_rows_match_single_rows():
    voc = _tiny_vocos()
    torch.manual_seed(1)
    mel = torch.randn(3, 100, 21) * 0.5 - 5
    lengths = torch.tensor([21, 7, 13])
    wav = vocode_masked(voc, mel, lengths)
    assert wav.shape == (3, 21 * HOP_LENGTH)
    for b, t in enumerate(lengths.tolist()):
        torch.testing.assert_close(wav[b, : t * HOP_LENGTH], voc(mel[b: b + 1, :, :t])[0], rtol=0, atol=1e-5)
        assert not wav[b, t * HOP_LENGTH:].any()


@pytest.mark.parametrize("t", [5, 12, 13, 40, 41, 97])
def test_stream_windows_are_those_of_stream_vocoder(t):
    def voc(x):  # each sample holds the index of its frame in the window
        return torch.arange(x.shape[-1]).repeat_interleave(4)[None]

    mel = torch.zeros(1, 100, t)
    pieces = list(stream_vocoder(voc, mel, hop=4, first=4, chunk=16, context=8))
    windows = stream_windows(t, first=4, chunk=16, context=8)
    assert len(windows) == len(pieces)
    for (_a, _b, start, n), piece in zip(windows, pieces):
        assert piece.numel() == n * 4 and int(piece[0]) == start  # the window's frame index of the kept part


def _synth(tmp_path, prosody: bool) -> Synthesizer:
    model, pred = _models()
    torch.save({"ema": model.state_dict(), "config": {"data": {"root": str(tmp_path)}, "model": MODEL},
                "num_speakers": 3, "stats": {"mean": -5.0, "std": 2.0, "backend": "bigvgan"}}, tmp_path / "tts.pt")
    kw = {}
    if prosody:
        torch.save({"ema": pred.net.state_dict(), "stats": pred.stats.state_dict(), "net_cfg": NET, "cond_dim": 26,
                    "temperature": 0.5}, tmp_path / "prosody.pt")
        kw = {"prosody": str(tmp_path / "prosody.pt"), "prosody_durations": "regressor"}
    synth = Synthesizer(tmp_path / "tts.pt", "cpu", vocoder="griffin-lim", **kw)
    synth.vocoder = _tiny_vocos()
    return synth


@pytest.mark.parametrize("prosody", [False, True])
def test_stream_batched_matches_stream(tmp_path, prosody):
    synth = _synth(tmp_path, prosody)
    texts = ["Merhaba! Bugün hava çok güzel.", "Bu bir deneme.", "Kısa bir cümle daha? Evet, bir tane daha.",
             "Yapay zekâ modelleri her geçen gün daha hızlı ve daha verimli hâle geliyor."]
    kw = dict(speaker=2, cfg_scale=1.5, temperature=0.5, pause=0.05, first=4, chunk=16)
    got = {i: [] for i in range(len(texts))}
    rounds = list(stream_batched(synth, texts, seeds=[3, 4, 5, 6], **kw))
    assert {i for i, _ in rounds[0]} == set(range(len(texts)))  # every request has a piece after the first round
    assert len(rounds) > 3  # the mels stream in several windows
    for out in rounds:
        for i, piece in out:
            got[i].append(piece)
    for i, text in enumerate(texts):
        ref = list(synth.stream(text, seed=3 + i, **kw))
        assert [p.numel() for p in got[i]] == [p.numel() for p in ref]
        torch.testing.assert_close(torch.cat(got[i]), torch.cat(ref), rtol=0, atol=1e-5)

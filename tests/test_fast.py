import pytest
import torch
from torch import nn

from drifting_tts.config import Config
from drifting_tts.fast import GraphedAcoustic, stream_vocoder
from drifting_tts.models.tts import DriftingTTS
from drifting_tts.text import text_to_ids


class ToyVocoder(nn.Module):
    """Convolutions with a receptive field of +-6 frames, then ``hop`` samples per frame."""

    def __init__(self, hop: int = 4):
        super().__init__()
        self.hop = hop
        self.net = nn.Sequential(nn.Conv1d(100, 8, 5, padding=2), nn.Tanh(), nn.Conv1d(8, 8, 5, padding=2), nn.Tanh(),
                                 nn.Conv1d(8, 1, 5, padding=2))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net(mel)[:, 0].repeat_interleave(self.hop, dim=-1)


@pytest.mark.parametrize("frames", [10, 12, 13, 40, 41, 97])
def test_stream_vocoder_matches_whole_utterance(frames):
    torch.manual_seed(0)
    voc = ToyVocoder().eval()
    mel = torch.randn(1, 100, frames)
    with torch.no_grad():
        full = voc(mel)[0]
        pieces = list(stream_vocoder(voc, mel, hop=4, first=4, chunk=16, context=8))
    assert sum(p.numel() for p in pieces) == full.numel() == frames * 4
    torch.testing.assert_close(torch.cat(pieces), full, rtol=0, atol=1e-6)
    if frames > 12:  # the first piece is the first ``first`` frames only
        assert pieces[0].numel() == 4 * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("text", ["merhaba.", "bu bir deneme cümlesidir, tek adımda ve hızlı üretilir."])
def test_graphed_acoustic_matches_eager(text):
    cfg = {"text": {"d": 16, "heads": 2, "layers": 2, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
           "gen": {"hidden": 32, "depth": 2, "heads": 2, "patch": 2, "mlp_ratio": 4.0, "n_registers": 4,
                   "noise_classes": 8, "noise_coords": 3, "residual_prior": True, "num_steps": 1},
           "pitch": {"enabled": True}}
    torch.manual_seed(0)
    model = DriftingTTS(Config(cfg), num_speakers=3).cuda().eval()
    with torch.no_grad():
        for p in model.parameters():  # zero-initialised output layers would hide most of the network
            p.add_(torch.randn_like(p) * 0.1)
    ids = torch.tensor([text_to_ids(text)], device="cuda")
    spk = torch.tensor([2], device="cuda")
    fast = GraphedAcoustic(model, token_bucket=16, frame_bucket=32)
    with torch.no_grad():
        for seed in range(2):
            g = torch.Generator(device="cuda").manual_seed(seed)
            ref, _ = model.synthesize(ids, torch.tensor([ids.shape[1]], device="cuda"), spk, cfg_scale=1.5,
                                      temperature=0.5, length_scale=1.3, generator=g)
            g = torch.Generator(device="cuda").manual_seed(seed)
            mel = fast(ids, spk, 1.5, 0.5, 1.3, generator=g)
            assert mel.shape == ref.shape
            torch.testing.assert_close(mel, ref, rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("durations", ["sampled", "regressor"])
def test_graphed_acoustic_with_a_prosody_predictor_matches_eager(durations):
    """The drift prosody predictor inside the encoder's graph: the same draws and the same mel as
    ``ProsodyPredictor.predict`` followed by ``synthesize``."""
    from drifting_tts.models.prosody_net import ProsodyPredictor

    cfg = {"text": {"d": 16, "heads": 2, "layers": 2, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
           "gen": {"hidden": 32, "depth": 2, "heads": 2, "patch": 2, "mlp_ratio": 4.0, "n_registers": 4,
                   "noise_classes": 8, "noise_coords": 3, "residual_prior": True, "num_steps": 1},
           "pitch": {"enabled": True}}
    torch.manual_seed(0)
    model = DriftingTTS(Config(cfg), num_speakers=3).cuda().eval()
    pred = ProsodyPredictor({"kind": "drift", "d": 32, "layers": 1, "heads": 2, "ffn": 64, "noise_tok": 4,
                             "noise_glob": 4, "out_init": 1.0}, cond_dim=16 + 8 + 2).cuda().eval()
    with torch.no_grad():
        for p in [*model.parameters(), *pred.parameters()]:
            p.add_(torch.randn_like(p) * 0.1)
    ids = torch.tensor([text_to_ids("bu bir deneme cümlesidir, tek adımda ve hızlı üretilir.")], device="cuda")
    n, spk = torch.tensor([ids.shape[1]], device="cuda"), torch.tensor([2], device="cuda")
    fast = GraphedAcoustic(model, token_bucket=16, frame_bucket=32, prosody=pred, prosody_durations=durations)
    with torch.no_grad():
        for seed in range(2):
            g = torch.Generator(device="cuda").manual_seed(seed)
            frames, pitch = pred.predict(model, ids, n, spk, 0.7, 1.3, generator=g)
            ref, _ = model.synthesize(ids, n, spk, cfg_scale=1.5, temperature=0.5, length_scale=1.3, generator=g,
                                      durations=frames if durations == "sampled" else None, pitch=pitch)
            g = torch.Generator(device="cuda").manual_seed(seed)
            mel = fast(ids, spk, 1.5, 0.5, 1.3, generator=g, prosody_temperature=0.7)
            assert mel.shape == ref.shape
            torch.testing.assert_close(mel, ref, rtol=0, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("duration_temperature,rhythm", [(0.3, None), (0.7, 0), (None, 1), (0.0, None)])
def test_graphed_acoustic_with_a_duration_row_matches_eager(duration_temperature, rhythm):
    """Durations at their own temperature or another speaker's rhythm (two rows) inside the encoder's graph."""
    from drifting_tts.models.prosody_net import ProsodyPredictor

    cfg = {"text": {"d": 16, "heads": 2, "layers": 2, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
           "gen": {"hidden": 32, "depth": 2, "heads": 2, "patch": 2, "mlp_ratio": 4.0, "n_registers": 4,
                   "noise_classes": 8, "noise_coords": 3, "residual_prior": True, "num_steps": 1},
           "pitch": {"enabled": True}}
    torch.manual_seed(0)
    model = DriftingTTS(Config(cfg), num_speakers=3).cuda().eval()
    pred = ProsodyPredictor({"kind": "drift", "d": 32, "layers": 1, "heads": 2, "ffn": 64, "noise_tok": 4,
                             "noise_glob": 4, "out_init": 1.0}, cond_dim=16 + 8 + 2).cuda().eval()
    with torch.no_grad():
        for p in [*model.parameters(), *pred.parameters()]:
            p.add_(torch.randn_like(p) * 0.1)
    ids = torch.tensor([text_to_ids("bu bir deneme cümlesidir, tek adımda ve hızlı üretilir.")], device="cuda")
    n, spk = torch.tensor([ids.shape[1]], device="cuda"), torch.tensor([2], device="cuda")
    rs = None if rhythm is None else torch.tensor([rhythm], device="cuda")
    dt = 0.7 if duration_temperature is None else duration_temperature  # a voice of its own: the row runs anyway
    fast = GraphedAcoustic(model, token_bucket=16, frame_bucket=32, prosody=pred, duration_row=True)
    with torch.no_grad():
        for seed in range(2):
            g = torch.Generator(device="cuda").manual_seed(seed)
            frames, pitch = pred.predict(model, ids, n, spk, 0.7, 1.3, generator=g, duration_temperature=dt,
                                         duration_speaker=rs)
            ref, _ = model.synthesize(ids, n, spk, cfg_scale=1.5, temperature=0.5, length_scale=1.3, generator=g,
                                      durations=frames, pitch=pitch)
            g = torch.Generator(device="cuda").manual_seed(seed)
            mel = fast(ids, spk, 1.5, 0.5, 1.3, generator=g, prosody_temperature=0.7, duration_temperature=dt,
                       rhythm=rs)
            assert mel.shape == ref.shape
            torch.testing.assert_close(mel, ref, rtol=0, atol=1e-4)

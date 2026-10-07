import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from drifting_tts.config import Config  # noqa: E402
from drifting_tts.mlx.convert import convert_acoustic  # noqa: E402
from drifting_tts.mlx.model import DriftingTTS as MlxTTS  # noqa: E402
from drifting_tts.mlx.model import durations  # noqa: E402
from drifting_tts.models.text_encoder import durations_to_alignment  # noqa: E402
from drifting_tts.models.tts import DriftingTTS  # noqa: E402
from drifting_tts.text import SYMBOLS, text_to_ids  # noqa: E402

CFG = {"text": {"d": 16, "heads": 2, "layers": 2, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
       "gen": {"hidden": 32, "depth": 2, "heads": 2, "patch": 2, "mlp_ratio": 4.0, "n_registers": 4,
               "noise_classes": 8, "noise_coords": 3, "residual_prior": True, "num_steps": 1},
       "pitch": {"enabled": True}}


@pytest.fixture(scope="module")
def models():
    mx.set_default_device(mx.cpu)
    torch.manual_seed(0)
    tm = DriftingTTS(Config(CFG), num_speakers=3).eval()
    with torch.no_grad():  # zero-initialised adaLN / output layers would hide most of the network
        for p in tm.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    mm = MlxTTS(CFG, num_speakers=3, n_vocab=len(SYMBOLS))
    mm.load_weights([(k, mx.array(v)) for k, v in convert_acoustic(tm.state_dict()).items()], strict=True)
    return tm, mm


@pytest.mark.parametrize("text", ["merhaba.", "bu bir deneme cümlesidir, tek adımda."])
def test_mlx_acoustic_model_matches_pytorch(models, text):
    tm, mm = models
    ids = text_to_ids(text)
    spk = torch.tensor([2])
    with torch.no_grad():
        h, mu, logw, xm = tm.encoder(torch.tensor([ids]), torch.tensor([len(ids)]), spk)
        h, _ = tm.pitch_condition(h, xm, spk)
        attn, y_len = durations_to_alignment(logw, xm, 1.3)
        cond = tm.frame_condition(h, mu, attn)
        z = torch.randn(1, 100, cond.shape[-1])
        labels = torch.randint(0, 8, (1, 3))
        ref = tm.generate(z, cond, spk, torch.tensor([1.5]), noise_labels=labels)[0].T.numpy()

    tokens, mlx_logw = mm.encode(mx.array([ids]), mx.array([2]))
    np.testing.assert_allclose(np.array(tokens[0]), torch.cat([mu, h], 1)[0].T.numpy(), atol=1e-4, rtol=1e-4)
    w = durations(np.array(mlx_logw[0]), 1.3)
    assert w.sum() == int(y_len)
    frames = tokens[:, mx.array(np.repeat(np.arange(len(ids)), w))]
    mel = mm.generate(mx.array(z.numpy().transpose(0, 2, 1)), frames, mx.array([2]), mx.array([1.5]),
                      mx.array(labels.numpy()))
    assert mel.shape == (1, int(y_len), 100)  # odd and even frame counts: patch padding is trimmed
    np.testing.assert_allclose(np.array(mel[0]), ref, atol=1e-4 * np.abs(ref).max(), rtol=0)

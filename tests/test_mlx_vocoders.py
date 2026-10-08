"""The MLX Vocos port, the vocoder files and their registry, against PyTorch (tiny random models), plus a parity check
of the released small vocoders on real mels when their checkpoints are cached."""

import json

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
pytest.importorskip("vocos")

from mlx.utils import tree_flatten  # noqa: E402

from drifting_tts.mlx.convert import convert_vocoder, parse_vocoder  # noqa: E402
from drifting_tts.mlx.synthesize import chunk_windows, decode_window  # noqa: E402
from drifting_tts.mlx.vocoder import METADATA_KEY, VOCODERS, load_vocoder, save_vocoder, vocoder_path  # noqa: E402
from drifting_tts.mlx.vocos import Vocos, overlap_add, vocos_context_frames  # noqa: E402
from drifting_tts.vocoder import Vocoder  # noqa: E402
from tests.test_vocoder_registry import _tiny_vocos  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _torch_vocos(tmp_path, layers: int = 2, std: float | None = None) -> Vocoder:
    model, _ = _tiny_vocos(tmp_path, layers=layers, padding="same")
    if std is not None:  # Vocos's init (std 0.02, layer scale 1 / layers) makes far frames' influence vanish
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() > 1:
                    p.normal_(0, std)
                elif name.endswith("gamma"):
                    p.fill_(1.0)
    return Vocoder.wrap(model.eval(), "vocos", "cpu", "bigvgan", "tiny", 32)


def _mlx(voc: Vocoder, path) -> Vocos:
    kind, hparams, weights = convert_vocoder(voc)
    save_vocoder(path, weights, kind, hparams, fp16=False)
    return load_vocoder(path)


@pytest.mark.parametrize("frames", [1, 2, 9, 40])
def test_mlx_vocos_matches_torch(tmp_path, frames):
    voc = _torch_vocos(tmp_path)
    model = _mlx(voc, tmp_path / "vocos.safetensors")
    assert isinstance(model, Vocos) and model.kind == "vocos" and model.hop_length == 256
    mel = torch.randn(2, 100, frames, generator=torch.Generator().manual_seed(frames)) * 2 - 5
    with torch.no_grad():
        ref = voc._vocos_bigvgan(mel).numpy()  # unclamped
    out = np.array(model(mx.array(mel.numpy())))
    assert out.shape == ref.shape == (2, frames * 256)
    assert np.abs(out - ref).max() <= 1e-5 * np.abs(ref).max()


def test_overlap_add_matches_fold():
    frames = mx.random.normal((2, 7, 16), key=mx.random.key(0))
    ref = torch.nn.functional.fold(torch.tensor(np.array(frames)).transpose(1, 2), (1, 6 * 4 + 16), (1, 16),
                                   stride=(1, 4))[:, 0, 0]
    np.testing.assert_allclose(np.array(overlap_add(frames, 4)), ref.numpy(), atol=1e-6)


def test_vocos_streaming_context_is_exact(tmp_path):
    """The derived context (29 frames for 8 layers) makes every chunk equal the whole-utterance output."""
    voc = _torch_vocos(tmp_path, layers=8, std=1.0)
    model = _mlx(voc, tmp_path / "vocos.safetensors")
    assert model.context_frames == 29 == vocos_context_frames({"num_layers": 8, "n_fft": 1024, "hop_length": 256})
    mel = mx.random.normal((1, 150, 100), key=mx.random.key(1)) * 2 - 5  # [1, T, n_mels], as the synthesizer holds it
    full = np.array(mx.clip(model(mel.transpose(0, 2, 1))[0], -1, 1))

    def streamed(context):
        return np.concatenate([np.array(decode_window(model, mel, a, b, context, 256))
                               for a, b in chunk_windows(150, 24, 16)])

    assert np.abs(streamed(29) - full).max() < 1e-6  # bit-exact on the CPU
    assert np.abs(streamed(26) - full).max() > 1e-5  # less context is not exact


def test_vocoder_files_describe_themselves(tmp_path):
    voc = _torch_vocos(tmp_path)
    kind, hparams, weights = convert_vocoder(voc)
    assert kind == "vocos" and hparams["padding"] == "same" and not any("window" in k for k in weights)
    save_vocoder(tmp_path / "v16.safetensors", weights, kind, hparams, fp16=True)
    stored, meta = mx.load(str(tmp_path / "v16.safetensors"), return_metadata=True)
    assert json.loads(meta[METADATA_KEY]) == {"kind": "vocos", "hparams": hparams}
    assert {v.dtype for v in stored.values()} == {mx.float16}
    model = load_vocoder(tmp_path / "v16.safetensors")
    assert {v.dtype for _, v in tree_flatten(model.parameters())} == {mx.float32}
    with pytest.raises(NotImplementedError):
        Vocos({**hparams, "padding": "center"})
    mx.save_safetensors(str(tmp_path / "bare.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    with pytest.raises(ValueError, match="does not describe"):
        load_vocoder(tmp_path / "bare.safetensors")


def test_vocoder_names_and_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="--vocoder vocos-ft"):
        vocoder_path("vocos-ft", tmp_path)
    for file in VOCODERS.values():
        (tmp_path / file).touch()
    assert vocoder_path(None, tmp_path) == tmp_path / "vocoder.safetensors"
    assert vocoder_path("vocos-ft", tmp_path) == tmp_path / VOCODERS["vocos-ft"]
    (tmp_path / "mine.safetensors").touch()
    assert vocoder_path(str(tmp_path / "mine.safetensors"), "unused") == tmp_path / "mine.safetensors"
    with pytest.raises(ValueError, match="unknown MLX vocoder"):
        vocoder_path("griffin-lim", tmp_path)
    assert parse_vocoder("vocos-ft") == ("vocos-ft", "vocos-ft")
    assert parse_vocoder("bigvgan-base-ft=/x/ft.pt") == ("bigvgan-base-ft", "/x/ft.pt")
    assert parse_vocoder(str(tmp_path / "mine.safetensors")) == ("bigvgan-v2-ft", str(tmp_path / "mine.safetensors"))
    for bad in ("bigvgan-v2", "griffin-lim=/x.pt"):
        with pytest.raises(ValueError):
            parse_vocoder(bad)


def _hub_cached(*files: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache("Vyvo/drifting-tts-tr", f), str) for f in files)


@pytest.mark.parametrize("name,file", [("vocos-ft", "vocos_ft.pt"), ("bigvgan-base-ft", "bigvgan_base_ft.pt")])
def test_released_vocoders_match_pytorch_on_real_mels(name, file):
    """A real mel of the released acoustic model, cropped to 48 frames (the MLX CPU backend is slow)."""
    if not _hub_cached("drifting_tts_v3.1.pt", file):
        pytest.skip("released checkpoints not cached")
    if name.startswith("bigvgan"):
        pytest.importorskip("librosa")
    from drifting_tts.vocoder import load_vocoder as load_torch_vocoder
    from scripts.check_mlx_parity import compare, mlx_vocoder, real_mels

    mel = real_mels(None, ["Merhaba, nasılsınız?"])[0][..., :48]
    torch_vocoder = load_torch_vocoder(name, "cpu")
    r = compare(torch_vocoder, mlx_vocoder(name, torch_vocoder, None), mel, chunk_frames=16, first_chunk_frames=8)
    assert r["mlx_vs_torch_db"] > 80 and r["mlx_streamed_vs_torch_db"] > 80 and r["mlx_streamed_vs_mlx_db"] > 100

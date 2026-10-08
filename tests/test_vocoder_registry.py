import os

import pytest
import torch
from huggingface_hub.constants import HF_HUB_CACHE

from drifting_tts.audio import HOP_LENGTH, BigVGANLogMel, LogMel
from drifting_tts.fast import stream_vocoder
from drifting_tts.vocoder import (
    BASE_REPO,
    BIGVGAN_REPO,
    VOCODERS,
    GriffinLim,
    Vocoder,
    checkpoint_kind,
    load_vocoder,
    ola_istft,
)


def _code_cached(repo: str) -> bool:
    snaps = os.path.join(HF_HUB_CACHE, f"models--{repo.replace('/', '--')}", "snapshots")
    return os.path.isdir(snaps) and any(os.path.exists(os.path.join(snaps, s, "bigvgan.py")) for s in os.listdir(snaps))


def _chirp(seconds: float = 0.5) -> torch.Tensor:
    t = torch.arange(int(24_000 * seconds)) / 24_000
    return 0.3 * torch.sin(2 * torch.pi * (150 + 400 * t) * t)[None]


def _tiny_vocos(tmp_path, layers: int = 2, padding: str = "center"):
    """A Vocos with the real 100-band / n_fft 1024 / hop 256 interface but a tiny backbone (random init)."""
    from vocos import Vocos

    cfg = tmp_path / "tiny_vocos.yaml"
    cfg.write_text(
        "feature_extractor:\n  class_path: vocos.feature_extractors.MelSpectrogramFeatures\n"
        "  init_args: {sample_rate: 24000, n_fft: 1024, hop_length: 256, n_mels: 100, padding: center}\n"
        "backbone:\n  class_path: vocos.models.VocosBackbone\n"
        f"  init_args: {{input_channels: 100, dim: 16, intermediate_dim: 32, num_layers: {layers}}}\n"
        "head:\n  class_path: vocos.heads.ISTFTHead\n"
        f"  init_args: {{dim: 16, n_fft: 1024, hop_length: 256, padding: {padding}}}\n")
    torch.manual_seed(0)
    return Vocos.from_hparams(str(cfg)), str(cfg)


def test_registry_entries():
    assert {"bigvgan-v2-ft", "bigvgan-v2", "bigvgan-v1", "bigvgan-base", "bigvgan-base-ft", "vocos-ft",
            "griffin-lim"} <= set(VOCODERS)
    for name, e in VOCODERS.items():
        assert e.kind in ("bigvgan", "vocos", "griffin-lim"), name
        assert e.context is None or e.context > 0
        if e.kind == "bigvgan":
            assert e.repo.startswith("nvidia/") and e.mel == "bigvgan"
    assert VOCODERS["griffin-lim"].context is None and VOCODERS["griffin-lim"].mel is None
    assert VOCODERS["bigvgan-base-ft"].repo == BASE_REPO and VOCODERS["vocos-ft"].mel == "bigvgan"


def test_unknown_vocoder_and_checkpoint_kinds(tmp_path):
    with pytest.raises(ValueError, match="griffin-lim"):
        load_vocoder("no-such-vocoder", "cpu")
    assert checkpoint_kind({"generator": {}, "hparams": {}}) == "bigvgan"
    assert checkpoint_kind({"vocos": {}, "init": "x", "mel": "bigvgan"}) == "vocos"
    with pytest.raises(ValueError, match="not a vocoder checkpoint"):
        checkpoint_kind({"ema": {}})
    torch.save({"ema": {}}, tmp_path / "tts.pt")
    with pytest.raises(ValueError):
        load_vocoder(str(tmp_path / "tts.pt"), "cpu")


@pytest.mark.parametrize("centred", [False, True])
def test_ola_istft_inverts_the_mel_stft(centred):
    gl = GriffinLim("vocos" if centred else "bigvgan", n_iter=0)
    y = _chirp()[..., : 40 * HOP_LENGTH]
    spec = gl.stft(y)
    assert spec.shape[-1] == (41 if centred else 40)  # T frames <-> (T - 1) * hop (centred) or T * hop samples
    out = ola_istft(spec, gl.window, centred)
    assert out.shape == y.shape
    torch.testing.assert_close(out, y, rtol=0, atol=1e-5)
    if centred:
        ref = torch.istft(spec, 1024, 256, 1024, gl.window, center=True, length=y.shape[-1])
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)


@pytest.mark.parametrize("backend", ["bigvgan", "vocos"])
def test_griffin_lim(backend):
    voc = load_vocoder("griffin-lim", "cpu", backend=backend)
    assert voc.kind == "griffin-lim" and voc.mel == backend and voc.context is None and not voc.graphs
    assert voc.num_params == 0
    front = BigVGANLogMel() if backend == "bigvgan" else LogMel()
    mel = front(_chirp())
    voc.model.n_iter = 16
    wav = voc(mel)
    t = mel.shape[-1]
    assert wav.shape == (1, (t if backend == "bigvgan" else t - 1) * HOP_LENGTH)
    assert wav.abs().max() <= 1 and torch.isfinite(wav).all()
    inner = slice(4, t - 4)
    assert (front(wav)[..., inner] - mel[..., inner]).abs().mean() < 0.5  # the mel survives the round trip
    pieces = list(stream_vocoder(voc, mel, context=voc.context))  # cannot stream: one piece per sentence
    assert len(pieces) == 1 and torch.equal(pieces[0], wav[0])


@pytest.mark.parametrize("padding", ["same", "center"])
def test_vocos_checkpoint_on_bigvgan_mels(tmp_path, padding):
    pytest.importorskip("vocos")
    model, cfg = _tiny_vocos(tmp_path)
    ck = {"vocos": model.state_dict(), "init": cfg, "mel": "bigvgan", "step": 1}
    if padding == "same":
        ck["head_padding"] = "same"
    torch.save(ck, tmp_path / "vocos_ft.pt")
    voc = load_vocoder(str(tmp_path / "vocos_ft.pt"), "cpu")
    assert voc.kind == "vocos" and voc.mel == "bigvgan" and voc.context == VOCODERS["vocos-ft"].context
    assert voc.model.head.istft.padding == padding and voc.num_params == sum(p.numel() for p in model.parameters())
    mel = torch.randn(1, 100, 30) - 5
    wav = voc(mel)
    assert wav.shape == (1, 30 * HOP_LENGTH) and torch.isfinite(wav).all()  # BigVGAN framing: T * hop samples
    if padding == "same":  # the CUDA-graph-safe head equals Vocos's own
        ref = voc.model.head(voc.model.backbone(mel)).clamp(-1, 1)
        torch.testing.assert_close(wav, ref, rtol=0, atol=1e-5)


def test_legacy_vocos_checkpoint(tmp_path):
    """A ``vocos_ft.pt`` without ``mel`` is a Vocos on its own mels: ``Vocoder(finetuned=...)`` as before."""
    pytest.importorskip("vocos")
    model, cfg = _tiny_vocos(tmp_path)
    torch.save({"vocos": model.state_dict(), "init": cfg, "step": 1}, tmp_path / "vocos_ft.pt")
    voc = Vocoder("cpu", finetuned=str(tmp_path / "vocos_ft.pt"), backend="vocos")
    assert voc.kind == "vocos" and voc.mel == "vocos"
    mel = torch.randn(1, 100, 20) - 5
    torch.testing.assert_close(voc(mel), model.decode(mel).clamp(-1, 1))
    assert voc(mel).shape == (1, 19 * HOP_LENGTH)


@pytest.mark.parametrize("padding", ["same", "center"])
def test_vocos_streaming_context_is_exact(tmp_path, padding):
    """8 ConvNeXt layers with 7-tap depthwise convolutions, as the real backbone: a receptive field of 3 + 8 * 3
    frames, plus 2 for the ISTFT overlap. With the registry's context, streamed audio equals whole-utterance audio."""
    pytest.importorskip("vocos")
    model, _ = _tiny_vocos(tmp_path, layers=8, padding=padding)
    for p in model.parameters():
        if p.dim() > 1:  # Vocos's std 0.02 init makes far frames' influence vanish below float64 precision
            p.data.normal_(0, 0.3)
    voc = Vocoder.wrap(model.double(), "vocos", "cpu", "bigvgan", "tiny", VOCODERS["vocos-ft"].context)
    mel = torch.randn(1, 100, 150, dtype=torch.float64) - 5
    vocode = voc._vocos_bigvgan  # float64 (``__call__`` runs in float32)
    with torch.no_grad():
        full = vocode(mel)[0]
        pieces = {c: torch.cat(list(stream_vocoder(vocode, mel, first=16, chunk=24, context=c)))
                  for c in (28, 29, voc.context)}
    assert voc.context >= 29 and pieces[voc.context].shape == full.shape
    torch.testing.assert_close(pieces[voc.context], full, rtol=0, atol=1e-12)
    torch.testing.assert_close(pieces[29], full, rtol=0, atol=1e-12)
    assert (pieces[28] - full).abs().max() > 1e-12  # one frame less is not exact


@pytest.mark.skipif(not _code_cached(BIGVGAN_REPO), reason="BigVGAN code (HF repo) not cached")
@pytest.mark.parametrize("repo", [BIGVGAN_REPO, BASE_REPO])
def test_bigvgan_checkpoint_resolution(tmp_path, repo):
    pytest.importorskip("librosa")
    if not _code_cached(repo):
        pytest.skip(f"{repo} code not cached")
    from drifting_tts.vocoder import build_bigvgan
    from tests.test_vocoder import TINY

    g = build_bigvgan(repo, TINY, pretrained=False)
    torch.save({"generator": g.state_dict(), "hparams": TINY, "repo": repo, "step": 1}, tmp_path / "bigvgan_ft.pt")
    voc = load_vocoder(str(tmp_path / "bigvgan_ft.pt"), "cpu")
    entry = next(e for e in VOCODERS.values() if e.repo == repo and e.hub_file)  # its fine-tune entry
    assert voc.kind == "bigvgan" and voc.mel == "bigvgan" and voc.context == entry.context and voc.graphs
    assert not any(k.endswith("weight_v") for k in voc.model.state_dict())  # weight norm removed
    out = voc(torch.randn(1, 100, 12) - 5)
    assert out.shape == (1, 12 * HOP_LENGTH) and torch.isfinite(out).all()


def _tiny_tts_checkpoint(tmp_path, backend: str) -> str:
    from drifting_tts.config import load_config
    from drifting_tts.models.tts import DriftingTTS
    from drifting_tts.utils import save_checkpoint

    cfg = load_config("configs/tts.yaml", [f"data.root={tmp_path / 'none'}", "model.gen.hidden=32",
                                           "model.gen.depth=1", "model.gen.heads=2", "model.gen.noise_coords=2",
                                           "model.text.d=16", "model.text.layers=1", "model.text.ffn=32",
                                           "model.text.spk_dim=8"])
    torch.manual_seed(0)
    tts = DriftingTTS(cfg.model, num_speakers=2)
    path = tmp_path / "tts.pt"
    save_checkpoint(path, ema=tts.state_dict(), config=cfg.to_dict(), num_speakers=2,
                    stats={"mean": -5.0, "std": 2.0, "backend": backend})
    return str(path)


def test_synthesizer_with_registry_names(tmp_path):
    from drifting_tts.synthesize import Synthesizer

    synth = Synthesizer(_tiny_tts_checkpoint(tmp_path, "bigvgan"), "cpu", vocoder="griffin-lim")
    synth.vocoder.model.n_iter = 4
    wav, info = synth("merhaba dünya. nasılsın?", speaker=0, pause=0.1)
    mels = synth.mels("merhaba dünya. nasılsın?", speaker=0)
    assert len(mels) == 2 and wav.numel() == sum(m.shape[-1] for m in mels) * HOP_LENGTH + 2400
    pieces = list(synth.stream("merhaba dünya. nasılsın?", speaker=0, pause=0.1))
    assert len(pieces) == 3  # sentence, pause, sentence: Griffin-Lim yields each sentence whole
    torch.testing.assert_close(torch.cat(pieces), wav)
    pytest.importorskip("vocos")
    model, cfg = _tiny_vocos(tmp_path)
    torch.save({"vocos": model.state_dict(), "init": cfg, "step": 1}, tmp_path / "vocos_ft.pt")  # Vocos mels
    with pytest.raises(ValueError, match="expects vocos mels"):
        Synthesizer(_tiny_tts_checkpoint(tmp_path, "bigvgan"), "cpu", vocoder=str(tmp_path / "vocos_ft.pt"))

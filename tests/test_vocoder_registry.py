import os

import pytest
import torch
from huggingface_hub.constants import HF_HUB_CACHE

from drifting_tts.audio import HOP_LENGTH, BigVGANLogMel, LogMel, world_f0
from drifting_tts.fast import stream_vocoder
from drifting_tts.vocoder import (
    BASE_REPO,
    BIGVGAN_REPO,
    REVOX_REPO,
    REVOX_REVISION,
    VOCODERS,
    VOCOS_REPO,
    GriffinLim,
    RevoxLogMel,
    Vocoder,
    checkpoint_kind,
    load_vocoder,
    ola_istft,
    resample_sharp,
    revox_frames,
    revox_pitch,
    to_revox_frames,
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
        assert e.kind in ("bigvgan", "vocos", "griffin-lim", "revox"), name
        assert e.context is None or e.context > 0
        if e.kind == "bigvgan":
            assert e.repo.startswith("nvidia/") and e.mel == "bigvgan"
    assert VOCODERS["griffin-lim"].context is None and VOCODERS["griffin-lim"].mel is None
    assert VOCODERS["bigvgan-base-ft"].repo == BASE_REPO and VOCODERS["vocos-ft"].mel == "bigvgan"
    v2 = VOCODERS["vocos-v2"]  # same network as vocos-ft: same mel and streaming context
    assert (v2.kind, v2.mel, v2.context, v2.repo) == ("vocos", "bigvgan", VOCODERS["vocos-ft"].context, VOCOS_REPO)
    revox = VOCODERS["revox"]
    assert revox.context is None and revox.mel == "bigvgan" and revox.repo == REVOX_REPO
    assert "non-commercial" in revox.about and "Minori Live" in revox.about  # CC BY-NC-SA 4.0, attributed


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


def _harmonic(seconds: float = 1.0, f0: float = 140.0, glide: float = 20.0) -> torch.Tensor:
    """A gliding 29-harmonic tone with a 3 Hz amplitude modulation, 24 kHz ``[1, samples]``."""
    t = torch.arange(int(24_000 * seconds)) / 24_000
    phase = 2 * torch.pi * torch.cumsum(f0 + glide * t, 0) / 24_000
    return 0.1 * sum(torch.sin(h * phase) / h for h in range(1, 30))[None] * (0.6 + 0.4 * torch.sin(6 * torch.pi * t))


def test_revox_frames_and_time_interpolation():
    assert [revox_frames(t) for t in (1, 15, 30, 31)] == [2, 16, 32, 34]  # ceil(512 T / 480) at 48 kHz
    t = 40
    times = (torch.arange(t) + 0.5) * HOP_LENGTH / 24_000  # BigVGAN frame centres
    y = to_revox_frames(3 + 2 * times[None])
    k = torch.arange(revox_frames(t)) / 100  # Revox frame centres
    inner = (k * 93.75 - 0.5 >= 0) & (k * 93.75 - 0.5 <= t - 1)
    torch.testing.assert_close(y[0, inner], 3 + 2 * k[inner])  # a linear ramp is reproduced exactly
    assert not inner[0] and y[0, 0] == 3 + 2 * times[0]  # frame 0 (0 s) precedes the first centre: held


def test_revox_mel_conversion_matches_upsampled_audio():
    """Revox's mel converted from the 24 kHz magnitude matches its mel of the audio upsampled x2, in dB."""
    x, gl, front = _harmonic(), GriffinLim("bigvgan"), RevoxLogMel()
    frames = x.shape[-1] // HOP_LENGTH
    truth = front(resample_sharp(x, 24_000, 48_000))[..., : revox_frames(frames)]
    top = (front.fb.shape[-1] - 1 - (front.fb.flip(-1) > 0).float().argmax(-1)) * 24_000 / (front.fb.shape[-1] - 1)
    low = top < 11_500  # below the upsampler's transition band

    def error_db(mel: torch.Tensor) -> tuple[float, float]:
        d, ref = (mel - truth)[0, low, 2:-2], truth[0, low, 2:-2]  # the edge frames see different padding
        d = 20 / torch.log(torch.tensor(10.0)) * d[ref > ref.max() - 6.9]  # within 60 dB of the peak
        return float(d.abs().mean()), float(d.mean())

    mae, bias = error_db(front.convert(gl.stft(x).abs()[..., :frames]))  # exact magnitude: x2 and timing
    assert mae < 0.5 and abs(bias) < 0.2
    mae, bias = error_db(front.convert(gl.magnitude(BigVGANLogMel()(x))))  # through our mel and NNLS
    assert mae < 2.0 and abs(bias) < 0.5


@pytest.mark.parametrize("method", ["dio", "harvest"])
def test_revox_pitch(method):
    x = _harmonic(glide=0.0)[0]
    k = revox_frames(x.numel() // HOP_LENGTH)
    f0, voiced, valid = revox_pitch(x, 24_000, k, method)
    assert f0.shape == voiced.shape == valid.shape == (k,) and valid.all() and torch.equal(voiced, f0 > 0)
    assert voiced[5:-5].all() and abs(float(f0[voiced].median()) / 140 - 1) < 0.02


def _revox_cached() -> bool:
    pytest.importorskip("onnxruntime")
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(REVOX_REPO, "vocoder.onnx", revision=REVOX_REVISION, local_files_only=True)
    except Exception:
        return False
    return True


def test_revox_vocoder(tmp_path):
    """Needs the Revox ONNX in the HF cache (non-commercial weights, never bundled): skipped otherwise."""
    if not _revox_cached():
        pytest.skip(f"{REVOX_REPO} not in the HF cache")
    voc = load_vocoder("revox:griffin-lim:harvest", "cpu")
    assert voc.kind == "revox" and voc.mel == "bigvgan" and voc.context is None and not voc.graphs
    assert voc.num_params == 4_463_874 and voc.model.f0 == "griffin-lim" and voc.model.method == "harvest"
    mel = BigVGANLogMel()(_harmonic(glide=0.0))
    wav = voc(mel)
    assert wav.shape == (1, mel.shape[-1] * HOP_LENGTH) and wav.abs().max() <= 1 and torch.isfinite(wav).all()
    f0 = world_f0(wav[0].double().numpy(), 24_000, 10.0, "harvest")
    assert abs(float(torch.tensor(f0[f0 > 0]).median()) / 140 - 1) < 0.03  # the pitch survives
    pieces = list(stream_vocoder(voc, mel, context=voc.context))
    assert len(pieces) == 1 and torch.equal(pieces[0], wav[0])  # one piece per sentence

    from drifting_tts.synthesize import Synthesizer

    synth = Synthesizer(_tiny_tts_checkpoint(tmp_path, "bigvgan"), "cpu", vocoder="revox:none")
    wav, _ = synth("merhaba dünya. nasılsın?", speaker=0, pause=0.1)
    mels = synth.mels("merhaba dünya. nasılsın?", speaker=0)
    assert wav.numel() == sum(m.shape[-1] for m in mels) * HOP_LENGTH + 2400
    assert len(list(synth.stream("merhaba dünya. nasılsın?", speaker=0, pause=0.1))) == 3

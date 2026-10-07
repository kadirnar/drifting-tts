"""Local checkpoints exercise the public MLX API without downloading model weights."""

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from drifting_tts.mlx.bigvgan import BigVGAN  # noqa: E402
from drifting_tts.mlx.model import DriftingTTS  # noqa: E402
from drifting_tts.mlx.synthesize import Synthesizer  # noqa: E402
from drifting_tts.text import SYMBOLS, normalize, split_sentences, text_to_ids  # noqa: E402

MODEL = {
    "text": {"d": 8, "heads": 2, "layers": 1, "ffn": 16, "dropout": 0.0, "spk_dim": 4},
    "gen": {"hidden": 16, "depth": 1, "heads": 2, "patch": 2, "mlp_ratio": 2.0, "n_registers": 2,
            "noise_classes": 4, "noise_coords": 2, "residual_prior": True, "num_steps": 1},
    "pitch": {"enabled": True},
}
VOCODER = {
    "num_mels": 4, "upsample_rates": [2, 2], "upsample_kernel_sizes": [4, 4], "upsample_initial_channel": 8,
    "resblock": "1", "resblock_kernel_sizes": [3], "resblock_dilation_sizes": [[1, 3]],
    "activation": "snakebeta", "snake_logscale": True, "use_tanh_at_final": False, "use_bias_at_final": False,
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        path = tmp_path_factory.mktemp("mlx-synthesis")
        mx.random.seed(7)
        model = DriftingTTS(MODEL, num_speakers=2, n_vocab=len(SYMBOLS), n_mels=4)
        # Exactly two frames per token, independent of the text, voice and noise seed.
        model.encoder.duration.proj.weight = mx.zeros_like(model.encoder.duration.proj.weight)
        model.encoder.duration.proj.bias = mx.array([np.log(1.25)], dtype=mx.float32)
        vocoder = BigVGAN(VOCODER)
        mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
        mx.save_safetensors(str(path / "vocoder.safetensors"), dict(tree_flatten(vocoder.parameters())))
        config = {
            "model": MODEL, "vocoder": VOCODER, "num_speakers": 2, "n_vocab": len(SYMBOLS), "n_mels": 4,
            "sample_rate": 24_000, "voices": {"studio": 0, "female": 1}, "default_voice": "studio",
            "temperature": 0.3, "duration_scale": 1.0, "duration_scales": {}, "stats": {"mean": -2.0, "std": 1.0},
        }
        (path / "config.json").write_text(json.dumps(config))
        yield path
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def synth(checkpoint):
    return Synthesizer(checkpoint, compile=False)


@pytest.mark.parametrize("chunk_frames", [1, 11, 128])
def test_stream_matches_whole_waveform(synth, chunk_frames):
    text = "Merhaba dünya."
    kwargs = {"speaker": "female", "seed": 19, "temperature": 0.4, "cfg_scale": 1.5, "pitch_shift": 2.0}
    expected, _ = synth(text, **kwargs)
    chunks = list(synth.stream(text, chunk_frames=chunk_frames, **kwargs))
    actual = np.concatenate([audio for audio, _ in chunks])
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_stream_has_exactly_one_pause_between_sentences(synth):
    text, pause, chunk_frames = "Merhaba. Nasılsın?", 0.002, 11
    expected, _ = synth(text, pause=pause, seed=2)
    chunks = list(synth.stream(text, pause=pause, seed=2, chunk_frames=chunk_frames))
    silence = [audio for audio, info in chunks if info["is_silence"]]
    speech = [(audio, info) for audio, info in chunks if not info["is_silence"]]
    assert len(silence) == 1
    assert np.array_equal(silence[0], np.zeros(int(pause * synth.sample_rate), np.float32))
    assert not chunks[0][1]["is_silence"] and not chunks[-1][1]["is_silence"]
    assert all(0 < len(audio) <= chunk_frames * synth.vocoder.hop_length for audio, _ in speech)
    assert {info["sentence_index"] for _, info in speech} == {0, 1}
    for i, sentence in enumerate(split_sentences(normalize(text))):
        sentence_audio = [audio for audio, info in speech if info["sentence_index"] == i]
        assert sum(map(len, sentence_audio)) == 2 * len(text_to_ids(sentence, normalized=True)) * 4
    np.testing.assert_allclose(np.concatenate([audio for audio, _ in chunks]), expected, atol=2e-6, rtol=2e-5)


def test_zero_pause_does_not_emit_empty_audio(synth):
    chunks = list(synth.stream("Merhaba. Nasılsın?", pause=0, chunk_frames=11))
    assert chunks and all(len(audio) > 0 and not info["is_silence"] for audio, info in chunks)


def test_chunk_ramp_preserves_waveform_and_caps_decode_work(synth):
    text = "merhaba dünya " * 10 + "."
    expected, _ = synth(text, length_scale=6)
    chunks = list(synth.stream(text, length_scale=6, chunk_frames=512))
    sizes = [len(audio) // synth.vocoder.hop_length for audio, _ in chunks]
    assert sizes[:4] == [24, 128, 256, 512]
    assert 0 < sizes[-1] <= 512
    np.testing.assert_allclose(np.concatenate([audio for audio, _ in chunks]), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("text", ["", "  ", "...!?", "☀️"])
def test_empty_normalized_input(synth, text):
    assert list(synth.stream(text)) == []
    audio, info = synth(text)
    assert audio.dtype == np.float32 and audio.shape == (0,)
    assert info["rtf_total"] == 0


def test_seed_is_independent_of_global_random_state(synth):
    def generate(seed, global_seed):
        mx.random.seed(global_seed)
        return np.concatenate([audio for audio, _ in synth.stream("Merhaba dünya.", seed=seed, chunk_frames=11)])

    first = generate(1, 7)
    np.testing.assert_array_equal(first, generate(1, 999))
    assert not np.allclose(first, generate(2, 7), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("kwargs", [
    {"chunk_frames": 0}, {"chunk_frames": -1}, {"chunk_frames": 1.5}, {"chunk_frames": True},
    {"first_chunk_frames": 0}, {"first_chunk_frames": 1.5}, {"context_frames": -1}, {"context_frames": 0},
    {"pause": -0.1}, {"temperature": -0.1}, {"temperature": float("nan")},
    {"length_scale": 0}, {"length_scale": float("inf")}, {"cfg_scale": float("nan")},
    {"pitch_shift": float("inf")}, {"speaker": 2},
])
def test_invalid_stream_parameters(synth, kwargs):
    with pytest.raises((ValueError, TypeError)):
        list(synth.stream("Merhaba.", **kwargs))


def test_first_next_decodes_only_the_first_audio_chunk(synth, monkeypatch):
    encoded, decoded = [], []
    encode, vocode = synth._encode, synth._vocode

    def record_encode(*args):
        encoded.append(args[0].shape[1])
        return encode(*args)

    def record_vocode(mel):
        decoded.append(mel.shape[-1])
        return vocode(mel)

    monkeypatch.setattr(synth, "_encode", record_encode)
    monkeypatch.setattr(synth, "_vocode", record_vocode)
    stream = synth.stream("Merhaba dünya. Nasılsın?", chunk_frames=11, first_chunk_frames=3)
    assert not encoded and not decoded
    audio, first = next(stream)
    total_frames = 2 * len(text_to_ids("merhaba dünya.", normalized=True))
    assert len(encoded) == len(decoded) == 1
    assert decoded == [3 + synth.vocoder.context_frames]
    assert decoded[0] < total_frames and len(audio) == 3 * synth.vocoder.hop_length
    assert first["sentence_index"] == first["chunk_index"] == 0
    assert 0 < first["ttfa_seconds"] <= first["elapsed_seconds"]
    audio, second = next(stream)
    assert len(encoded) == 1 and len(decoded) == 2
    assert len(audio) == 11 * synth.vocoder.hop_length
    assert second["chunk_index"] == 1 and second["acoustic_seconds"] == 0
    assert second["ttfa_seconds"] == first["ttfa_seconds"]
    stream.close()


def test_default_temperature_and_duration_scaling(synth):
    implicit, _ = synth("Merhaba.", seed=3)
    explicit, _ = synth("Merhaba.", seed=3, temperature=synth.config["temperature"])
    np.testing.assert_array_equal(implicit, explicit)
    slower, _ = synth("Merhaba.", seed=3, length_scale=2.0)
    # ceil(1.25) = 2; ceil(1.25 * 2) = 3 frames per token.
    assert len(slower) * 2 == len(implicit) * 3


def test_compiled_synthesis_matches_eager_with_pitch_and_shape_changes(checkpoint, synth):
    compiled = Synthesizer(checkpoint, compile=True)
    assert compiled.compiled and not synth.compiled
    for text, pitch in (("Merhaba.", 0.0), ("Merhaba dünya.", 2.0), ("Merhaba dünya.", -2.0)):
        kwargs = {"seed": 23, "pitch_shift": pitch, "speaker": "female"}
        expected, _ = synth(text, **kwargs)
        actual, _ = compiled(text, **kwargs)
        streamed = np.concatenate([audio for audio, _ in compiled.stream(text, chunk_frames=11, **kwargs)])
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(streamed, expected, atol=2e-5, rtol=2e-4)

import torch

from drifting_tts.synthesize import split_sentences
from drifting_tts.text import normalize


def test_split_sentences():
    text = normalize("Merhaba! Bugün hava çok güzel. Peki ya yarın?")
    assert split_sentences(text) == ["merhaba!", "bugün hava çok güzel.", "peki ya yarın?"]


def test_split_long_sentence():
    words = " ".join(["kelime"] * 60) + "."
    parts = split_sentences(words, max_chars=100)
    assert all(len(p) <= 101 for p in parts)
    assert " ".join(parts) == words


def _tiny_tts():
    from drifting_tts.config import Config
    from drifting_tts.models.tts import DriftingTTS

    cfg = Config({"text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
                  "gen": {"hidden": 32, "depth": 1, "heads": 2, "patch": 2, "mlp_ratio": 2.0, "n_registers": 2,
                          "noise_classes": 4, "noise_coords": 2, "num_steps": 1}})
    torch.manual_seed(0)
    model = DriftingTTS(cfg, num_speakers=2).eval()
    for p in model.generator.final.parameters():  # the output layer is zero-initialised: make the noise visible
        torch.nn.init.normal_(p, std=0.5)
    return model


def test_seed_fully_determines_the_output():
    """``z`` *and* the style codes come from the seeded generator, so the global RNG state must not matter."""
    model = _tiny_tts()
    args = (torch.tensor([[1, 5, 1, 6, 1]]), torch.tensor([5]), torch.tensor([0]))

    def synth(seed, global_seed, temperature):
        torch.manual_seed(global_seed)
        return model.synthesize(*args, temperature=temperature, generator=torch.Generator().manual_seed(seed))[0]

    for t in (0.0, 0.5):  # temperature 0: the output depends on the style codes alone
        assert torch.equal(synth(0, 1, t), synth(0, 2, t))
        assert not torch.allclose(synth(0, 1, t), synth(3, 1, t))


def test_inference_modules_import_without_training_dependencies():
    """Loading and synthesis must not need torchaudio or tensorboard (e.g. a ZeroGPU Space with torch only)."""
    import subprocess
    import sys

    code = ("import sys; sys.modules['torchaudio'] = None; sys.modules['tensorboard'] = None\n"
            "from drifting_tts.synthesize import Synthesizer\nfrom drifting_tts.train import load_tts\n"
            "from drifting_tts.vocoder import Vocoder\nprint('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_voice_names_resolve_to_the_curated_speakers():
    import pytest

    from drifting_tts.voices import DEFAULT_VOICE, VOICES, voice_id

    assert DEFAULT_VOICE == "studio" and voice_id("male") == VOICES["male"]["id"] and voice_id("female") == 323
    assert voice_id("studio") == 722
    assert voice_id(17) == 17 and voice_id("17") == 17  # raw training speaker IDs still work
    with pytest.raises(KeyError):
        voice_id("robot")

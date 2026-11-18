"""The models of the pilot demo (``app.py``) without the user interface, so that they run and can be tested without
Gradio: ARDrift v2 (the text as a prefix, #103) at 20,000 and 40,000 steps, and the aligned model with a voice
reference and guidance on the voice (#96, #99). The weights are the files next to this module (a Space that
holds them, ``scripts/deploy_pilots_space.sh``) or come from the pilots' model repo; the vocoder and the voices
come from the released English model's repo."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.data import MelStats
from drifting_tts.synthesize import Synthesizer
from drifting_tts.train import load_tts

REPO = os.environ.get("DRIFTING_TTS_PILOTS_REPO", "kadirnar/drifting-tts-en-pilots")
RELEASE = os.environ.get("DRIFTING_TTS_EN_REPO", "Vyvo/drifting-tts-en")
V2 = {"40,000 steps": "ardrift_v2_clock_40k.pt", "20,000 steps": "ardrift_v2_clock_20k.pt"}
VOICE = "ardrift_en_voice.pt"
MODES = {"reference": "reference", "prompt": "prefix", "prompt + reference": "both"}  # label -> Synthesizer's mode
MAX_CHARS_V2, MAX_CHARS_CLONE = 220, 400  # one utterance of the length v2 was trained on; a few sentences
REFERENCE_SECONDS = 6.0  # what the cloning models were measured with


def weights(name: str, repo: str = REPO) -> str | None:
    """``name`` next to this module, else from ``repo``; ``None`` where neither has it (a private repo without a
    token)."""
    local = Path(__file__).parent / name
    if local.exists():
        return str(local)
    try:
        return hf_hub_download(repo, name)
    except Exception as e:  # not found, or not readable without a token
        print(f"{name}: not next to engine.py and not readable in {repo} ({type(e).__name__})", flush=True)
        return None


def default_device() -> str:
    """``DEMO_DEVICE``, else the GPU where there is one (a ZeroGPU Space has it inside ``spaces.GPU`` calls)."""
    return os.environ.get("DEMO_DEVICE") or (
        "cuda" if torch.cuda.is_available() or os.environ.get("SPACES_ZERO_GPU") else "cpu")


class Pilots:
    def __init__(self, device: str | None = None, repo: str = REPO, release: str = RELEASE,
                 v2: dict[str, str] | None = None):
        self.device = device or default_device()
        self.voices = {v["label"]: int(v["id"])
                       for v in json.loads(open(hf_hub_download(release, "voices.json")).read())["voices"]}
        voice = weights(VOICE, repo)
        if voice is None:
            raise FileNotFoundError(f"{VOICE} is neither next to engine.py nor readable in {repo}")
        self.cloner = Synthesizer(voice, self.device, vocoder=hf_hub_download(release, "bigvgan_en.pt"))
        self.cloner.guidance = "voice"  # the null branch of guidance is the model without the voice, not the text
        self.vocoder, self.text = self.cloner.vocoder, self.cloner.text
        self.v2 = {}
        for label, name in (V2 if v2 is None else v2).items():
            path = weights(name, repo)
            if path is None:  # a checkpoint the Space does not hold: not offered
                continue
            model, _, stats = load_tts(path, self.device)
            self.v2[label] = (model.eval(), MelStats.from_dict(stats))
        if not self.v2:
            raise FileNotFoundError(f"no ARDrift v2 checkpoint next to engine.py or readable in {repo}")

    def _clean(self, text: str, limit: int) -> tuple[str, str]:
        text = " ".join((text or "").split())
        if not text:
            raise ValueError("Please enter some English text.")
        if len(text) > limit:
            raise ValueError(f"Please keep the text under {limit} characters.")
        normalized = self.text.normalize(text)
        if not normalized:
            raise ValueError("Nothing to read after normalisation.")
        return text, normalized

    @torch.no_grad()
    def speak_v2(self, text: str, voice: str, steps: str = "40,000 steps", temperature: float = 1.0,
                 speed: float = 1.0, seed: int = 0) -> tuple[np.ndarray, str]:
        """ARDrift v2: one utterance of ``text`` in a voice of the speaker table. ``speed`` scales the tempo (the
        model's clock: patches per character); the utterance ends where the text ends on the clock, or earlier if
        the end head fires."""
        _, normalized = self._clean(text, MAX_CHARS_V2)
        model, stats = self.v2[steps]
        t0 = time.time()
        ids = torch.tensor([self.text.text_to_ids(normalized, normalized=True)], device=self.device)
        rate = model.rate / float(speed)
        mel, _ = model.synthesize(ids, torch.tensor([ids.shape[1]], device=self.device),
                                  torch.tensor([self.voices[voice]], device=self.device),
                                  temperature=float(temperature), rate=rate,
                                  generator=torch.Generator(device=self.device).manual_seed(int(seed)))
        wav = self.vocoder(stats.denormalize(mel))[0].float().cpu().numpy().squeeze()
        bound = math.ceil((ids.shape[1] // 2) * rate) * model.patch  # the text's end on the clock, in frames
        end = "the end head ended it" if mel.shape[-1] < bound else "it ended where the text ends on the clock"
        info = (f"{len(wav) / SAMPLE_RATE:.1f} s of audio in {time.time() - t0:.1f} s ({self.device}); {end}. "
                f"Normalised text: *{normalized}*")
        return np.clip(wav, -1, 1), info

    @torch.no_grad()
    def clone(self, text: str, recording: str, mode: str = "reference", scale: float = 3.0,
              temperature: float = 0.5, seed: int = 0) -> tuple[np.ndarray, str]:
        """``text`` in the voice of ``recording`` (a file; its first six seconds are used). ``mode``: a label of
        :data:`MODES`. ``scale``: guidance on the voice, for the modes with a reference (a prompt alone was
        measured without guidance and runs at 1)."""
        text, normalized = self._clean(text, MAX_CHARS_CLONE)
        if not recording:
            raise ValueError("Please upload or record a few seconds of the voice.")
        t0 = time.time()
        try:
            ref = self.cloner.reference(recording, mode=MODES[mode], seconds=REFERENCE_SECONDS,
                                        reference_seconds=REFERENCE_SECONDS)
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"The recording could not be used: {e}") from e
        scale = float(scale) if MODES[mode] != "prefix" else 1.0
        wav, out = self.cloner(text, ref_audio=ref, cfg_scale=scale, temperature=float(temperature), seed=int(seed))
        info = (f"{out['seconds']:.1f} s of audio in {time.time() - t0:.1f} s ({self.device}); voice given as "
                f"{mode}, guidance {scale:g}. Normalised text: *{normalized}*")
        return np.clip(wav.float().numpy(), -1, 1), info

"""Published artefacts: the files of each release in :data:`HUB_REPO` and the named prosody predictors.

A release is the acoustic model, a vocoder, a prosody source and a pause rule
(:meth:`drifting_tts.synthesize.Synthesizer.from_pretrained`). Files are downloaded from the Hub, or read from a local
directory laid out like the repo (``$DRIFTING_TTS_HUB_DIR``, e.g. files staged for an upload) when it holds them.
"""

from __future__ import annotations

import os
from pathlib import Path

HUB_REPO = "Vyvo/drifting-tts-tr"
HUB_DIR_ENV = "DRIFTING_TTS_HUB_DIR"

# name -> (file in HUB_REPO, local fallback read first when it exists, like the vocoder registry's ``local``)
PROSODY_MODELS: dict[str, tuple[str, str | None]] = {
    # stochastic prosody predictor trained with drifting on v3.1 (docs/PROSODY_MODEL.md): temperature 0.5 and
    # per-voice duration factors stored in the checkpoint
    "drift": ("prosody_drift_v3.2.pt", None),
}

# release -> Synthesizer arguments. ``model`` is a file of HUB_REPO; the other values are what Synthesizer takes.
# v3.2 keeps v3.1's acoustic weights (``drifting_tts_v3.2.pt``: the same weights with a sanitised config). It samples
# only the token pitch: sampled durations cost intelligibility on new text for the voices with little data (Freya-495
# WER 1.74% -> 5.78% male, 3.02% -> 11.28% female), so the durations stay v3.1's regressors'.
RELEASES: dict[str, dict] = {
    "v3.1": {"model": "drifting_tts_v3.1.pt", "vocoder": "bigvgan-v2-ft", "prosody": None, "pause": 0.15},
    "v3.2": {"model": "drifting_tts_v3.2.pt", "vocoder": "vocos-v2", "prosody": "drift",
             "prosody_durations": "regressor", "pause": "punct"},
}
LATEST = "v3.2"


def hub_file(filename: str, what: str | None = None) -> str:
    """Local path of ``filename`` of :data:`HUB_REPO`: ``$DRIFTING_TTS_HUB_DIR/<filename>`` if that exists, else the
    Hub download. ``what`` names the caller's object in the error raised when the file is not published."""
    local = os.environ.get(HUB_DIR_ENV)
    if local and (Path(local) / filename).is_file():
        return str(Path(local) / filename)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(HUB_REPO, filename)
    except EntryNotFoundError as e:
        raise FileNotFoundError(f"{what or filename}: {HUB_REPO}/{filename} not found (not published yet, or "
                                f"offline); pass the checkpoint path instead, or set {HUB_DIR_ENV}") from e


def resolve_prosody(spec: str | Path) -> str:
    """A prosody predictor name of :data:`PROSODY_MODELS` (e.g. ``drift``) or a checkpoint path -> a local path."""
    if str(spec) in PROSODY_MODELS:
        filename, local = PROSODY_MODELS[str(spec)]
        return local if local and Path(local).is_file() else hub_file(filename, f"prosody {spec!r}")
    if not Path(spec).is_file():
        raise ValueError(f"unknown prosody predictor {str(spec)!r}: expected a checkpoint path or one of "
                         f"{', '.join(PROSODY_MODELS)}")
    return str(spec)

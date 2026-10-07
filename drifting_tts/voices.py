"""The voices of the released model: the best-rated of its 722 training speakers.

``male`` and ``female`` were chosen with ``scripts/select_voices.py`` on 30 held-out ``dev`` sentences per candidate
(UTMOSv2 and Whisper large-v3 CER), never on a test set: the best male and the best female voice of the 24
best-covered speakers. ``studio`` (the default) was added later by fine-tuning (speaker 722, checkpoint v3.1).
Other training speaker IDs still load, but these are the supported, documented voices.
"""

from __future__ import annotations

VOICES: dict[str, dict] = {
    "male": {"id": 389, "pitch_hz": 104, "utmosv2": 2.84, "cer": 0.0066, "train_minutes": 59},
    "female": {"id": 323, "pitch_hz": 174, "utmosv2": 2.82, "cer": 0.0102, "train_minutes": 19},
    # added by fine-tuning (configs/tts_v3_add_voice.yaml) on one studio voice; needs v3.1.
    # Freya-TR-Eval WER 1.23% / UTMOSv2 2.94, the best of the three, hence the default
    "studio": {"id": 722, "pitch_hz": 103, "utmosv2": 2.94, "cer": 0.0024, "train_minutes": 2058},
}
DEFAULT_VOICE = "studio"


def voice_id(voice: str | int) -> int:
    """Speaker ID of a voice name (``male`` / ``female``) or of a raw training speaker ID."""
    if isinstance(voice, int) or str(voice).isdigit():
        return int(voice)
    if voice not in VOICES:
        raise KeyError(f"unknown voice {voice!r}; choose one of {', '.join(VOICES)}")
    return VOICES[voice]["id"]

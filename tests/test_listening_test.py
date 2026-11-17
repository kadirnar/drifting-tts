"""The blind A/B listening page (#85): pairing, the random order and its hidden key, and the score of saved answers."""

import base64
import json
import re

import numpy as np
import pytest
import soundfile as sf

from scripts import listening_test as lt


def _folders(tmp_path, n: int = 8):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    t = np.arange(4800) / 24000
    for i in range(n):
        sf.write(a / f"utt{i:02d}.wav", 0.1 * np.sin(2 * np.pi * 220 * t), 24000)
        sf.write(b / f"utt{i:02d}.wav", 0.1 * np.sin(2 * np.pi * 440 * t), 24000)
    sf.write(a / "only_a.wav", np.zeros(100), 24000)
    return a, b


def test_pairs_are_the_files_of_both_folders(tmp_path):
    a, b = _folders(tmp_path)
    assert lt.pairs(a, b) == [f"utt{i:02d}.wav" for i in range(8)]
    assert lt.pairs(a, b, match="utt03") == ["utt03.wav"]


def test_items_are_balanced_reproducible_and_keep_the_systems_apart(tmp_path):
    a, b = _folders(tmp_path)
    names = lt.pairs(a, b)
    items = lt.build_items(a, b, names, seed=1)
    assert sorted(it["id"] for it in items) == names and [it["id"] for it in items] != names  # shuffled
    assert sum(it["flip"] for it in items) == 4  # half of the pairs play system B first
    assert [(it["id"], it["flip"]) for it in items] == [(it["id"], it["flip"]) for it in lt.build_items(a, b, names, 1)]
    assert [it["flip"] for it in items] != [it["flip"] for it in lt.build_items(a, b, names, 2)]
    wav_a, wav_b = lt.encode_wav(a / "utt00.wav"), lt.encode_wav(b / "utt00.wav")
    for it in items:
        if it["id"] == "utt00.wav":
            assert (it["left"], it["right"]) == ((wav_b, wav_a) if it["flip"] else (wav_a, wav_b))
    short = lt.encode_wav(a / "utt00.wav", max_seconds=0.05)
    assert len(short) < len(wav_a) and short.startswith("data:audio/wav;base64,")


def test_page_hides_the_key_and_carries_every_pair(tmp_path):
    a, b = _folders(tmp_path, n=4)
    items = lt.build_items(a, b, lt.pairs(a, b), seed=0)
    page = lt.render(items, "baseline", "candidate", "Listening")
    assert "baseline" not in page and "candidate" not in page and '"flip"' not in page  # nothing readable gives it away
    key = json.loads(base64.b64decode(re.search(r'atob\("([^"]+)"\)', page).group(1)))
    assert key == {"a": "baseline", "b": "candidate", "flips": [it["flip"] for it in items]}
    shown = json.loads(re.search(r"const ITEMS = (\[.*?\]), SCALE", page, re.S).group(1))
    assert [s["id"] for s in shown] == [it["id"] for it in items] and page.count("data:audio/wav;base64,") == 8


def test_score_is_from_a_to_b_whatever_the_side():
    answers = [{"id": "1", "flip": False, "rating": 2}, {"id": "2", "flip": True, "rating": -2},  # B better, twice
               {"id": "3", "flip": False, "rating": 1}, {"id": "4", "flip": True, "rating": 1},   # B, then A
               {"id": "5", "flip": False, "rating": 0}, {"id": "6", "flip": False, "rating": None}]
    s = lt.score(answers)
    assert s["n"] == 5 and s["cmos"] == pytest.approx(0.8)
    assert (s["b_preferred"], s["a_preferred"], s["ties"]) == (3, 1, 1)
    assert s["sign_test_p"] == pytest.approx(2 * (1 + 4) / 16)  # two-sided, 1 of 4 against
    assert lt.score([{"id": "1", "flip": False, "rating": None}]) == {"n": 0}
    one_sided = [{"id": str(i), "flip": i % 2 == 1, "rating": 3} for i in range(8)]  # always "B much better"
    assert lt.score(one_sided)["cmos"] == 0.0

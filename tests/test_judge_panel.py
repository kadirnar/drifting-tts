"""The judge panel (#85) without its judges: the paired comparison, the summary and the script with a stub judge."""

import json
import sys

import numpy as np
import pytest
import soundfile as sf

from scripts import judge_panel as jp


def test_paired_difference_and_its_interval():
    rng = np.random.default_rng(0)
    base = rng.normal(3.0, 0.5, 200)
    better = jp.paired(base + 0.1 + rng.normal(0, 0.05, 200), base)  # a small shift on every sentence
    assert better["diff"] == pytest.approx(0.1, abs=0.02) and better["significant"] and better["higher"] > 0.9
    assert better["lo"] < better["diff"] < better["hi"]
    noise = jp.paired(base + rng.normal(0, 0.3, 200), base)  # the same on average
    assert not noise["significant"] and noise["lo"] < 0 < noise["hi"]
    unpaired_would_miss = jp.paired(rng.normal(3.1, 0.5, 200), base)  # other sentences: the shift drowns
    assert better["hi"] - better["lo"] < 0.2 * (unpaired_would_miss["hi"] - unpaired_would_miss["lo"])


def test_summarize_means_and_comparisons_with_the_baseline():
    scores = {"mos": {"base": [3.0, 3.2, 3.4], "new": [3.5, 3.7, 3.9], "old": [2.0, 2.2, 2.4]}}
    s = jp.summarize(scores, "base")["mos"]
    assert s["mean"] == {"base": pytest.approx(3.2), "new": pytest.approx(3.7), "old": pytest.approx(2.2)}
    assert set(s["vs_baseline"]) == {"new", "old"}
    assert s["vs_baseline"]["new"]["diff"] == pytest.approx(0.5)
    assert s["vs_baseline"]["old"]["diff"] == pytest.approx(-1.0)


def test_panel_scores_shared_files_caches_them_and_counts_the_judges(tmp_path, monkeypatch, capsys):
    t = np.arange(4000) / 16000
    for name, amp in (("rec", 0.5), ("base", 0.3), ("new", 0.4)):
        (tmp_path / name).mkdir()
        for i in range(5):
            sf.write(tmp_path / name / f"{i:02d}.wav", amp * np.sin(2 * np.pi * (200 + 10 * i) * t), 16000)
    sf.write(tmp_path / "rec" / "only_here.wav", np.zeros(100), 16000)
    built = []

    def fake_build(name, device):  # the louder, the better; squim also takes its reference
        built.append(name)
        score = lambda paths, reference=None: {jp.PRIMARY[name]: [float(np.abs(jp.load16(p)).max()) for p in paths]}  # noqa: E731
        return score

    monkeypatch.setattr(jp, "build", fake_build)
    argv = ["judge_panel.py", "--systems", f"rec={tmp_path / 'rec'}", f"base={tmp_path / 'base'}",
            f"new={tmp_path / 'new'}", "--baseline", "base", "--judges", "utmos22", "squim", "--out",
            str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv)
    jp.main()
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["sentences"] == 5 and set(summary["metrics"]) == {"utmos22", "squim_mos"}
    m = summary["metrics"]["utmos22"]
    assert m["mean"]["new"] == pytest.approx(0.4, abs=0.01) and m["vs_baseline"]["new"]["significant"]
    assert m["vs_baseline"]["rec"]["diff"] == pytest.approx(0.2, abs=0.01)
    table = (tmp_path / "out" / "summary.md").read_text()
    assert "| new | 2 of 2 | 2 | 0 |" in table and "| rec | 2 of 2 | 2 | 0 |" in table
    assert built == ["utmos22", "squim"]
    built.clear()
    monkeypatch.setattr(sys, "argv", [*argv, "--groups", "00", "nothing"])
    jp.main()  # every score comes from the cache: no judge is built again
    printed = capsys.readouterr().out
    assert built == [] and "2 of 2" in printed
    assert "**00** (1 sentences)" in printed and "**nothing**" not in printed  # the files that start with "00"


def test_forget_speechmos_takes_the_hub_repository_off_the_path(monkeypatch):
    monkeypatch.setattr(sys, "path", [*sys.path, "/x/torch/hub/tarepan_SpeechMOS_v1.2.0"])
    monkeypatch.setitem(sys.modules, "speechmos", object())
    monkeypatch.setitem(sys.modules, "speechmos.utmos22", object())
    jp._forget_speechmos(hub=True)
    assert "speechmos" not in sys.modules and "speechmos.utmos22" not in sys.modules
    assert any("SpeechMOS" in p for p in sys.path)  # UTMOS22 is about to import from it
    jp._forget_speechmos(hub=False)
    assert not any("SpeechMOS" in p for p in sys.path)

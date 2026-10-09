"""Rule-based sentence-type and position features of Turkish text for the prosody predictor (#40).

Words are the space-separated pieces of the normalised text (punctuation stays attached), i.e. the segments of
:func:`drifting_tts.models.prosody_net.word_index`; sentences end at a word ending in ``.``, ``!`` or ``?``. Per word
(:data:`SENTENCE_FEATURES`):

* the type of its sentence: statement, polar question (a question with the particle mI), wh-question, other question,
  exclamation; and two question flags, a tag (``..., değil mi?``) and an alternative (``... mı, yoksa ... mı?``);
* its role in a question: the host of mI (the word before the particle, which carries the peak in Turkish polar
  questions), the mI word itself, after the last mI, a wh-word, after the wh-word;
* phrasing: last word of the sentence, a comma after it, its relative position in the sentence, the number of words to
  the sentence end (/ 10, at most 2), the relative position of its sentence in the text, last sentence of the text.

Turkish polar questions mostly peak on the syllable before mI and end low; wh-questions peak on the wh-word
(research memo §4.2). The character-level network can in principle find these cues itself, but questions are 5% of the
training sentences.
"""

from __future__ import annotations

import re

import numpy as np

SENTENCE_FEATURES = ("statement", "polar_q", "wh_q", "other_q", "exclamation", "tag_q", "alt_q",
                     "mi_host", "mi", "post_mi", "wh_word", "post_wh",
                     "sentence_final", "comma", "pos_in_sentence", "words_to_end", "sentence_pos", "last_sentence")
DIM = len(SENTENCE_FEATURES)
SENTENCE_TYPES = ("statement", "polar_q", "wh_q", "other_q", "exclamation")

MI = re.compile(r"m[ıiuü](?:s[ıiuü]n(?:[ıiuü]z)?|y[ıiuü][mz]|d[ıiuü]r(?:l[ae]r)?|yd[ıiuü][mnk]?|ym[ıiuü]ş|ys[ae])?")
WH = re.compile(r"ne|neden|niye|niçin|nasıl|nasılsın(?:ız)?|ner[ea](?:de|ye|den|si|sinde|ye)?|nereli|kim(?:i|e|in|den|le|"
                r"inle|ler|lerin?)?|hangi(?:si|sini|sine|sinde|lerini?)?|kaç(?:ta|ar|ıncı|ın[ıa]?)?|ney[ie]|neyle|"
                r"nedir|niçindir")


def _bare(word: str) -> str:
    return word.strip(".,!?")


def sentence_type(words: list[str]) -> str:
    """One of :data:`SENTENCE_TYPES` for a sentence given as its words (normalised, punctuation attached)."""
    end = words[-1][-1:] if words and words[-1] else ""
    if end == "?":
        bare = [_bare(w) for w in words]
        if any(MI.fullmatch(w) for w in bare[1:]):
            return "polar_q"
        return "wh_q" if any(WH.fullmatch(w) for w in bare) else "other_q"
    return "exclamation" if end == "!" else "statement"


def sentence_features(text: str) -> np.ndarray:
    """Normalised text -> ``[n_words, DIM]`` float32 features (see the module docstring)."""
    words = text.split(" ")
    out = np.zeros((len(words), DIM), np.float32)
    col = {k: i for i, k in enumerate(SENTENCE_FEATURES)}
    sentences, start = [], 0
    for i, w in enumerate(words):
        if w.endswith((".", "!", "?")) or i == len(words) - 1:
            sentences.append((start, i + 1))
            start = i + 1
    for si, (a, b) in enumerate(sentences):
        ws = words[a:b]
        bare = [_bare(w) for w in ws]
        kind = sentence_type(ws)
        out[a:b, col[kind]] = 1
        n = b - a
        if kind in ("polar_q", "wh_q", "other_q"):
            if n >= 2 and bare[-2] == "değil" and MI.fullmatch(bare[-1]):
                out[a:b, col["tag_q"]] = 1
            if "yoksa" in bare:
                out[a:b, col["alt_q"]] = 1
            mis = [j for j in range(1, n) if MI.fullmatch(bare[j])]
            for j in mis:
                out[a + j, col["mi"]] = 1
                out[a + j - 1, col["mi_host"]] = 1
            if mis:
                out[a + mis[-1] + 1: b, col["post_mi"]] = 1
            whs = [j for j in range(n) if WH.fullmatch(bare[j])]
            for j in whs:
                out[a + j, col["wh_word"]] = 1
            if whs:
                out[a + whs[0] + 1: b, col["post_wh"]] = 1
        out[b - 1, col["sentence_final"]] = 1
        for j, w in enumerate(ws):
            out[a + j, col["comma"]] = float(w.endswith(","))
            out[a + j, col["pos_in_sentence"]] = j / max(n - 1, 1)
            out[a + j, col["words_to_end"]] = min((n - 1 - j) / 10, 2.0)
        out[a:b, col["sentence_pos"]] = si / max(len(sentences) - 1, 1)
        out[a:b, col["last_sentence"]] = float(si == len(sentences) - 1)
    return out

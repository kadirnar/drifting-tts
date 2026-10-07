"""Write web/tests/text_fixtures.json: Python outputs of the text frontend that web/text.js must reproduce.

    PYTHONPATH=$PWD python scripts/make_web_fixtures.py && node web/tests/text.test.mjs

Inputs: every string in tests/test_text.py, the Freya-TR-Eval sentences and seeded random stress strings.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

from drifting_tts.benchmark import FREYA, load_texts
from drifting_tts.synthesize import split_sentences
from drifting_tts.text import (
    _SPELLED,
    _UNITS,
    BLANK_ID,
    PAD_ID,
    SYMBOL_TO_ID,
    SYMBOLS,
    normalize,
    number_to_words,
    ordinal_to_words,
    text_to_ids,
)
from tests import test_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/tests/text_fixtures.json"
N_STRESS = 2500


def unit_test_strings() -> list[str]:
    """The parametrised normalisation cases plus every string literal of tests/test_text.py."""
    strings = [s for case in test_text.CASES for s in case]
    tree = ast.parse((ROOT / "tests/test_text.py").read_text(encoding="utf-8"))
    strings += [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return list(dict.fromkeys(strings))


def unit_test_numbers() -> list[int]:
    """The parametrised inputs of test_number_to_words and test_ordinal_to_words."""
    return [n for fn in (test_text.test_number_to_words, test_text.test_ordinal_to_words)
            for mark in fn.pytestmark for n, _ in mark.args[1]]


WORDS = [
    "Ali", "ali", "İstanbul", "İSTANBUL", "istanbul", "Işık", "IŞIK", "ışık", "ıIiİ", "İıIi", "Ömer", "çiçek", "Çağrı",
    "ŞEKER", "ğ", "Ğ", "kitap", "yüzyıl", "Yüzyıl", "lig", "Lig", "Ankara", "ANKARA'DA", "yağmur", "Dünya", "Savaşı",
    "kâr", "Kâğıt", "hâlâ", "Âdem", "îman", "ÛMİT", "Î", "café", "Müller", "Straße", "niño", "côté", "àla", "Ärger",
    "Abdülhamid", "Selim'in", "Murat", "Kemal", "Ronaldo", "Ahmet", "Ayşe", "Yılmaz", "Üyesi", "Başkan", "sezonu",
    "elma", "armut", "şeyler", "gün", "saat", "fiyat", "nüfus", "indirim", "hız", "derece", "metre", "x", "X", "q", "w",
    "evet", "hayır", "Anadolu'ya", "Ali'nin", "2021'de", "kadar", "arası", "Mart", "ocak", "Ağustos'ta", "Türkiye'nin",
    "a", "e", "i", "o", "u", "İ", "I", "Q", "WhatsApp", "iPhone", "e-posta", "rock'n'roll", "o'nun", "ı", "ö", "ü",
]
OPENERS = ["Sonra", "Bu", "Ama", "Ve", "O", "Ben", "Bir", "Şimdi", "İşte", "Öyle", "Ki", "Bunu", "Daha"]
SUFFIXES = ["'da", "'de", "'ta", "'te", "'ya", "'ye", "'nin", "'nın", "'in", "'ı", "'i", "'u", "'ü", "'dan", "'den",
            "'lik", "'lık", "'deki", "'daki", "'si", "'sı", "'yi", "'yı", "'a", "'e", "'DA", "'DE", "'Nİn", "'IN",
            "'ler", "'lar", "'lı", "'li", "'ten", "'tan", "'ki", "'dı", "'di", "'tı", "'yle", "'la", "'le", "'ncı"]
SYMBOLS_ODD = [
    "😀", "❤\ufe0f", "👍🏽", "🇹🇷", "→", "©", "™", "#", "*", "_", "~", "^", "<", ">", "|", "\\", "{", "}", "[", "]",
    "(", ")", "§", "¶", "•", "·", "×", "÷", "±", "√", "∞", "π", "α", "ΑΣ", "Σ", "ж", "Ж", "الله", "中文", "日本",
    "한국", "ſ", "ﬁ",
    "\u200b", "\u200d", "\ufeff", "¿", "¡", "«", "»", "“", "”", "‘", "’", "ʼ", "`", "´", "…", "–", "—", "−", ";", ":",
    "&", "+", "=", "@", "%", "$", "€", "£", "₺", "°", "²", "³", "½", "Ⅻ", "١٢٣", "１２３", "𝟏𝟐", "٣:٤٥", "𝐀𝐁", "ǅ",
    "i\u0307", "I\u0307", "e\u0301", "s\u0327", "S\u0327", "g\u0306", "c\u0327", "o\u0308", "U\u0308", "\u0301",
    "a\u0302", "A\u0302", "I\u0302", "\u0338", "\u0345", "\u0334", "\u093f", "\ufe0f", "/", "\"", "'", "-", ".",
]
PUNCT = ["?!", "...", "!!!", ",,", " . ", "?.!", ";", ":", "…", ".", ",", "!", "?", " ?", "!.", ". .", ", ,", "'", "\""]
SPACES = [" "] * 12 + ["", "  ", "\t", "\n", "\u00a0", "\u2009", "\u3000", "\x1c", "\x85", "\u2028", " \n "]
WHITESPACE = [" ", "", "\t", "\n", "\r\n", "\v", "\f", "\x1c", "\x1f", "\x85", "\u00a0", "\u1680", "\u2009", "\u202f",
              "\u205f", "\u2028", "\u2029", "\u3000", "\ufeff", "\u180e", "\u200b", "\u200d"]
MARKS = ["\u0307", "\u0301", "\u0327", "\u0306", "\u0308", "\u0302", "\u0338", "\ufe0f", "\u200d", "\u0345", "\u0334",
         "\u0e31", "\u094d", "\u093f", "\u05b0", "\u0670", "\U0001d165", "\u0340", "\u0344", "\u0f73", "\u20dd"]


def stress_token(r: random.Random) -> str:
    def num(max_digits: int = 7) -> str:
        n = str(r.randrange(10 ** r.randint(1, max_digits)))
        return ("0" * r.randint(1, 3) + n) if r.random() < 0.08 else n

    def thousands() -> str:
        n = f"{r.randrange(1000, 10 ** r.randint(4, 16)):,}".replace(",", ".")
        return n + (f",{num(3)}" if r.random() < 0.4 else "")

    def suffix(p: float = 0.4) -> str:
        return r.choice(SUFFIXES) if r.random() < p else ""

    def scale() -> str:
        return r.choice(["bin", "milyon", "milyar", "trilyon"])

    kind = r.randrange(25)
    if kind == 0:
        return num(r.choice([3, 7, 16, 22]))
    if kind == 1:  # decimals with leading / trailing zeros, "," or "."
        frac = r.choice(["0" * r.randint(1, 4) + num(3), num(5), "0" * r.randint(1, 3), num(3) + "0" * r.randint(1, 2),
                         num(18)])
        return f"{num(4)}{r.choice(',.')}{frac}"
    if kind == 2:
        return thousands() + suffix()
    if kind == 3:  # negatives
        n = r.choice([num(4), thousands(), f"{num(2)},{num(2)}"])
        return r.choice([f"-{n}", f"−{n}", f"(-{n})", f"x=-{n}", f"a-{n}", f"{n}-a", f" -{n}", f"--{n}", f".-{n}"])
    if kind == 4:  # ranges
        a, b = num(4), num(4)
        return r.choice([f"{a}-{b}", f"{a}–{b}", f"{a} - {b}", f"{a}-{b}'de", f"{a}—{b}"])
    if kind == 5:  # dates and versions
        d, m, y = r.randint(1, 31), r.randint(1, 12), r.randint(1900, 2099)
        return r.choice([f"{d}.{m}.{y}", f"{d:02d}.{m:02d}.{y}", f"{d}/{m}/{y}", f"{d}.{m}.{y}'te", f"{y}.{m}",
                         f"v{num(1)}.{num(2)}.{num(2)}", f"{d},{m}.{y}", f"1.{num(5)}", f"{num(3)}.{num(3)}.{num(2)}"])
    if kind == 6:  # times
        h, mm = r.randint(0, 29), r.randint(0, 69)
        return r.choice([f"{h}:{mm:02d}", f"{h:02d}:{mm:02d}{suffix(0.6)}", f"{h}:{mm:02d}:{r.randint(0, 59):02d}",
                         f"{r.randint(0, 9)}:{r.randint(0, 9)}", f"{h}.{mm:02d}{suffix()}", f"{num(3)}:{mm:02d}",
                         f"{h}:{mm}", f"saat {h:02d}:{mm:02d}{suffix()}", f"{h:02d}:00{suffix()}"])
    if kind == 7:  # ordinals before lower-case, capitalised and opener words
        n = r.choice([num(1), num(2), num(3), num(4), num(6), num(7)])
        nxt = r.choice([r.choice(WORDS), r.choice(OPENERS), r.choice(OPENERS) + "ler", "yüzyıl", "Yüzyıl", "ÇAĞ", ""])
        return f"{n}.{r.choice([' ', '  ', '', chr(10)])}{nxt}"
    if kind == 8:  # Roman numerals
        roman = r.choice(["I", "II", "III", "IV", "V", "VI", "IX", "X", "XIV", "XV", "XIX", "XX", "XXXIX", "XL", "IIII",
                          "VX", "M", "C", "L", "MCM", "IIX", "XXXX", "Ⅻ"])
        nxt = r.choice([r.choice(WORDS), r.choice(OPENERS), "yüzyıl", "Dünya", ""])
        return f"{roman}{r.choice(['.', '. ', '.  ', ' ', ''])}{nxt}"
    if kind == 9:  # prefix currencies
        sym = r.choice("₺$€£")
        return r.choice([f"{sym}{num(4)}", f"{sym} {thousands()}", f"{sym}{num(2)},{num(1)} {scale()}",
                         f"{sym}{num(1)} binlik", f"{sym}'{r.choice(['a', 'ın', 'DAN'])}", f"{sym} bölgesi",
                         f"{sym}{suffix(0.8)}"])
    if kind == 10:  # units and currencies after a number or a scale word
        unit = r.choice([*_UNITS, "km/s", "km/sa", "km/h", "kmh", "kms", "mt", "Km", "KG", "cm3", "M", "tlx"])
        n = r.choice([num(3), f"{num(2)},{num(1)}", thousands(), f"{num(2)} {scale()}"])
        return f"{n}{r.choice([' ', '', '  '])}{unit}{suffix(0.5)}"
    if kind == 11:  # percentages
        n = r.choice([num(2), f"{num(2)},{num(1)}", f"{num(1)}.{num(1)}"])
        return r.choice([f"%{n}", f"% {n}", f"{n}%", f"{n} %", f"%{n}{suffix()}", f"yüzde {n}", f"{n}%'si", "%",
                         "%'si"])
    if kind == 12:  # titles and abbreviations
        return r.choice(["Dr.", "Prof.", "Doç.", "Yrd.", "Öğr.", "Av.", "Op.", "Uzm.", "Sn.", "DR.", "dr.", "Dr",
                         "vb.", "VB.", "Vb.", "vs.", "vd.", "örn.", "Örn.", "ÖRN.", "bkz.", "Bkz.", "No.", "No:", "no",
                         "No", "NO.", "xvb.", "vbx."]) + r.choice([" ", "", "  ", ". "]) + r.choice(
            [r.choice(WORDS), r.choice(OPENERS), num(2), "", "İstanbul", "ÖZ", "Ç", "Ahmet", "Şule", "Ümit", "Ilgaz"])
    if kind == 13:  # acronyms with suffixes
        acr = r.choice([*_SPELLED, "THY", "COVID", "FIFA", "NATO", "İTÜ", "ODTÜ", "IŞİD", "UEFA", "AB", "TC", "XY",
                        "ÇĞÖŞÜİ", "IIII", "Iİ", "ABDLER"])
        return acr + r.choice([suffix(0.7), "-19", "-" + num(2), "'ye", "'nin", "'DE", ""])
    if kind == 14:  # quotes
        w = r.choice(WORDS)
        q = r.choice([("'", "'"), ('"', '"'), ("“", "”"), ("«", "»"), ("‘", "’"), ("`", "´"), ("'", "'i"), ("„", "“")])
        return f"{q[0]}{w}{q[1]}"
    if kind == 15:
        return r.choice(SYMBOLS_ODD)
    if kind == 16:
        return r.choice(PUNCT)
    if kind == 17:  # apostrophe positions
        return r.choice(["'80'ler", "'", "x'", "'x", "5'i", "5'", "'5", "%'", "°'de", "$'a", "€'ya", "''", "' '",
                         "a''b", "TL'", "'TL", "km'", "2'nci", "₺'"]) + suffix(0.2)
    if kind == 18:  # glued number / letter combinations
        return r.choice([f"{num(2)}{r.choice(WORDS)}", f"{r.choice(WORDS)}{num(2)}", f"{num(2)}_{num(2)}",
                         f"{num(2)}.{r.choice(WORDS)}", f"{num(1)},{r.choice(WORDS)}", f"{num(3)}{suffix(1)}"])
    if kind == 19:  # Unicode digits
        return r.choice(["١٢٣", "٠,٠٥", "١٢:٣٠", "１２３", "１.２５０.０００", "𝟏𝟐", "𝟎𝟗:𝟎𝟓", "٣.", "²", "½",
                         "₂"]) + suffix(0.2)
    if kind == 20:
        return r.choice(WORDS) + suffix(0.3)
    if kind == 21:  # combining sequences inside words
        w = r.choice(WORDS)
        i = r.randrange(len(w) + 1)
        return w[:i] + r.choice(MARKS) + w[i:]
    if kind == 22:
        return r.choice(OPENERS)
    if kind == 23:  # rules around unusual whitespace (Python's \s differs from JS's) and non-ASCII digits
        ws = r.choice(WHITESPACE)
        n = r.choice([num(3), "١٢٣", "٣", "１２", "𝟓", f"{num(2)},{num(2)}"])
        return r.choice([f"{n}{ws}TL{suffix()}", f"{n}{ws}%", f"%{ws}{n}", f"Dr.{ws}Ahmet", f"No.{ws}{n}", f"No{ws}{n}",
                         f"{n}.{ws}yüzyıl", f"{n}.{ws}Dünya", f"XV.{ws}yüzyıl", f"₺{ws}{n}{ws}milyon", f"{n}{ws}km/s",
                         f"vb.{ws}Sonra", f"vs.{ws}", f"{n}:{n}", f"{n}{ws}bin{ws}TL", f"saat{ws}{n}:{n}", f"-{n}",
                         f"{ws}{n}{ws}", "x²'de", "Ⅻ'de", "5 TL'²", "5 kg'½", "5 m'ǅ", f"{n}'ǅa", "5 TL'²e",
                         "5 m'½i", "$'²e", "€'Ⅻü", f"{n} kg'³ye"])
    return r.choice(WORDS).upper() + suffix(0.3)


def stress_string(r: random.Random) -> str:
    n_tokens = r.randint(20, 45) if r.random() < 0.04 else r.choice([r.randint(1, 4), r.randint(2, 10)])
    out = r.choice(["", "", "", " ", "  ", "\t", "\n", "...", "-", "'"])
    for i in range(n_tokens):
        tok = stress_token(r)
        if r.random() < 0.05:
            tok = r.choice([str.upper, str.lower, str.title, str.swapcase])(tok)
        out += tok + (r.choice(SPACES) if i < n_tokens - 1 else "")
        if r.random() < 0.15:
            out += r.choice(PUNCT) + r.choice(SPACES)
    return out + r.choice(["", "", ".", "!", "?", "?!", "...", " ", "  ", "\n", ". ", "'"])


def case(source: str, raw: str) -> dict:
    norm = normalize(raw)
    return {
        "source": source,
        "input": raw,
        "normalized": norm,
        "ids": text_to_ids(raw),
        "split": split_sentences(norm),
        "split_short": split_sentences(norm, 40),
        "split_raw": split_sentences(raw, 30),
    }


def main() -> None:
    r = random.Random(0)
    numbers = unit_test_numbers() + [r.randrange(10 ** r.randint(1, 30)) for _ in range(300)]
    numbers += [10**k for k in range(31)]
    inputs = [("test_text", s) for s in unit_test_strings()]
    inputs += [("freya", t["text"]) for t in load_texts(FREYA)]
    inputs += [("stress", stress_string(r)) for _ in range(N_STRESS)]
    cases = [case(source, raw) for source, raw in inputs]
    head = {
        "sources": f"tests/test_text.py; {FREYA} (CC-BY-4.0); random stress strings (seed 0)",
        "symbols": SYMBOLS,
        "symbol_to_id": SYMBOL_TO_ID,
        "pad_id": PAD_ID,
        "blank_id": BLANK_ID,
        "number_to_words": [[str(n), number_to_words(n)] for n in numbers],
        "ordinal_to_words": [[str(n), ordinal_to_words(n)] for n in numbers if n > 0],
    }
    dumps = lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    lines = ",\n".join(map(dumps, cases))  # one case per line
    body = dumps(head)[:-1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f'{body},"cases":[\n{lines}\n]}}\n', encoding="utf-8")
    counts = {s: sum(src == s for src, _ in inputs) for s in ("test_text", "freya", "stress")}
    print(f"wrote {len(cases)} cases {counts} to {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

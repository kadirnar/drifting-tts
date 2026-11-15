"""Turkish text frontend: normalisation and a character vocabulary; :func:`frontend` gives a language's frontend
(``tr``: this module, ``en``: :mod:`drifting_tts.text_en`).

Turkish orthography is close to phonemic, so characters are used directly as input symbols
(no G2P). Normalisation lower-cases with Turkish rules, verbalises numbers, clock times, ordinals, fractions, phone
numbers, units, currencies, common abbreviations, initials, e-mail and web addresses, and drops symbols that are not
pronounced. It leaves a ``.`` only where a sentence ends (:func:`split_sentences` splits there): the dots of
abbreviations, dotted acronyms and initials are resolved here.
"""

from __future__ import annotations

import re
import unicodedata

PAD, BLANK = "<pad>", "<blank>"
LETTERS = "abcçdefgğhıijklmnoöpqrsştuüvwxyz"
PUNCTUATION = " .,!?"
SYMBOLS = [PAD, BLANK, *LETTERS, *PUNCTUATION]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
PAD_ID, BLANK_ID = SYMBOL_TO_ID[PAD], SYMBOL_TO_ID[BLANK]

_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_SCALES = [(10**12, "trilyon"), (10**9, "milyar"), (10**6, "milyon"), (10**3, "bin")]
_DIGIT_NAMES = ["sıfır", *_ONES[1:]]

_CHAR_MAP = {
    "â": "a", "î": "i", "û": "u", "Â": "A", "Î": "İ", "Û": "U",
    "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'", "“": '"', "”": '"', "«": '"', "»": '"',
    "…": "...", "–": "-", "—": "-", "−": "-", ";": ",",
    "&": " ve ", "+": " artı ", "=": " eşittir ", "@": " et ",
    "é": "e", "è": "e", "à": "a", "ä": "a", "ñ": "n", "ß": "ss", "ô": "o",
}
_UPPER, _LOWER = "A-ZÇĞİÖŞÜ", "a-zçğıöşü"
_NUMBER = r"\d+(?:[.,]\d+)*"
_SCALE_WORD = r"\b(?:bin|milyon|milyar|trilyon)\b"

# Units and currencies, expanded only after a number ("250 TL", "5 bin km"); "₺250" is first rewritten to "250 ₺"
_UNITS = {
    "₺": "lira", "TL": "lira", "tl": "lira", "$": "dolar", "USD": "dolar", "€": "avro", "EUR": "avro", "£": "sterlin",
    "km²": "kilometrekare", "km2": "kilometrekare", "m²": "metrekare", "m2": "metrekare", "m³": "metreküp",
    "m3": "metreküp", "km": "kilometre", "m": "metre", "cm": "santimetre", "mm": "milimetre", "kg": "kilogram",
    "g": "gram", "gr": "gram", "mg": "miligram", "lt": "litre", "ml": "mililitre", "sn": "saniye", "dk": "dakika",
    "°C": "derece", "°": "derece",
}
_UNIT = re.compile(rf"(\d|{_SCALE_WORD})\s*({'|'.join(map(re.escape, sorted(_UNITS, key=len, reverse=True)))})"
                   r"(?!\w)(?:'([^\W\d_]+))?")
_PREFIX_CURRENCY = re.compile(rf"([₺$€£])\s*({_NUMBER}(?:\s+{_SCALE_WORD})?)")
_SPEED = re.compile(rf"({_NUMBER})\s*km/(?:sa|s|h)(?!\w)")
_TIME = re.compile(r"(?<![\d.,:])(\d{1,2}):(\d{2})(?![\d:])")

# Titles are expanded only before a capitalised name ("Av. Ali" but not "Av. sezonu")
_TITLES = {"Dr": "doktor", "Prof": "profesör", "Doç": "doçent", "Yrd": "yardımcı", "Öğr": "öğretim", "Av": "avukat",
           "Op": "operatör", "Uzm": "uzman", "Sn": "sayın", "Hz": "hazreti"}
_TITLE = re.compile(rf"(?<!\w)({'|'.join(_TITLES)})\.(?=\s*[{_UPPER}])")
_ABBREVIATIONS = {"vb": "ve benzeri", "vs": "vesaire", "vd": "ve diğerleri", "örn": "örneğin", "bkz": "bakınız"}
_ABBREVIATION = re.compile(rf"(?<!\w)((?i:{'|'.join(_ABBREVIATIONS)}))\.")  # also "Örn.", all-caps "VB."
_SENTENCE_START = re.compile(rf"\s*(?:$|[{_UPPER}](?![{_UPPER}]))")  # end of text or a capitalised (not all-caps) word

# Acronyms read letter by letter (K is "ka" as in PKK, KDV); other all-caps words are just lower-cased (NATO, ODTÜ)
_LETTER_NAMES = dict(zip("ABCÇDEFGHIİJKLMNOÖPRSŞTUÜVYZ",
                         "a be ce çe de e fe ge he ı i je ka le me ne o ö pe re se şe te u ü ve ye ze".split()))
_SPELLED = "AB ABD AİHM AKP AVM BM CHP DSÖ HDP İBB KDV KKTC MHP PKK PTT SGK TBMM TC TL TRT TSK".split()
_ACRONYMS = {a: " ".join(_LETTER_NAMES[c] for c in a) for a in _SPELLED}
_ACRONYMS |= {"THY": "te ha ye", "COVID": "kovid", "FIFA": "fifa"}

# "N." before a capitalised word is ambiguous: "1. Dünya Savaşı" (ordinal) vs "Sayı 5. Sonra ..." (sentence end).
# A 1-3 digit number (or a Roman numeral) is read as an ordinal unless the next word is one of these sentence
# openers, closed-class words that never follow an ordinal. Before a lower-case word it is always an ordinal.
_OPENERS = (
    "Ama Ancak Ardından Artık Aslında Ayrıca Bazen Belki Ben Bence Bir Biz Böyle Böylece Bu Buna Bunda Bundan Bunlar "
    "Bunu Bunun Burada Çünkü Da Daha De Dolayısıyla Elbette Evet Fakat Halbuki Hatta Hayır Hem Hemen Her Herkes Hiç "
    "İşte Kim Ki Mesela Ne Neden Nasıl Niye O Ona Onda Ondan Onlar Onu Onun Orada Oysa Öyle Örneğin Peki Sadece Sen "
    "Siz Son Sonra Şimdi Şu Tabii Ve Veya Ya Yani Yine Zaten"
).split()
_TITLE_WORD = rf"\s+(?!(?:{'|'.join(_OPENERS)})(?!\w))[{_UPPER}]"
_ORDINAL = re.compile(rf"(?<![\d.,])(?:(\d{{1,6}})\.(?=\s+[{_LOWER}])|(\d{{1,3}})\.(?={_TITLE_WORD}))")
# Regnal numbers and centuries (II. Abdülhamid, XV. yüzyıl): only I, V, X (1-39), so initials such as "M. Kemal" or
# "C. Ronaldo" are left alone
_ROMAN_ORDINAL = re.compile(rf"(?<![\w.])(?=[IVX])(X{{0,3}}(?:IX|IV|V?I{{0,3}}))\.(?=\s+[{_LOWER}]|{_TITLE_WORD})")

# The dot of an abbreviation is not a sentence end ("T.C. Sağlık Bakanlığı", "Gül Sok. No: 5", "M. Kemal"), unless the
# text ends there or a sentence opener follows ("... A.Ş. Sonra kapandı.")
_STOP = re.compile(rf"\s*$|\s+(?:{'|'.join(_OPENERS)})(?!\w)")
# Dotted letter sequences are read letter by letter (T.C., A.B.D., t.c.); a few stand for words
_DOTTED = re.compile(rf"(?<![\w.])(?:[{_UPPER}{_LOWER}]\.){{2,}}")
_DOTTED_WORDS = {"A.Ş.": "anonim şirketi"}
# Initials before a capitalised name (M. Kemal, A. O. Yılmaz) are read by their letter names; before a sentence opener
# the dot ends the sentence ("Plan B. Sonra gittik."). A lone I, V or X stays a regnal number (_ROMAN_ORDINAL)
_INITIALS = re.compile(rf"(?<![\w.-])[{_UPPER}]\.(?:\s+[{_UPPER}]\.){{0,5}}(?={_TITLE_WORD})")
# ... but a lone letter after one of these words names a thing ("Hepatit B. Aşısı var.", "Grup A. Türkiye lider.")
_LETTERED = re.compile(r"(?<!\w)(?:grup|hepatit|plan|tip|seçenek|şık|blok|kapı|sınıf|şube|kategori|model|madde|bölüm"
                       r"|vitamin)\s+$")
# Abbreviations that follow a name, in addresses and company names (Gül Sok., Atatürk Cad., Yılmaz Ltd. Şti.)
_AFTER_NAME = {"Mah": "mahallesi", "Cad": "caddesi", "Sok": "sokak", "Sk": "sokak", "Apt": "apartmanı",
               "Bul": "bulvarı", "Blv": "bulvarı", "Ltd": "limited", "Şti": "şirketi", "Tic": "ticaret",
               "San": "sanayi"}
_AFTER_NAME_ABBREVIATION = re.compile(rf"(?<=[\w.] )({'|'.join(_AFTER_NAME)})\.")
_TELEPHONE = re.compile(r"(?<!\w)(?:Tel[.:]|tel:)\s*(?=[+(\d])")  # "ince bir tel. 5 metre" is a wire
# Phone numbers, read group by group ("0532 123 45 67": sıfır beş yüz otuz iki, yüz yirmi üç, kırk beş, altmış yedi):
# with a country code, a trunk zero or an area code in brackets; or a mobile number in its four spaced groups
_PHONE = re.compile(r"(?<![\w.,+/-])(?:(?:(?:\+90|0090)[ -]?(?:\([0-9]{3}\)|[0-9]{3})|0 ?(?:\([0-9]{3}\)|[0-9]{3})"
                    r"|\(0[0-9]{3}\)|\([0-9]{3}\))[ -]?[0-9]{3}[ -]?[0-9]{2}[ -]?[0-9]{2}"
                    r"|5[0-9]{2}[ -][0-9]{3}[ -][0-9]{2}[ -][0-9]{2})(?![0-9])(?![.,][0-9])")
# Fractions ("3/4'ü" -> dörtte üçü): a numerator below a denominator of 2-10 or 100; not a date or a door number
_FRACTION = re.compile(r"(?<![\w/.,-])(?<!numara )(\d{1,2})/(\d{1,3})(?![\w/])(?![.,]\d)(?:'([^\W\d_]+))?")
# E-mail and web addresses are spelled out ("nokta", "et"); a bare domain needs one of these endings
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_DOMAIN_ENDINGS = "com|net|org|edu|gov|tr|io|info|app|dev"
_URL = re.compile(r"(?<![\w@./-])(?:https?://[A-Za-z0-9.-]+|www\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
                  rf"|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:{_DOMAIN_ENDINGS})(?![A-Za-z0-9-]))(?:/[^\s]*)?")
_ADDRESS_WORDS = {"www": "ve ve ve", "com": "kom", "tr": "te re"}
_ADDRESS_CHARS = {".": " nokta ", "/": " bölü ", "-": " tire ", "_": " alt çizgi ", "@": " et ", ":": " "}
_TRAILING = ".,!?;:)]}'\""
_APOSTROPHES = ("'", "’", "ʼ")  # before _CHAR_MAP


def _below_thousand(n: int) -> list[str]:
    words = []
    h, rest = divmod(n, 100)
    if h:
        words += ([] if h == 1 else [_ONES[h]]) + ["yüz"]
    t, o = divmod(rest, 10)
    if t:
        words.append(_TENS[t])
    if o:
        words.append(_ONES[o])
    return words


def number_to_words(n: int) -> str:
    """Verbalise a non-negative integer in Turkish (``1100 -> "bin yüz"``)."""
    if n == 0:
        return "sıfır"
    words: list[str] = []
    for value, name in _SCALES:
        q, n = divmod(n, value)
        if q:
            # Turkish says "bin" (not "bir bin") but "bir milyon"
            words += ([] if (q == 1 and name == "bin") else _below_thousand(q) if q < 1000
                      else number_to_words(q).split()) + [name]
    words += _below_thousand(n)
    return " ".join(words)


def _digits_to_words(s: str) -> str:
    return " ".join(_DIGIT_NAMES[int(c)] for c in s)


_VOWELS = "aeıioöuü"
_HARMONY = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}


def ordinal_to_words(n: int) -> str:
    """Turkish ordinal (``14 -> "on dördüncü"``) using four-way vowel harmony."""
    *head, last = number_to_words(n).split()
    v = _HARMONY[[c for c in last if c in _VOWELS][-1]]
    if last[-1] in _VOWELS:
        last = last + "nc" + v
    else:
        last = ("dörd" if last == "dört" else last) + v + "nc" + v
    return " ".join([*head, last])


def _roman_to_int(s: str) -> int:
    values = [{"I": 1, "V": 5, "X": 10}[c] for c in s]
    return sum(-v if v < nxt else v for v, nxt in zip(values, [*values[1:], 0]))


def _attach(word: str, suffix: str | None) -> str:
    """Re-harmonise a suffix written for an abbreviation onto its spoken form (``lira + ye -> liraya``)."""
    if not suffix:
        return word
    suffix = turkish_lower(suffix)
    if word[-1] not in _VOWELS and len(suffix) > 1 and suffix[0] in "nsy" and suffix[1] in _VOWELS:
        suffix = suffix[1:]  # buffer consonants only follow a vowel: kg'ye -> kilograma
    if suffix[0] in "dt":
        suffix = ("t" if word[-1] in "çfhkpsşt" else "d") + suffix[1:]
    for c in suffix:
        last = next(v for v in reversed(word) if v in _VOWELS)
        if c in "ae":
            c = "a" if last in "aıou" else "e"
        elif c in "ıiuü" and not word.endswith("k"):  # "-ki" does not harmonise: TL'deki -> liradaki
            c = _HARMONY[last]
        word += c
    return word


def _verbalise_time(match: re.Match) -> str:
    """Clock time: ``14:30 -> "on dört otuz"``, ``10:00 -> "on"``, ``09:05 -> "dokuz sıfır beş"``."""
    hour, minute = int(match.group(1)), match.group(2)
    words = number_to_words(hour)
    if minute != "00":
        words += " " + (_digits_to_words(minute) if minute[0] == "0" else number_to_words(int(minute)))
    elif hour == 0:
        words += " sıfır"
    return f" {words} "


def _verbalise_number(match: re.Match) -> str:
    raw = match.group(0)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", raw):  # 1.250.000(,5): dots are thousands separators
        raw = raw.replace(".", "")
    if re.fullmatch(r"\d+(,\d+)?", raw):
        integer, _, fraction = raw.partition(",")
        text = number_to_words(int(integer)) if len(integer) <= 15 else _digits_to_words(integer)
        if fraction:
            lead = len(fraction) - len(fraction.lstrip("0"))
            frac = ["sıfır"] * lead + ([number_to_words(int(fraction))] if fraction.strip("0") else [])
            text += " virgül " + " ".join(frac or ["sıfır"])
    else:  # dates, versions: read every group on its own (28.10.2014, 14.30)
        text = " ".join(number_to_words(int(g)) if len(g) <= 15 else _digits_to_words(g)
                        for g in re.split(r"[.,]", raw))
    return f" {text} "


def _stop(match: re.Match) -> str:
    """"." when the abbreviation that ``match`` covers also ends the sentence (:data:`_STOP`), else ""."""
    return "." if _STOP.match(match.string, match.end()) else ""


def _letter_name(letter: str) -> str:
    return _LETTER_NAMES.get(letter.replace("i", "İ").upper(), letter)


def _glue(match: re.Match) -> str:
    """"" when a suffix follows ``match`` (``T.C.'nin``: the apostrophe must stay next to a letter), else " "."""
    return "" if match.string.startswith(_APOSTROPHES, match.end()) else " "


def _spell_dotted(match: re.Match) -> str:
    """``T.C.`` -> "te ce" (a suffix stays attached: ``T.C.'nin`` -> "te cenin")."""
    word = _DOTTED_WORDS.get(match.group()) or " ".join(_letter_name(c) for c in match.group() if c != ".")
    return f" {word}{_stop(match)}{_glue(match)}"


def _spell_initials(match: re.Match) -> str:
    letters = re.findall(rf"[{_UPPER}]", match.group())
    if len(letters) == 1 and (letters[0] in "IVX" or _LETTERED.search(turkish_lower(match.string[:match.start()]))):
        return match.group()
    return " ".join(_letter_name(c) for c in letters)


def _digit_group(group: str) -> str:
    """A group of a phone number: ``532`` -> "beş yüz otuz iki", ``05`` -> "sıfır beş"."""
    rest = group.lstrip("0")
    return " ".join(["sıfır"] * (len(group) - len(rest)) + ([number_to_words(int(rest))] if rest else []))


def _verbalise_phone(match: re.Match) -> str:
    digits = "".join(c for c in match.group() if c in "0123456789")
    parts = []
    if match.group().startswith("+") or digits.startswith("0090"):
        parts.append("artı doksan")
        digits = digits[4 if digits.startswith("0090") else 2:]
    trunk, digits = ("sıfır ", digits[1:]) if len(digits) == 11 else ("", digits)
    parts += [trunk + _digit_group(digits[:3]), *(_digit_group(digits[a:b]) for a, b in ((3, 6), (6, 8), (8, 10)))]
    return f" {', '.join(parts)}{_glue(match)}"


def _verbalise_fraction(match: re.Match) -> str:
    numerator, denominator = int(match.group(1)), int(match.group(2))
    if not (0 < numerator < denominator and (denominator <= 10 or denominator == 100)):
        return match.group()
    return f" {_attach(number_to_words(denominator), 'de')} {_attach(number_to_words(numerator), match.group(3))} "


def _speak_address(address: str) -> str:
    """An e-mail or web address without its scheme, query and fragment: dots, slashes and dashes by name."""
    address = re.split("[?#]", re.sub("^https?://", "", address), maxsplit=1)[0].rstrip("/")
    host, slash, path = address.partition("/")
    local, at, domain = host.rpartition("@")
    domain = ".".join(_ADDRESS_WORDS.get(label, label) for label in domain.split("."))
    return "".join(_ADDRESS_CHARS.get(c, c) for c in local + at + domain + slash + path)


def _verbalise_address(match: re.Match) -> str:
    address = match.group().rstrip(_TRAILING)  # the punctuation after it belongs to the sentence
    tail = match.group()[len(address):]
    return f" {_speak_address(address)}{tail or _glue(match)}{' ' if tail else ''}"


def _expand_abbreviation(match: re.Match) -> str:
    abbreviation = match.group(1).lower()
    word = _ABBREVIATIONS[abbreviation]
    # "vb.", "vs.", "vd." close a list, so their dot may also end the sentence ("elma vb. Sonra ...")
    ends_list = abbreviation in ("vb", "vs", "vd") and _SENTENCE_START.match(match.string, match.end())
    return word + "." if ends_list else word


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize(text: str) -> str:
    """Normalise raw Turkish text to the symbol alphabet (lower-case letters, space and ``.,!?``)."""
    # drop combining marks left after NFC, e.g. the U+0307 in "i̇" that Python's "İ".lower() produces
    text = "".join(c for c in unicodedata.normalize("NFC", text) if not unicodedata.combining(c))
    text = _EMAIL.sub(_verbalise_address, text)  # before "@" and "+" are mapped
    text = _URL.sub(_verbalise_address, text)
    text = _TELEPHONE.sub("telefon ", text)
    text = _PHONE.sub(_verbalise_phone, text)
    text = "".join(_CHAR_MAP.get(c, c) for c in text)
    text = _DOTTED.sub(_spell_dotted, text)  # before the apostrophes: a suffix stays on the last letter's name
    text = _AFTER_NAME_ABBREVIATION.sub(lambda m: f"{_AFTER_NAME[m.group(1)]}{_stop(m)}", text)
    # an apostrophe between a letter/digit/unit and a letter starts a suffix (Anadolu'ya, 2021'de, $'a); others are
    # quote marks ('evet') and become spaces
    text = re.sub(r"(?<![^\W_])(?<![%$₺€£°])'|'(?![^\W\d_])", " ", text)
    text = _TITLE.sub(lambda m: f"{_TITLES[m.group(1)]} ", text)
    text = _ABBREVIATION.sub(_expand_abbreviation, text)
    text = re.sub(r"\b[Nn]o[.:]?\s*(?=\d)", "numara ", text)
    text = _FRACTION.sub(_verbalise_fraction, text)
    text = _TIME.sub(_verbalise_time, text)  # before ":" becomes ","
    text = re.sub(r"(?<=\d):(?=\d)", " ", text).replace(":", ",")  # 3:1 -> "üç bir"
    text = _SPEED.sub(r"saatte \1 kilometre", text)
    text = _PREFIX_CURRENCY.sub(r"\2 \1", text)
    text = _UNIT.sub(lambda m: f"{m.group(1)} {_attach(_UNITS[m.group(2)], m.group(3))}", text)
    text = re.sub(r"([₺$€£])(?:'([^\W\d_]+))?", lambda m: f" {_attach(_UNITS[m.group(1)], m.group(2))} ", text)
    text = re.sub(r"%\s*(\d)", r" yüzde \1", text)
    text = re.sub(rf"({_NUMBER})\s*%", r" yüzde \1", text)
    text = re.sub(rf"(?<!\w)[{_UPPER}]{{2,}}(?!\w)", lambda m: _ACRONYMS.get(m.group(), m.group()), text)
    text = re.sub(r"(?<![\w.,])-(?=\d)", " eksi ", text)  # a minus starts a word; 3-5 stays a range
    text = _INITIALS.sub(_spell_initials, text)
    text = _ROMAN_ORDINAL.sub(lambda m: f" {ordinal_to_words(_roman_to_int(m.group(1)))} ", text)
    text = _ORDINAL.sub(lambda m: f" {ordinal_to_words(int(m.group(1) or m.group(2)))} ", text)
    text = re.sub(_NUMBER, _verbalise_number, text)
    text = re.sub(r"\s*'", "", text)  # glue suffixes: "iki bin yirmi bir 'de" -> "iki bin yirmi birde"
    text = turkish_lower(text)
    text = text.replace('"', " ").replace("-", " ").replace("/", " ")
    text = "".join(c if c in SYMBOL_TO_ID else " " for c in text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # no space before punctuation
    text = re.sub(r"([.,!?])(?=[^\s.,!?])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[.,!? ]+", "", text)
    return text


def text_to_ids(text: str, intersperse_blank: bool = True, normalized: bool = False) -> list[int]:
    """Map text to symbol ids; optionally intersperse a blank token (Grad-TTS) for alignment."""
    if not normalized:
        text = normalize(text)
    ids = [SYMBOL_TO_ID[c] for c in text]
    if intersperse_blank:
        out = [BLANK_ID] * (2 * len(ids) + 1)
        out[1::2] = ids
        ids = out
    return ids


def ids_to_text(ids: list[int]) -> str:
    return "".join(SYMBOLS[i] for i in ids if i not in (PAD_ID, BLANK_ID))


def group_sentences(sentences: list[str], max_chars: int = 180) -> list[str]:
    """Paragraph mode: consecutive sentences joined (with a space) into units of at most ``max_chars`` characters, each
    synthesised in one pass. The model is trained on utterances of up to 16 s, most of several sentences, so in one
    pass it renders the gaps and the intonation across the sentences' boundaries itself. A sentence longer than
    ``max_chars`` stays alone."""
    out: list[str] = []
    for s in sentences:
        if out and len(out[-1]) + 1 + len(s) <= max_chars:
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out


def split_sentences(text: str, max_chars: int = 180) -> list[str]:
    """Split normalised text into sentences (the generator is trained on <= 16 s utterances)."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    out: list[str] = []
    for s in parts:
        while len(s) > max_chars:  # very long sentences: split at the last comma / space before the limit
            cut = max(s.rfind(",", 0, max_chars), s.rfind(" ", 0, max_chars))
            cut = cut if cut > 0 else max_chars
            out.append(s[: cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out


LANGUAGES = ("tr", "en")


def frontend(language: str | None = "tr"):
    """The text frontend of ``language`` (a model's ``model.text.language``, a dataset's ``stats.json``; default
    Turkish): a module with ``SYMBOLS``, ``normalize`` and ``text_to_ids``. ``PAD_ID``, ``BLANK_ID`` and the sentence
    helpers are the same for every language."""
    if language in (None, "tr"):
        import sys

        return sys.modules[__name__]
    if language == "en":
        from . import text_en

        return text_en
    raise ValueError(f"unknown language {language!r}; choose one of {', '.join(LANGUAGES)}")

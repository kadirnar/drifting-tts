"""English text frontend: normalisation and a character vocabulary (``language: en``).

English spelling is far from phonemic, but with thousands of hours of speech the text encoder learns the letter-to-
sound mapping itself, as character-input TTS models do, so characters are the input symbols (no G2P dependency). The
alphabet is the 26 letters, the apostrophe (``don't``, ``we'll``: it changes the pronunciation) and the Turkish
frontend's punctuation; ``<pad>`` and ``<blank>`` keep their ids, so ``PAD_ID`` / ``BLANK_ID`` are shared.

Normalisation lower-cases and writes out what is not spelled as it is read: numbers (cardinals, decimals, years,
decades, ordinals, fractions, dates, phone numbers, Roman numerals), money, percentages, units, clock times, common
abbreviations, initials, e-mail and web addresses and all-caps acronyms (letter by letter). It leaves a ``.`` only
where a sentence ends (:func:`split_sentences` splits there): the dots of abbreviations, dotted acronyms and initials
are resolved here.
"""

from __future__ import annotations

import re
import unicodedata

from .text import BLANK, PAD, PUNCTUATION, group_sentences, split_sentences  # noqa: F401 (re-exported)

LETTERS = "abcdefghijklmnopqrstuvwxyz'"
SYMBOLS = [PAD, BLANK, *LETTERS, *PUNCTUATION]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
PAD_ID, BLANK_ID = SYMBOL_TO_ID[PAD], SYMBOL_TO_ID[BLANK]

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")]
_ORDINAL_WORDS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                  "nine": "ninth", "twelve": "twelfth"}
_LETTER_NAMES = dict(zip("abcdefghijklmnopqrstuvwxyz",
                         "ay bee see dee ee eff gee aitch eye jay kay el em en oh pee cue ar ess tee you vee "
                         "double-you ex why zee".split()))
# all-caps words spelled letter by letter: those without a vowel (TV, HR, NFL) and these; other all-caps words are
# read as words (emphasis: "I was SO tired"; NASA, NATO)
_SPELLED = {"USA", "UK", "EU", "UN", "AI", "CEO", "CFO", "CIA", "FBI", "DNA", "UFO", "IOU", "ATM", "BMW", "HIV", "IQ",
            "ID", "UI", "API", "URL", "USB", "GPU", "CPU", "OS", "EA", "ESPN", "MIT", "UCLA", "ICU", "IRS", "ER",
            "PE", "AA", "BA", "MA", "VIP", "OJ"}
_INTERJECTIONS = {"HMM", "HM", "MM", "MMM", "SHH", "SH", "PSST", "BRR", "GRR", "ZZZ", "PFFT", "TSK", "NTH", "PST"}
_SPECIAL = {"OK": "okay", "Ok": "okay", "BBQ": "barbecue", "Mr": "mister", "Mrs": "missus", "Ms": "miz",
            "Dr": "doctor", "Jr": "junior", "Sr": "senior", "St": "saint", "Prof": "professor"}
_ABBREVIATIONS = {"vs": "versus", "etc": "et cetera", "approx": "approximately", "e.g": "for example",
                  "i.e": "that is"}
_CHAR_MAP = {
    "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'", "“": '"', "”": '"', "«": '"', "»": '"',
    "…": "...", "—": ", ", "–": "-", "−": "-", ";": ",",
    "&": " and ", "+": " plus ", "=": " equals ", "@": " at ", "#": " number ",
}
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}
_NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"
_SCALE_WORD = r"(?:thousand|million|billion|trillion|k|m|bn)\b"
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])(\s*(?:[ap]\.?m\b\.?))?", re.I)
_AMPM = re.compile(r"(?<=\d)\s*([ap])\.?m\b\.?", re.I)
_HOUR_WORD = r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|o'clock)"
_AMPM_WORDS = re.compile(rf"({_HOUR_WORD})\s+([AaPp])\.?[Mm]\b\.?")  # "three AM", "nine p.m."

# The dot of an abbreviation is not a sentence end ("the U.S. economy", "Apple Inc. CEO Tim Cook", "J. K. Rowling"),
# unless the text ends there or one of these sentence openers follows ("... in the U.S. The economy grew.")
_OPENERS = (
    "A After All Also Although An And Another Are As At Because Before Both But Can Could Did Do Does Each Even Every "
    "Finally For From He Her Here His How However I If In Instead Is It Its Just Later Let Many Maybe Meanwhile Most "
    "My Next No Not Now Oh On Once One Only Or Other Our Perhaps Please She Should Since So Some Still Thank Thanks "
    "That The Their Then There These They This Those Though To Today Tomorrow Was We Well Were What When Where Which "
    "While Who Why Will With Would Yes Yesterday Yet You Your"
).split()
_OPENER = rf"(?:{'|'.join(_OPENERS)})\b"
_STOP = re.compile(rf"\s*$|\s+{_OPENER}")
_NAME = rf"\s+(?!{_OPENER})[A-Z]"  # a capitalised word that does not open a sentence follows
# Titles before a name and abbreviations after one; "St." is a saint before a name and a street after one
_TITLES = {"Mt": "mount", "Gen": "general", "Gov": "governor", "Sen": "senator", "Rep": "representative",
           "Capt": "captain", "Lt": "lieutenant", "Col": "colonel", "Sgt": "sergeant", "Rev": "reverend",
           "Pres": "president"}
_TITLE = re.compile(rf"\b({'|'.join(_TITLES)})\.(?={_NAME})")
_AFTER_NAME = {"Jr": "junior", "Sr": "senior", "Inc": "incorporated", "Ltd": "limited", "Co": "company",
               "Corp": "corporation", "Bros": "brothers", "Ave": "avenue", "Blvd": "boulevard", "Rd": "road"}
_AFTER_NAME_ABBREVIATION = re.compile(rf"\b({'|'.join(_AFTER_NAME)})\.")
_SAINT = re.compile(rf"(?<!(?:st|nd|rd|th) )\bSt\.(?={_NAME})")
_STREET = re.compile(r"\b([A-Z][a-z]+|\d+(?:st|nd|rd|th))(\s+)St\.")
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_MONTH_ABBREVIATIONS = {m[:3].capitalize(): m for m in _MONTHS if m != "may"} | {"Sept": "september"}
_MONTH = re.compile(rf"\b({'|'.join(sorted(_MONTH_ABBREVIATIONS, key=len, reverse=True))})\.(?=\s*\d)")
# Dotted letter sequences are read letter by letter (U.S., J.K., a.m.); initials before a capitalised name too
# (J. K. Rowling). "So did I. Mary left." keeps its dot: a lone "I." is the pronoun
_DOTTED = re.compile(r"(?<![\w.])(?:[A-Za-z]\.){2,}")
_INITIALS = re.compile(rf"(?<![\w.'-])[A-Z]\.(?:\s+[A-Z]\.){{0,5}}(?={_NAME})")
# ... but a lone letter after one of these words names a thing ("vitamin C. Take it daily.", "an A. Mom was happy.")
_LETTERED = re.compile(r"\b(?:a|an|the|vitamin|vitamins|plan|grade|hepatitis|type|model|exhibit|option|group|class|"
                       r"level|team|section|part|phase|point|figure|appendix|schedule|series|terminal|gate|platform|"
                       r"row|block|unit|ward|wing|room|letter|category|tier|division|side|item|answer|choice|column|"
                       r"zone|building|suite|lot)\s+$", re.I)
_BEFORE_NUMBER = {"fig": "figure", "vol": "volume", "ch": "chapter", "sec": "section", "eq": "equation"}
_BEFORE_NUMBER |= {k.capitalize(): v for k, v in _BEFORE_NUMBER.items()} | {"pp": "pages", "p": "page"}  # not "P.808"
_BEFORE_NUMBER_ABBREVIATION = re.compile(rf"\b({'|'.join(_BEFORE_NUMBER)})\.\s*(?=\d)")

# Phone numbers digit by digit, group by group: (555) 123-4567, 555-123-4567, 1-800-555-1234, +1 555 123 4567. A
# seven-digit number is one only after a word such as "call" ("pages 100-2000" is a range)
_PHONE = re.compile(r"(?<![\w.,+/-])(?:\+1[ .-]?(?:\([0-9]{3}\) ?|[0-9]{3}[ .-]?)[0-9]{3}[ .-]?[0-9]{4}"
                    r"|(?:1[ .-])?(?:\([0-9]{3}\) ?|[0-9]{3}[.-])[0-9]{3}[.-][0-9]{4})(?![\w-])(?![.,][0-9])")
_LOCAL_PHONE = re.compile(r"\b((?:call|dial|phone|text|fax|tel|telephone|cell|mobile|number)[:.]?\s+)"
                          r"([0-9]{3}-[0-9]{4})(?![\w-])", re.I)
# Dates: month/day/year (day/month/year when only that is possible) and ISO year-month-day
_DATE = re.compile(r"(?<![\w/.-])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\w/])")
_ISO_DATE = re.compile(r"(?<![\w/.-])(\d{4})-(\d{2})-(\d{2})(?![\w-])")
# Fractions: a numerator below one of these denominators ("9/11" and "24/7" are not fractions), "1 1/2"
_FRACTION = re.compile(r"(?<![\w/.,-])(?:(\d{1,2}) )?(\d{1,2})/(\d{1,3})(?![\w/])(?![.,]\d)")
_DENOMINATORS = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
                 10: "tenth", 12: "twelfth", 16: "sixteenth", 32: "thirty second", 64: "sixty fourth",
                 100: "hundredth"}
# Units after a number, glued or spaced ("10km", "20 °C"): singular after 1. A glued "m" is left alone ("5m users")
_UNITS = {
    "km": "kilometer", "m": "meter", "cm": "centimeter", "mm": "millimeter", "mi": "mile", "yd": "yard",
    "km2": "square kilometer", "m2": "square meter", "kg": "kilogram", "g": "gram", "mg": "milligram", "lb": "pound",
    "lbs": "pound", "oz": "ounce", "ml": "milliliter", "hr": "hour", "hrs": "hour", "min": "minute", "mins": "minute",
    "sec": "second", "secs": "second", "ms": "millisecond", "GB": "gigabyte", "MB": "megabyte", "KB": "kilobyte",
    "TB": "terabyte", "kW": "kilowatt", "kWh": "kilowatt hour",
    "ft": ("foot", "feet"), "Hz": ("hertz", "hertz"), "kHz": ("kilohertz", "kilohertz"),
    "MHz": ("megahertz", "megahertz"), "GHz": ("gigahertz", "gigahertz"),
    "mph": ("mile per hour", "miles per hour"), "kph": ("kilometer per hour", "kilometers per hour"),
    "km/h": ("kilometer per hour", "kilometers per hour"), "m/s": ("meter per second", "meters per second"),
    "°C": ("degree celsius", "degrees celsius"), "° C": ("degree celsius", "degrees celsius"),
    "°F": ("degree fahrenheit", "degrees fahrenheit"), "° F": ("degree fahrenheit", "degrees fahrenheit"),
    "°": "degree",
}
_UNIT = re.compile(rf"({_NUMBER})(\s?)({'|'.join(map(re.escape, sorted(_UNITS, key=len, reverse=True)))})(?![\w/°])")
# Roman numerals: after a word that numbers things (World War II, Chapter IV: cardinal), after a regnal name (Henry
# VIII: "the eighth"), and two or more of I, V, X after any capitalised word (Rocky II: cardinal; not "IV", a drip)
_ROMAN = r"(?=[IVXL])(?:XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
_NUMBERED = ("War|Chapter|Part|Act|Scene|Volume|Vol|Book|Phase|Type|Section|Bowl|Title|Article|Appendix|Stage|Level|"
             "Class|Episode|Season|Apollo|Figure|Table")
_REGNAL = ("Henry|Elizabeth|George|Edward|Charles|James|William|Richard|Louis|Paul|John|Pius|Benedict|Philip|"
           "Frederick|Catherine|Nicholas|Alexander|Peter|Mary|Leo|Gregory|Innocent|Clement|Napoleon|Ivan|Ramses|"
           "Constantine|Ferdinand|Leopold|Albert|Wilhelm|Victor")
_ROMAN_NUMBERED = re.compile(rf"\b({_NUMBERED})(\s+)({_ROMAN})\b")
_ROMAN_REGNAL = re.compile(rf"\b({_REGNAL})(\s+)({_ROMAN})\b")
_ROMAN_AFTER_NAME = re.compile(r"\b([A-Z][a-z]+)(\s+)(?!IV\b)((?=[IVX]{2})X{0,3}(?:IX|IV|V?I{0,3}))\b")
# E-mail and web addresses are spelled out ("dot", "at", "slash"); a bare domain needs one of these endings
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_DOMAIN_ENDINGS = "com|net|org|edu|gov|io|co|uk|ai|app|dev|info"
_URL = re.compile(r"(?<![\w@./-])(?:https?://[A-Za-z0-9.-]+|www\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
                  rf"|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:{_DOMAIN_ENDINGS})(?![A-Za-z0-9-]))(?:/[^\s]*)?")
_ADDRESS_CHARS = {".": " dot ", "/": " slash ", "-": " dash ", "_": " underscore ", "@": " at ", ":": " "}
_TRAILING = ".,!?;:)]}'\""
_APOSTROPHES = ("'", "’", "ʼ")  # before _CHAR_MAP


def _below_thousand(n: int) -> list[str]:
    out = []
    if n >= 100:
        out += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        out.append(_TENS[n // 10])
        n %= 10
        if n:
            out.append(_ONES[n])
    elif n or not out:
        out.append(_ONES[n])
    return out


def number_to_words(n: int) -> str:
    """Cardinal number in words: 2024 -> "two thousand twenty four" (American: no "and")."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n >= 10**15:
        return " ".join(_ONES[int(d)] for d in str(n))
    out = []
    for value, name in _SCALES:
        if n >= value:
            out += _below_thousand(n // value) + [name]
            n %= value
    if n or not out:
        out += _below_thousand(n)
    return " ".join(out)


def ordinal_to_words(n: int) -> str:
    words = number_to_words(n).split()
    last = words[-1]
    if last in _ORDINAL_WORDS:
        words[-1] = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year_to_words(n: int) -> str:
    """1999 -> "nineteen ninety nine", 1900 -> "nineteen hundred", 2008 -> "two thousand eight", 2024 -> "twenty twenty
    four"."""
    hi, lo = divmod(n, 100)
    if 2000 <= n < 2010 or n % 1000 == 0:
        return number_to_words(n)
    if lo == 0:
        return f"{number_to_words(hi)} hundred"
    return f"{number_to_words(hi)} {'oh ' if lo < 10 else ''}{number_to_words(lo)}"


def _plural(words: str) -> str:
    """ "nineteen ninety" -> "nineteen nineties" (decades)."""
    head, _, last = words.rpartition(" ")
    last = last[:-1] + "ies" if last.endswith("y") else last + ("es" if last.endswith("x") else "s")
    return f"{head} {last}".strip()


def _number(s: str, year: bool = True) -> str:
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return f"{number_to_words(int(whole))} point {' '.join(_ONES[int(d)] for d in frac)}"
    n = int(s)
    if year and 1100 <= n < 2100 and len(s) == 4:
        return year_to_words(n)
    if len(s) > 1 and s[0] == "0":  # 007, 0800: digit by digit
        return " ".join("oh" if d == "0" else _ONES[int(d)] for d in s)
    return number_to_words(n)


def _scale(word: str | None) -> str:
    return {"k": "thousand", "m": "million", "bn": "billion"}.get((word or "").lower(), (word or "").lower())


def _money(m: re.Match) -> str:
    sym, amount, scale = m.group(1), m.group(2).replace(",", ""), _scale(m.group(3))
    one, many, sub_one, sub_many = _CURRENCY[sym]
    if scale:
        return f" {_number(amount, year=False)} {scale} {many} "
    whole, _, cents = amount.partition(".")
    out = f"{number_to_words(int(whole))} {one if int(whole) == 1 else many}"
    if cents and int(cents[:2].ljust(2, "0")):
        c = int(cents[:2].ljust(2, "0"))
        out += f" {number_to_words(c)} {sub_one if c == 1 else sub_many}"
    return f" {out} "


def _stop(m: re.Match) -> str:
    """"." when the match swallowed the full stop that ends a sentence ("at 5 p.m. Then"), else ""."""
    return "." if m.group().endswith(".") and re.match(r"\s*($|[A-Z\"'])", m.string[m.end():]) else ""


def _time(m: re.Match) -> str:
    h, mm, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if h > 24 or mm > 59:
        return f"{number_to_words(h)} {number_to_words(mm)}"
    words = number_to_words(h)
    if mm:
        words += f" {'oh ' if mm < 10 else ''}{number_to_words(mm)}"
    elif not ampm:
        words += " o'clock"
    if ampm:
        words += " ay em" if ampm.strip().lower().startswith("a") else " pee em"
    return f" {words}{_stop(m)} "


def _opener_stop(m: re.Match) -> str:
    """"." when the abbreviation that ``m`` covers also ends the sentence (:data:`_STOP`), else ""."""
    return "." if _STOP.match(m.string, m.end()) else ""


def _glue(m: re.Match) -> str:
    """"" when an apostrophe follows ``m`` (``the U.S.'s``: it must stay next to a letter), else " "."""
    return "" if m.string.startswith(_APOSTROPHES, m.end()) else " "


def _spell(letters: str) -> str:
    return " ".join(_LETTER_NAMES[c] for c in letters.lower() if c in _LETTER_NAMES)


def _initials(m: re.Match) -> str:
    if len(m.group()) == 2 and (m.group() == "I." or _LETTERED.search(m.string[:m.start()])):
        return m.group()
    return _spell(m.group())


def _digits(s: str) -> str:
    return " ".join("oh" if c == "0" else _ONES[int(c)] for c in s)


def _phone(m: re.Match) -> str:
    """``(555) 123-4567`` -> "five five five, one two three, four five six seven"."""
    digits = "".join(c for c in m.group() if c in "0123456789")
    country, digits = ([], digits) if len(digits) == 10 else ([f"{'plus ' * m.group().startswith('+')}one"], digits[1:])
    return f" {', '.join([*country, _digits(digits[:3]), _digits(digits[3:6]), _digits(digits[6:])])}{_glue(m)}"


def _date(month: int, day: int, year: str) -> str | None:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    y = int(year)
    words = f"{'oh ' if y < 10 else ''}{number_to_words(y)}" if len(year) == 2 else _number(year)
    return f" {_MONTHS[month - 1]} {ordinal_to_words(day)}, {words} "


def _slash_date(m: re.Match) -> str:
    a, b = int(m.group(1)), int(m.group(2))
    return _date(a, b, m.group(3)) or _date(b, a, m.group(3)) or m.group()


def _fraction(m: re.Match) -> str:
    whole, n, d = m.group(1), int(m.group(2)), int(m.group(3))
    if not (0 < n < d and d in _DENOMINATORS):
        return m.group()
    part = _DENOMINATORS[d]
    if n > 1:
        part = "halves" if d == 2 else part + "s"
    if whole is None:
        return f" {number_to_words(n)} {part} "
    return f" {number_to_words(int(whole))} and {'a' if n == 1 else number_to_words(n)} {part} "


def _unit(m: re.Match) -> str:
    if m.group(3) == "m" and not m.group(2):
        return m.group()
    names = _UNITS[m.group(3)]
    one, many = names if isinstance(names, tuple) else (names, names + "s")
    return f" {_number(m.group(1), year=False)} {one if m.group(1) == '1' else many} "


def _roman_to_int(s: str) -> int:
    values = [{"I": 1, "V": 5, "X": 10, "L": 50}[c] for c in s]
    return sum(-v if v < nxt else v for v, nxt in zip(values, [*values[1:], 0]))


def _roman(m: re.Match, ordinal: bool = False) -> str:
    """``World War II`` -> "world war two", ``Henry VIII`` -> "henry the eighth"; "John I think" keeps its pronoun."""
    numeral = m.group(3)
    if numeral == "I" and (ordinal or m.group(1) == "Act") and not re.match(r"\s*(?:$|[,.;:!?)])", m.string[m.end():]):
        return m.group()
    n = _roman_to_int(numeral)
    return f"{m.group(1)}{m.group(2)}{'the ' + ordinal_to_words(n) if ordinal else number_to_words(n)}"


def _speak_address(address: str) -> str:
    """An e-mail or web address without its scheme, query and fragment: dots, slashes and dashes by name."""
    address = re.split("[?#]", re.sub("^https?://", "", address), maxsplit=1)[0].rstrip("/")
    www, address = (f"{_spell('www')} dot ", address[4:]) if address.startswith("www.") else ("", address)
    return www + "".join(_ADDRESS_CHARS.get(c, c) for c in address)


def _address(m: re.Match) -> str:
    address = m.group().rstrip(_TRAILING)  # the punctuation after it belongs to the sentence
    tail = m.group()[len(address):]
    return f" {_speak_address(address)}{tail or _glue(m)}{' ' if tail else ''}"


def _abbreviation(m: re.Match) -> str:
    word = _ABBREVIATIONS.get(m.group(1).lower())  # IGNORECASE also lets "i" match "ı", "s" match "ſ"
    return m.group() if word is None else f" {word}{_stop(m)} "


def _caps(m: re.Match) -> str:
    """An all-caps word (or a known abbreviation without its period): spelled, special-cased or kept."""
    word = m.group()
    if word in _SPECIAL:
        return _SPECIAL[word]
    plural = len(word) > 2 and word.endswith("s") and word[:-1].isupper()  # "TVs", "CDs"
    letters = word[:-1] if plural else word
    if not letters.isupper() or len(letters) < 2 or letters in _INTERJECTIONS:
        return word
    if letters in _SPELLED or (len(letters) <= 5 and not re.search(r"[AEIOUY]", letters)):
        return f" {' '.join(_LETTER_NAMES[c.lower()] for c in letters)}{'s' if plural else ''} "
    return word


def normalize(text: str) -> str:
    """Normalise raw English text to the symbol alphabet (lower-case letters, apostrophe, space and ``.,!?``)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))  # café -> cafe
    text = _EMAIL.sub(_address, text)  # before "@" and "+" are mapped
    text = _URL.sub(_address, text)
    text = _PHONE.sub(_phone, text)
    text = _LOCAL_PHONE.sub(lambda m: f"{m.group(1)}{_digits(m.group(2)[:3])}, {_digits(m.group(2)[4:])}{_glue(m)}",
                            text)
    text = "".join(_CHAR_MAP.get(c, c) for c in text)
    text = _SAINT.sub("saint", text)
    text = _STREET.sub(lambda m: f"{m.group(1)}{m.group(2)}street{_opener_stop(m)}", text)
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|St|Prof)\.(?=\s)", lambda m: _SPECIAL[m.group(1)], text)
    text = _TITLE.sub(lambda m: _TITLES[m.group(1)], text)
    text = _AFTER_NAME_ABBREVIATION.sub(lambda m: f"{_AFTER_NAME[m.group(1)]}{_opener_stop(m)}", text)
    text = _MONTH.sub(lambda m: _MONTH_ABBREVIATIONS[m.group(1)], text)
    text = re.sub(r"\bPh\.D\.?", lambda m: f"{_spell('phd')}{_opener_stop(m) if m.group().endswith('.') else ''}",
                  text)
    text = re.sub(r"\b(vs|etc|approx|e\.g|i\.e)\.", _abbreviation, text, flags=re.I)
    text = re.sub(r"\b[Nn]o\.\s*(?=\d)", "number ", text)
    text = _BEFORE_NUMBER_ABBREVIATION.sub(lambda m: f"{_BEFORE_NUMBER[m.group(1)]} ", text)
    text = _TIME.sub(_time, text)
    text = _AMPM.sub(lambda m: f" {'ay' if m.group(1).lower() == 'a' else 'pee'} em{_stop(m)} ", text)
    text = _AMPM_WORDS.sub(lambda m: f"{m.group(1)} {'ay' if m.group(2).lower() == 'a' else 'pee'} em{_stop(m)} ",
                           text)
    text = _DOTTED.sub(lambda m: f" {_spell(m.group())}{_opener_stop(m)}{_glue(m)}", text)
    text = text.replace(":", ",")
    text = re.sub(rf"([$£€])\s*({_NUMBER})(?:\s*({_SCALE_WORD}))?", _money, text, flags=re.I)
    text = re.sub(rf"({_NUMBER})\s*%", lambda m: f" {_number(m.group(1), year=False)} percent ", text)
    text = _DATE.sub(_slash_date, text)
    text = _ISO_DATE.sub(lambda m: _date(int(m.group(2)), int(m.group(3)), m.group(1)) or m.group(), text)
    text = _FRACTION.sub(_fraction, text)
    text = re.sub(r"'?(\d{2}|\d{4})'?s\b", lambda m: f" {_plural(_number(m.group(1)))} ", text)  # the 90s, 1980's
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: f" {ordinal_to_words(int(m.group(1)))} ", text, flags=re.I)
    text = re.sub(r"(?<![\w.,])-(?=\d)", " minus ", text)
    text = re.sub(rf"({_NUMBER})\s*-\s*(?={_NUMBER})", r"\1 to ", text)  # 20-30
    text = _UNIT.sub(_unit, text)
    text = re.sub(_NUMBER, lambda m: f" {_number(m.group())} ", text)
    text = _INITIALS.sub(_initials, text)
    text = _ROMAN_NUMBERED.sub(_roman, text)
    text = _ROMAN_REGNAL.sub(lambda m: _roman(m, ordinal=True), text)
    text = _ROMAN_AFTER_NAME.sub(_roman, text)
    text = re.sub(r"\b[A-Z][A-Za-z]*\b", _caps, text)
    text = text.lower()
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)  # quote marks; keep don't, we'll, rock'n'roll
    text = text.replace('"', " ").replace("-", " ").replace("/", " ")
    text = "".join(c if c in SYMBOL_TO_ID else " " for c in text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # no space before punctuation
    text = re.sub(r",+", ",", text)
    text = re.sub(r",(?=[.!?])", "", text)
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

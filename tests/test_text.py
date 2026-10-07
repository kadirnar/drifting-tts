import pytest

from drifting_tts.text import (
    BLANK_ID,
    SYMBOL_TO_ID,
    SYMBOLS,
    ids_to_text,
    normalize,
    number_to_words,
    ordinal_to_words,
    text_to_ids,
)


@pytest.mark.parametrize(
    "n, words",
    [
        (0, "sıfır"), (7, "yedi"), (10, "on"), (19, "on dokuz"), (100, "yüz"), (101, "yüz bir"),
        (342, "üç yüz kırk iki"), (1000, "bin"), (1100, "bin yüz"), (2017, "iki bin on yedi"),
        (10_000, "on bin"), (1_000_000, "bir milyon"), (1_250_000, "bir milyon iki yüz elli bin"),
        (3_000_000_001, "üç milyar bir"),
    ],
)
def test_number_to_words(n, words):
    assert number_to_words(n) == words


@pytest.mark.parametrize(
    "n, words",
    [(1, "birinci"), (2, "ikinci"), (3, "üçüncü"), (4, "dördüncü"), (6, "altıncı"), (9, "dokuzuncu"), (10, "onuncu"),
     (14, "on dördüncü"), (20, "yirminci"), (40, "kırkıncı"), (100, "yüzüncü"), (1000, "bininci")],
)
def test_ordinal_to_words(n, words):
    assert ordinal_to_words(n) == words


BASIC = [
    (
        "İstanbul'da 2017 yılında %50 indirim vardı.",
        "istanbulda iki bin on yedi yılında yüzde elli indirim vardı.",
    ),
    ("IŞIK ve Işık", "ışık ve ışık"),
    ("Fiyat 3,5 lira; nüfus 1.250.000.", "fiyat üç virgül beş lira, nüfus bir milyon iki yüz elli bin."),
    ("Hmm…  peki — tamam?!", "hmm... peki tamam?!"),
    ("Kâr ve hüzün “güzel”", "kar ve hüzün güzel"),
    ("Arapça: الله kelimesi", "arapça, kelimesi"),
    ("0,05 olasılık", "sıfır virgül sıfır beş olasılık"),
    ("2021'de bitti.", "iki bin yirmi birde bitti."),
    ("Hristiyanlık 14. yüzyılda", "hristiyanlık on dördüncü yüzyılda"),
    ("28.10.2014 tarihi", "yirmi sekiz on iki bin on dört tarihi"),
    ("Bu 2. kez, 1. ve 100. sırada", "bu ikinci kez, birinci ve yüzüncü sırada"),
]

APOSTROPHES = [  # only a suffix apostrophe is deleted; quote marks become spaces
    ("Ali 'evet' dedi.", "ali evet dedi."),
    ("Ali ‘evet’ dedi.", "ali evet dedi."),
    ("Ali `evet´ dedi.", "ali evet dedi."),
    ("Anadolu'ya gitti", "anadoluya gitti"),
    ("Ömer’in kitabı", "ömerin kitabı"),
    ("ANKARA'DA YAĞMUR", "ankarada yağmur"),
    ("Ali'nin 'evet'i", "alinin eveti"),
    ("'80'ler", "seksenler"),
    ("5'inci kat", "beşinci kat"),
    ("1990'lı yıllar", "bin dokuz yüz doksanlı yıllar"),
]

TIMES = [  # clock times are read before ":" is mapped to ","
    ("Toplantı saat 14:30'da.", "toplantı saat on dört otuzda."),
    ("10:00", "on"),
    ("Saat 14:00'te", "saat on dörtte"),
    ("09:05", "dokuz sıfır beş"),
    ("00:00'da", "sıfır sıfırda"),
    ("14:30-15:00 arası", "on dört otuz on beş arası"),
    ("12:30:45", "on iki otuz kırk beş"),
    ("Skor 3:1", "skor üç bir"),
    ("Saat 09.15'te geldi.", "saat dokuz on beşte geldi."),
]

COMBINING_DOT = [  # "İ".lower() == "i̇" (i + U+0307)
    ("i\u0307stanbul", "istanbul"),
    ("İSTANBUL".lower() + " İzmir", "istanbul izmir"),
]

ORDINALS = [
    ("1. Dünya Savaşı", "birinci dünya savaşı"),
    ("14. Yüzyıl", "on dördüncü yüzyıl"),
    ("ABD'nin 45. Başkanı", "a be denin kırk beşinci başkanı"),
    ("Takım 2. Lig'e düştü.", "takım ikinci lige düştü."),
    ("21. yüzyılın", "yirmi birinci yüzyılın"),
    # a number before a sentence opener ends the sentence; 4+ digits before a capital stay cardinal
    ("Sayı 5. Sonra geldi.", "sayı beş. sonra geldi."),
    ("Puan 85. Bu iyi.", "puan seksen beş. bu iyi."),
    ("Bölüm 3. Burada", "bölüm üç. burada"),
    ("Yıl 2014. Ali geldi.", "yıl iki bin on dört. ali geldi."),
]

ROMAN = [
    ("II. Abdülhamid", "ikinci abdülhamid"),
    ("XV. yüzyıl", "on beşinci yüzyıl"),
    ("III. Selim'in", "üçüncü selimin"),
    ("I. Dünya Savaşı", "birinci dünya savaşı"),
    ("XIX. yüzyılda", "on dokuzuncu yüzyılda"),
    ("IV. Murat", "dördüncü murat"),
    ("M. Kemal", "m. kemal"),  # initials are not Roman numerals
    ("C. Ronaldo", "c. ronaldo"),
    ("Bölüm IV", "bölüm ıv"),  # no dot, no ordinal
]

SIGNS = [  # a minus at the start of a word is "eksi"; a hyphen between numbers is a range
    ("-5 derece", "eksi beş derece"),
    ("Sıcaklık −3 °C'ye düştü.", "sıcaklık eksi üç dereceye düştü."),
    ("(-12)", "eksi on iki"),
    ("x=-5", "x eşittir eksi beş"),
    ("3-5 gün", "üç beş gün"),
    ("Maç 2-1 bitti.", "maç iki bir bitti."),
    ("1990-1995", "bin dokuz yüz doksan bin dokuz yüz doksan beş"),
    ("COVID-19 salgını", "kovid on dokuz salgını"),
]

UNITS = [
    ("250 TL", "iki yüz elli lira"),
    ("₺250", "iki yüz elli lira"),
    ("250 ₺", "iki yüz elli lira"),
    ("250 tl", "iki yüz elli lira"),
    ("2,5 milyar TL", "iki virgül beş milyar lira"),
    ("$2,5 milyon", "iki virgül beş milyon dolar"),
    ("100 $", "yüz dolar"),
    ("€50", "elli avro"),
    ("€ bölgesi", "avro bölgesi"),
    # suffixes written for the abbreviation are re-harmonised to the spoken word
    ("250 TL'ye", "iki yüz elli liraya"),
    ("250 TL'lik", "iki yüz elli liralık"),
    ("100 TL'deki", "yüz liradaki"),
    ("Fiyatı 10 $'dı", "fiyatı on dolardı"),
    ("5 kg'dan", "beş kilogramdan"),
    ("5 kg'ye", "beş kilograma"),
    ("2,5 kg'si", "iki virgül beş kilogramı"),
    ("10 km", "on kilometre"),
    ("5 m'den", "beş metreden"),
    ("30 cm", "otuz santimetre"),
    ("120 m² daire", "yüz yirmi metrekare daire"),
    ("25°C", "yirmi beş derece"),
    ("90° açı", "doksan derece açı"),
    ("100 km/s hız", "saatte yüz kilometre hız"),
    ("5 dk sonra", "beş dakika sonra"),
    ("kaç km", "kaç km"),  # units only after a number
    ("50%", "yüzde elli"),
    ("%3,5 faiz", "yüzde üç virgül beş faiz"),
    ("%5'lik", "yüzde beşlik"),
]

ABBREVIATIONS = [
    ("Dr. Ahmet geldi.", "doktor ahmet geldi."),
    ("Prof. Dr. Ayşe Yılmaz", "profesör doktor ayşe yılmaz"),
    ("Doç. Dr. Ali", "doçent doktor ali"),
    ("Dr. Öğr. Üyesi Ayşe", "doktor öğretim üyesi ayşe"),
    ("Av. Mehmet", "avukat mehmet"),
    ("Av. sezonu", "av. sezonu"),  # titles only before a name
    ("elma, armut vb. şeyler", "elma, armut ve benzeri şeyler"),
    ("elma vb. Sonra geldi.", "elma ve benzeri. sonra geldi."),
    ("kalem, defter vs.", "kalem, defter vesaire."),
    ("örn. kedi", "örneğin kedi"),
    ("Örn. İstanbul", "örneğin istanbul"),
    ("Kapı No. 5", "kapı numara beş"),
    ("No:12", "numara on iki"),
]

ACRONYMS = [
    ("TBMM'de", "te be me mede"),
    ("ABD ve AB", "a be de ve a be"),
    ("BM'nin", "be menin"),
    ("TRT", "te re te"),
    ("THY'nin", "te ha yenin"),
    ("AKP CHP MHP HDP PKK", "a ka pe ce he pe me he pe he de pe pe ka ka"),
    ("TL'nin değeri", "te lenin değeri"),
    ("İTÜ ODTÜ NATO'ya", "itü odtü natoya"),  # read as words
    ("IŞİD", "ışid"),
]

CASES = BASIC + APOSTROPHES + TIMES + COMBINING_DOT + ORDINALS + ROMAN + SIGNS + UNITS + ABBREVIATIONS + ACRONYMS


@pytest.mark.parametrize("raw, norm", CASES)
def test_normalize(raw, norm):
    assert normalize(raw) == norm


@pytest.mark.parametrize("raw", [raw for raw, _ in CASES])
def test_normalize_is_idempotent_and_in_vocabulary(raw):
    norm = normalize(raw)
    assert set(norm) <= set(SYMBOL_TO_ID)
    assert normalize(norm) == norm


def test_symbols_are_frozen():  # checkpoints and prepared indices store these ids
    assert SYMBOLS[:2] == ["<pad>", "<blank>"]
    assert "".join(SYMBOLS[2:]) == "abcçdefgğhıijklmnoöpqrsştuüvwxyz .,!?"


def test_text_to_ids_intersperse_roundtrip():
    ids = text_to_ids("Merhaba dünya!")
    assert ids[0] == BLANK_ID and ids[-1] == BLANK_ID and len(ids) % 2 == 1
    assert ids_to_text(ids) == "merhaba dünya!"
    assert all(0 <= i < len(SYMBOLS) for i in ids)

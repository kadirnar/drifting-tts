"""English text frontend (``language: en``): normalisation, symbols, and the language switch of the pipeline."""

import pytest

from drifting_tts import text_en
from drifting_tts.text import BLANK_ID, PAD_ID, frontend

CASES = [
    ("Uh, you know, it's— well, I got maybe four hours?", "uh, you know, it's, well, i got maybe four hours?"),
    ("I left at 7:45 to beat the rush.", "i left at seven forty five to beat the rush."),
    ("Sleeping at 4am is great.", "sleeping at four ay em is great."),
    ("Meet me at 10:00 or 9:05 p.m.", "meet me at ten o'clock or nine oh five pee em."),
    ("You ever lay there at three AM?", "you ever lay there at three ay em?"),
    ("Those were the 90s, the 1980's.", "those were the nineties, the nineteen eighties."),
    ("Since 2008, in 1999, by 2024, in 1900.",
     "since two thousand eight, in nineteen ninety nine, by twenty twenty four, in nineteen hundred."),
    ("1,250,000 views and 3.14", "one million two hundred fifty thousand views and three point one four"),
    ("It's $15, or $4.99, or $1, or $2.5 million.",
     "it's fifteen dollars, or four dollars ninety nine cents, or one dollar, or two point five million dollars."),
    ("Like 20% off.", "like twenty percent off."),
    ("The 21st or the 2nd, the 40th.", "the twenty first or the second, the fortieth."),
    ("His BBQ, an old TV, a call from HR.", "his barbecue, an old tee vee, a call from aitch ar."),
    ("The FBI and NASA, in the USA.", "the eff bee eye and nasa, in the you ess ay."),
    ("I was SO tired, hmm, HMM. OK?", "i was so tired, hmm, hmm. okay?"),
    ("Mr. Smith vs. Dr. Jones, etc.", "mister smith versus doctor jones, et cetera."),
    ("It ends at 5 p.m. Then we go, e.g. home.", "it ends at five pee em. then we go, for example home."),
    ("It's -5 degrees, pages 20-30, No. 7.", "it's minus five degrees, pages twenty to thirty, number seven."),
    ("We'll see, rock'n'roll, the students' books, 'quoted'.",
     "we'll see, rock'n'roll, the students books, quoted."),
    ("café naïve “quoted” & more", "cafe naive quoted and more"),
]

DOTS = [  # only a sentence end keeps its dot: dotted acronyms and initials are read by their letter names
    ("The U.S. economy grew.", "the you ess economy grew."),
    ("He works in the U.S. The economy grew.", "he works in the you ess. the economy grew."),  # an opener follows
    ("the U.S.'s policy in Washington D.C.", "the you ess's policy in washington dee see."),
    ("J. K. Rowling wrote it.", "jay kay rowling wrote it."),
    ("J.K. Rowling and John F. Kennedy", "jay kay rowling and john eff kennedy"),
    ("I. M. Pei designed it.", "eye em pei designed it."),
    ("So did I. Mary left.", "so did i. mary left."),  # the pronoun
    ("Plan B. Then we left.", "plan b. then we left."),  # not an initial: an opener follows
    ("Take vitamin C. Take it daily.", "take vitamin c. take it daily."),  # a letter that names a thing
    ("He got a B. His sister got an A. Mom was happy.", "he got a b. his sister got an a. mom was happy."),
    ("hello. how are you.", "hello. how are you."),  # plain sentences keep their dots
    ("She has a Ph.D. in physics. He has a Ph.D. Then he left.",
     "she has a pee aitch dee in physics. he has a pee aitch dee. then he left."),
    ("Apple Inc. CEO Tim Cook spoke.", "apple incorporated see ee oh tim cook spoke."),
    ("Martin Luther King Jr. He spoke too.", "martin luther king junior. he spoke too."),
    ("Smith & Co. Ltd. and Acme Corp. merged.", "smith and company limited and acme corporation merged."),
    ("Gen. Patton and Sen. Smith met Capt. Kirk at Mt. Everest.",
     "general patton and senator smith met captain kirk at mount everest."),
    ("I live on Main St. near St. Louis.", "i live on main street near saint louis."),
    ("He lives on Elm St. She lives on 5th St. NW.",
     "he lives on elm street. she lives on fifth street en double you."),
    ("On Jan. 5 and Sept. 2020, but I met Jan. He left.",
     "on january five and september twenty twenty, but i met jan. he left."),
    ("See Fig. 3 and pp. 12-15, vol. 2, p. 5.", "see figure three and pages twelve to fifteen, volume two, page five."),
]

NUMBERS = [
    ("Call 555-1234 on 3/4/2021, it is 10km away at 20°C.",
     "call five five five, one two three four on march fourth, twenty twenty one, it is ten kilometers away at twenty "
     "degrees celsius."),
    ("Call (555) 123-4567 or 1-800-555-1234.",
     "call five five five, one two three, four five six seven or one, eight oh oh, five five five, one two three "
     "four."),
    ("+1 555 123 4567", "plus one, five five five, one two three, four five six seven"),
    ("pages 100-2000", "pages one hundred to two thousand"),  # a range, not a phone number
    ("On 2021-03-04, 25/12/2020 and 3/4/21.",
     "on march fourth, twenty twenty one, december twenty fifth, twenty twenty and march fourth, twenty one."),
    ("13/13/2020", "thirteen thirteen twenty twenty"),  # not a date
    ("Add 1/2 cup, 3/4 of it, 2/3, 5/8 inch and 1 1/2 hours.",
     "add one half cup, three quarters of it, two thirds, five eighths inch and one and a half hours."),
    ("9/11, 24/7 and 50/50", "nine eleven, twenty four seven and fifty fifty"),  # not fractions
    ("5 kg, 1 lb, 3.5 oz, 60 mph, 100 km/h, 1 km, 6 ft.",
     "five kilograms, one pound, three point five ounces, sixty miles per hour, one hundred kilometers per hour, one "
     "kilometer, six feet."),
    ("-5°C, 72 °F, 20-30 km, 1999 km, 5 min, 36 ms, 24 kHz, 16 GB.",
     "minus five degrees celsius, seventy two degrees fahrenheit, twenty to thirty kilometers, one thousand nine "
     "hundred ninety nine kilometers, five minutes, thirty six milliseconds, twenty four kilohertz, sixteen "
     "gigabytes."),
    ("It is 10 m away, with 5m users and $5m.", "it is ten meters away, with five m users and five million dollars."),
    ("World War II, Chapter IV, Type I error, Super Bowl XLII.",
     "world war two, chapter four, type one error, super bowl forty two."),
    ("Henry VIII, Elizabeth II, Louis XIV and Elizabeth I.",
     "henry the eighth, elizabeth the second, louis the fourteenth and elizabeth the first."),
    ("Rocky II, the IV drip, John I think, Malcolm X.", "rocky two, the iv drip, john i think, malcolm x."),
]

ADDRESSES = [
    ("Email john.doe@example.com or visit https://www.example.com/about-us?x=1.",
     "email john dot doe at example dot com or visit double you double you double you dot example dot com slash about "
     "dash us."),
    ("See example.org/docs, okay.", "see example dot org slash docs, okay."),
    ("It ended.Then we left.", "it ended. then we left."),  # a missing space is not an address
]

CASES += DOTS + NUMBERS + ADDRESSES


@pytest.mark.parametrize("raw,expected", CASES)
def test_normalize(raw, expected):
    assert text_en.normalize(raw) == expected


@pytest.mark.parametrize("raw", [raw for raw, _ in CASES])
def test_normalize_is_idempotent_and_in_vocabulary(raw):
    norm = text_en.normalize(raw)
    assert set(norm) <= set(text_en.SYMBOL_TO_ID)
    assert text_en.normalize(norm) == norm


@pytest.mark.parametrize(
    "raw, sentences",
    [
        ("The U.S. economy grew. J. K. Rowling wrote it.", ["the you ess economy grew.", "jay kay rowling wrote it."]),
        ("Dr. Smith met Mr. Jones at 5 p.m. Then they left!",
         ["doctor smith met mister jones at five pee em.", "then they left!"]),
        ("hello. how are you? fine.", ["hello.", "how are you?", "fine."]),
        ("Plan B. Then we left.", ["plan b.", "then we left."]),
        ("Call (555) 123-4567. Or write to a.b@example.com. Thanks.",
         ["call five five five, one two three, four five six seven.", "or write to a dot b at example dot com.",
          "thanks."]),
    ],
)
def test_split_sentences(raw, sentences):  # abbreviations, initials and addresses do not break a sentence
    assert text_en.split_sentences(text_en.normalize(raw)) == sentences


def test_numbers():
    assert text_en.number_to_words(0) == "zero"
    assert text_en.number_to_words(1001) == "one thousand one"
    assert text_en.number_to_words(-12) == "minus twelve"
    assert text_en.ordinal_to_words(22) == "twenty second"
    assert text_en.ordinal_to_words(12) == "twelfth"
    assert text_en.year_to_words(2015) == "twenty fifteen"
    assert text_en.year_to_words(1905) == "nineteen oh five"


def test_symbols_and_frontend_switch():
    en, tr = frontend("en"), frontend("tr")
    assert en is text_en and tr.SYMBOLS[:2] == en.SYMBOLS[:2]
    assert (en.PAD_ID, en.BLANK_ID) == (PAD_ID, BLANK_ID)
    assert "'" in en.SYMBOLS and "ç" not in en.SYMBOLS
    ids = en.text_to_ids("Don't stop!")
    assert ids[0::2] == [BLANK_ID] * (len(ids) // 2 + 1) and en.ids_to_text(ids) == "don't stop!"
    assert all(c in en.SYMBOL_TO_ID for c in en.normalize("Ünïcödé — 100% “ok” @ #3 ½"))
    with pytest.raises(ValueError):
        frontend("de")


def test_english_ar_model_vocabulary():
    from drifting_tts.config import load_config
    from drifting_tts.models.ar import ARDriftTTS

    cfg = load_config("configs/ar_tts.yaml", ["model.text.language=en", "model.ar.depth=1", "model.ar.head_depth=1"])
    model = ARDriftTTS(cfg.model, num_speakers=2)
    assert model.encoder.emb.num_embeddings == len(text_en.SYMBOLS)


def _english_parquet(path, n, start=0, seconds=1.0):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.test_data import _wav_bytes

    rows = {
        "audio": [{"bytes": _wav_bytes(seconds + 0.1 * i, freq=200 + 10 * i), "path": f"{i}.wav"} for i in range(n)],
        "text": [f"Hello there, it's {start + i} o'clock." for i in range(n)],
        "speaker_id": [f"spk_{i % 3}" for i in range(n)],
    }
    pq.write_table(pa.table(rows), path)


def test_prepare_english_held_out_files_and_stream(tmp_path):
    import json

    import torch

    from drifting_tts.cli import main
    from drifting_tts.data import MelDataset
    from drifting_tts.stream import ParquetStream

    _english_parquet(tmp_path / "train-0.parquet", 12)
    _english_parquet(tmp_path / "held-0.parquet", 6, start=100, seconds=2.5)
    out = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--val-glob", str(tmp_path / "held-*.parquet"),
          "--val-size", "4", "--out", str(out), "--workers", "1", "--no-trim", "--backend", "bigvgan",
          "--language", "en", "--speaker-column", "speaker_id"])
    stats = json.loads((out / "stats.json").read_text())
    assert stats["language"] == "en" and stats["speakers"] == 3
    index = [json.loads(line) for line in open(out / "index.jsonl")]
    assert index[0]["norm_text"] == "hello there, it's zero o'clock." and index[0]["speaker"] == "spk_0"
    splits = [e["split"] for e in index]
    assert splits[:12] == ["train"] * 12 and sorted(splits[12:]) == ["dev"] * 2 + ["val"] * 4  # held-out file only
    ds = MelDataset(out, "train", min_frames=1)
    assert ds.language == "en" and text_en.ids_to_text(ds[0]["text"].tolist()) == index[0]["norm_text"]

    stream = ParquetStream([str(tmp_path / "train-0.parquet")], out, speaker_column="speaker_id", min_frames=1,
                           batch_frames=10**6, max_batch=4, trim=False)
    batch = next(iter(stream))
    assert sorted(batch) == ["index", "mel", "mel_len", "spk", "text", "text_len"]
    assert batch["mel"].shape[:2] == (4, 100) and batch["text"].shape[0] == 4
    k = int(batch["mel_len"].argmin())  # the streamed mel equals the prepared one (fp16 storage aside)
    i = next(j for j, e in enumerate(index) if e["frames"] == int(batch["mel_len"][k]) and e["split"] == "train")
    torch.testing.assert_close(batch["mel"][k, :, : index[i]["frames"]], ds[i]["mel"], atol=2e-3, rtol=0)

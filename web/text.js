// Turkish text frontend for the browser: a port of drifting_tts/text.py (normalize, text_to_ids) and
// drifting_tts.synthesize.split_sentences. web/tests/text.test.mjs checks it against Python outputs.
//
// Porting notes: Python's \w \d \b are Unicode-aware but JS's are ASCII-only (even with the u flag), and JS's \s is a
// different set, so the classes below spell out Python's; int() reads any Unicode decimal digit and has no size limit.

export const PAD = "<pad>", BLANK = "<blank>";
const LETTERS = "abcçdefgğhıijklmnoöpqrsştuüvwxyz";
const PUNCTUATION = " .,!?";
export const SYMBOLS = [PAD, BLANK, ...LETTERS, ...PUNCTUATION];
export const SYMBOL_TO_ID = new Map(SYMBOLS.map((s, i) => [s, i]));
export const PAD_ID = SYMBOL_TO_ID.get(PAD), BLANK_ID = SYMBOL_TO_ID.get(BLANK);

// Python character classes (bracket contents)
const W = String.raw`\p{L}\p{N}_`; // \w
const D = String.raw`\p{Nd}`; // \d
const S = String.raw`\t\n\v\f\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000`; // \s
const ALNUM = String.raw`\p{L}\p{N}`; // [^\W_]
const LETTER = String.raw`\p{L}\p{Nl}\p{No}`; // [^\W\d_]
const re = (source, flags = "gu") => new RegExp(source, flags);
const escape = (s) => s.replace(/[$^\\.*+?()[\]{}|/]/g, "\\$&");

const ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"];
const TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"];
const SCALES = [[10n ** 12n, "trilyon"], [10n ** 9n, "milyar"], [10n ** 6n, "milyon"], [10n ** 3n, "bin"]];
const DIGIT_NAMES = ["sıfır", ...ONES.slice(1)];

const CHAR_MAP = {
  "â": "a", "î": "i", "û": "u", "Â": "A", "Î": "İ", "Û": "U",
  "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'", "“": '"', "”": '"', "«": '"', "»": '"',
  "…": "...", "–": "-", "—": "-", "−": "-", ";": ",",
  "&": " ve ", "+": " artı ", "=": " eşittir ", "@": " et ",
  "é": "e", "è": "e", "à": "a", "ä": "a", "ñ": "n", "ß": "ss", "ô": "o",
};
const UPPER = "A-ZÇĞİÖŞÜ", LOWER = "a-zçğıöşü";
const NUMBER = `[${D}]+(?:[.,][${D}]+)*`;
const SCALE_WORD = `(?<![${W}])(?:bin|milyon|milyar|trilyon)(?![${W}])`; // \b(?:...)\b

// Units and currencies, expanded only after a number ("250 TL", "5 bin km"); "₺250" is first rewritten to "250 ₺"
const UNITS = {
  "₺": "lira", "TL": "lira", "tl": "lira", "$": "dolar", "USD": "dolar", "€": "avro", "EUR": "avro", "£": "sterlin",
  "km²": "kilometrekare", "km2": "kilometrekare", "m²": "metrekare", "m2": "metrekare", "m³": "metreküp",
  "m3": "metreküp", "km": "kilometre", "m": "metre", "cm": "santimetre", "mm": "milimetre", "kg": "kilogram",
  "g": "gram", "gr": "gram", "mg": "miligram", "lt": "litre", "ml": "mililitre", "sn": "saniye", "dk": "dakika",
  "°C": "derece", "°": "derece",
};
const UNIT_NAMES = Object.keys(UNITS).sort((a, b) => b.length - a.length).map(escape).join("|"); // stable, as sorted()
const UNIT = re(`([${D}]|${SCALE_WORD})[${S}]*(${UNIT_NAMES})(?![${W}])(?:'([${LETTER}]+))?`);
const PREFIX_CURRENCY = re(`([₺$€£])[${S}]*(${NUMBER}(?:[${S}]+${SCALE_WORD})?)`);
const SPEED = re(`(${NUMBER})[${S}]*km/(?:sa|s|h)(?![${W}])`);
const TIME = re(`(?<![${D}.,:])([${D}]{1,2}):([${D}]{2})(?![${D}:])`);

// Titles are expanded only before a capitalised name ("Av. Ali" but not "Av. sezonu")
const TITLES = {
  Dr: "doktor", Prof: "profesör", Doç: "doçent", Yrd: "yardımcı", Öğr: "öğretim", Av: "avukat",
  Op: "operatör", Uzm: "uzman", Sn: "sayın", Hz: "hazreti",
};
const TITLE = re(`(?<![${W}])(${Object.keys(TITLES).join("|")})\\.(?=[${S}]*[${UPPER}])`);
const ABBREVIATIONS = { vb: "ve benzeri", vs: "vesaire", vd: "ve diğerleri", örn: "örneğin", bkz: "bakınız" };
// (?i:...) of Python: also "Örn.", all-caps "VB." (Python's IGNORECASE also lets "s" match "ſ", then fails the lookup)
const caseless = (word) => [...word].map((c) => `[${c}${c.toUpperCase()}]`).join("");
const ABBREVIATION = re(`(?<![${W}])(${Object.keys(ABBREVIATIONS).map(caseless).join("|")})\\.`);
const SENTENCE_START = re(`[${S}]*(?:$|[${UPPER}](?![${UPPER}]))`, "uy"); // end of text or a capitalised word

// Acronyms read letter by letter (K is "ka" as in PKK, KDV); other all-caps words are just lower-cased (NATO, ODTÜ)
const NAMES = "a be ce çe de e fe ge he ı i je ka le me ne o ö pe re se şe te u ü ve ye ze".split(" ");
const LETTER_NAMES = Object.fromEntries([..."ABCÇDEFGHIİJKLMNOÖPRSŞTUÜVYZ"].map((c, i) => [c, NAMES[i]]));
const SPELLED = "AB ABD AİHM AKP AVM BM CHP DSÖ HDP İBB KDV KKTC MHP PKK PTT SGK TBMM TC TL TRT TSK".split(" ");
const ACRONYMS = {
  ...Object.fromEntries(SPELLED.map((a) => [a, [...a].map((c) => LETTER_NAMES[c]).join(" ")])),
  THY: "te ha ye", COVID: "kovid", FIFA: "fifa",
};

// "N." before a capitalised word is ambiguous: "1. Dünya Savaşı" (ordinal) vs "Sayı 5. Sonra ..." (sentence end).
// A 1-3 digit number (or a Roman numeral) is read as an ordinal unless the next word is one of these sentence
// openers, closed-class words that never follow an ordinal. Before a lower-case word it is always an ordinal.
const OPENERS = (
  "Ama Ancak Ardından Artık Aslında Ayrıca Bazen Belki Ben Bence Bir Biz Böyle Böylece Bu Buna Bunda Bundan Bunlar " +
  "Bunu Bunun Burada Çünkü Da Daha De Dolayısıyla Elbette Evet Fakat Halbuki Hatta Hayır Hem Hemen Her Herkes Hiç " +
  "İşte Kim Ki Mesela Ne Neden Nasıl Niye O Ona Onda Ondan Onlar Onu Onun Orada Oysa Öyle Örneğin Peki Sadece Sen " +
  "Siz Son Sonra Şimdi Şu Tabii Ve Veya Ya Yani Yine Zaten"
).split(" ");
const TITLE_WORD = `[${S}]+(?!(?:${OPENERS.join("|")})(?![${W}]))[${UPPER}]`;
const ORDINAL = re(`(?<![${D}.,])(?:([${D}]{1,6})\\.(?=[${S}]+[${LOWER}])|([${D}]{1,3})\\.(?=${TITLE_WORD}))`);
// Regnal numbers and centuries (II. Abdülhamid, XV. yüzyıl): only I, V, X (1-39), so initials such as "M. Kemal" or
// "C. Ronaldo" are left alone
const ROMAN_ORDINAL = re(`(?<![${W}.])(?=[IVX])(X{0,3}(?:IX|IV|V?I{0,3}))\\.(?=[${S}]+[${LOWER}]|${TITLE_WORD})`);

// The dot of an abbreviation is not a sentence end ("T.C. Sağlık Bakanlığı", "Gül Sok. No: 5", "M. Kemal"), unless the
// text ends there or a sentence opener follows ("... A.Ş. Sonra kapandı.")
const STOP = re(`[${S}]*$|[${S}]+(?:${OPENERS.join("|")})(?![${W}])`, "uy");
// Dotted letter sequences are read letter by letter (T.C., A.B.D., t.c.); a few stand for words
const DOTTED = re(`(?<![${W}.])(?:[${UPPER}${LOWER}]\\.){2,}`);
const DOTTED_WORDS = new Map([["A.Ş.", "anonim şirketi"]]);
// Initials before a capitalised name (M. Kemal, A. O. Yılmaz) are read by their letter names; before a sentence opener
// the dot ends the sentence ("Plan B. Sonra gittik."). A lone I, V or X stays a regnal number (ROMAN_ORDINAL)
const INITIALS = re(`(?<![${W}.-])[${UPPER}]\\.(?:[${S}]+[${UPPER}]\\.){0,5}(?=${TITLE_WORD})`);
const INITIAL = re(`[${UPPER}]`);
// ... but a lone letter after one of these words names a thing ("Hepatit B. Aşısı var.", "Grup A. Türkiye lider.")
const LETTERED = re(`(?<![${W}])(?:grup|hepatit|plan|tip|seçenek|şık|blok|kapı|sınıf|şube|kategori|model|madde|bölüm` +
  `|vitamin)[${S}]+$`, "u");
// Abbreviations that follow a name, in addresses and company names (Gül Sok., Atatürk Cad., Yılmaz Ltd. Şti.)
const AFTER_NAME = {
  Mah: "mahallesi", Cad: "caddesi", Sok: "sokak", Sk: "sokak", Apt: "apartmanı", Bul: "bulvarı", Blv: "bulvarı",
  Ltd: "limited", Şti: "şirketi", Tic: "ticaret", San: "sanayi",
};
const AFTER_NAME_ABBREVIATION = re(`(?<=[${W}.] )(${Object.keys(AFTER_NAME).join("|")})\\.`);
const TELEPHONE = re(`(?<![${W}])(?:Tel[.:]|tel:)[${S}]*(?=[+(${D}])`); // "ince bir tel. 5 metre" is a wire
// Phone numbers, read group by group ("0532 123 45 67": sıfır beş yüz otuz iki, yüz yirmi üç, kırk beş, altmış yedi):
// with a country code, a trunk zero or an area code in brackets; or a mobile number in its four spaced groups
const PHONE = re(`(?<![${W}.,+/-])(?:(?:(?:\\+90|0090)[ -]?(?:\\([0-9]{3}\\)|[0-9]{3})|0 ?(?:\\([0-9]{3}\\)|[0-9]{3})` +
  `|\\(0[0-9]{3}\\)|\\([0-9]{3}\\))[ -]?[0-9]{3}[ -]?[0-9]{2}[ -]?[0-9]{2}` +
  `|5[0-9]{2}[ -][0-9]{3}[ -][0-9]{2}[ -][0-9]{2})(?![0-9])(?![.,][0-9])`);
// Fractions ("3/4'ü" -> dörtte üçü): a numerator below a denominator of 2-10 or 100; not a date or a door number
const FRACTION = re(`(?<![${W}/.,-])(?<!numara )([${D}]{1,2})/([${D}]{1,3})(?![${W}/])(?![.,][${D}])(?:'([${LETTER}]+))?`);
// E-mail and web addresses are spelled out ("nokta", "et"); a bare domain needs one of these endings
const EMAIL = re("(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)+");
const DOMAIN_ENDINGS = "com|net|org|edu|gov|tr|io|info|app|dev";
const URL_ = re(`(?<![${W}@./-])(?:https?://[A-Za-z0-9.-]+|www\\.[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)+` +
  `|[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)*\\.(?:${DOMAIN_ENDINGS})(?![A-Za-z0-9-]))(?:/[^${S}]*)?`);
const ADDRESS_WORDS = new Map([["www", "ve ve ve"], ["com", "kom"], ["tr", "te re"]]);
const ADDRESS_CHARS = new Map([[".", " nokta "], ["/", " bölü "], ["-", " tire "], ["_", " alt çizgi "], ["@", " et "],
  [":", " "]]);
const TRAILING = ".,!?;:)]}'\"";
const APOSTROPHES = ["'", "’", "ʼ"]; // before CHAR_MAP

// int() of a decimal digit: Unicode encodes every digit set (Nd) as a contiguous run 0-9
const ND = /\p{Nd}/u;
// JavaScriptCore can inspect only one UTF-16 unit in Unicode character-class lookbehind. Read neighboring
// code points explicitly where a supplementary-plane digit may precede an apostrophe or a colon.
const beforePoint = (text, offset) => [...text.slice(Math.max(0, offset - 2), offset)].at(-1) ?? "";
const afterPoint = (text, offset) => String.fromCodePoint(text.codePointAt(offset + 1) ?? 0);
const SUFFIX_BEFORE = re(`^[${ALNUM}%$₺€£°]$`, "u"), SUFFIX_AFTER = re(`^[${LETTER}]$`, "u");
function digitValue(c) {
  const cp = c.codePointAt(0);
  if (cp <= 0x39) return cp - 0x30;
  let zero = cp;
  while (ND.test(String.fromCodePoint(zero - 1))) zero--;
  return (cp - zero) % 10;
}
const toInt = (digits) => BigInt([...digits].map(digitValue).join(""));

function belowThousand(n) {
  const words = [];
  const h = Math.floor(n / 100), rest = n % 100;
  if (h) words.push(...(h === 1 ? [] : [ONES[h]]), "yüz");
  const t = Math.floor(rest / 10), o = rest % 10;
  if (t) words.push(TENS[t]);
  if (o) words.push(ONES[o]);
  return words;
}

/** Verbalise a non-negative integer (number or BigInt) in Turkish (1100 -> "bin yüz"). */
export function numberToWords(n) {
  n = BigInt(n);
  if (n === 0n) return "sıfır";
  const words = [];
  for (const [value, name] of SCALES) {
    const q = n / value;
    n %= value;
    if (q) {
      // Turkish says "bin" (not "bir bin") but "bir milyon"
      words.push(...(q === 1n && name === "bin" ? [] : q < 1000n ? belowThousand(Number(q))
        : numberToWords(q).split(" ")), name);
    }
  }
  return [...words, ...belowThousand(Number(n))].join(" ");
}

const digitsToWords = (s) => [...s].map((c) => DIGIT_NAMES[digitValue(c)]).join(" ");

const VOWELS = "aeıioöuü";
const HARMONY = { a: "ı", ı: "ı", e: "i", i: "i", o: "u", u: "u", ö: "ü", ü: "ü" };
const isVowel = (c) => c.length > 0 && VOWELS.includes(c);

/** Turkish ordinal (14 -> "on dördüncü") using four-way vowel harmony. */
export function ordinalToWords(n) {
  const head = numberToWords(n).split(" ");
  let last = head.pop();
  const v = HARMONY[[...last].filter(isVowel).at(-1)];
  last = isVowel(last.at(-1)) ? last + "nc" + v : (last === "dört" ? "dörd" : last) + v + "nc" + v;
  return [...head, last].join(" ");
}

function romanToInt(s) {
  const values = [...s].map((c) => ({ I: 1, V: 5, X: 10 })[c]);
  return values.reduce((sum, v, i) => sum + (v < (values[i + 1] ?? 0) ? -v : v), 0);
}

/** Re-harmonise a suffix written for an abbreviation onto its spoken form (lira + ye -> liraya). */
function attach(word, suffix) {
  if (!suffix) return word;
  let chars = [...turkishLower(suffix)];
  if (!isVowel(word.at(-1)) && chars.length > 1 && "nsy".includes(chars[0]) && isVowel(chars[1])) {
    chars = chars.slice(1); // buffer consonants only follow a vowel: kg'ye -> kilograma
  }
  if ("dt".includes(chars[0])) chars[0] = "çfhkpsşt".includes(word.at(-1)) ? "t" : "d";
  for (let c of chars) {
    const last = [...word].reverse().find(isVowel);
    if ("ae".includes(c)) c = "aıou".includes(last) ? "a" : "e";
    else if ("ıiuü".includes(c) && !word.endsWith("k")) c = HARMONY[last]; // "-ki" does not harmonise: TL'deki
    word += c;
  }
  return word;
}

/** Clock time: 14:30 -> "on dört otuz", 10:00 -> "on", 09:05 -> "dokuz sıfır beş". */
function verbaliseTime(_, hourDigits, minute) {
  const hour = toInt(hourDigits);
  let words = numberToWords(hour);
  if (minute !== "00") words += " " + (minute[0] === "0" ? digitsToWords(minute) : numberToWords(toInt(minute)));
  else if (hour === 0n) words += " sıfır";
  return ` ${words} `;
}

const THOUSANDS = re(`^[${D}]{1,3}(?:\\.[${D}]{3})+(?:,[${D}]+)?$`, "u"); // 1.250.000(,5): dots group thousands
const DECIMAL = re(`^[${D}]+(?:,[${D}]+)?$`, "u");
const cardinal = (g) => ([...g].length <= 15 ? numberToWords(toInt(g)) : digitsToWords(g));

function verbaliseNumber(raw) {
  if (THOUSANDS.test(raw)) raw = raw.replaceAll(".", "");
  let text;
  if (DECIMAL.test(raw)) {
    const [integer, fraction] = raw.split(",");
    text = cardinal(integer);
    if (fraction) {
      const lead = fraction.length - fraction.replace(/^0+/, "").length;
      const frac = [...Array(lead).fill("sıfır"), ...(/[^0]/.test(fraction) ? [numberToWords(toInt(fraction))] : [])];
      text += " virgül " + (frac.length ? frac : ["sıfır"]).join(" ");
    }
  } else { // dates, versions: read every group on its own (28.10.2014, 14.30)
    text = raw.split(/[.,]/).map(cardinal).join(" ");
  }
  return ` ${text} `;
}

/** "." when the abbreviation `match` (at `offset` of `string`) also ends the sentence (STOP), else "". */
function stop(match, offset, string) {
  STOP.lastIndex = offset + match.length;
  return STOP.test(string) ? "." : "";
}

const letterName = (letter) => LETTER_NAMES[letter.replace("i", "İ").toUpperCase()] ?? letter;

/** "" when a suffix follows the match (T.C.'nin: the apostrophe must stay next to a letter), else " ". */
const glue = (match, offset, string) => (APOSTROPHES.includes(string[offset + match.length]) ? "" : " ");

/** T.C. -> "te ce" (a suffix stays attached: T.C.'nin -> "te cenin"). */
function spellDotted(match, offset, string) {
  const word = DOTTED_WORDS.get(match) ?? [...match].filter((c) => c !== ".").map(letterName).join(" ");
  return ` ${word}${stop(match, offset, string)}${glue(match, offset, string)}`;
}

function spellInitials(match, offset, string) {
  const letters = match.match(INITIAL);
  if (letters.length === 1 && ("IVX".includes(letters[0]) || LETTERED.test(turkishLower(string.slice(0, offset))))) {
    return match;
  }
  return letters.map(letterName).join(" ");
}

/** A group of a phone number: 532 -> "beş yüz otuz iki", 05 -> "sıfır beş". */
function digitGroup(group) {
  const rest = group.replace(/^0+/, "");
  return [...Array(group.length - rest.length).fill("sıfır"), ...(rest ? [numberToWords(BigInt(rest))] : [])].join(" ");
}

function verbalisePhone(match, offset, string) {
  let digits = [...match].filter((c) => c >= "0" && c <= "9").join("");
  const parts = [];
  if (match.startsWith("+") || digits.startsWith("0090")) {
    parts.push("artı doksan");
    digits = digits.slice(digits.startsWith("0090") ? 4 : 2);
  }
  const trunk = digits.length === 11 ? "sıfır " : "";
  if (trunk) digits = digits.slice(1);
  parts.push(trunk + digitGroup(digits.slice(0, 3)), digitGroup(digits.slice(3, 6)), digitGroup(digits.slice(6, 8)),
    digitGroup(digits.slice(8, 10)));
  return ` ${parts.join(", ")}${glue(match, offset, string)}`;
}

function verbaliseFraction(match, a, b, suffix) {
  const numerator = toInt(a), denominator = toInt(b);
  if (!(0n < numerator && numerator < denominator && (denominator <= 10n || denominator === 100n))) return match;
  return ` ${attach(numberToWords(denominator), "de")} ${attach(numberToWords(numerator), suffix)} `;
}

/** An e-mail or web address without its scheme, query and fragment: dots, slashes and dashes by name. */
function speakAddress(address) {
  address = address.replace(/^https?:\/\//, "").split(/[?#]/)[0].replace(/\/+$/, "");
  const slash = address.indexOf("/");
  const host = slash < 0 ? address : address.slice(0, slash), path = slash < 0 ? "" : address.slice(slash);
  const at = host.lastIndexOf("@");
  const domain = host.slice(at + 1).split(".").map((label) => ADDRESS_WORDS.get(label) ?? label).join(".");
  return [...(host.slice(0, at + 1) + domain + path)].map((c) => ADDRESS_CHARS.get(c) ?? c).join("");
}

function verbaliseAddress(match, offset, string) {
  let end = match.length; // the punctuation after it belongs to the sentence
  while (end > 0 && TRAILING.includes(match[end - 1])) end--;
  const tail = match.slice(end);
  return ` ${speakAddress(match.slice(0, end))}${tail || glue(match, offset, string)}${tail ? " " : ""}`;
}

function expandAbbreviation(match, abbreviation, offset, string) {
  abbreviation = abbreviation.toLowerCase();
  const word = ABBREVIATIONS[abbreviation];
  // "vb.", "vs.", "vd." close a list, so their dot may also end the sentence ("elma vb. Sonra ...")
  SENTENCE_START.lastIndex = offset + match.length;
  const endsList = ["vb", "vs", "vd"].includes(abbreviation) && SENTENCE_START.test(string);
  return endsList ? word + "." : word;
}

// unicodedata.combining(c) != 0, a non-zero canonical combining class: canonical reordering (NFD) moves such a mark
// in front of U+0345 (class 240) or behind U+0334 (class 1). Code points that decompose are starters once in NFC.
const isCombining = (c) => /\p{M}/u.test(c) && c.normalize("NFD") === c &&
  (("\u0345" + c).normalize("NFD")[0] !== "\u0345" || (c + "\u0334").normalize("NFD")[0] === "\u0334");

const SPACES = re(`^[${S}]+|[${S}]+$`);
const strip = (s) => s.replace(SPACES, ""); // str.strip()

export const turkishLower = (text) => text.replaceAll("I", "ı").replaceAll("İ", "i").toLowerCase();

/** Normalise raw Turkish text to the symbol alphabet (lower-case letters, space and ".,!?"). */
export function normalize(text) {
  // drop combining marks left after NFC, e.g. the U+0307 in "i\u0307" that "İ".toLowerCase() produces
  text = [...text.normalize("NFC")].filter((c) => !isCombining(c)).join("");
  text = text.replace(EMAIL, verbaliseAddress); // before "@" and "+" are mapped
  text = text.replace(URL_, verbaliseAddress);
  text = text.replace(TELEPHONE, "telefon ");
  text = text.replace(PHONE, verbalisePhone);
  text = [...text].map((c) => CHAR_MAP[c] ?? c).join("");
  text = text.replace(DOTTED, spellDotted); // before the apostrophes: a suffix stays on the last letter's name
  text = text.replace(AFTER_NAME_ABBREVIATION, (m, name, offset, string) => `${AFTER_NAME[name]}${stop(m, offset, string)}`);
  // an apostrophe between a letter/digit/unit and a letter starts a suffix (Anadolu'ya, 2021'de, $'a); others are
  // quote marks ('evet') and become spaces
  text = text.replace(/'/gu, (_, offset, input) =>
    SUFFIX_BEFORE.test(beforePoint(input, offset)) && SUFFIX_AFTER.test(afterPoint(input, offset)) ? "'" : " ");
  text = text.replace(TITLE, (_, title) => `${TITLES[title]} `);
  text = text.replace(ABBREVIATION, expandAbbreviation);
  text = text.replace(re(`(?<![${W}])[Nn]o[.:]?[${S}]*(?=[${D}])`), "numara ");
  text = text.replace(FRACTION, verbaliseFraction);
  text = text.replace(TIME, verbaliseTime); // before ":" becomes ","
  text = text.replace(/:/gu, (_, offset, input) =>
    ND.test(beforePoint(input, offset)) && ND.test(afterPoint(input, offset)) ? " " : ","); // 3:1 -> "üç bir"
  text = text.replace(SPEED, "saatte $1 kilometre");
  text = text.replace(PREFIX_CURRENCY, "$2 $1");
  text = text.replace(UNIT, (_, n, unit, suffix) => `${n} ${attach(UNITS[unit], suffix)}`);
  text = text.replace(re(`([₺$€£])(?:'([${LETTER}]+))?`), (_, c, suffix) => ` ${attach(UNITS[c], suffix)} `);
  text = text.replace(re(`%[${S}]*([${D}])`), " yüzde $1");
  text = text.replace(re(`(${NUMBER})[${S}]*%`), " yüzde $1");
  text = text.replace(re(`(?<![${W}])[${UPPER}]{2,}(?![${W}])`), (m) => ACRONYMS[m] ?? m);
  text = text.replace(re(`(?<![${W}.,])-(?=[${D}])`), " eksi "); // a minus starts a word; 3-5 stays a range
  text = text.replace(INITIALS, spellInitials);
  text = text.replace(ROMAN_ORDINAL, (_, roman) => ` ${ordinalToWords(romanToInt(roman))} `);
  text = text.replace(ORDINAL, (_, a, b) => ` ${ordinalToWords(toInt(a ?? b))} `);
  text = text.replace(re(NUMBER), verbaliseNumber);
  text = text.replace(re(`[${S}]*'`), ""); // glue suffixes: "iki bin yirmi bir 'de" -> "iki bin yirmi birde"
  text = turkishLower(text);
  text = text.replaceAll('"', " ").replaceAll("-", " ").replaceAll("/", " ");
  text = [...text].map((c) => (SYMBOL_TO_ID.has(c) ? c : " ")).join("");
  text = text.replace(re(`[${S}]+([.,!?])`), "$1"); // no space before punctuation
  text = text.replace(re(`([.,!?])(?=[^${S}.,!?])`), "$1 ");
  text = strip(text.replace(re(`[${S}]+`), " "));
  return text.replace(/^[.,!? ]+/u, "");
}

/** Map text to symbol ids; optionally intersperse a blank token (Grad-TTS) for alignment. */
export function textToIds(text, { intersperseBlank = true, normalized = false } = {}) {
  if (!normalized) text = normalize(text);
  const ids = [...text].map((c) => {
    if (!SYMBOL_TO_ID.has(c)) throw new Error(`not in the symbol alphabet: ${JSON.stringify(c)}`);
    return SYMBOL_TO_ID.get(c);
  });
  if (!intersperseBlank) return ids;
  const out = new Array(2 * ids.length + 1).fill(BLANK_ID);
  ids.forEach((id, i) => { out[2 * i + 1] = id; });
  return out;
}

/** Split normalised text into sentences (the generator is trained on <= 16 s utterances). */
export function splitSentences(text, maxChars = 180) {
  const out = [];
  for (const part of text.split(re(`(?<=[.!?])[${S}]+`, "u"))) {
    let s = [...strip(part)]; // code points, as Python indexes strings
    const rfind = (c) => (maxChars > 0 ? s.lastIndexOf(c, maxChars - 1) : -1); // s.rfind(c, 0, max_chars)
    while (s.length > maxChars) { // very long sentences: split at the last comma / space before the limit
      let cut = Math.max(rfind(","), rfind(" "));
      cut = cut > 0 ? cut : maxChars;
      out.push(strip(s.slice(0, cut + 1).join("")));
      s = [...strip(s.slice(cut + 1).join(""))];
    }
    if (s.length) out.push(s.join(""));
  }
  return out;
}

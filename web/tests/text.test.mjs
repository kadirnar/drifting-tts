// Parity test of web/text.js against the Python frontend: node web/tests/text.test.mjs
// Fixtures: PYTHONPATH=$PWD python scripts/make_web_fixtures.py
import { readFileSync } from "node:fs";
import {
  BLANK_ID, PAD_ID, SYMBOL_TO_ID, SYMBOLS, normalize, numberToWords, ordinalToWords, splitSentences, textToIds,
} from "../text.js";

const fx = JSON.parse(readFileSync(new URL("./text_fixtures.json", import.meta.url), "utf8"));
const failures = [];
let checks = 0;

function check(name, input, expected, run) {
  checks++;
  let actual;
  try {
    actual = run();
  } catch (e) {
    actual = `<throws ${e}>`;
  }
  if (JSON.stringify(actual) !== JSON.stringify(expected)) failures.push({ name, input, expected, actual });
  return JSON.stringify(actual) === JSON.stringify(expected);
}

check("SYMBOLS", null, fx.symbols, () => SYMBOLS);
check("SYMBOL_TO_ID", null, fx.symbol_to_id, () => Object.fromEntries(SYMBOL_TO_ID));
check("PAD_ID, BLANK_ID", null, [fx.pad_id, fx.blank_id], () => [PAD_ID, BLANK_ID]);
for (const [n, words] of fx.number_to_words) check("numberToWords", n, words, () => numberToWords(BigInt(n)));
for (const [n, words] of fx.ordinal_to_words) check("ordinalToWords", n, words, () => ordinalToWords(BigInt(n)));

const passed = {};
for (const c of fx.cases) {
  const ok = [
    check("normalize", c.input, c.normalized, () => normalize(c.input)),
    check("textToIds", c.input, c.ids, () => textToIds(c.input)),
    check("textToIds plain", c.input, c.ids.filter((_, i) => i % 2), () => textToIds(c.input, { intersperseBlank: false })),
    check("textToIds normalized", c.normalized, c.ids, () => textToIds(c.normalized, { normalized: true })),
    check("splitSentences", c.normalized, c.split, () => splitSentences(c.normalized)),
    check("splitSentences 40", c.normalized, c.split_short, () => splitSentences(c.normalized, 40)),
    check("splitSentences raw 30", c.input, c.split_raw, () => splitSentences(c.input, 30)),
  ].every(Boolean);
  passed[c.source] ??= [0, 0];
  passed[c.source][0] += ok;
  passed[c.source][1] += 1;
}

for (const f of failures.slice(0, 15)) {
  console.log(`FAIL ${f.name}\n  input:    ${JSON.stringify(f.input)}\n  expected: ${JSON.stringify(f.expected)}\n` +
    `  actual:   ${JSON.stringify(f.actual)}`);
}
for (const [source, [ok, n]] of Object.entries(passed)) console.log(`${source}: ${ok}/${n} cases pass`);
console.log(`${checks - failures.length}/${checks} checks pass` + (failures.length ? `, ${failures.length} FAIL` : ""));
process.exitCode = failures.length ? 1 : 0;

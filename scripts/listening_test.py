"""A blind A/B listening page from two folders of renditions of the same sentences (#85).

No judge in this repository hears what a listener calls robotic (docs/HEAD_REALISM.md), so a change that is meant to
be heard needs a listening test. This script pairs the wav files of two systems by file name, puts each pair in a
random left / right order and writes **one self-contained HTML file** (the audio is embedded): the listener plays A
and B, says which sounds more natural on a seven-point scale (comparison MOS: -3 "A much better" ... +3 "B much
better") and at the end the page reveals the result and offers the answers as a JSON file.

    python scripts/listening_test.py --a outputs/eval/base/wav --b outputs/eval/new/wav \
        --name-a baseline --name-b "MAE kernel" --num 20 --out outputs/listening/base_vs_new.html
    python scripts/listening_test.py --score answers.json        # CMOS, its 95% interval and a sign test

The score is from system A to system B: positive means B is preferred. ``--match`` keeps only the files whose name
contains a string (e.g. one temperature of ``drifting-tts evaluate``'s ``wav`` folder).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
from pathlib import Path

SCALE = [(-3, "A much better"), (-2, "A better"), (-1, "A slightly better"), (0, "the same"),
         (1, "B slightly better"), (2, "B better"), (3, "B much better")]


def pairs(a: Path, b: Path, match: str | None = None) -> list[str]:
    """File names present in both folders (sorted), optionally only those containing ``match``."""
    names = sorted({p.name for p in a.glob("*.wav")} & {p.name for p in b.glob("*.wav")})
    return [n for n in names if match is None or match in n]


def encode_wav(path: Path, max_seconds: float | None = None) -> str:
    """A wav file as a base64 ``data:`` URI: mono, 16-bit, its own sample rate, at most ``max_seconds``."""
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if max_seconds:
        wav = wav[: int(max_seconds * sr)]
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def build_items(a: Path, b: Path, names: list[str], seed: int, max_seconds: float | None = None) -> list[dict]:
    """One item per file: the two renditions in a random order. ``flip`` is true when system B is played as "A";
    half of the items are flipped (to within one), so a listener who always answers one side scores zero."""
    rng = random.Random(seed)
    flips = [i % 2 == 1 for i in range(len(names))]
    rng.shuffle(flips)
    order = list(range(len(names)))
    rng.shuffle(order)
    items = []
    for i in order:
        first, second = (b, a) if flips[i] else (a, b)
        items.append({"id": names[i], "flip": flips[i], "left": encode_wav(first / names[i], max_seconds),
                      "right": encode_wav(second / names[i], max_seconds)})
    return items


def score(answers: list[dict]) -> dict:
    """Comparison MOS of system B over system A from answers ``{"id", "flip", "rating"}`` (rating: -3 .. 3 for the
    side played as B): the mean, a 95% interval (normal approximation of the mean) and a two-sided sign test over
    the items with a preference."""
    vals = [(-r["rating"] if r["flip"] else r["rating"]) for r in answers if r.get("rating") is not None]
    n = len(vals)
    if not n:
        return {"n": 0}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    wins, losses = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
    m, k = wins + losses, min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
    return {"n": n, "cmos": mean, "ci95": 1.96 * sd / math.sqrt(n), "b_preferred": wins, "a_preferred": losses,
            "ties": n - m, "sign_test_p": p}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg: #fbfaf7; --fg: #1c1b19; --muted: #6b675f; --line: #dedad0; --accent: #2f5d50; --card: #ffffff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #171614; --fg: #ecebe6; --muted: #a09c92; --line: #34322d; --accent: #8fc7b5; --card: #201f1c; } }
:root[data-theme="dark"] { --bg: #171614; --fg: #ecebe6; --muted: #a09c92; --line: #34322d; --accent: #8fc7b5;
  --card: #201f1c; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
main { max-width: 720px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.35rem; margin: 0 0 4px; }
p.lead { color: var(--muted); margin: 0 0 20px; }
.item { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px;
  margin: 0 0 16px; }
.item h2 { font-size: .95rem; margin: 0 0 10px; color: var(--muted); font-weight: 600; }
.players { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.players label { font-weight: 600; display: block; margin-bottom: 4px; }
audio { width: 100%; }
.scale { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.scale button { flex: 1 1 84px; padding: 8px 6px; border: 1px solid var(--line); border-radius: 8px;
  background: transparent; color: var(--fg); font: inherit; font-size: .82rem; cursor: pointer; }
.scale button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--bg); }
.bar { position: sticky; bottom: 0; background: var(--bg); border-top: 1px solid var(--line); padding: 12px 16px;
  display: flex; gap: 12px; align-items: center; justify-content: center; flex-wrap: wrap; }
.bar button { padding: 9px 16px; border-radius: 8px; border: 1px solid var(--accent); background: var(--accent);
  color: var(--bg); font: inherit; cursor: pointer; }
.bar button:disabled { opacity: .45; cursor: default; }
#result { background: var(--card); border: 1px solid var(--accent); border-radius: 10px; padding: 16px;
  margin: 0 0 16px; }
#result table { border-collapse: collapse; width: 100%; }
#result td { padding: 4px 8px; border-bottom: 1px solid var(--line); }
@media (max-width: 520px) { .players { grid-template-columns: 1fr; } }
</style></head><body><main>
<h1>__TITLE__</h1>
<p class="lead">__N__ pairs. Each pair is the same sentence from two systems, in a random order. Play A and B, then
say which sounds more natural, less robotic. The systems are revealed at the end.</p>
<div id="result" hidden></div>
<div id="items"></div>
</main>
<div class="bar"><span id="progress"></span><button id="finish" disabled>Show the result</button>
<button id="save" disabled>Save answers (JSON)</button></div>
<script>
const ITEMS = __ITEMS__, SCALE = __SCALE__, KEY = JSON.parse(atob("__KEY__"));
const answers = new Array(ITEMS.length).fill(null);
const root = document.getElementById("items");
ITEMS.forEach((it, i) => {
  const d = document.createElement("div"); d.className = "item";
  d.innerHTML = `<h2>Pair ${i + 1} of ${ITEMS.length}</h2><div class="players">` +
    `<div><label>A</label><audio controls preload="none" src="${it.left}"></audio></div>` +
    `<div><label>B</label><audio controls preload="none" src="${it.right}"></audio></div></div>` +
    `<div class="scale" role="group" aria-label="Which sounds more natural?"></div>`;
  const s = d.querySelector(".scale");
  SCALE.forEach(([v, text]) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = text; b.setAttribute("aria-pressed", "false");
    b.onclick = () => {
      answers[i] = v;
      s.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      update();
    };
    s.appendChild(b);
  });
  root.appendChild(d);
});
function update() {
  const n = answers.filter(a => a !== null).length;
  document.getElementById("progress").textContent = `${n} / ${ITEMS.length} answered`;
  document.getElementById("finish").disabled = n === 0;
  document.getElementById("save").disabled = n === 0;
}
function rows() { return ITEMS.map((it, i) => ({id: it.id, flip: KEY.flips[i], rating: answers[i]})); }
function score() {
  const v = rows().filter(r => r.rating !== null).map(r => r.flip ? -r.rating : r.rating), n = v.length;
  const mean = v.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
  return {n, cmos: mean, ci95: 1.96 * sd / Math.sqrt(n), b: v.filter(x => x > 0).length,
          a: v.filter(x => x < 0).length};
}
document.getElementById("finish").onclick = () => {
  const s = score(), r = document.getElementById("result");
  r.hidden = false;
  r.innerHTML = `<strong>${KEY.b} against ${KEY.a}</strong><table>` +
    `<tr><td>comparison MOS (positive: ${KEY.b} preferred)</td>` +
    `<td>${s.cmos.toFixed(2)} ± ${s.ci95.toFixed(2)}</td></tr>` +
    `<tr><td>pairs where ${KEY.b} was preferred</td><td>${s.b} of ${s.n}</td></tr>` +
    `<tr><td>pairs where ${KEY.a} was preferred</td><td>${s.a} of ${s.n}</td></tr></table>`;
  r.scrollIntoView({behavior: "smooth"});
};
document.getElementById("save").onclick = () => {
  const data = JSON.stringify({title: document.title, a: KEY.a, b: KEY.b, answers: rows()}, null, 1);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([data], {type: "application/json"}));
  a.download = "listening_answers.json"; a.click();
};
update();
</script></body></html>
"""


def render(items: list[dict], name_a: str, name_b: str, title: str) -> str:
    key = base64.b64encode(json.dumps({"a": name_a, "b": name_b, "flips": [it["flip"] for it in items]}).encode())
    shown = [{"id": it["id"], "left": it["left"], "right": it["right"]} for it in items]  # no ``flip`` in the page
    return (PAGE.replace("__TITLE__", title).replace("__N__", str(len(items)))
            .replace("__ITEMS__", json.dumps(shown)).replace("__SCALE__", json.dumps(SCALE))
            .replace("__KEY__", key.decode()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, help="folder of system A's wav files")
    ap.add_argument("--b", type=Path, help="folder of system B's wav files (the same file names)")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--match", default=None, help="only files whose name contains this")
    ap.add_argument("--num", type=int, default=20, help="pairs on the page (a random subset if there are more)")
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="Which sounds more natural?")
    ap.add_argument("--out", type=Path, default=Path("outputs/listening/page.html"))
    ap.add_argument("--score", type=Path, default=None, help="score a saved answers JSON instead of making a page")
    args = ap.parse_args()

    if args.score:
        saved = json.loads(args.score.read_text())
        s = score(saved["answers"])
        print(f"{saved.get('b', 'B')} against {saved.get('a', 'A')}: CMOS {s['cmos']:+.2f} ± {s['ci95']:.2f} "
              f"({s['n']} pairs; preferred {s['b_preferred']} / {s['a_preferred']}, {s['ties']} ties; "
              f"sign test p = {s['sign_test_p']:.3f})")
        return
    if not args.a or not args.b:
        ap.error("--a and --b are required")
    names = pairs(args.a, args.b, args.match)
    if not names:
        raise SystemExit(f"no wav files with the same name in {args.a} and {args.b}")
    names = sorted(random.Random(args.seed).sample(names, min(args.num, len(names))))
    page = render(build_items(args.a, args.b, names, args.seed, args.max_seconds), args.name_a, args.name_b,
                  args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"{len(names)} pairs -> {args.out} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

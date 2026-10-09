"""Prepare the files of a release for upload to Vyvo/drifting-tts-tr: sanitise, verify, scan (no upload).

Writes ``<out>/vocos_v2.pt`` (``--vocoder``), ``<out>/prosody_drift_v3.2.pt`` (``--prosody``) and, only when the
acoustic model changes, ``<out>/drifting_tts_v3.2.pt`` (``--model``), with only what the loaders read
(:mod:`drifting_tts.publish`). Each file is reloaded on the CPU and checked against its source (identical weights /
outputs), then scanned for absolute paths and for the data root's private names (speaker, show and dataset names,
counted, never printed). ``MANIFEST.json`` lists sizes and SHA-256 (it stays local: it names the sources).

    python scripts/prepare_release.py --out runs/rel_publish --prosody runs/pm_drift_final/prosody_ema.pt \
        --vocoder runs/voc_long/vocos_ft_160000.pt --tts runs/release/drifting_tts_v3.1.pt --data data/train

Then ``DRIFTING_TTS_HUB_DIR=<out> drifting-tts synthesize --release v3.2 ...`` runs the release from these files
exactly as users will get it from the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from drifting_tts.models.prosody_net import ProsodyPredictor, export_checkpoint
from drifting_tts.publish import export_tts, export_vocos, private_terms, scan
from drifting_tts.text import text_to_ids

TEXTS = ["merhaba, bu cümle tek adımda üretildi.", "yarın saat on dörtte toplantı var mı?"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_prosody(src: str, dst: Path, tts) -> None:
    a, b = ProsodyPredictor.load(src, "cpu", tts=tts), ProsodyPredictor.load(dst, "cpu", tts=tts)
    assert a.temperature == b.temperature and a.duration_scales == b.duration_scales, "metadata changed"
    for text in TEXTS:
        ids = torch.tensor([text_to_ids(text, normalized=True)])
        for spk in (722, 389, 323):
            out = []
            for p in (a, b):
                g = torch.Generator().manual_seed(spk)
                out.append(p.predict(tts, ids, torch.tensor([ids.shape[1]]), torch.tensor([spk]), p.temperature or 1.0,
                                     p.duration_scales.get(spk, 1.0), generator=g))
            assert all(torch.equal(x, y) for x, y in zip(*out)), f"prosody differs: {text!r}, speaker {spk}"


def check_vocoder(src: str, dst: Path) -> None:
    from drifting_tts.vocoder import load_vocoder

    a, b = load_vocoder(src, "cpu"), load_vocoder(str(dst), "cpu")
    assert (a.kind, a.mel, a.context) == (b.kind, b.mel, b.context)
    mel = torch.randn(1, 100, 64, generator=torch.Generator().manual_seed(0)) - 5
    assert torch.equal(a(mel), b(mel)), "vocoder output differs"


def check_tts(src: str, dst: Path) -> None:
    from drifting_tts.train import load_tts

    (a, _, sa), (b, _, sb) = load_tts(src), load_tts(dst)
    assert sa == sb and a.duration_scales == b.duration_scales and a.temperature == b.temperature
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="staging directory (laid out like the Hub repo)")
    p.add_argument("--prosody", help="train-prosody checkpoint (prosody_ema.pt) -> prosody_drift_v3.2.pt")
    p.add_argument("--vocoder", help="finetune-vocoder Vocos checkpoint (vocos_ft_<step>.pt) -> vocos_v2.pt")
    p.add_argument("--model", help="a new acoustic model (model_ema.pt) -> drifting_tts_v3.2.pt; omit to keep v3.1's")
    p.add_argument("--tts", default="runs/release/drifting_tts_v3.1.pt",
                   help="the release's acoustic model: the prosody predictor's fingerprint and check")
    p.add_argument("--data", default=None, help="prepared data root whose private names must not appear")
    p.add_argument("--scan", nargs="*", default=[], help="also scan these files (e.g. already published ones)")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    terms = private_terms(args.data) if args.data else {}
    manifest, ok = {}, True

    def done(name: str, src: str | None) -> None:
        path = out / name
        manifest[name] = {"bytes": path.stat().st_size, "sha256": sha256(path), "source": src}
        print(f"{name}: {path.stat().st_size / 2**20:.1f} MB, sha256 {manifest[name]['sha256'][:16]}..., "
              f"verified against its source")

    tts_path = args.tts
    if args.model:
        torch.save(export_tts(torch.load(args.model, map_location="cpu", weights_only=False)),
                   out / "drifting_tts_v3.2.pt")
        check_tts(args.model, out / "drifting_tts_v3.2.pt")
        done("drifting_tts_v3.2.pt", args.model)
        tts_path = str(out / "drifting_tts_v3.2.pt")
    if args.prosody:
        from drifting_tts.train import load_tts

        tts = load_tts(tts_path)[0]
        ck = torch.load(args.prosody, map_location="cpu", weights_only=False)
        torch.save(export_checkpoint(ck, tts), out / "prosody_drift_v3.2.pt")
        check_prosody(args.prosody, out / "prosody_drift_v3.2.pt", tts)
        done("prosody_drift_v3.2.pt", args.prosody)
    if args.vocoder:
        torch.save(export_vocos(torch.load(args.vocoder, map_location="cpu", weights_only=False)), out / "vocos_v2.pt")
        check_vocoder(args.vocoder, out / "vocos_v2.pt")
        done("vocos_v2.pt", args.vocoder)

    print("scan (hits per category; 0 = clean):")
    for path in [out / n for n in manifest] + [Path(x) for x in args.scan]:
        hits = scan(path, terms)
        ok &= path.name not in manifest or not any(hits.values())
        print(f"  {path.name}: " + ", ".join(f"{k} {v}" for k, v in hits.items()))
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    if not ok:
        sys.exit("a staged file contains private strings: do not upload it")


if __name__ == "__main__":
    main()

"""Bundle the browser frontend and Python-generated parity fixtures into the Swift core package.

Run from any directory. --check checks committed resources without writing them.
Only top-level named const/function exports are accepted; imports or other module syntax fail closed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "swift" / "DriftingTTSCore"


def bundle(source: str) -> str:
    export = re.compile(r"^export (?=(?:const|function)\b)", re.MULTILINE)
    script, count = export.subn("", source)
    if not count or re.search(r"^\s*(?:import|export)\b", script, re.MULTILINE):
        raise ValueError("frontend must contain only supported named const/function exports and no imports")
    return (
        "// Generated from web/text.js by scripts/sync_swift_frontend.py; do not edit.\n"
        '"use strict";\nglobalThis.DriftingText = (() => {\n' + script
        + "\nreturn Object.freeze({normalize, textToIds, splitSentences, numberToWords, ordinalToWords,\n"
        + "symbols: Object.freeze(SYMBOLS), padID: PAD_ID, blankID: BLANK_ID});\n})();\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    resources = {
        PACKAGE / "Sources/DriftingTTSCore/Resources/TurkishFrontend.js": bundle((ROOT / "web/text.js").read_text()),
        PACKAGE / "Tests/DriftingTTSCoreTests/Resources/text_fixtures.json": (
            ROOT / "web/tests/text_fixtures.json"
        ).read_text(),
    }
    for path, content in resources.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"stale Swift resource: {path.relative_to(ROOT)}; run {Path(__file__).name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print("Swift frontend resources are current" if args.check else "Updated Swift frontend resources")


if __name__ == "__main__":
    main()

"""Command-line entry point: ``drifting-tts <command> [args]``."""

from __future__ import annotations

import argparse
import importlib
import sys

# command name -> module exposing ``add_args(parser)`` and ``run(args)``
COMMANDS: dict[str, str] = {
    "prepare": "drifting_tts.prepare",
    "score": "drifting_tts.score",
    "train-mae": "drifting_tts.train_mae",
    "train": "drifting_tts.train",
    "synthesize": "drifting_tts.synthesize",
    "evaluate": "drifting_tts.evaluate",
    "prosody": "drifting_tts.prosody_eval",
    "benchmark": "drifting_tts.benchmark",
    "merge-data": "drifting_tts.merge_data",
    "calibrate-durations": "drifting_tts.calibrate",
    "finetune-vocoder": "drifting_tts.finetune_vocoder",
    "extract-latents": "drifting_tts.extract_latents",
    "prosody-cache": "drifting_tts.prosody_cache",
    "train-prosody": "drifting_tts.train_prosody",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="drifting-tts")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, module_name in COMMANDS.items():
        module = importlib.import_module(module_name)
        module.add_args(sub.add_parser(name, help=(module.__doc__ or "").strip().splitlines()[0]))
    args = parser.parse_args(argv)
    importlib.import_module(COMMANDS[args.command]).run(args)


if __name__ == "__main__":
    main(sys.argv[1:])

"""`drifting-tts prosody` (drifting_tts.prosody_eval) with the stochastic prosody predictor as extra systems (#39).

Adds systems ``<name>-T<t>[-S<spread>][-pitch]`` for every ``--prosody-model name=checkpoint``: one pass over the whole
text, durations and token pitch sampled by that predictor at prosody temperature t (and output-space spread), with
its own per-voice duration factor, the generator noise as for the other one-pass systems. They join the prosody
evaluation of prosody-eval, so every row uses the same metrics, judges and recordings. Each sampled prosody is
cached per (utterance, seed): the runner asks for it several times (synthesis, token report, F0 alignment, seed
diversity). ``-pitch``: only the token pitch is sampled; the durations stay the regressors' (with v3.1's per-voice
factors). Other arguments: those of ``drifting-tts prosody``.

    python scripts/eval_prosody_audio.py --prosody-model drift=runs/pm_drift/prosody_ema.pt \
        --systems recording onepass oracle-both drift-T0.7 --out runs/pm_audio/val722
"""

from __future__ import annotations

import argparse
import re
import sys

import torch

import drifting_tts.prosody_eval as pe
from drifting_tts.models.prosody_net import ProsodyPredictor

_parse_system = pe.parse_system


def parse_system(name: str):
    """``<predictor>-T<t>``, else the systems of :func:`drifting_tts.prosody_eval.parse_system`."""
    m = re.fullmatch(r"([A-Za-z0-9_]+)-T([\d.]+)(?:-S([\d.]+))?(-pitch)?", name)
    if m and m.group(1) in Runner.predictors:
        spec = ("sampler", m.group(1), float(m.group(2)), float(m.group(3) or 1.0))
        return pe.System(name, "onepass", dur=("pred",) if m.group(4) else spec, pitch=spec)
    return _parse_system(name)


class Runner(pe.Runner):
    predictors: dict[str, ProsodyPredictor] = {}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._seed: dict[int, int] = {}
        self._cache: dict[tuple, tuple] = {}

    def __call__(self, u, s, seed=None, **kw):
        self._seed[u.index] = u.index if seed is None else seed  # the seed of the rendition being made
        return super().__call__(u, s, seed=seed, **kw)

    def prosody(self, u, s, *a, **kw):
        if s.pitch[0] != "sampler":
            return super().prosody(u, s, *a, **kw)
        if s.dur[0] != "sampler":  # -pitch: sampled token pitch, the regressors' durations
            return None, self.prosody(u, pe.System(s.name, "onepass", dur=s.pitch, pitch=s.pitch))[1]
        seed = self._seed.get(u.index, u.index)
        key = (s.name, u.index, seed)
        if key not in self._cache:
            pred = self.predictors[s.pitch[1]]
            g = torch.Generator(device=self.device).manual_seed(seed)
            tempo = pred.duration_scales.get(self.speaker, 1.0)
            frames, pitch = pred.predict(self.model, u.ids, torch.tensor([u.ids.shape[1]], device=self.device),
                                         self.spk, s.pitch[2], tempo, generator=g, spread=s.pitch[3])
            self._cache[key] = (frames, pitch)
        return self._cache[key]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--prosody-model", nargs="+", required=True, help="name=checkpoint (names: letters, digits, _)")
    own, rest = p.parse_known_args(argv)
    parser = argparse.ArgumentParser(prog="eval_prosody_audio")
    pe.add_args(parser)
    args = parser.parse_args(rest)
    for spec in own.prosody_model:
        name, path = spec.split("=", 1)
        Runner.predictors[name] = ProsodyPredictor.load(path, args.device)
    pe.parse_system, pe.Runner = parse_system, Runner
    pe.run(args)


if __name__ == "__main__":
    main(sys.argv[1:])

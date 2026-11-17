"""Evaluation inside a training loop (#102).

Kyutai on the training dynamics of their drifting models: the word error rate saturates within the first few percent
of a run, the predicted MOS sits on a plateau for a long time and then jumps within a few thousand steps. Three
samples on disk cannot show such a jump. Every ``train.eval_every`` steps the training loops

* draw the training objective on fixed held-out batches, without a gradient and with the same noise every time
  (:func:`heldout_metrics`): the loss and its parts, ``p_data`` and the drift loss of every feature map, the spread
  of the samples;
* synthesise ``train.eval.num`` held-out sentences with every EMA and score them (:class:`AudioEval`): WER and CER
  (Whisper), a predicted MOS, the F0 spread, the pauses, and the length against the recording's.

The numbers go to TensorBoard (``eval/...``; ``eval_<decay>/...`` for another EMA) and to ``eval_history.json`` in
the run's directory. The training's random streams are put back afterwards (:func:`fixed_rng`), so a run with
evaluation takes the same steps as one without.

    train:
      eval_every: 5000         # 0: off
      ema_decays: [0.999, 0.9999]   # the first is model_ema.pt, another is model_ema_<decay>.pt
      eval:
        num: 24                # held-out sentences (the val split)
        batches: 8             # held-out batches of the objective
        asr: large-v3          # Whisper model, or none; asr_device: cpu beside a full GPU
        mos: utmosv2           # utmosv2 | utmos22 | none
        f0: true               # F0 spread of the generated audio (WORLD harvest, on the CPU)
        temperature: 0.5       # noise temperature of the generated sentences
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .audio import HOP_LENGTH, SAMPLE_RATE
from .utils import EMA, rng_state, set_rng_state


@contextmanager
def fixed_rng(seed: int):
    """Run a block on its own seed and give the global random streams back as they were."""
    state = rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        set_rng_state(state)


def ema_name(decay: float) -> str:
    """``0.9999`` -> ``"0.9999"``: how an EMA is called in file names and TensorBoard tags."""
    return f"{decay:g}"


def build_emas(model, tc) -> dict[str, EMA]:
    """The run's EMAs by name, the main one first: ``train.ema_decays`` (a list), else ``train.ema_decay`` alone.
    Kyutai keeps several, up to 0.9999; a slow one only pays off late in a long run."""
    decays = [float(d) for d in (tc.get("ema_decays") or [tc.ema_decay])]
    if len(set(decays)) != len(decays):
        raise ValueError(f"train.ema_decays holds a decay twice: {decays}")
    return {ema_name(d): EMA(model, d) for d in decays}


def load_emas(emas: dict[str, EMA], ck: dict) -> None:
    """The EMAs of a ``last.pt``: ``ema`` is the main one, ``emas`` the others by name. An EMA the checkpoint does
    not have (a decay added on resuming) starts from the main one."""
    names = list(emas)
    emas[names[0]].model.load_state_dict(ck["ema"])
    for name in names[1:]:
        state = (ck.get("emas") or {}).get(name)
        if state is None:
            print(f"EMA {name}: not in the checkpoint, it starts from the main EMA", flush=True)
        emas[name].model.load_state_dict(ck["ema"] if state is None else state)


def other_emas(emas: dict[str, EMA]) -> dict[str, dict]:
    """The state dicts of the EMAs after the first, for ``last.pt``'s ``emas``."""
    return {name: e.model.state_dict() for name, e in list(emas.items())[1:]}


@torch.no_grad()
def heldout_metrics(step: Callable[[dict], dict], batches: list[dict], seed: int = 1234) -> dict[str, float]:
    """Mean of ``step(batch)``'s metrics over fixed held-out batches. The noise is seeded the same at every call, so
    two evaluations differ by the model alone."""
    total: dict[str, float] = {}
    with fixed_rng(seed):
        for b in batches:
            for k, v in step(b).items():
                total[k] = total.get(k, 0.0) + float(v)
    return {k: v / max(len(batches), 1) for k, v in total.items()}


class AudioEval:
    """Held-out sentences through a model and a vocoder, judged: see the module docstring.

    ``synthesize(model, item, generator) -> mel`` renders one dataset item (normalised mel ``[1, n_mels, T]``).
    The judges are loaded at the first call and kept; a judge whose package is missing is skipped with a message.
    """

    def __init__(self, ds, vocoder, device, synthesize: Callable, num: int = 24, asr: str | None = "large-v3",
                 mos: str | None = "utmosv2", f0: bool = True, asr_device: str | None = None):
        self.ds, self.vocoder, self.device, self.synthesize = ds, vocoder, device, synthesize
        self.idx = list(range(min(int(num), len(ds))))
        self.asr, self.mos, self.f0, self.asr_device = asr, mos, bool(f0), asr_device
        self.judges = None

    def _judges(self):
        if self.judges is None:
            from .judges import load_judges

            self.judges = load_judges(self.asr, None, self.mos, self.device, language=self.ds.language,
                                      asr_device=self.asr_device)
        return self.judges

    @torch.no_grad()
    def __call__(self, model, seed: int = 0) -> dict[str, float]:
        from .evaluate import _plain, score_utterance, summarize
        from .prosody import FRAME_MS, MIN_PAUSE_S, f0_contour, f0_statistics, frame_level_db, pauses, silent_frames

        judges, rows, extra = self._judges(), [], {"f0_std": [], "pauses": [], "length_ratio": []}
        min_pause = int(round(MIN_PAUSE_S * 1000 / FRAME_MS))
        was = model.training
        model.eval()
        with fixed_rng(seed):
            for i in self.idx:
                it = self.ds[i]
                g = torch.Generator(device=self.device).manual_seed(seed + i)
                wav = self.vocoder(self.ds.stats.denormalize(self.synthesize(model, it, g)))[0].float().cpu()
                rows.append(score_utterance(judges, wav, _plain(self.ds.items[i]["norm_text"], self.ds.language))[0])
                x = wav.numpy().astype(np.float64)
                runs, _ = pauses(silent_frames(frame_level_db(x, SAMPLE_RATE)), min_pause)
                extra["pauses"].append(len(runs))
                extra["length_ratio"].append(len(x) / (it["mel"].shape[-1] * HOP_LENGTH))
                if self.f0:
                    extra["f0_std"].append(f0_statistics(f0_contour(x, SAMPLE_RATE)).get("f0_std", float("nan")))
        model.train(was)
        out = {k: float(v) for k, v in summarize(rows).items() if not k.endswith("_ci")}
        for k, v in extra.items():  # an utterance without voiced frames has no F0 spread
            v = [x for x in v if np.isfinite(x)]
            if v:
                out[k] = float(np.mean(v))
        return out


def record(writer, work: Path, step: int, name: str, main: bool, results: dict[str, float]) -> None:
    """One evaluation of one EMA: TensorBoard scalars (``eval/`` for the main EMA, ``eval_<name>/`` for another), a
    line on the console and a row of ``eval_history.json``."""
    tag = "eval" if main else f"eval_{name}"
    for k, v in results.items():
        writer.add_scalar(f"{tag}/{k}", v, step)
    path = Path(work) / "eval_history.json"
    history = json.loads(path.read_text()) if path.exists() else []
    history = [r for r in history if not (r["step"] == step and r["ema"] == name)]  # a resumed run repeats a step
    history.append({"step": step, "ema": name, **results})
    path.write_text(json.dumps(history, indent=1))
    short = {k: v for k, v in results.items() if "/" not in k}  # the per-map values stay in the files
    print(f"  eval (EMA {name}) " + " ".join(f"{k}={v:.4g}" for k, v in short.items()), flush=True)

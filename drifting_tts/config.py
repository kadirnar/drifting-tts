"""Tiny YAML config system with attribute access and ``a.b.c=value`` overrides."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader that also parses ``5e-4`` (no dot) as a float, unlike YAML 1.1."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


class Config(dict):
    """A ``dict`` whose (nested) keys are also readable as attributes."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(self.to_dict(), memo))

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def merge(base: dict, override: dict) -> Config:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base.to_dict() if isinstance(base, Config) else base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v).to_dict()
        else:
            out[k] = v
    return Config(out)


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply CLI overrides of the form ``train.lr=1e-4`` (values parsed as YAML)."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = out
        *parents, leaf = key.strip().split(".")
        for p in parents:
            if p not in node or not isinstance(node[p], dict):
                node[p] = Config()
            node = node[p]
        node[leaf] = _yaml_load(raw)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML file. A top-level ``base: other.yaml`` key inherits from another file."""
    path = Path(path)
    data = _yaml_load(path.read_text()) or {}
    base = data.pop("base", None)
    cfg = merge(load_config(path.parent / base), data) if base else Config(data)
    return apply_overrides(cfg, overrides or [])


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True))

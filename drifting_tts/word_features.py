"""Contextual word features for the prosody predictor (#40): a Turkish BERT run over the normalised text.

Words are the space-separated pieces of the normalised text (punctuation stays attached), i.e. the segments of
:func:`drifting_tts.models.prosody_net.word_index`. Each word's vector is the mean over its sub-word pieces of the
mean of BERT's last ``layers`` hidden layers. ``dbmdz/bert-base-turkish-cased`` (MIT) is the default: the text is
lower-cased by normalisation, but the uncased model strips diacritics (``gelmiş`` -> ``gelmis``), the cased one
keeps them. Needs ``transformers`` (the ``eval`` extra).
"""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_WORD_MODEL = "dbmdz/bert-base-turkish-cased"


class WordEncoder:
    def __init__(self, name: str = DEFAULT_WORD_MODEL, device: str = "cuda", layers: int = 4):
        from transformers import AutoModel, AutoTokenizer

        self.name, self.device, self.layers = name, device, layers
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> list[Tensor]:
        """Normalised texts -> one ``[n_words, dim]`` float32 tensor per text (CPU)."""
        words = [t.split(" ") for t in texts]
        enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True,
                             max_length=512)
        hs = self.model(**enc.to(self.device), output_hidden_states=True).hidden_states
        h = torch.stack(hs[-self.layers:]).mean(0).float()  # [B, L, dim]
        out = []
        for b, ws in enumerate(words):
            ids = torch.tensor([-1 if w is None else w for w in enc.word_ids(b)], device=h.device)
            onehot = (ids[:, None] == torch.arange(len(ws), device=h.device)[None]).float()  # [L, W]
            cnt = onehot.sum(0)
            out.append(((onehot.T @ h[b]) / cnt.clamp_min(1)[:, None]).cpu())  # words cut by truncation: zeros
        return out

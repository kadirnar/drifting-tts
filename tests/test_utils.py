import random

import numpy as np
import torch
from torch import nn

from drifting_tts.utils import EMA, rng_state, set_rng_state


def test_ema_update_matches_per_tensor_loop():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
    ema = EMA(model, 0.9)
    ref = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        model.train()(torch.randn(5, 4))  # changes the BatchNorm running statistics (buffers)
        ema.update(model)
        ref = [r.lerp(p.detach(), 0.1) for r, p in zip(ref, model.parameters())]
    for e, r in zip(ema.model.parameters(), ref):
        assert torch.allclose(e, r, atol=1e-6)
    for e, b in zip(ema.model.buffers(), model.buffers()):
        assert torch.equal(e, b)
    EMA(nn.Linear(2, 2), 0.5).update(nn.Linear(2, 2))  # a model without buffers


def test_rng_state_roundtrip():
    state = rng_state()
    a = (random.random(), np.random.rand(), torch.rand(3))
    set_rng_state(state)
    b = (random.random(), np.random.rand(), torch.rand(3))
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])

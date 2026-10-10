"""Counter-based noise (drifting_tts.noise): the scheme's properties, the Triton kernel against the PyTorch
reference, and the same audio for a request alone or in a batch. The kernel tests need a GPU."""

import pytest
import torch

from drifting_tts import noise
from drifting_tts.batched import stream_batched
from drifting_tts.chunked import Chunking
from drifting_tts.noise import DiTNoise, Philox, philox_normal, philox_randint

from .test_batched import _synth


def test_philox_normal_is_a_function_of_seed_sentence_stream_and_index():
    seeds, sent = torch.tensor([0, 7, 2**40 + 3]), torch.tensor([0, 2, 1])
    x = philox_normal(seeds, sent, "dit", 100, 64)
    assert x.shape == (3, 100, 64) and abs(float(x.mean())) < 0.05 and abs(float(x.std()) - 1) < 0.05
    # a window from frame 10 is the same values as frames 10.. of the whole draw: frame-major indices
    w = philox_normal(seeds, sent, "dit", 100, 20, starts=torch.tensor([10, 10, 10]))
    torch.testing.assert_close(w, x[..., 10:30], rtol=0, atol=0)
    # other rows of the batch, other padding: the same values
    one = philox_normal(seeds[1:2], sent[1:2], "dit", 100, 30)
    torch.testing.assert_close(one, x[1:2, :, :30], rtol=0, atol=0)
    # lengths zero the rest; streams and sentences differ
    z = philox_normal(seeds, sent, "dit", 100, 64, lengths=torch.tensor([64, 5, 0]))
    assert not z[1, :, 5:].any() and not z[2].any() and torch.equal(z[1, :, :5], x[1, :, :5])
    assert not torch.equal(philox_normal(seeds, sent, "prosody_tok", 100, 64), x)
    assert not torch.equal(philox_normal(seeds, sent + 1, "dit", 100, 64), x)
    labels = philox_randint(seeds, sent, "style", 32, 64)
    assert labels.shape == (3, 32) and labels.min() >= 0 and labels.max() < 64


@pytest.mark.skipif(not (torch.cuda.is_available() and noise.HAS_TRITON), reason="the Triton kernel needs a GPU")
def test_triton_kernel_matches_the_pytorch_reference():
    seeds, sent = torch.tensor([0, 1, 12345, 2**40 + 7]), torch.tensor([0, 0, 3, 1])
    starts, lengths = torch.tensor([0, 5, 0, 64]), torch.tensor([300, 200, 300, 17])
    ref = philox_normal(seeds, sent, "dit", 100, 300, starts=starts, lengths=lengths, scale=0.3)
    got = philox_normal(seeds.cuda(), sent.cuda(), "dit", 100, 300, starts=starts.cuda(), lengths=lengths.cuda(),
                        scale=0.3)
    torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=2e-6)  # libdevice vs PyTorch log / cos / sin
    out = torch.zeros(4, 100, 400, device="cuda")[..., 50:350]  # a strided output
    philox_normal(seeds.cuda(), sent.cuda(), "dit", 100, 300, starts=starts.cuda(), lengths=lengths.cuda(),
                  scale=0.3, out=out)
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=2e-6)


def test_dit_noise_windows():
    n = DiTNoise(Philox([3, 4], [0, 1]), 100, torch.tensor([50, 20]), 0.5)
    full = n.full()
    assert full.shape == (2, 100, 50) and not full[1, :, 20:].any()
    w = n.window(torch.tensor([1, 0]), torch.tensor([4, 10]), 12, torch.tensor([12, 12]))
    torch.testing.assert_close(w[0], full[1, :, 4:16], rtol=0, atol=0)
    torch.testing.assert_close(w[1], full[0, :, 10:22], rtol=0, atol=0)


@pytest.mark.parametrize("prosody", [False, True])
def test_a_request_sounds_the_same_alone_and_in_a_batch(tmp_path, prosody):
    synth = _synth(tmp_path, prosody)
    texts = ["Merhaba! Bugün hava çok güzel.", "Bu bir deneme.", "Yapay zekâ modelleri her geçen gün daha hızlı."]
    ch = Chunking(right=6, left=4, chunk=12, crossfade=2, noise="philox")
    kw = dict(speaker=2, cfg_scale=1.5, temperature=0.5, pause=0.05, first=4)
    got = {i: [] for i in range(len(texts))}
    for out in stream_batched(synth, texts, seeds=[11, 12, 13], chunked=ch, **kw):
        for i, p in out:
            got[i].append(p)
    for i, text in enumerate(texts):
        ref = list(synth.stream(text, seed=11 + i, chunked=ch, **kw))
        assert [p.numel() for p in got[i]] == [p.numel() for p in ref]
        torch.testing.assert_close(torch.cat(got[i]), torch.cat(ref), rtol=0, atol=1e-5)
    # another batch composition: the same audio for request 1
    alone = {0: []}
    for out in stream_batched(synth, [texts[1]], seeds=[12], chunked=ch, **kw):
        for i, p in out:
            alone[i].append(p)
    torch.testing.assert_close(torch.cat(alone[0]), torch.cat(got[1]), rtol=0, atol=1e-5)

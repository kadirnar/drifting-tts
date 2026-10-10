"""Streaming the acoustic model (drifting_tts.chunked): DiT windows, noise slicing, joins, the single-request stream
and its batched form. CPU, tiny models, no downloads (the CUDA-graph test needs a GPU)."""

import pytest
import torch

from drifting_tts.audio import HOP_LENGTH
from drifting_tts.batched import batched_generator, stream_batched, stream_windows, synth_prepare
from drifting_tts.chunked import Chunking, WindowedMel, dit_windows
from drifting_tts.fast import stream_vocoder
from drifting_tts.text import normalize, split_sentences

from .test_batched import _synth


@pytest.mark.parametrize("t", [1, 40, 64, 96, 97, 128, 129, 300, 511, 1000])
@pytest.mark.parametrize("head,chunk,left,right", [(64, 256, 32, 64), (64, 192, 32, 32), (64, 256, 0, 0),
                                                   (16, 8, 4, 6)])
def test_dit_windows_commit_every_frame_once(t, head, chunk, left, right):
    w = dit_windows(t, head, chunk, left, right)
    assert w[0][:3] == (0, min(t, head + right), 0)
    assert [x[2] for x in w[1:]] == [x[3] for x in w[:-1]] and w[-1][3] == t  # contiguous commits up to t
    for a, b, c0, c1 in w:
        assert 0 <= a <= c0 < c1 <= b <= t and c0 - a <= left
        assert b == t or b - c1 == right  # lookahead except at the end
    assert all(c1 - c0 == chunk for _, b, c0, c1 in w[1:] if b < t)
    assert (len(w) == 1) == (t <= head + right)
    # with the vocoder's windows (first + context = head, the same chunk), DiT window k has committed the frames of
    # vocoder window k: one DiT window per streaming round
    commits = [x[3] for x in w]
    for k, (_, b, _, _) in enumerate(stream_windows(t, first=head // 2, chunk=chunk, context=head // 2)):
        assert commits[min(k, len(commits) - 1)] >= b


def _inputs(tmp_path, texts, seeds):
    synth = _synth(tmp_path, prosody=True)
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    sentences = [split_sentences(normalize(t))[0] for t in texts]
    return synth, synth_prepare(synth, sentences, 2, 1.5, 0.5, gens)


def test_one_window_is_the_whole_sentence(tmp_path):
    synth, (z, cond, spk, alpha, labels, lens) = _inputs(tmp_path, ["Bu bir deneme cümlesidir."], [3])
    gen = batched_generator(synth.model)
    t = int(lens[0])
    wm = WindowedMel(gen, z, cond, spk, alpha, labels, [t], Chunking(right=t, crossfade=0), head=4)
    wm.step()
    assert wm.committed == [t] and not wm.pending(0)
    whole = gen(z, cond, spk, alpha, torch.ones(1, t, dtype=torch.bool), labels)
    assert torch.equal(wm.mel, whole)


def test_windows_slice_the_sentence_noise_and_condition(tmp_path):
    """Without crossfade, each window's committed frames are the DiT on that window's slice of the sentence's noise
    and condition (drawn once, in the single-request order)."""
    synth, (z, cond, spk, alpha, labels, lens) = _inputs(tmp_path, ["Yapay zekâ modelleri her geçen gün daha hızlı ve "
                                                                    "daha verimli hâle geliyor."], [5])
    gen = batched_generator(synth.model)
    t = int(lens[0])
    ch = Chunking(right=6, left=4, chunk=10, crossfade=0)
    wm = WindowedMel(gen, z, cond, spk, alpha, labels, [t], ch, head=8)
    while wm.pending(0):
        wm.step()
    assert len(wm.windows[0]) > 3
    for a, b, c0, c1 in wm.windows[0]:
        ref = gen(z[..., a:b], cond[..., a:b], spk, alpha, torch.ones(1, b - a, dtype=torch.bool), labels)
        torch.testing.assert_close(wm.mel[..., c0:c1], ref[..., c0 - a: c1 - a], rtol=0, atol=1e-5)


def test_crossfade_blends_the_previous_lookahead():
    """A fake DiT that outputs the window's index: the crossfade frames ramp from one window to the next."""
    calls = []

    def gen(z, cond, spk, alpha, mask, labels):
        calls.append(z.shape[-1])
        return torch.full_like(z, float(len(calls)))

    t, X = 60, 4
    z = torch.zeros(1, 3, t)
    wm = WindowedMel(gen, z, torch.zeros(1, 2, t), torch.zeros(1, dtype=torch.long), torch.ones(1),
                     torch.zeros(1, 1, dtype=torch.long), [t], Chunking(right=6, left=2, chunk=16, crossfade=X),
                     head=10)
    while wm.pending(0):
        wm.step()
    mel = wm.mel[0, 0]
    for k, (_, _, c0, c1) in enumerate(wm.windows[0]):
        if k == 0:
            assert torch.all(mel[c0:c1] == 1)
            continue
        ramp = (torch.arange(X) + 0.5) / X
        torch.testing.assert_close(mel[c0: c0 + X], k * (1 - ramp) + (k + 1) * ramp)
        assert torch.all(mel[c0 + X: c1] == k + 1)


def test_batch_rows_match_single_rows(tmp_path):
    texts = ["Merhaba!", "Bu bir deneme cümlesidir, tek adımda üretilir.", "Yapay zekâ modelleri her geçen gün daha "
             "hızlı ve daha verimli hâle geliyor."]
    synth, (z, cond, spk, alpha, labels, lens) = _inputs(tmp_path, texts, [0, 1, 2])
    gen, ch = batched_generator(synth.model), Chunking(right=6, left=4, chunk=10, crossfade=3)
    lens = lens.tolist()
    wm = WindowedMel(gen, z, cond, spk, alpha, labels, lens, ch, head=8)
    while any(wm.pending(r) for r in range(3)):
        wm.step()
    for b, t in enumerate(lens):
        one = WindowedMel(gen, z[b: b + 1, :, :t], cond[b: b + 1, :, :t], spk[b: b + 1], alpha[b: b + 1],
                          labels[b: b + 1], [t], ch, head=8)
        while one.pending(0):
            one.step()
        torch.testing.assert_close(wm.mel[b: b + 1, :, :t], one.mel, rtol=0, atol=1e-5)


@pytest.mark.parametrize("prosody", [False, True])
def test_chunked_stream_vocodes_the_windowed_mel(tmp_path, prosody):
    """Synthesizer.stream(chunked=...): the pieces of the streaming vocoder over the mel the windows commit, with the
    whole-sentence path's draws (a window covering the sentence gives its audio)."""
    synth = _synth(tmp_path, prosody)
    text = "Merhaba! Bugün hava çok güzel, dışarı çıkalım mı?"
    kw = dict(speaker=2, cfg_scale=1.5, temperature=0.5, pause=0.05, first=4, seed=7)
    ch = Chunking(right=6, left=4, chunk=12, crossfade=2)
    got = list(synth.stream(text, chunked=ch, **kw))
    whole = list(synth.stream(text, chunk=12, **kw))
    assert [p.numel() for p in got] == [p.numel() for p in whole]  # same durations, same pieces
    big = list(synth.stream(text, chunked=Chunking(right=10_000, left=0, crossfade=0, chunk=12), **kw))
    torch.testing.assert_close(torch.cat(big), torch.cat(whole), rtol=0, atol=1e-5)
    # the first sentence's pieces: stream_vocoder over the windowed mel
    spk, tempo = synth._speaker(2)
    g = torch.Generator().manual_seed(7)
    s0 = split_sentences(normalize(text))[0]
    z, cond, alpha, labels = synth._prepare(s0, spk, g, 1.5, 0.5, tempo)
    wm = WindowedMel(synth._window_generator(), z, cond, spk, alpha, labels, [z.shape[-1]], ch, head=4 + 8)
    while wm.pending(0):
        wm.step()
    ref = list(stream_vocoder(synth.vocoder, synth.stats.denormalize(wm.mel), first=4, chunk=12, context=8))
    assert len(ref) > 2
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5)


@pytest.mark.parametrize("prosody,buckets", [(False, 1), (True, 1), (True, 2)])
def test_chunked_stream_batched_matches_chunked_stream(tmp_path, prosody, buckets):
    from drifting_tts.batched import Serving

    synth = _synth(tmp_path, prosody)
    texts = ["Merhaba! Bugün hava çok güzel.", "Bu bir deneme.", "Kısa bir cümle daha? Evet, bir tane daha.",
             "Yapay zekâ modelleri her geçen gün daha hızlı ve daha verimli hâle geliyor."]
    ch = Chunking(right=6, left=4, chunk=12, crossfade=2)
    kw = dict(speaker=2, cfg_scale=1.5, temperature=0.5, pause=0.05, first=4)
    got = {i: [] for i in range(len(texts))}
    rounds = list(stream_batched(synth, texts, seeds=[3, 4, 5, 6], chunked=ch, serving=Serving(buckets=buckets),
                                 **kw))
    assert {i for i, _ in rounds[0]} == set(range(len(texts)))
    for out in rounds:
        for i, piece in out:
            got[i].append(piece)
    for i, text in enumerate(texts):
        ref = list(synth.stream(text, seed=3 + i, chunked=ch, **kw))
        assert [p.numel() for p in got[i]] == [p.numel() for p in ref]
        torch.testing.assert_close(torch.cat(got[i]), torch.cat(ref), rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graphed_windows_match_eager_windows():
    from drifting_tts.config import Config
    from drifting_tts.fast import GraphedAcoustic
    from drifting_tts.models.tts import DriftingTTS
    from drifting_tts.text import text_to_ids

    from .test_batched import MODEL

    torch.manual_seed(0)
    model = DriftingTTS(Config(MODEL), num_speakers=3).cuda().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ids = torch.tensor([text_to_ids("yapay zekâ modelleri her geçen gün daha hızlı ve daha verimli hâle geliyor.")],
                       device="cuda")
    spk = torch.tensor([2], device="cuda")
    fast = GraphedAcoustic(model, token_bucket=16, frame_bucket=32)
    ch = Chunking(right=12, left=8, chunk=24, crossfade=4)
    with torch.no_grad():
        z, cond, alpha, labels = fast.prepare(ids, spk, 1.5, 0.5, 1.3, torch.Generator(device="cuda").manual_seed(1))
        t = z.shape[-1]
        mels = []
        for gen in (fast.generate_window, batched_generator(model)):
            wm = WindowedMel(gen, z, cond, spk, alpha, labels, [t], ch, head=16)
            while wm.pending(0):
                wm.step()
            mels.append(wm.mel)
    assert len(wm.windows[0]) > 2
    torch.testing.assert_close(mels[0], mels[1], rtol=0, atol=1e-4)
    assert HOP_LENGTH == 256

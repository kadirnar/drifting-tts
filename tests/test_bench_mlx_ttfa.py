"""Consumer-side latency must include lazy host conversion and account for playback buffering; no MLX required."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import bench_mlx_ttfa as bench


class Clock:
    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds


class PendingPCM:
    """An audio result whose host conversion still has pending work."""

    def __init__(self, clock, seconds):
        self.clock, self.seconds = clock, seconds

    def __array__(self, dtype=None, copy=None):
        self.clock.advance(self.seconds)
        return np.ones(100, dtype=dtype)


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock.now)
    return clock


def measure(synth, **kwargs):
    return bench.run(synth, "Merhaba.", mode=kwargs.pop("mode", "stream"), chunk_frames=512,
                     first_chunk_frames=24, seed=0, options={}, synchronize=kwargs.pop("synchronize", lambda: None),
                     **kwargs)


def test_ttfa_includes_request_setup_and_pending_host_pcm(clock):
    def stream(*args, **kwargs):
        clock.advance(0.01)  # The public stream API may do work before returning its iterator.

        def chunks():
            yield np.zeros(0, np.float32), {}
            clock.advance(0.03)
            yield PendingPCM(clock, 0.2), {}

        return chunks()

    syncs = []

    def synchronize():
        syncs.append(clock.now)
        clock.advance(0.05)

    result = measure(SimpleNamespace(sample_rate=1000, stream=stream), synchronize=synchronize)
    assert result["ttfa_ms"] == pytest.approx(240)
    assert result["total_ms"] == pytest.approx(290)
    assert result["first_audio_s"] == 0.1
    assert result["chunks"] == 1 and len(syncs) == 2


def test_playback_deficit_uses_previously_delivered_pcm_including_silence(clock):
    def stream(*args, **kwargs):
        clock.advance(0.1)
        yield np.ones(100, np.float32), {}
        clock.advance(0.05)
        yield np.zeros(200, np.float32), {"is_silence": True}
        clock.advance(0.35)
        yield np.ones(100, np.float32), {}

    result = measure(SimpleNamespace(sample_rate=1000, stream=stream))
    assert result["ttfa_ms"] == pytest.approx(100)
    assert result["playback_deficit_ms"] == pytest.approx(100)
    assert result["samples"] == 400 and result["chunks"] == 3
    assert result["audio_s"] == pytest.approx(0.4)
    assert result["rtf"] == pytest.approx(1.25)


def test_buffered_ttfa_waits_for_the_complete_public_result(clock):
    class Buffered:
        sample_rate = 1000

        def __call__(self, *args, **kwargs):
            clock.advance(0.2)
            return PendingPCM(clock, 0.1), {}

    result = measure(Buffered(), mode="buffered")
    assert result["ttfa_ms"] == pytest.approx(300)
    assert result["total_ms"] == pytest.approx(300)
    assert result["playback_deficit_ms"] == 0


@pytest.mark.parametrize("chunks", [[], [(np.zeros(100, np.float32), {"is_silence": True})]])
def test_benchmark_rejects_missing_speech_audio(clock, chunks):
    synth = SimpleNamespace(sample_rate=1000, stream=lambda *args, **kwargs: iter(chunks))
    with pytest.raises(ValueError, match="no speech audio"):
        measure(synth)

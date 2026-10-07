"""Pretrained judges for :mod:`drifting_tts.evaluate`. They load lazily, and a missing optional dependency
(``pip install "drifting-tts[eval]"``) disables only the metric that needs it, with a message.

* **ASR**: Whisper large-v3 through ``faster-whisper`` (Turkish, beam 5, temperature 0 with no fallback, so
  transcripts are deterministic). Whisper reports 6.7% FLEURS-tr WER for large-v3; there is no published
  Turkish figure for large-v3-turbo, which stays available as a fast option.
* **Speaker embeddings**: WavLM-Large + ECAPA-TDNN fine-tuned for speaker verification (UniSpeech, VoxCeleb1-O
  EER 0.43%). This is the SIM model of Seed-TTS-eval, F5-TTS and CosyVoice, so similarities are on the published
  scale (same speaker ≈ 0.6–0.7, which is far below the 0.9+ of ``wavlm-base-plus-sv``). The weights are
  the ``prj-beatrice`` safetensors port at a pinned revision. They load into ``transformers.WavLMModel`` plus
  the ECAPA head below, and no remote code runs. Alternatives: ``ecapa-voxceleb`` (SpeechBrain, EER 0.80%, needs
  ``speechbrain``) and ``wavlm-base-plus-sv`` (the previous default).
* **Naturalness**: UTMOSv2 (VoiceMOS Challenge 2024), or UTMOS22 strong. Both are trained on English MOS data, so
  compare systems against the vocoded ground truth rather than reading the values as absolute MOS.

Every judge takes 16 kHz mono float32 numpy audio.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

SR = 16_000
SV_MODELS = {  # alias -> (HF repo, pinned revision, loader); the repo id is accepted as well
    "wavlm-large-ecapa": ("prj-beatrice/unispeech-wavlm-large-ecapa-tdnn-torch-native",
                          "13ad4fcc740cbd47992bcc2aaa59186d4072b7c9", "ecapa"),
    "ecapa-voxceleb": ("speechbrain/spkrec-ecapa-voxceleb", "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286", "speechbrain"),
    "wavlm-base-plus-sv": ("microsoft/wavlm-base-plus-sv", None, "xvector"),
}
MOS_MODELS = ("utmosv2", "utmos22")
UTMOSV2 = ("sarulab-speech/UTMOSv2", "fold0_s42_best_model.pth", "506474f2b33dc77c234d668cc419be1861899cad")
INSTALL = 'pip install "drifting-tts[eval]"  (UTMOSv2: pip install git+https://github.com/sarulab-speech/UTMOSv2.git)'


@dataclass
class Judges:
    """Callables on 16 kHz audio. ``None`` disables a metric."""

    asr: Callable[[np.ndarray], str] | None = None  # transcript
    sv: Callable[[np.ndarray], Tensor] | None = None  # L2-normalised speaker embedding
    mos: Callable[[np.ndarray], float] | None = None  # predicted MOS


class WhisperASR:
    def __init__(self, model: str = "large-v3", device: str = "cuda", beam_size: int = 5):
        from faster_whisper import WhisperModel

        kind, _, index = device.partition(":")
        self.model = WhisperModel(model, device=kind, device_index=int(index or 0),
                                  compute_type="float16" if kind == "cuda" else "int8")
        self.beam_size = beam_size

    def __call__(self, wav16: np.ndarray) -> str:
        segments, _ = self.model.transcribe(wav16, language="tr", beam_size=self.beam_size, temperature=0.0,
                                            condition_on_previous_text=False)
        return " ".join(s.text for s in segments).strip()


# ECAPA-TDNN head of UniSpeech's speaker-verification model; attribute names follow its state dict
class _ConvReluBn(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel, padding=padding)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(F.relu(self.conv(x)))


class _Res2ConvReluBn(nn.Module):
    def __init__(self, dilation: int, channels: int = 512, scale: int = 8):
        super().__init__()
        self.width = channels // scale
        self.convs = nn.ModuleList(nn.Conv1d(self.width, self.width, 3, padding=dilation, dilation=dilation)
                                   for _ in range(scale - 1))
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.width) for _ in range(scale - 1))

    def forward(self, x: Tensor) -> Tensor:
        pieces, out, y = torch.split(x, self.width, 1), [], None
        for piece, conv, bn in zip(pieces, self.convs, self.bns):
            y = bn(F.relu(conv(piece if y is None else y + piece)))
            out.append(y)
        return torch.cat([*out, pieces[-1]], 1)


class _SEConnect(nn.Module):
    def __init__(self, channels: int = 512, bottleneck: int = 128):
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(channels, bottleneck), nn.Linear(bottleneck, channels)

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.linear2(F.relu(self.linear1(x.mean(2)))))[:, :, None]


class _SERes2Block(nn.Module):
    def __init__(self, dilation: int, channels: int = 512):
        super().__init__()
        self.Conv1dReluBn1 = _ConvReluBn(channels, channels)
        self.Res2Conv1dReluBn = _Res2ConvReluBn(dilation, channels)
        self.Conv1dReluBn2 = _ConvReluBn(channels, channels)
        self.SE_Connect = _SEConnect(channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.SE_Connect(self.Conv1dReluBn2(self.Res2Conv1dReluBn(self.Conv1dReluBn1(x))))


class _AttentiveStatsPool(nn.Module):
    def __init__(self, channels: int = 1536, bottleneck: int = 128):
        super().__init__()
        self.linear1, self.linear2 = nn.Conv1d(channels, bottleneck, 1), nn.Conv1d(bottleneck, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x))), dim=2)
        mean = (alpha * x).sum(2)
        std = ((alpha * x**2).sum(2) - mean**2).clamp_min(1e-9).sqrt()
        return torch.cat([mean, std], 1)


class _EcapaTDNN(nn.Module):
    def __init__(self, feat_dim: int = 1024, channels: int = 512, emb_dim: int = 256):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.layer1 = _ConvReluBn(feat_dim, channels, kernel=5, padding=2)
        self.layer2, self.layer3, self.layer4 = (_SERes2Block(d, channels) for d in (2, 3, 4))
        self.conv = nn.Conv1d(3 * channels, 3 * channels, 1)
        self.pooling = _AttentiveStatsPool(3 * channels)
        self.bn = nn.BatchNorm1d(6 * channels)
        self.linear = nn.Linear(6 * channels, emb_dim)

    def forward(self, feats: Tensor) -> Tensor:  # [B, feat_dim, T] -> [B, emb_dim]
        out1 = self.layer1(self.instance_norm(feats))
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        x = F.relu(self.conv(torch.cat([out2, out3, out4], 1)))
        return self.linear(self.bn(self.pooling(x)))


class WavLMEcapa(nn.Module):
    """WavLM-Large + ECAPA-TDNN (UniSpeech speaker verification): one waveform -> a 256-d speaker embedding."""

    def __init__(self):
        super().__init__()
        from transformers import WavLMConfig, WavLMModel

        cfg = WavLMConfig(hidden_size=1024, intermediate_size=4096, num_attention_heads=16, num_hidden_layers=24,
                          feat_extract_norm="layer", do_stable_layer_norm=True, apply_spec_augment=False)
        self.wavlm = WavLMModel(cfg)
        self.feature_weight = nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
        self.ecapa = _EcapaTDNN(cfg.hidden_size)

    @classmethod
    def from_hub(cls, repo: str, revision: str | None = None) -> WavLMEcapa:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        model = cls()
        model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors", revision=revision)))
        return model.eval()

    def forward(self, wav16: Tensor) -> Tensor:
        """``[T]`` 16 kHz waveform -> ``[256]`` (s3prl's per-utterance waveform normalisation first)."""
        hidden = self.wavlm(F.layer_norm(wav16, wav16.shape)[None], output_hidden_states=True).hidden_states
        feats = sum(w * h for w, h in zip(self.feature_weight.softmax(0), hidden))  # learned layer weights
        return self.ecapa(feats.transpose(1, 2) + 1e-6)[0]


def sv_spec(name: str) -> tuple[str, str | None, str]:
    for alias, spec in SV_MODELS.items():
        if name in (alias, spec[0]):
            return spec
    raise ValueError(f"unknown speaker model {name!r}; choose from {', '.join(SV_MODELS)}")


class SpeakerEmbedder:
    def __init__(self, name: str = "wavlm-large-ecapa", device: str = "cuda"):
        repo, revision, self.kind = sv_spec(name)
        self.device = device
        if self.kind == "ecapa":
            self.model = WavLMEcapa.from_hub(repo, revision).to(device)
        elif self.kind == "speechbrain":
            from huggingface_hub import constants, snapshot_download
            from speechbrain.inference.speaker import EncoderClassifier

            path = snapshot_download(repo, revision=revision)
            self.model = EncoderClassifier.from_hparams(source=path, run_opts={"device": device},
                                                        savedir=f"{constants.HF_HOME}/speechbrain/{repo}")
        else:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            self.fe = AutoFeatureExtractor.from_pretrained(repo)
            self.model = WavLMForXVector.from_pretrained(repo).to(device).eval()

    @torch.no_grad()
    def __call__(self, wav16: np.ndarray) -> Tensor:
        x = torch.from_numpy(wav16).float().to(self.device)
        if self.kind == "ecapa":
            e = self.model(x)
        elif self.kind == "speechbrain":
            e = self.model.encode_batch(x[None])[0, 0]
        else:
            e = self.model(**self.fe(wav16, sampling_rate=SR, return_tensors="pt").to(self.device)).embeddings[0]
        return F.normalize(e.float(), dim=-1).cpu()


class UTMOSv2:
    """UTMOSv2 (``fusion_stage3``, fold 0). It scores random crops, so ``repetitions`` crops are averaged under a
    fixed NumPy seed, which makes the score deterministic. Its mel spectrograms (``n_fft`` 4096, hop 32, 512 bins)
    are computed on the GPU with the same STFT, filters and dB scaling as its librosa pipeline (identical scores on
    docs/samples); on the CPU they take about a second per crop and dominate the run time."""

    def __init__(self, device: str = "cuda", repetitions: int = 1):
        import utmosv2
        from huggingface_hub import hf_hub_download

        repo, filename, revision = UTMOSV2
        self.model = utmosv2.create_model(pretrained=True, device=device,
                                          checkpoint_path=hf_hub_download(repo, filename, revision=revision))
        self.device, self.repetitions, self._fb = device, repetitions, {}

    @torch.no_grad()
    def _melspec(self, cfg, spec_cfg, y: np.ndarray) -> np.ndarray:
        """``librosa.feature.melspectrogram`` + ``power_to_db(ref=np.max)`` as in ``utmosv2.dataset.multi_spec``."""
        import librosa

        key = (cfg.sr, spec_cfg.n_fft, spec_cfg.n_mels)
        if key not in self._fb:
            fb = librosa.filters.mel(sr=cfg.sr, n_fft=spec_cfg.n_fft, n_mels=spec_cfg.n_mels)
            self._fb[key] = torch.from_numpy(fb).to(self.device)
        x = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).to(self.device)
        window = torch.hann_window(spec_cfg.win_length, device=self.device)  # periodic, centred in n_fft
        spec = torch.stft(x, spec_cfg.n_fft, spec_cfg.hop_length, spec_cfg.win_length, window, center=True,
                          pad_mode="constant", return_complex=True).abs() ** 2
        db = 10 * torch.log10((self._fb[key] @ spec).clamp_min(1e-10))
        db = (db - db.max()).clamp_min(-80.0)
        if spec_cfg.norm is not None:
            db = (db + spec_cfg.norm) / spec_cfg.norm
        return db.cpu().numpy()

    def __call__(self, wav16: np.ndarray) -> float:
        from utmosv2.dataset import multi_spec

        state, librosa_melspec = np.random.get_state(), multi_spec._make_melspec
        np.random.seed(0)
        multi_spec._make_melspec = self._melspec
        try:
            pred = self.model.predict(data=wav16, sr=SR, device=self.device, num_workers=0,
                                      num_repetitions=self.repetitions, verbose=False)
        finally:
            np.random.set_state(state)
            multi_spec._make_melspec = librosa_melspec
        return float(np.asarray(pred).reshape(-1)[0])


class UTMOS22:
    """UTMOS22 strong learner (``tarepan/SpeechMOS``, the F5-TTS evaluation route); deterministic."""

    def __init__(self, device: str = "cuda"):
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, wav16: np.ndarray) -> float:
        return float(self.model(torch.from_numpy(wav16).float()[None].to(self.device), SR)[0])


def load_judges(asr: str | None = "large-v3", sv: str | None = "wavlm-large-ecapa", mos: str | None = "utmosv2",
                device: str = "cuda", mos_repetitions: int = 1) -> Judges:
    """Build the requested judges (``None`` / ``"none"`` skips one). A judge whose dependency is missing is
    disabled with a message, and evaluation goes on without that metric."""

    def build(what: str, name: str | None, factory: Callable):
        if name in (None, "none"):
            return None
        try:
            return factory()
        except ImportError as e:
            print(f"[evaluate] {what} judge {name!r} disabled ({e}); install with: {INSTALL}", flush=True)
            return None

    if mos not in (None, "none", *MOS_MODELS):
        raise ValueError(f"unknown MOS model {mos!r}; choose from {', '.join(MOS_MODELS)}")
    return Judges(
        asr=build("ASR", asr, lambda: WhisperASR(asr, device)),
        sv=build("speaker", sv, lambda: SpeakerEmbedder(sv, device)),
        mos=build("MOS", mos, lambda: UTMOSv2(device, mos_repetitions) if mos == "utmosv2" else UTMOS22(device)),
    )

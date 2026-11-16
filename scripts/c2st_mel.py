"""Can a classifier tell the model's mels from the recordings'? A classifier two-sample test (C2ST, Lopez-Paz &
Oquab, arXiv 1610.06545) on held-out utterances, at several window lengths and views of the mel.

The drifting loss stops moving the head when its samples match the data *in the kernel's space* (single patches of
raw log-mel). Whether the mels match the recordings beyond that space is not measured anywhere: UTMOSv2 scores the
model at its vocoder's ceiling, and the prosody statistics of ``scripts/prosody_gap.py`` agree with the recordings'.
This test asks the question directly. For every held-out utterance the model generates a mel **under the recording's
own alignment** (free-running, or teacher-forced with ``--mode gta``), so the two mels have the same frames and the
same text at every frame, and durations cannot give the answer away. A small CNN is trained on windows of both
(utterances split 80 / 20) and its accuracy on the held-out utterances is reported: 0.5 = indistinguishable.

Views: ``full`` (the window), ``low`` / ``mid`` / ``high`` (a third of the mel bins), ``envelope`` (the mel smoothed
along frequency: formants, energy), ``detail`` (mel minus that envelope: harmonics), ``delta`` (frame-to-frame
change), ``contour`` (per-frame mean level only: the energy contour over time) and ``shuffled`` (the window's frames
in random order: what single frames give away, without their order).

    python scripts/c2st_mel.py --model checkpoints/en/ardrift_en.pt --data data/en_val --split train \
        --temperature 0.5 --windows 4 16 64 --out outputs/diag/c2st_T0.5.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

VIEWS = ("full", "low", "mid", "high", "envelope", "detail", "delta", "contour", "shuffled")


def view(x: torch.Tensor, name: str) -> torch.Tensor:
    """Windows ``[N, C, W]`` -> the classifier's input ``[N, 1, C', W']``."""
    from drifting_tts.models.mae import spectral_detail

    C = x.shape[1]
    if name == "low":
        x = x[:, : C // 3]
    elif name == "mid":
        x = x[:, C // 3: 2 * C // 3]
    elif name == "high":
        x = x[:, 2 * C // 3:]
    elif name == "envelope":
        x = x - spectral_detail(x)
    elif name == "detail":
        x = spectral_detail(x)
    elif name == "delta":
        x = x[:, :, 1:] - x[:, :, :-1]
    elif name == "contour":
        x = x.mean(1, keepdim=True)
    elif name == "shuffled":
        idx = torch.argsort(torch.rand(x.shape[0], x.shape[2], device=x.device), 1)
        x = torch.gather(x, 2, idx[:, None].expand(-1, C, -1))
    return x[:, None]


class Net(nn.Module):
    def __init__(self, width: int = 48):
        super().__init__()
        chans = [1, width, width * 2, width * 2, width * 4]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(chans[:-1], chans[1:]))
        self.norms = nn.ModuleList(nn.GroupNorm(8, b) for b in chans[1:])
        self.out = nn.Linear(chans[-1] * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = F.silu(norm(conv(x)))
            if i < len(self.convs) - 1:  # pool what is still larger than one cell
                x = F.max_pool2d(x, (2 if x.shape[2] >= 2 else 1, 2 if x.shape[3] >= 2 else 1))
        return self.out(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))[:, 0]


class Pairs:
    """Recorded and generated mels of the same utterances, concatenated on the device, and the frames where a window
    of ``w`` frames starts inside one utterance; :meth:`sample` draws the same windows of both."""

    def __init__(self, real: list[torch.Tensor], fake: list[torch.Tensor], w: int, dev: str,
                 active: float | None = None, dither: float = 0.0, stride: int = 1):
        """``active``: only windows whose every frame of the recording has a mean level above it (speech, not the
        pauses, where the recording sits on the log-mel floor). ``dither``: Gaussian noise of this std on both sides
        (hides the float16 grid of the stored recordings and the exact floor, which are not the model's doing).
        ``stride``: windows start at multiples of it inside an utterance (the model's patch: with ``w`` = the patch,
        every window is one whole generated patch, with no patch boundary inside)."""
        self.real, self.fake, self.w, self.dither = torch.cat(real, 1).to(dev), torch.cat(fake, 1).to(dev), w, dither
        self.fake = self.fake.clamp_min(self.real.min())  # the log-mel floor of the front end
        starts, t0 = [], 0
        for m in real:
            if m.shape[1] >= w:
                s = torch.arange(t0, t0 + m.shape[1] - w + 1)
                keep = (s - t0) % stride == 0
                if active is not None:  # the quietest frame of each window
                    keep &= -F.max_pool1d(-m.mean(0)[None, None], w, 1)[0, 0] > active
                s = s[keep]
                starts.append(s)
            t0 += m.shape[1]
        self.starts = torch.cat(starts).to(dev)
        self.offsets = torch.arange(w, device=dev)

    def sample(self, n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """``[n, C, w]`` windows of the recordings and the same windows of the generated mels."""
        pick = torch.randint(len(self.starts), (n,), generator=g, device=self.starts.device)
        idx = self.starts[pick][:, None] + self.offsets  # [n, w]
        a, b = self.real[:, idx].transpose(0, 1), self.fake[:, idx].transpose(0, 1)
        if self.dither:
            a = a + self.dither * torch.randn(a.shape, generator=g, device=a.device)
            b = b + self.dither * torch.randn(b.shape, generator=g, device=b.device)
        return a, b


def c2st(real: list[torch.Tensor], fake: list[torch.Tensor], w: int, name: str, dev: str, steps: int, batch: int,
         seed: int, active: float | None = None, dither: float = 0.0, stride: int = 1) -> dict:
    n_test = max(len(real) // 5, 1)
    train = Pairs(real[n_test:], fake[n_test:], w, dev, active, dither, stride)
    test = Pairs(real[:n_test], fake[:n_test], w, dev, active, dither, stride)
    g = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    net.train()
    y = torch.cat([torch.ones(batch // 2), torch.zeros(batch // 2)]).to(dev)
    for _ in range(steps):
        a, b = train.sample(batch // 2, g)
        loss = F.binary_cross_entropy_with_logits(net(view(torch.cat([a, b]), name)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        ge = torch.Generator(device=dev).manual_seed(seed + 1)
        la, lb = [], []
        for _ in range(8):
            a, b = test.sample(500, ge)
            la.append(net(view(a, name)))
            lb.append(net(view(b, name)))
        la, lb = torch.cat(la).cpu(), torch.cat(lb).cpu()
    acc = 0.5 * float((la > 0).float().mean() + (lb <= 0).float().mean())
    pair = float((la > lb).float().mean())  # which of the two renditions of one window is the recording
    return {"accuracy": acc, "paired_accuracy": pair, "train_loss": float(loss.detach())}


@torch.no_grad()
def generate(args, dev: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    from drifting_tts.alignment import sequence_mask
    from drifting_tts.data import MelDataset, collate
    from drifting_tts.models.text_encoder import align
    from drifting_tts.train import load_tts

    model, cfg, stats = load_tts(args.model, dev)
    if args.head:  # a head fine-tuned alone (scripts/head_finetune.py)
        model.head.load_state_dict(torch.load(args.head, map_location=dev)["head"])
    ds = MelDataset(args.data, args.split, min_frames=1, max_frames=cfg.data.get("max_frames", 1500))
    ds.stats.mean, ds.stats.std = float(stats["mean"]), float(stats["std"])
    if dev == "cuda":
        model.enable_fast()
    order = sorted(range(min(args.num, len(ds))), key=ds.frames)  # similar lengths per batch
    real, fake = [], []
    for b0 in range(0, len(order), args.batch):
        batch = collate([ds[i] for i in order[b0: b0 + args.batch]])
        text, text_len, y, y_len, spk = (batch[n].to(dev) for n in ("text", "text_len", "mel", "mel_len", "spk"))
        y_mask = sequence_mask(y_len, y.shape[-1])[:, None].float()
        h, mu, _, x_mask = model.encoder(text, text_len, spk)
        attn, _ = align(mu, x_mask, y, y_mask)
        cond = model.frame_condition(h, mu, attn)
        g = torch.Generator(device=dev).manual_seed(args.seed + b0)
        if args.mode == "gta" and args.input_noise:  # as model.gta, the history noised as in training
            from drifting_tts.models.ar import patchify, unpatchify

            cp, yp = model.pad_frames(cond), model.pad_frames(y)
            s = model.teacher_forced(cp, yp, spk, torch.full((y.shape[0],), args.input_noise, device=dev))
            mu_p = patchify(cp[:, : model.n_mels], model.patch)
            z = torch.randn(*s.shape[:2], model.head.noise_dim, device=dev, generator=g) * args.temperature
            x = unpatchify(mu_p + model.head(s.flatten(0, 1), z.flatten(0, 1)).view_as(mu_p), model.patch,
                           model.n_mels)[..., : y.shape[-1]]
        elif args.mode == "gta":
            x = model.gta(cond, y, spk, args.temperature, g)
        elif args.mode == "prior":
            x = cond[:, : model.n_mels]
        elif args.mode == "null":
            x = y.clone()
        else:
            x = (model.fast_generator or model.generate)(cond, spk, args.temperature, g,
                                                         history_noise=args.history_noise)
        for i, n in enumerate(y_len.tolist()):
            real.append(y[i, :, :n].cpu())
            fake.append(x[i, :, :n].float().cpu())
        print(f"generated {len(real)}/{len(order)}", flush=True)
    perm = torch.randperm(len(real), generator=torch.Generator().manual_seed(args.seed)).tolist()
    return [real[i] for i in perm], [fake[i] for i in perm]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default=None, help="a head.pt of scripts/head_finetune.py, loaded over the model")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--mode", choices=("free", "gta", "prior", "null"), default="free",
                    help="free-running generation, teacher-forced samples, the aligned prior mean, or the recording "
                         "itself (the test's own floor: 0.5 expected)")
    ap.add_argument("--active", type=float, default=None,
                    help="only windows whose frames all have a mean level above this in the recording (e.g. -0.5)")
    ap.add_argument("--dither", type=float, default=0.02, help="noise added to both sides (normalised mel units)")
    ap.add_argument("--stride", type=int, default=1,
                    help="windows start at multiples of this many frames (the model's patch: whole patches)")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--input-noise", type=float, default=0.0, help="gta: noise on the teacher-forced history")
    ap.add_argument("--history-noise", type=float, default=0.0,
                    help="free: noise on the model's own history, as on the recorded one in training (#101)")
    ap.add_argument("--windows", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--train-batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None, help="keep the generated mels here (reused when the file exists)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cache and Path(args.cache).exists():
        real, fake = torch.load(args.cache)
    else:
        real, fake = generate(args, dev)
        if args.cache:
            Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save((real, fake), args.cache)
    results: dict = {"model": args.model, "mode": args.mode, "temperature": args.temperature,
                     "utterances": len(real), "active": args.active, "dither": args.dither, "stride": args.stride,
                     "windows": {}}
    print(f"\n{len(real)} utterances, {args.mode}, T {args.temperature:g}: accuracy on held-out utterances "
          "(0.5 = indistinguishable) / paired accuracy")
    print(f"{'window':>8s}" + "".join(f"{v:>15s}" for v in args.views))
    for w in args.windows:
        row = {}
        for v in args.views:
            if (v == "delta" and w < 2) or (v == "shuffled" and w < 2):
                continue
            row[v] = c2st(real, fake, w, v, dev, args.steps, args.train_batch, args.seed, args.active, args.dither,
                          args.stride)
        results["windows"][str(w)] = row
        print(f"{w:8d}" + "".join(f"{row[v]['accuracy']:9.3f}/{row[v]['paired_accuracy']:.3f}" if v in row
                                  else f"{'-':>15s}" for v in args.views), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

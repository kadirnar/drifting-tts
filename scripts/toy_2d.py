"""Toy 2-D drifting experiment (paper Fig. 4): an MLP generator trained only with the drift loss.

    python scripts/toy_2d.py --dataset checkerboard --out docs/toy_checkerboard.png
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from drifting_tts.drift import drift_loss  # noqa: E402


def checkerboard(n: int) -> torch.Tensor:
    b = torch.randint(0, 2, (n,))
    i = torch.randint(0, 2, (n,)) * 2 + b
    j = torch.randint(0, 2, (n,)) * 2 + b
    pts = torch.stack([i + torch.rand(n), j + torch.rand(n)], 1) - 2.0
    return pts / 2.0 + 0.02 * torch.randn(n, 2)


def swiss_roll(n: int) -> torch.Tensor:
    t = 0.5 * math.pi + 4.0 * math.pi * torch.rand(n)
    pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1)
    return pts / pts.abs().max() + 0.03 * torch.randn(n, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["checkerboard", "swiss_roll"], default="checkerboard")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out", default="toy.png")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler = {"checkerboard": checkerboard, "swiss_roll": swiss_roll}[args.dataset]

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(32, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    snaps, forces = {}, []
    for step in range(args.steps + 1):
        if step in (0, args.steps // 20, args.steps // 4, args.steps):
            with torch.no_grad():
                snaps[step] = net(torch.randn(5000, 32, device=device)).cpu()
        if step == args.steps:
            break
        x = net(torch.randn(args.batch, 32, device=device))
        loss, info = drift_loss(x[None], sampler(args.batch).to(device)[None], affinity_floor=0.0)
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
        forces.append(sum(v.item() for k, v in info.items() if k.startswith("force_")))

    fig, axes = plt.subplots(1, len(snaps) + 2, figsize=(3 * (len(snaps) + 2), 3))
    target = sampler(5000)
    axes[0].scatter(target[:, 0], target[:, 1], s=1, c="k", alpha=0.3)
    axes[0].set_title("target p")
    for ax, (step, pts) in zip(axes[1:], snaps.items()):
        ax.scatter(pts[:, 0], pts[:, 1], s=1, c="tab:orange", alpha=0.3)
        ax.set_title(f"q at step {step}")
    for ax in axes[:-1]:
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[-1].semilogy(forces)
    axes[-1].set_title("raw drift norm $||V||^2$")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}; final drift norm {forces[-1]:.3e}")


if __name__ == "__main__":
    main()

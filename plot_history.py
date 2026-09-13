"""Plot the training curves from outputs/history.csv — one figure per file.

Produces three separate figures, because each answers a different question and
each needs its own caption in a report:

  fig_train_loss{tag}.png    did the model fit the training data?
  fig_val_metrics{tag}.png   did it generalise, and which epoch was kept?
  fig_lr_schedule{tag}.png   what did the learning rate actually do?

Run:
    python plot_history.py
    python plot_history.py --tag seed42
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import config as C
import plotstyle as S


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"No history at {path}. Run `python train.py` first.")
    with open(path) as f:
        return list(csv.DictReader(f))


def _phase_marker(ax, boundary: int, label: bool = True) -> None:
    """Mark where the backbone unfroze — the sharpest event in every curve."""
    if not boundary:
        return
    ax.axvline(boundary - 0.5, ls="--", lw=1, color=S.FAINT, zorder=1)
    if label:
        ax.annotate("backbone unfrozen",
                    xy=(boundary - 0.5, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=8, color=S.MUTED, ha="left", va="top")


def fig_train_loss(rows, boundary, out: Path) -> None:
    epochs = [int(r["epoch"]) for r in rows]
    loss = [float(r["train_loss"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(epochs, loss, color=S.ACCENT, marker="o", markevery=[0, len(epochs) - 1])
    _phase_marker(ax, boundary)
    # Direct-label the endpoints only, never every point.
    ax.annotate(f"{loss[0]:.2f}", (epochs[0], loss[0]), textcoords="offset points",
                xytext=(6, 6), fontsize=9, color=S.INK)
    ax.annotate(f"{loss[-1]:.2f}", (epochs[-1], loss[-1]), textcoords="offset points",
                xytext=(-4, 10), fontsize=9, color=S.INK, ha="right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    S.save(fig, out, "label smoothing 0.1, so the floor sits above zero")


def fig_val_metrics(rows, boundary, out: Path) -> None:
    epochs = [int(r["epoch"]) for r in rows]
    acc = [float(r["val_accuracy"]) for r in rows]
    f1 = [float(r["val_macro_f1"]) for r in rows]
    best_i = max(range(len(f1)), key=f1.__getitem__)

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(epochs, acc, color=S.SERIES[0], label="accuracy")
    ax.plot(epochs, f1, color=S.SERIES[1], label="macro-F1")
    _phase_marker(ax, boundary)

    # The selected epoch, labelled once.
    ax.plot([epochs[best_i]], [f1[best_i]], marker="o", ms=8,
            color=S.SERIES[1], mec="white", mew=1.5, zorder=5)
    ax.annotate(f"kept: epoch {epochs[best_i]}\nmacro-F1 {f1[best_i]:.3f}",
                (epochs[best_i], f1[best_i]), textcoords="offset points",
                xytext=(8, -22), fontsize=9, color=S.INK)

    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    S.save(fig, out, "validation split held out of train; the test split is not used here")


def fig_lr(rows, boundary, out: Path) -> None:
    epochs = [int(r["epoch"]) for r in rows]
    lr = [float(r["head_lr"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(epochs, lr, color=S.ACCENT)
    ax.fill_between(epochs, lr, color=S.ACCENT, alpha=0.10)
    _phase_marker(ax, boundary)
    peak_i = max(range(len(lr)), key=lr.__getitem__)
    ax.annotate("peak after warmup", (epochs[peak_i], lr[peak_i]),
                textcoords="offset points", xytext=(8, -4),
                fontsize=9, color=S.INK)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    S.save(fig, out, "backbone groups run at 0.1x to 0.6x of this, and at zero while frozen")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the training curves")
    ap.add_argument("--tag", default="", help="artifact suffix, matching train.py --tag")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    S.apply_style()
    rows = read_history(C.OUT_DIR / f"history{suffix}.csv")
    boundary = next((int(r["epoch"]) for r in rows if r["phase"] == "e2e"), 0)

    print("training curves:")
    fig_train_loss(rows, boundary, C.OUT_DIR / f"fig_train_loss{suffix}.png")
    fig_val_metrics(rows, boundary, C.OUT_DIR / f"fig_val_metrics{suffix}.png")
    fig_lr(rows, boundary, C.OUT_DIR / f"fig_lr_schedule{suffix}.png")

    f1 = [float(r["val_macro_f1"]) for r in rows]
    best = max(range(len(f1)), key=f1.__getitem__)
    print(f"\nbest epoch {rows[best]['epoch']}: val macro-F1 {f1[best]:.4f}, "
          f"val accuracy {float(rows[best]['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()

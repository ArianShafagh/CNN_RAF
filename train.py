"""Fine-tuning of ResNet-18 on RAF-DB Basic.

Phase 1 (PHASE1_EPOCHS): backbone frozen — its learning rates are held at zero
    and its BatchNorm layers are pinned to eval — so only the fresh 7-class head
    moves while it settles.
Phase 2 (PHASE2_EPOCHS): the whole network trains end to end.

Both phases share ONE AdamW optimizer and ONE warmup+cosine schedule. Building a
second optimizer at the phase boundary would throw away the head's accumulated
moment estimates and restart the cosine, which is what an earlier version did.
Learning rates are layer-wise (see model.build_param_groups): the stem barely
moves, the head moves fully.

Model selection uses macro-F1 on a held-out validation split carved from the
train directory. The official test partition is never read here — evaluate.py
scores it once, at the end, so the reported test number is not selected on.

Checkpoints:
    best_model.pth : best validation macro-F1 (weights + metrics + config)
    last_model.pth : every epoch, including optimizer/scheduler state so that
                     --resume genuinely continues rather than restarts.

Run:
    python train.py
    python train.py --seed 43
    python train.py --resume
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

import config as C
import dataset as D
import model as M

# sklearn for the per-epoch metrics + confusion matrix — CPU numpy arrays.
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible training runs."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------
def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """One AdamW over layer-wise parameter groups, built once for the whole run.

    Norm and bias parameters sit in their own zero-weight-decay groups; see
    model.build_param_groups for why.
    """
    groups = M.build_param_groups(model, C.HEAD_LR, C.WEIGHT_DECAY)
    return torch.optim.AdamW(groups, lr=C.HEAD_LR)


def make_cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup -> cosine annealing over *steps* (stepped per batch).

    LambdaLR multiplies each group's own base lr by this factor, so the
    layer-wise ratios set up in build_param_groups are preserved throughout.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return (step + 1) / max(1, warmup_steps)   # 0->1 linear
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_backbone_lr_scale(optimizer, scale: float, scheduler=None) -> None:
    """Scale every non-head group's base LR by `scale`.

    `scale=0` implements the frozen phase without touching the optimizer's
    structure, so AdamW's moment estimates and the cosine schedule both survive
    the phase boundary intact.

    Careful: LambdaLR snapshots `initial_lr` into `scheduler.base_lrs` when it is
    constructed and multiplies *that* list by the lambda on every step. Editing
    the param group alone is silently ignored once a scheduler exists, which
    would leave the backbone pinned at lr=0 for the whole run. So the snapshot
    has to be rewritten too.
    """
    for i, group in enumerate(optimizer.param_groups):
        if group["stage"] == "fc":
            continue
        scaled = group["base_lr"] * scale
        group["initial_lr"] = scaled
        group["lr"] = scaled
        if scheduler is not None:
            scheduler.base_lrs[i] = scaled


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device,
                    scheduler=None, frozen: bool = False):
    """Run one epoch over `loader`, stepping the per-batch scheduler.

    When `frozen`, BatchNorm layers are pinned to eval so their running
    statistics do not drift while the backbone is meant to be fixed.

    Returns mean train loss across the epoch.
    """
    model.train()
    if frozen:
        M.set_bn_eval(model)
    running, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running += loss.item() * labels.size(0)
        n += labels.size(0)
    return running / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Compute validation metrics; returns a dict including y_true/y_pred."""
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        y_pred.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=list(range(C.NUM_CLASSES))),
        "report": classification_report(
            y_true, y_pred, target_names=C.EMOTION_NAMES,
            labels=list(range(C.NUM_CLASSES)), digits=4, zero_division=0,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def _snapshot_config(seed: int) -> dict:
    """Minimal config snapshot embedded in each checkpoint for traceability."""
    return {
        "image_size": C.IMAGE_SIZE,
        "batch_size": C.BATCH_SIZE,
        "num_classes": C.NUM_CLASSES,
        "emotion_names": C.EMOTION_NAMES,
        "label_smoothing": C.LABEL_SMOOTHING,
        "phase1_epochs": C.PHASE1_EPOCHS,
        "phase2_epochs": C.PHASE2_EPOCHS,
        "head_lr": C.HEAD_LR,
        "layer_lr_mults": C.LAYER_LR_MULTS,
        "imbalance_power": C.IMBALANCE_POWER,
        "val_fraction": C.VAL_FRACTION,
        "split_seed": C.SPLIT_SEED,
        "seed": seed,
        "selection": "validation macro-F1 (held out of train; test untouched)",
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, metrics, seed):
    """Write a checkpoint that is genuinely resumable."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "config": _snapshot_config(seed),
        },
        path,
    )


# ---------------------------------------------------------------------------
# History CSV  (replaces the 25 per-epoch confusion PNGs)
# ---------------------------------------------------------------------------
def init_history(path: Path) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "phase", "head_lr", "train_loss", "val_accuracy",
             "val_macro_f1", "seconds"]
        )


def append_history(path: Path, row: list) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def save_confusion_png(metrics: dict, epoch: int, path: Path) -> None:
    """Confusion matrix PNG for the best epoch only.

    matplotlib is imported lazily so training still runs if it is missing or a
    headless backend issue crops up.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # pragma: no cover - viz is best-effort
        print(f"    (skipping confusion PNG: {exc})")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        metrics["confusion"], annot=True, fmt="d", cmap="Blues",
        xticklabels=C.EMOTION_NAMES, yticklabels=C.EMOTION_NAMES,
        cbar_kws={"label": "count"}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune ResNet-18 on RAF-DB Basic")
    ap.add_argument("--seed", type=int, default=C.SEED,
                    help=f"random seed (default: {C.SEED})")
    ap.add_argument("--resume", action="store_true",
                    help="continue from last_model.pth")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for artifact filenames, e.g. --tag seed43")
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    best_ckpt = C.CKPT_DIR / f"best_model{suffix}.pth"
    last_ckpt = C.CKPT_DIR / f"last_model{suffix}.pth"
    history_csv = C.OUT_DIR / f"history{suffix}.csv"
    best_cm_png = C.OUT_DIR / f"val_confusion_best{suffix}.png"

    set_seed(args.seed)
    device = D.select_device()
    print(f"device: {device} | seed: {args.seed}")

    train_loader, val_loader, _ = D.build_loaders(device=device)
    print(f"train: {len(train_loader.dataset)} images in {len(train_loader)} batches")
    print(f"val  : {len(val_loader.dataset)} images in {len(val_loader)} batches"
          "   (test split untouched until evaluate.py)")

    model = M.build_resnet18(num_classes=C.NUM_CLASSES, pretrained=True).to(device)
    loss_fn = M.build_loss(device)

    # One optimizer and one schedule for the entire run.
    optimizer = build_optimizer(model)
    for g in optimizer.param_groups:          # remember the unscaled LR
        g["base_lr"] = g["lr"]
    steps_per_epoch = max(1, len(train_loader))
    total_steps = C.TOTAL_EPOCHS * steps_per_epoch
    set_backbone_lr_scale(optimizer, 0.0)     # phase 1 starts frozen
    scheduler = make_cosine_with_warmup(
        optimizer, total_steps, C.WARMUP_EPOCHS * steps_per_epoch
    )

    start_epoch = 0
    best = {"macro_f1": -1.0, "accuracy": 0.0, "epoch": -1}
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {last_ckpt.name} at epoch {start_epoch + 1}")
    else:
        init_history(history_csv)

    print(f"\n=== {C.TOTAL_EPOCHS} epochs "
          f"({C.PHASE1_EPOCHS} frozen + {C.PHASE2_EPOCHS} end-to-end) | "
          f"head lr {C.HEAD_LR:g} | {steps_per_epoch} steps/epoch ===")

    for epoch in range(start_epoch, C.TOTAL_EPOCHS):
        frozen = epoch < C.PHASE1_EPOCHS
        if frozen:
            M.freeze_backbone(model)
        elif epoch == C.PHASE1_EPOCHS or (args.resume and epoch == start_epoch):
            # Entering (or resuming into) the end-to-end phase.
            M.unfreeze_backbone(model)
            set_backbone_lr_scale(optimizer, 1.0, scheduler)
            print(f"\n--- backbone unfrozen at epoch {epoch + 1} ---")

        # Read the LR before the epoch: reading it afterwards reports the
        # post-anneal value, which on the last epoch is ~0 and misleading.
        head_lr = next(g["lr"] for g in optimizer.param_groups if g["stage"] == "fc")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                     device, scheduler=scheduler, frozen=frozen)
        metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        phase = "frozen" if frozen else "e2e"

        print(f"[{phase:>6}] epoch {epoch + 1:>2}/{C.TOTAL_EPOCHS} | "
              f"head lr {head_lr:.2e} | train_loss {train_loss:.4f} | "
              f"val_acc {metrics['accuracy']:.4f} | "
              f"val_macroF1 {metrics['macro_f1']:.4f} | {dt:.1f}s")
        append_history(history_csv, [
            epoch + 1, phase, f"{head_lr:.6e}", f"{train_loss:.6f}",
            f"{metrics['accuracy']:.6f}", f"{metrics['macro_f1']:.6f}", f"{dt:.1f}",
        ])

        if metrics["macro_f1"] > best["macro_f1"]:
            best.update(macro_f1=metrics["macro_f1"],
                        accuracy=metrics["accuracy"],
                        epoch=epoch,
                        report=metrics["report"])
            save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch,
                            metrics, args.seed)
            save_confusion_png(metrics, epoch, best_cm_png)
            print(f"    -> new best val macroF1={metrics['macro_f1']:.4f} "
                  f"(saved {best_ckpt.name})")

        save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch,
                        metrics, args.seed)

    print("\n==================== DONE ====================")
    print(f"best val macroF1: {best['macro_f1']:.4f} (acc {best['accuracy']:.4f}) "
          f"at epoch {best['epoch'] + 1}")
    print(f"best checkpoint : {best_ckpt}")
    print(f"history         : {history_csv}")
    print("\nValidation report at the best epoch "
          "(this is the SELECTION split, not the test score):")
    print(best.get("report", "(none)"))
    print("Now run `python evaluate.py` for the held-out test number.")


if __name__ == "__main__":
    main()

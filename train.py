"""Two-phase fine-tuning of ResNet-18 on RAF-DB Basic.

Phase 1 (PHASE1_EPOCHS=5): backbone frozen, train only the new 7-class head at
    PHASE1_LR=1e-3 with a 2-epoch linear warmup then cosine annealing.
Phase 2 (PHASE2_EPOCHS=20): backbone unfrozen, end-to-end fine-tune at
    PHASE2_LR=1e-4 with cosine annealing (no warmup).

AdamW + label-smoothing (0.1) loss. Batches are balanced by a
WeightedRandomSampler (see dataset.py) so Fear/Disgust are oversampled.

Each epoch we compute on the validation (test) split:
    - overall accuracy
    - per-class precision / recall / F1
    - macro-F1  (used for model selection)
    - confusion matrix  (logged + saved as PNG to outputs/)

Checkpoints:
    - best_model.pth : best validation macro-F1 so far (state_dict + meta)
    - last_model.pth : every epoch overwrite (resumable-ish)

Run:
    python train.py
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    """Seed torch + numpy for reproducible training runs."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------
def build_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """AdamW over currently-trainable params only.

    Weight decay per the plan; the new fc head is included when unfrozen-phased.
    """
    trainable = [p for _, p in M.trainable_parameters(model) if p.requires_grad]
    return torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=C.WEIGHT_DECAY,
    )


def make_cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup -> cosine annealing scheduler over *steps* (per-batch).

    Phase 1 uses warmup; phase 2 is cosine-only (warmup_steps=0).
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return (step + 1) / max(1, warmup_steps)   # 0->1 linear
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None):
    """Run one epoch over `loader`, stepping the (optional) per-batch scheduler.

    Returns mean train loss across the epoch.
    """
    model.train()
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
    """Compute metrics on `loader`; return a dict incl. y_true/y_pred for plots.

    Runs in eval mode (dropout/BN frozen) and on the CPU numpy side for sklearn.
    """
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)))
    report = classification_report(
        y_true, y_pred,
        target_names=C.EMOTION_NAMES,
        labels=list(range(C.NUM_CLASSES)),
        digits=4, zero_division=0,
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion": cm,
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "epoch_global": -1,   # filled in by caller
    }


# ---------------------------------------------------------------------------
# Phase orchestration
# ---------------------------------------------------------------------------
def run_phase(model, train_loader, val_loader, loss_fn, device,
              num_epochs: int, base_lr: float, phase_name: str,
              warmup_epochs: int, epoch_offset: int,
              best_state: dict) -> dict:
    """Train for `num_epochs` and update `best_state` (best macro-F1).

    epoch_offset is the global epoch index (for logging/plot filenames).
    Returns the (possibly updated) best_state dict.
    """
    optimizer = build_optimizer(model, base_lr)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    scheduler = make_cosine_with_warmup(optimizer, total_steps, warmup_steps)

    print(f"\n=== {phase_name}: {num_epochs} epochs | lr={base_lr:g} | "
          f"warmup={warmup_epochs} epoch(s) | {steps_per_epoch} steps/epoch ===")

    for local_ep in range(num_epochs):
        global_ep = epoch_offset + local_ep
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scheduler=scheduler
        )
        cur_lr = optimizer.param_groups[0]["lr"]
        metrics = evaluate(model, val_loader, device)
        metrics["epoch_global"] = global_ep
        dt = time.time() - t0

        print(
            f"[{phase_name}] epoch {local_ep + 1}/{num_epochs} "
            f"(global {global_ep + 1}) | lr {cur_lr:.2e} | "
            f"train_loss {train_loss:.4f} | val_acc {metrics['accuracy']:.4f} | "
            f"val_macroF1 {metrics['macro_f1']:.4f} | {dt:.1f}s"
        )

        # Save the per-epoch confusion matrix PNG for inspection.
        _save_confusion_png(metrics, global_ep, phase_name)

        # Checkpoint on best macro-F1 (the plan's selection criterion).
        if metrics["macro_f1"] > best_state["macro_f1"]:
            best_state.update(
                macro_f1=metrics["macro_f1"],
                accuracy=metrics["accuracy"],
                epoch_global=global_ep,
                confusion=metrics["confusion"],
                report=metrics["report"],
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "epoch": global_ep,
                    "config": _snapshot_config(),
                },
                C.BEST_CKPT,
            )
            print(f"    -> new best macroF1={metrics['macro_f1']:.4f} "
                  f"(saved {C.BEST_CKPT.name})")

        # Always keep a rolling last checkpoint.
        torch.save(
            {
                "model_state": model.state_dict(),
                "macro_f1": metrics["macro_f1"],
                "epoch": global_ep,
                "config": _snapshot_config(),
            },
            C.LAST_CKPT,
        )

    return best_state


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _save_confusion_png(metrics: dict, global_ep: int, phase_name: str) -> None:
    """Persist the epoch confusion matrix as PNG under outputs/.

    matplotlib is imported lazily so training still runs if it's missing or if
    a headless backend issue crops up.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # pragma: no cover - viz is best-effort
        print(f"    (skipping confusion PNG: {exc})")
        return

    cm = metrics["confusion"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=C.EMOTION_NAMES, yticklabels=C.EMOTION_NAMES,
        cbar_kws={"label": "count"}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"{phase_name} epoch {global_ep + 1} | "
        f"acc {metrics['accuracy']:.3f} | macroF1 {metrics['macro_f1']:.3f}"
    )
    plt.tight_layout()
    out = C.OUT_DIR / f"cm_{phase_name}_ep{global_ep + 1:02d}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Config snapshot for checkpoints
# ---------------------------------------------------------------------------
def _snapshot_config() -> dict:
    """Minimal config snapshot embedded in each checkpoint for traceability."""
    return {
        "image_size": C.IMAGE_SIZE,
        "batch_size": C.BATCH_SIZE,
        "num_classes": C.NUM_CLASSES,
        "emotion_names": C.EMOTION_NAMES,
        "label_smoothing": C.LABEL_SMOOTHING,
        "phase1_epochs": C.PHASE1_EPOCHS,
        "phase2_epochs": C.PHASE2_EPOCHS,
        "seed": C.SEED,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(C.SEED)
    device = D.select_device()
    print(f"device: {device}")

    train_loader, val_loader = D.build_loaders(device=device)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = M.build_resnet18(num_classes=C.NUM_CLASSES, pretrained=True).to(device)
    loss_fn = M.build_loss(device)

    best_state = {"macro_f1": -1.0, "accuracy": 0.0, "epoch_global": -1}

    # --- Phase 1: frozen backbone, head only -------------------------------
    M.freeze_backbone(model)
    best_state = run_phase(
        model, train_loader, val_loader, loss_fn, device,
        num_epochs=C.PHASE1_EPOCHS,
        base_lr=C.PHASE1_LR,
        phase_name="phase1",
        warmup_epochs=C.WARMUP_EPOCHS,
        epoch_offset=0,
        best_state=best_state,
    )

    # --- Phase 2: unfreeze, end-to-end -------------------------------------
    M.unfreeze_backbone(model)
    best_state = run_phase(
        model, train_loader, val_loader, loss_fn, device,
        num_epochs=C.PHASE2_EPOCHS,
        base_lr=C.PHASE2_LR,
        phase_name="phase2",
        warmup_epochs=0,                 # no warmup in phase 2
        epoch_offset=C.PHASE1_EPOCHS,
        best_state=best_state,
    )

    print("\n==================== DONE ====================")
    print(f"best val macroF1: {best_state['macro_f1']:.4f} "
          f"(acc {best_state['accuracy']:.4f}) "
          f"at global epoch {best_state['epoch_global'] + 1}")
    print(f"best checkpoint : {C.BEST_CKPT}")
    print("\nFinal per-class report at best epoch:")
    print(best_state["report"])


if __name__ == "__main__":
    main()

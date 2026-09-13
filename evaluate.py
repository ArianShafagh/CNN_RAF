"""Evaluate the trained ResNet-18 on RAF-DB Basic (final, full report).

Loads the best checkpoint (by validation macro-F1), prints every detail of
the model, and surfaces performance across many views:

  - overall accuracy, balanced accuracy, macro-F1, weighted-F1
  - per-class precision / recall / F1 / support
  - raw + row-normalised confusion matrices
  - per-class accuracy (recall-from-truth)
  - top cross-class confusions (what gets mistaken for what, and how often)
  - rare-class focus (Fear, Disgust) — the classes the imbalance targets
  - calibration glance: mean predicted-softmax confidence vs accuracy

Artifacts written to outputs/:
  - eval_report.txt   : the full printed report
  - eval_metrics.json : machine-readable metrics
  - fig_confusion_raw.png, fig_confusion_normalised.png,
    fig_per_class_recall.png  : one figure per view, never combined

This runs on the OFFICIAL TEST PARTITION, which training never reads: the
checkpoint was selected on a validation split held out of the train directory.
The number printed here is therefore a clean held-out result, not a figure that
was selected on.

Run:
    python evaluate.py            # uses best_model.pth
    python evaluate.py --last     # use last_model.pth instead
    python evaluate.py --tta      # average logits over the image and its mirror
    python evaluate.py --split val # score the validation split instead
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as C
import dataset as D
import model as M

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# The two rare classes the rebalancing targets (derived from config).
RARE_CLASSES = [idx for idx, name in enumerate(C.EMOTION_NAMES)
                if name in ("Fear", "Disgust")]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path, device: torch.device):
    """Build a ResNet-18, load state_dict from `path`, set eval mode.

    Returns (model, ckpt_meta) where ckpt_meta is the full checkpoint dict
    (so we can print what hyperparams produced this model).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run training first (python train.py) to produce it."
        )
    print(f"loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = M.build_resnet18(num_classes=C.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Model details (the "show all details" part)
# ---------------------------------------------------------------------------
def print_model_details(model: torch.nn.Module, ckpt_meta: dict, device, file=None):
    """Print architecture, parameter counts, per-layer shape/count table,
    and the config snapshot embedded in the checkpoint."""
    def p(*args, **kw):
        kw["file"] = file
        print(*args, **kw)

    p("=" * 70)
    p("MODEL DETAILS")
    p("=" * 70)
    p(f"checkpoint    : {ckpt_meta.get('_source', '?')}")
    p(f"selected at   : global epoch {ckpt_meta.get('epoch', '?')+1} "
      f"(stored epoch index {ckpt_meta.get('epoch', '?')})")
    p(f"device        : {device}")
    p("")

    # --- architecture ---
    p("-" * 70)
    p("Architecture (repr):")
    p(repr(model))
    p("")

    # --- per-layer table ---
    p("-" * 70)
    p(f"{'layer':48} {'shape':28} {'#params':>12}")
    p("-" * 70)
    total = total_train = 0
    n_layers = 0
    for name, param in model.named_parameters():
        shape = "x".join(str(s) for s in param.shape)
        n = param.numel()
        total += n
        if param.requires_grad:
            total_train += n
        p(f"{name:48} {shape:28} {n:>12,}")
        n_layers += 1
    p("-" * 70)
    p(f"{'TOTAL parameters':48} {'':28} {total:>12,}")
    p(f"{'TRAINABLE parameters':48} (frozen = {total-total_train:,})   {total_train:>12,}")
    p(f"{'parameter tensors':48} {'':28} {n_layers:>12}")
    p("")

    # --- head / classifier ---
    p("-" * 70)
    p("Classifier head (replaced for RAF-DB):")
    if hasattr(model, "fc"):
        p(f"  model.fc : Linear(in={model.fc.in_features}, "
          f"out={model.fc.out_features})")
        p(f"  classes  : {C.EMOTION_NAMES}")
    p("")

    # --- config snapshot ---
    p("-" * 70)
    p("Config snapshot from checkpoint:")
    snap = ckpt_meta.get("config", {})
    if snap:
        for k, v in snap.items():
            p(f"  {k}: {v}")
    else:
        p("  (no config embedded — older checkpoint?)")
    p("=" * 70)


# ---------------------------------------------------------------------------
# Prediction + metric computation
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(model, loader, device, tta: bool = False):
    """Return (y_true, y_pred, y_prob) over `loader`.

    y_prob is the per-sample softmax (for the calibration glance).

    With `tta`, logits are averaged over each image and its horizontal mirror.
    Facial expression is near-symmetric, so the mirror is a genuine second view
    of the same expression rather than a different input, and averaging the two
    cancels some of the left/right asymmetry the model picked up from training
    flips. Costs one extra forward pass.
    """
    model.eval()
    ys_true, ys_pred, ys_prob = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if tta:
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
        prob = torch.softmax(logits, dim=1)
        ys_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        ys_prob.extend(prob.cpu().numpy().tolist())
        ys_true.extend(labels.numpy().tolist())
    return (np.array(ys_true), np.array(ys_pred), np.array(ys_prob, dtype=np.float32))


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """All the aggregate + per-class + structural metrics used below."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)))
    cm_norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    per_class_recall = recall_score(y_true, y_pred, average=None,
                                    labels=list(range(C.NUM_CLASSES)), zero_division=0)

    return {
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_recall": per_class_recall,
        "confusion": cm,
        "confusion_norm": cm_norm,
        "report": classification_report(
            y_true, y_pred, target_names=C.EMOTION_NAMES,
            labels=list(range(C.NUM_CLASSES)), digits=4, zero_division=0,
        ),
        "y_acc": per_class_recall,   # recall-from-truth == per-class accuracy
    }


# ---------------------------------------------------------------------------
# Printing: all performance views
# ---------------------------------------------------------------------------
def print_overall(m, file=None):
    def p(*a, **k):
        k["file"] = file
        print(*a, **k)
    p("\n" + "=" * 70)
    p("OVERALL PERFORMANCE")
    p("=" * 70)
    p(f"  accuracy           : {m['accuracy']         :.4f}")
    p(f"  balanced accuracy  : {m['balanced_accuracy']:.4f}   (mean per-class recall)")
    p(f"  macro-F1           : {m['macro_f1']         :.4f}   <--- selection metric")
    p(f"  weighted-F1        : {m['weighted_f1']      :.4f}")
    p(f"  macro-precision    : {m['macro_precision'] :.4f}")
    p(f"  macro-recall       : {m['macro_recall']     :.4f}")


def print_per_class(m, file=None):
    def p(*a, **k):
        k["file"] = file
        print(*a, **k)
    p("\n" + "=" * 70)
    p("PER-CLASS PRECISION / RECALL / F1 / SUPPORT")
    p("=" * 70)
    p(m["report"])


def print_confusions(m, file=None):
    def p(*a, **k):
        k["file"] = file
        print(*a, **k)
    names = C.EMOTION_NAMES
    cm = m["confusion"]

    p("\n" + "=" * 70)
    p("CONFUSION MATRIX (rows=true, cols=predicted; raw counts)")
    p("=" * 70)
    header = f"{'true\\pred':>11} " + " ".join(f"{n[:6]:>8}" for n in names)
    p(header)
    for i, name in enumerate(names):
        row = " ".join(f"{cm[i, j]:>8d}" for j in range(C.NUM_CLASSES))
        p(f"{name:>11} {row}")

    p("\n" + "-" * 70)
    p("CONFUSION MATRIX (row-normalised = per-true-class recall split)")
    p("-" * 70)
    p(header)
    cmn = m["confusion_norm"]
    for i, name in enumerate(names):
        row = " ".join(f"{cmn[i, j]*100:>7.1f}%" for j in range(C.NUM_CLASSES))
        p(f"{name:>11} {row}")


def print_per_class_accuracy(m, file=None):
    def p(*a, **k):
        k["file"] = file
        print(*a, **k)
    names = C.EMOTION_NAMES
    p("\n" + "=" * 70)
    p("PER-CLASS ACCURACY (recall of each true class)")
    p("=" * 70)
    accs = m["y_acc"]
    for i, name in enumerate(names):
        bar = "#" * int(round(accs[i] * 40))
        p(f"  {name:<11} {accs[i]*100:6.2f}%  {bar}")


def print_top_confusions(m, file=None):
    def p(*a, **k):
        k["file"] = file
        print(*a, **k)
    names = C.EMOTION_NAMES
    cm = m["confusion"]
    pairs = []
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((names[i], names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    p("\n" + "=" * 70)
    p("TOP CROSS-CLASS CONFUSIONS  (true -> predicted, count)")
    p("=" * 70)
    for true_n, pred_n, cnt in pairs[:12]:
        p(f"  {true_n:<11} -> {pred_n:<11}  {cnt:>5d}")


def print_rare_focus(m, file=None):
    def p(*a, **k):
        k["file"] = file
        print(*a, **k)
    names = C.EMOTION_NAMES
    accs = m["y_acc"]
    cm = m["confusion"]
    p("\n" + "=" * 70)
    p("RARE-CLASS FOCUS (Fear, Disgust — rebuilt by the sampler)")
    p("=" * 70)
    for idx in RARE_CLASSES:
        correct = int(cm[idx, idx])
        total = int(cm[idx].sum())
        acc = accs[idx]
        p(f"  {names[idx]:<11} recall {acc*100:6.2f}%  "
          f"({correct}/{total} correct)")
        # show where the missed ones leaked to
        if total > correct:
            leaks = [(names[j], int(cm[idx, j]))
                     for j in range(C.NUM_CLASSES) if j != idx and cm[idx, j] > 0]
            leaks.sort(key=lambda x: x[1], reverse=True)
            leak_str = ", ".join(f"{n}:{c}" for n, c in leaks[:4])
            p(f"              misclassified as -> {leak_str}")


def print_calibration(m, file=None):
    def p(*a, **k):
        k["file"] = file
        print(*a, **k)
    y_pred = m["y_pred"]; y_prob = m["y_prob"]; y_true = m["y_true"]
    pred_conf = y_prob.max(axis=1)            # confidence on the chosen class
    correct = (y_pred == y_true)
    # mean confidence among correct vs wrong
    mean_conf_correct = float(pred_conf[correct].mean()) if correct.any() else 0.0
    mean_conf_wrong = float(pred_conf[~correct].mean()) if (~correct).any() else 0.0
    overall_acc = float(correct.mean())
    p("\n" + "=" * 70)
    p("CALIBRATION GLANCE (softmax confidence vs correctness)")
    p("=" * 70)
    p(f"  overall accuracy             : {overall_acc:.4f}")
    p(f"  mean confidence (correct)   : {mean_conf_correct:.4f}")
    p(f"  mean confidence (wrong)     : {mean_conf_wrong:.4f}")
    p(f"  -> gap correct-vs-wrong     : {mean_conf_correct - mean_conf_wrong:.4f}"
      "  (larger = model appropriately less confident when wrong)")
    # ECE-lite: bucket confidence into 5 bins, |avg_conf - acc| weighted.
    bins = np.linspace(0, 1, 6)
    ece = 0.0; N = len(pred_conf)
    for b in range(len(bins) - 1):
        lo, hi = bins[b], bins[b + 1]
        mask = (pred_conf >= lo) & (pred_conf < hi if b < len(bins)-2 else pred_conf <= hi)
        if mask.sum() == 0:
            continue
        conf_b = pred_conf[mask].mean()
        acc_b = correct[mask].mean()
        ece += abs(acc_b - conf_b) * mask.sum() / N
    p(f"  ECE-lite (5 bins)            : {float(ece):.4f}  (lower = better calibrated)")


# ---------------------------------------------------------------------------
# Multi-panel plot saved to outputs/
# ---------------------------------------------------------------------------
def save_eval_plots(m, out_dir: Path, variant: str) -> list:
    """Write each view as its own figure.

    Three separate files rather than one three-panel image: each answers a
    different question and each needs its own caption in a report.

      fig_confusion_raw          how many images landed in each cell
      fig_confusion_normalised   what share of each true class went where
      fig_per_class_recall       which emotions the model actually finds

    matplotlib is imported lazily; failure is non-fatal (the text report still
    prints).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as _np
        import plotstyle as S
    except Exception as exc:  # pragma: no cover
        print(f"(skipping plots: {exc})")
        return []
    S.apply_style()
    names = C.EMOTION_NAMES
    written = []

    def _heatmap(ax, data, fmt, vmax, cbar_label):
        """Sequential single hue, light to dark. Cell text takes an ink colour
        on light cells and a light colour on dark ones, never the series hue."""
        im = ax.imshow(data, cmap=S.GREEN_RAMP, vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        # 2px surface gap between cells, per the mark spec.
        ax.set_xticks(_np.arange(-.5, len(names), 1), minor=True)
        ax.set_yticks(_np.arange(-.5, len(names), 1), minor=True)
        ax.grid(which="minor", color=S.SURFACE, linewidth=2)
        ax.grid(which="major", visible=False)
        for i in range(len(names)):
            for j in range(len(names)):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=8.5,
                        color="#FFFFFF" if v > vmax * 0.55 else S.INK)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label(cbar_label, color=S.MUTED, fontsize=9)
        cb.ax.tick_params(labelsize=8, length=0, colors=S.MUTED)
        cb.outline.set_visible(False)

    # --- 1. raw counts ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    cm = m["confusion"]
    _heatmap(ax, cm, "d", float(cm.max()), "images")
    written.append(S.save(fig, out_dir / f"fig_confusion_raw{variant}.png",
                          "the diagonal is correct; every other cell is one kind of mistake"))

    # --- 2. row-normalised ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    _heatmap(ax, m["confusion_norm"] * 100, ".1f", 100.0, "% of true class")
    written.append(S.save(fig, out_dir / f"fig_confusion_normalised{variant}.png",
                          "rows sum to 100%, so classes of very different size are comparable"))

    # --- 3. per-class recall ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    accs = m["y_acc"] * 100
    order = _np.argsort(accs)[::-1]
    lbls = [names[i] for i in order]
    vals = [accs[i] for i in order]
    colors = [S.ALERT if i in RARE_CLASSES else S.ACCENT for i in order]
    bars = ax.bar(lbls, vals, color=colors, width=0.68)
    for b in bars:
        b.set_linewidth(0)
    ax.axhline(float(accs.mean()), ls="--", lw=1, color=S.MUTED, zorder=1)
    ax.annotate(f"balanced accuracy {accs.mean():.1f}%",
                xy=(len(lbls) - 0.5, accs.mean()), xytext=(0, 6),
                textcoords="offset points", ha="right", fontsize=8.5, color=S.MUTED)
    for b, v in zip(bars, vals):
        # A surface-coloured bbox so the balanced-accuracy rule cannot strike
        # through a value that happens to sit at the same height.
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9, color=S.INK, zorder=6,
                bbox=dict(facecolor=S.SURFACE, edgecolor="none", pad=1.5))
    ax.set_ylim(0, 105)
    ax.set_ylabel("recall (%)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    rare = ", ".join(names[i] for i in RARE_CLASSES)
    written.append(S.save(fig, out_dir / f"fig_per_class_recall{variant}.png",
                          f"orange marks the rare classes ({rare}); labels carry the value, not colour alone"))
    return written


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------
def save_metrics_json(m, ckpt_meta, path: Path, split: str = 'test', tta: bool = False):
    """Drop non-serialisable numpy arrays; keep the numbers + array lists."""
    cm = m["confusion"]
    cm_norm = m["confusion_norm"]
    payload = {
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "seed": ckpt_meta.get("config", {}).get("seed"),
        "overall": {
            "accuracy": float(m["accuracy"]),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "macro_f1": float(m["macro_f1"]),
            "weighted_f1": float(m["weighted_f1"]),
            "macro_precision": float(m["macro_precision"]),
            "macro_recall": float(m["macro_recall"]),
        },
        "per_class": {
            C.EMOTION_NAMES[i]: {
                "recall": float(m["y_acc"][i]),
                # include support from the confusion matrix diagonal row sum
                "support": int(cm[i].sum()),
            }
            for i in range(C.NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_rownorm": cm_norm.tolist(),
        "rare_classes": [C.EMOTION_NAMES[i] for i in RARE_CLASSES],
        "split": split,
        "tta": tta,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate RAF-DB ResNet-18")
    parser.add_argument("--last", action="store_true",
                        help="use last_model.pth instead of best_model.pth")
    parser.add_argument("--tta", action="store_true",
                        help="horizontal-flip test-time augmentation")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="which split to score (default: test)")
    parser.add_argument("--tag", type=str, default="",
                        help="checkpoint/artifact suffix, matching train.py --tag")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    device = D.select_device()
    name = "last_model" if args.last else "best_model"
    ckpt_path = C.CKPT_DIR / f"{name}{suffix}.pth"

    model, ckpt_meta = load_checkpoint(ckpt_path, device)
    ckpt_meta["_source"] = str(ckpt_path)

    # Deterministic, augmentation-free transforms on the requested split.
    _, val_ds, test_ds = D.build_datasets()
    ds = test_ds if args.split == "test" else val_ds
    loader = D.eval_loader(ds, device)
    print(f"scoring the {args.split} split: {len(ds)} images"
          + ("  [flip TTA]" if args.tta else ""))

    y_true, y_pred, y_prob = collect_predictions(model, loader, device, tta=args.tta)
    m = compute_metrics(y_true, y_pred, y_prob)

    # ---- everything to stdout ----
    print_model_details(model, ckpt_meta, device)
    print_overall(m)
    print_per_class(m)
    print_confusions(m)
    print_per_class_accuracy(m)
    print_top_confusions(m)
    print_rare_focus(m)
    print_calibration(m)
    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)

    # ---- persist artifacts ----
    variant = f"{suffix}_{args.split}" + ("_tta" if args.tta else "")
    report_txt = C.OUT_DIR / f"eval_report{variant}.txt"
    metrics_json = C.OUT_DIR / f"eval_metrics{variant}.json"

    with open(report_txt, "w") as f:
        print_model_details(model, ckpt_meta, device, file=f)
        print_overall(m, file=f)
        print_per_class(m, file=f)
        print_confusions(m, file=f)
        print_per_class_accuracy(m, file=f)
        print_top_confusions(m, file=f)
        print_rare_focus(m, file=f)
        print_calibration(m, file=f)
    figures = save_eval_plots(m, C.OUT_DIR, variant)
    save_metrics_json(m, ckpt_meta, metrics_json, args.split, args.tta)

    print(f"\nartifacts:")
    print(f"  {report_txt}")
    print(f"  {metrics_json}")
    for fpath in figures:
        print(f"  {fpath}")


if __name__ == "__main__":
    main()

"""RAF-DB Basic dataset, transforms, sampler, and DataLoaders.

Layout (already on disk, ImageFolder-compatible):
    Dataset/DATASET/train/<1..7>/*.jpg
    Dataset/DATASET/test/<1..7>/*.jpg
Folder index == emotion index (1..7 -> internal 0..6).

We use the *test* split as validation during training (it is the official
RAF-DB Basic test partition; the dataset has no separate val split). Metrics
are computed on it every epoch and model selection uses validation macro-F1.
"""
from __future__ import annotations

import math
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import config as C


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def _normalise() -> transforms.Normalize:
    return transforms.Normalize(mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD)


def train_transforms() -> transforms.Compose:
    """Augmentation for the small RAF-DB train set to curb overfitting."""
    return transforms.Compose([
        transforms.Resize((C.IMAGE_SIZE, C.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),              # ~10-15 deg
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        _normalise(),
        # Random erasing after normalisation (standard recipe)
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2),
                                 ratio=(0.3, 3.3)),
    ])


def eval_transforms() -> transforms.Compose:
    """No augmentation for validation / test."""
    return transforms.Compose([
        transforms.Resize((C.IMAGE_SIZE, C.IMAGE_SIZE)),
        transforms.ToTensor(),
        _normalise(),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def build_datasets():
    """Return (train_ds, val_ds) ImageFolder datasets.

    val_ds uses the test directory (official RAF-DB test partition) with
    deterministic, augmentation-free transforms.
    """
    train_ds = datasets.ImageFolder(str(C.TRAIN_DIR), transform=train_transforms())
    val_ds = datasets.ImageFolder(str(C.TEST_DIR), transform=eval_transforms())
    # Sanity: confirm folder->index mapping matches config ordering.
    if train_ds.classes != ["1", "2", "3", "4", "5", "6", "7"]:
        raise RuntimeError(
            f"Unexpected ImageFolder classes: {train_ds.classes}. "
            "Expected folders 1..7 mapping to labels 0..6."
        )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Class-imbalance handling
# ---------------------------------------------------------------------------
def class_counts(dataset) -> torch.Tensor:
    """Per-class sample counts as a float tensor indexed by label (0..6)."""
    counts = Counter(int(label) for _, label in dataset.samples)
    return torch.tensor(
        [counts[i] for i in range(C.NUM_CLASSES)], dtype=torch.float32
    )


def make_weighted_sampler(train_ds) -> WeightedRandomSampler:
    """Inverse-frequency WeightedRandomSampler so rare classes (Fear, Disgust)
    are oversampled each epoch."""
    counts = class_counts(train_ds)
    # weight_i = 1 / count_i  ; eps guards against division by zero
    weights = 1.0 / (counts + 1e-6)
    sample_weights = weights[torch.tensor(
        [int(lbl) for _, lbl in train_ds.samples], dtype=torch.long
    )]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),   # full epoch length, just rebalanced
        replacement=True,
    )


def class_weights_for_loss(train_ds, device: torch.device) -> torch.Tensor:
    """Inverse-frequency loss weights, normalised so they average to 1.

    Optional alternative to the sampler: pass these to CrossEntropyLoss.
    The training script uses the *sampler* by default; this is exposed for
    experimentation / the label-smoothing weighted-loss path.
    """
    counts = class_counts(train_ds)
    weights = counts.sum() / (counts * C.NUM_CLASSES + 1e-6)
    return weights.to(device)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def select_device() -> torch.device:
    """Prefer MPS (Apple Silicon GPU), fall back to CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def build_loaders(device: torch.device | None = None):
    """Return (train_loader, val_loader) with the weighted sampler applied
    to the train loader.

    Note: when a WeightedRandomSampler is used, shuffle must be False (the
    sampler already controls ordering); shuffling on top would be a no-op
    and torch warns about it.
    """
    train_ds, val_ds = build_datasets()
    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=C.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,            # sampler handles randomness
        num_workers=C.NUM_WORKERS,
        pin_memory=(device is not None and device.type == "mps"),
        drop_last=False,
        persistent_workers=(C.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=C.NUM_WORKERS,
        pin_memory=(device is not None and device.type == "mps"),
        persistent_workers=(C.NUM_WORKERS > 0),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick self-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tr, va = build_datasets()
    print(f"train samples : {len(tr)}")
    print(f"val/test  samples : {len(va)}")
    print("class -> folder mapping:", dict(zip(range(C.NUM_CLASSES), C.EMOTION_NAMES)))
    print("train class counts:", class_counts(tr).tolist())
    print("device:", select_device())
    tl, vl = build_loaders(device=select_device())
    xb, yb = next(iter(tl))
    print("batch shape:", xb.shape, "labels:", yb[:10].tolist())

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

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from rafdb.core import config as C


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def _normalise() -> transforms.Normalize:
    return transforms.Normalize(mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD)


def train_transforms() -> transforms.Compose:
    """Augmentation for the small RAF-DB train set to curb overfitting."""
    # RAF-DB ships pre-aligned 100x100 face crops, so the augmentation has to be
    # gentler than a generic ImageNet recipe:
    #   - RandomResizedCrop instead of a plain Resize gives scale/translation
    #     jitter, which is the variation a webcam actually introduces.
    #   - Rotation is cut to 8 deg: larger angles fight the alignment the
    #     dataset provides and bring in black corners.
    #   - Hue jitter is dropped; skin-tone shifts are not a cue for expression
    #     and only add noise.
    return transforms.Compose([
        transforms.RandomResizedCrop(C.IMAGE_SIZE, scale=(0.8, 1.0),
                                     ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.0),
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
def _check_classes(ds) -> None:
    """Confirm the folder->index mapping matches the config ordering."""
    if ds.classes != ["1", "2", "3", "4", "5", "6", "7"]:
        raise RuntimeError(
            f"Unexpected ImageFolder classes: {ds.classes}. "
            "Expected folders 1..7 mapping to labels 0..6."
        )


def stratified_split(labels, val_fraction: float, seed: int):
    """Split indices per class so val mirrors the train class proportions.

    A plain random split would put only ~28 Fear images in validation and the
    count would wobble run to run; stratifying keeps each class's share exact.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for cls in range(C.NUM_CLASSES):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def build_datasets():
    """Return (train_ds, val_ds, test_ds).

    train_ds and val_ds are disjoint stratified subsets of the *train* directory.
    They wrap two separate ImageFolder instances over the same files so that
    training gets augmentation while validation does not — a single dataset
    split after the fact would leak augmentation into validation.

    test_ds is the official RAF-DB test partition. Nothing in training touches
    it; it exists to be evaluated exactly once, at the end.
    """
    aug_base = datasets.ImageFolder(str(C.TRAIN_DIR), transform=train_transforms())
    clean_base = datasets.ImageFolder(str(C.TRAIN_DIR), transform=eval_transforms())
    test_ds = datasets.ImageFolder(str(C.TEST_DIR), transform=eval_transforms())
    for ds in (aug_base, test_ds):
        _check_classes(ds)

    labels = [lbl for _, lbl in aug_base.samples]
    train_idx, val_idx = stratified_split(labels, C.VAL_FRACTION, C.SPLIT_SEED)
    return Subset(aug_base, train_idx), Subset(clean_base, val_idx), test_ds


# ---------------------------------------------------------------------------
# Class-imbalance handling
# ---------------------------------------------------------------------------
def labels_of(dataset) -> list[int]:
    """Labels for an ImageFolder or a Subset of one, without decoding images."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [int(base.samples[i][1]) for i in dataset.indices]
    return [int(lbl) for _, lbl in dataset.samples]


def class_counts(dataset) -> torch.Tensor:
    """Per-class sample counts as a float tensor indexed by label (0..6)."""
    counts = Counter(labels_of(dataset))
    return torch.tensor(
        [counts[i] for i in range(C.NUM_CLASSES)], dtype=torch.float32
    )


def make_weighted_sampler(train_ds) -> WeightedRandomSampler:
    """WeightedRandomSampler that oversamples the rare classes (Fear, Disgust).

    Strength is controlled by C.IMBALANCE_POWER: weight = 1 / count**power.
    Full inverse frequency (power=1.0) gives Fear roughly a 17x edge over
    Happiness and overcorrects; the square root keeps the rare classes visible
    without starving the majority. See the config for the reasoning.
    """
    counts = class_counts(train_ds)
    weights = 1.0 / (counts.clamp(min=1.0) ** C.IMBALANCE_POWER)
    sample_weights = weights[torch.tensor(labels_of(train_ds), dtype=torch.long)]
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
def _pin(device: torch.device | None) -> bool:
    """Pinned host memory only helps CUDA's async host-to-device copies.

    MPS shares memory with the host, so pinning there buys nothing — the
    original code had this condition inverted, enabling it for MPS and
    disabling it for CUDA, which is exactly backwards.
    """
    return device is not None and device.type == "cuda"


def eval_loader(dataset, device: torch.device | None = None) -> DataLoader:
    """Deterministic loader for validation or test. Shared so that every
    evaluation path uses identical settings."""
    return DataLoader(
        dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=C.NUM_WORKERS,
        pin_memory=_pin(device),
        persistent_workers=(C.NUM_WORKERS > 0),
    )


def build_loaders(device: torch.device | None = None):
    """Return (train_loader, val_loader, test_loader).

    The train loader draws through the weighted sampler; shuffle must be False
    alongside a sampler, since the sampler already controls ordering.

    The test loader is built here for convenience but must not be consulted
    during training — model selection uses val_loader only.
    """
    train_ds, val_ds, test_ds = build_datasets()
    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=C.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,            # sampler handles randomness
        num_workers=C.NUM_WORKERS,
        pin_memory=_pin(device),
        drop_last=False,
        persistent_workers=(C.NUM_WORKERS > 0),
    )
    return train_loader, eval_loader(val_ds, device), eval_loader(test_ds, device)


# ---------------------------------------------------------------------------
# Quick self-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tr, va, te = build_datasets()
    print(f"train samples : {len(tr)}")
    print(f"val   samples : {len(va)}   ({C.VAL_FRACTION:.0%} held out of train)")
    print(f"test  samples : {len(te)}   (official partition, untouched in training)")
    print(f"train+val     : {len(tr) + len(va)}")

    print("\nclass -> folder mapping:", dict(zip(range(C.NUM_CLASSES), C.EMOTION_NAMES)))
    c_tr, c_va, c_te = class_counts(tr), class_counts(va), class_counts(te)
    print(f"\n{'class':<11} {'train':>7} {'val':>6} {'test':>7} {'train%':>8} {'val%':>7}")
    for i, name in enumerate(C.EMOTION_NAMES):
        print(f"{name:<11} {int(c_tr[i]):>7} {int(c_va[i]):>6} {int(c_te[i]):>7} "
              f"{c_tr[i]/c_tr.sum()*100:>7.2f}% {c_va[i]/c_va.sum()*100:>6.2f}%")

    # Overlap must be empty, or validation is meaningless.
    assert not (set(tr.indices) & set(va.indices)), "train/val overlap!"
    print("\ntrain/val overlap: none")

    # Sampler check: how balanced does a drawn epoch actually look?
    sampler = make_weighted_sampler(tr)
    drawn = Counter(labels_of(tr)[i] for i in list(sampler))
    print(f"\nsampler draw (power={C.IMBALANCE_POWER}):")
    for i, name in enumerate(C.EMOTION_NAMES):
        print(f"  {name:<11} {drawn[i]:>6}  ({drawn[i]/len(tr)*100:5.2f}%)")

    print("\ndevice:", select_device())
    tl, vl, tel = build_loaders(device=select_device())
    xb, yb = next(iter(tl))
    print("batch shape:", xb.shape, "labels:", yb[:10].tolist())

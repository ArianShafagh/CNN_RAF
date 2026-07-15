"""Model definition: ImageNet-pretrained ResNet-18 fine-tuned for RAF-DB Basic.

Exports:
    build_resnet18(num_classes, pretrained=True) -> nn.Module
    freeze_backbone(model)         # phase 1: train new head only
    unfreeze_backbone(model)       # phase 2: end-to-end fine-tune
    build_loss(device)             # label-smoothing CrossEntropyLoss

The final fully-connected layer of stock ResNet-18 is replaced with a fresh
7-class head. The rest of the conv stack keeps its ImageNet weights, so the
two-phase recipe (frozen head-only -> full fine-tune) has something useful to
build on.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

import config as C


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
def build_resnet18(num_classes: int = C.NUM_CLASSES,
                   pretrained: bool = True) -> nn.Module:
    """Return a ResNet-18 with a fresh `num_classes`-way fc head.

    Weights come from ImageNet when `pretrained` (the torchvision API picks the
    right enum for the installed version). The head is always re-initialised
    regardless, since ImageNet's 1000 classes are meaningless for RAF-DB.
    """
    try:
        # Newer torchvision (>=0.13) API
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
    except (ImportError, AttributeError):
        # Older torchvision fallback: bool pretrained= flag
        model = torchvision.models.resnet18(pretrained=pretrained)

    # Replace the classifier head. ResNet-18's final layer is `.fc`.
    in_features = model.fc.in_features          # 512 for resnet18
    model.fc = nn.Linear(in_features, num_classes)
    # Kaiming-init the new head (matches ResNet's conv init philosophy).
    nn.init.kaiming_normal_(model.fc.weight, mode="fan_out", nonlinearity="relu")
    nn.init.zeros_(model.fc.bias)

    return model


# ---------------------------------------------------------------------------
# Two-phase fine-tuning helpers
# ---------------------------------------------------------------------------
def freeze_backbone(model: nn.Module) -> nn.Module:
    """Phase 1: freeze everything except the new fc head.

    Returns the same model for chaining. Only `model.fc` receives gradients,
    so the ImageNet conv features act as a fixed feature extractor while the
    head warms up and stabilises.
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc")
    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """Phase 2: unfreeze the whole network for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def trainable_parameters(model: nn.Module):
    """Yield (name, param) pairs that require gradients — used to build the
    optimizer so it only ever updates trainable params."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def build_loss(device: torch.device) -> nn.CrossEntropyLoss:
    """Label-smoothing cross-entropy, matching the plan (smoothing ~0.1).

    No per-class weights here by default: the WeightedRandomSampler already
    balances batches, and stacking weighted loss on top tends to over-correct
    the rare classes during full fine-tuning. Both knobs remain available if
    you want to experiment — see dataset.class_weights_for_loss().
    """
    return nn.CrossEntropyLoss(label_smoothing=C.LABEL_SMOOTHING).to(device)


# ---------------------------------------------------------------------------
# Quick self-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dev = torch.device("cpu")
    net = build_resnet18().to(dev)
    n_train = sum(p.numel() for _, p in trainable_parameters(net) if p.requires_grad)
    print("head in_features:", net.fc.in_features, "-> out:", net.fc.out_features)
    print("params trainable before freeze:", n_train)

    freeze_backbone(net)
    n_head = sum(p.numel() for _, p in trainable_parameters(net) if p.requires_grad)
    print("params trainable after  freeze (head only):", n_head)

    unfreeze_backbone(net)
    n_all = sum(p.numel() for _, p in trainable_parameters(net) if p.requires_grad)
    print("params trainable after  unfreeze (all):    ", n_all)

    # Forward pass sanity
    net.eval()
    x = torch.randn(2, 3, C.IMAGE_SIZE, C.IMAGE_SIZE)
    with torch.no_grad():
        out = net(x)
    print("output shape:", tuple(out.shape))   # expect (2, 7)
    print("loss:", build_loss(dev))

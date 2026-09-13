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
    # Small-normal init, NOT Kaiming. Kaiming fan-out is designed for conv
    # layers feeding a ReLU; on a 512->7 classifier it yields std sqrt(2/7)
    # = 0.53, roughly 20x too large. That produced logits with std ~8 and a
    # starting loss near 11.9 against the ln(7) = 1.95 a uniform prior should
    # give, so the first epoch was spent undoing the initialisation.
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
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


def set_bn_eval(model: nn.Module) -> nn.Module:
    """Put every BatchNorm layer in eval mode, keeping its running statistics.

    `requires_grad = False` stops gradients but does NOT stop BatchNorm from
    updating `running_mean` / `running_var`: those are buffers, refreshed on
    every forward pass while the module is in train mode. During the frozen
    phase that quietly rewrites the ImageNet statistics the frozen weights were
    calibrated against, so the "fixed feature extractor" is not fixed at all.
    Call this after `model.train()` on each frozen-phase epoch.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return model


# ---------------------------------------------------------------------------
# Parameter groups: layer-wise LR + no weight decay on norms/biases
# ---------------------------------------------------------------------------
def _stage_of(param_name: str) -> str:
    """Map a parameter name onto a key of C.LAYER_LR_MULTS."""
    for stage in ("layer1", "layer2", "layer3", "layer4", "fc"):
        if param_name.startswith(stage):
            return stage
    return "stem"          # conv1.*, bn1.*


def build_param_groups(model: nn.Module, head_lr: float, weight_decay: float):
    """AdamW parameter groups with per-stage learning rates.

    Two things are happening here:

    1. Layer-wise LR. Early ResNet blocks hold generic edge/texture filters that
       transfer almost unchanged; later blocks and the fresh head need to move
       much further. A flat LR either under-trains the head or wrecks the stem.

    2. No weight decay on BatchNorm parameters or biases. Decaying a norm's
       scale/shift fights the normalisation itself, and it is standard practice
       to exclude them. Only weight matrices of conv/linear layers get decay.

    Each group carries a `name` so training can zero the backbone LRs during the
    frozen phase and restore them afterwards.
    """
    groups: dict[str, dict] = {}
    for name, param in model.named_parameters():
        stage = _stage_of(name)
        # ndim <= 1 catches every bias and every BatchNorm weight/bias.
        decay = param.ndim > 1
        key = f"{stage}_{'decay' if decay else 'nodecay'}"
        if key not in groups:
            groups[key] = {
                "name": key,
                "stage": stage,
                "params": [],
                "lr": head_lr * C.LAYER_LR_MULTS[stage],
                "weight_decay": weight_decay if decay else 0.0,
            }
        groups[key]["params"].append(param)
    # Stable ordering keeps checkpoint optimizer state loadable across runs.
    order = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]
    return sorted(groups.values(),
                  key=lambda g: (order.index(g["stage"]), g["name"]))


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

    print("\nparameter groups (layer-wise LR, decay split):")
    for g in build_param_groups(net, C.HEAD_LR, C.WEIGHT_DECAY):
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<16} lr={g['lr']:<9.2e} wd={g['weight_decay']:<7g} "
              f"tensors={len(g['params']):>3}  params={n:>10,}")

    # BatchNorm freeze check: running stats must not move when set_bn_eval is on.
    net.train(); freeze_backbone(net); set_bn_eval(net)
    before = net.bn1.running_mean.clone()
    net(torch.randn(4, 3, C.IMAGE_SIZE, C.IMAGE_SIZE))
    drift_frozen = float((net.bn1.running_mean - before).abs().max())
    net.train()                       # train mode WITHOUT set_bn_eval
    before = net.bn1.running_mean.clone()
    net(torch.randn(4, 3, C.IMAGE_SIZE, C.IMAGE_SIZE))
    drift_train = float((net.bn1.running_mean - before).abs().max())
    print(f"\nbn1.running_mean drift with set_bn_eval : {drift_frozen:.3e} (want 0)")
    print(f"bn1.running_mean drift without it       : {drift_train:.3e} (want > 0)")

    # Forward pass sanity
    net.eval()
    x = torch.randn(2, 3, C.IMAGE_SIZE, C.IMAGE_SIZE)
    with torch.no_grad():
        out = net(x)
    print("output shape:", tuple(out.shape))   # expect (2, 7)
    print("loss:", build_loss(dev))

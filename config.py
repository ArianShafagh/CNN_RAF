"""Central configuration for RAF-DB Basic emotion classification with ResNet-18.

All paths, hyperparameters, and constants live here so train/eval/export
all read from the same source of truth.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
# Dataset mirrors RAF-DB Basic, already in ImageFolder layout:
#   Dataset/DATASET/train/<1..7>/*.jpg   Dataset/DATASET/test/<1..7>/*.jpg
DATA_ROOT = ROOT / "Dataset" / "DATASET"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

# Where artifacts are written
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
OUT_DIR = ROOT / "outputs"          # confusion-matrix PNGs, reports
OUT_DIR.mkdir(exist_ok=True)
BEST_CKPT = CKPT_DIR / "best_model.pth"
LAST_CKPT = CKPT_DIR / "last_model.pth"
ONNX_PATH = OUT_DIR / "raf_resnet18.onnx"
HISTORY_CSV = OUT_DIR / "history.csv"   # per-epoch training curve

# ---------------------------------------------------------------------------
# RAF-DB Basic 7-class mapping
# Folder index  ==  label index (0-internal, 1..7 on disk).
# This is the canonical RAF-DB Basic ordering.
# ---------------------------------------------------------------------------
EMOTION_NAMES = [
    "Surprise",  # 1
    "Fear",      # 2  (rare)
    "Disgust",   # 3  (rare)
    "Happiness", # 4  (largest)
    "Sadness",   # 5
    "Anger",     # 6
    "Neutral",   # 7
]
NUM_CLASSES = 7

# ---------------------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------------------
# RAF-DB aligned crops are 100x100. Upscale to 224x224 for the ImageNet
# pre-trained ResNet input expectation.
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 64
NUM_WORKERS = 4        # safe default on M-series; tune up if you have headroom

# ---------------------------------------------------------------------------
# Validation split
# ---------------------------------------------------------------------------
# RAF-DB Basic ships only train/test. Carving a validation slice out of *train*
# keeps the official test partition untouched until the final evaluation, so the
# reported test number is not contaminated by checkpoint selection.
VAL_FRACTION = 0.10
# Deliberately separate from SEED: the split must stay identical across seeded
# runs, otherwise multi-seed results are not comparable.
SPLIT_SEED = 1234

# Two-phase fine-tuning. One optimizer and one cosine schedule span both
# phases; "phase 1" simply holds the backbone groups at lr=0 while the fresh
# head settles, which avoids the momentum reset a second optimizer would cause.
PHASE1_EPOCHS = 3      # frozen backbone, train new head only
PHASE2_EPOCHS = 27     # unfreeze, end-to-end
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

HEAD_LR = 1e-3         # peak LR for the new fc head
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1  # tolerates RAF-DB label noise/ambiguity
WARMUP_EPOCHS = 2      # linear warmup at the very start

# Layer-wise LR multipliers applied to HEAD_LR. Early ImageNet features are
# generic and need barely any adjustment; later blocks are task-specific.
LAYER_LR_MULTS = {
    "stem":   0.1,     # conv1 + bn1
    "layer1": 0.1,
    "layer2": 0.2,
    "layer3": 0.4,
    "layer4": 0.6,
    "fc":     1.0,
}

# ---------------------------------------------------------------------------
# Class-imbalance strength
# ---------------------------------------------------------------------------
# Sampler weight = 1 / count**IMBALANCE_POWER.
#   1.0 = full inverse frequency (Fear sampled ~17x Happiness) — overcorrects,
#         and measurably cost majority-class accuracy in the first run.
#   0.5 = square-root inverse frequency — keeps a real boost for the rare
#         classes without distorting the majority.
#   0.0 = no rebalancing at all.
IMBALANCE_POWER = 0.5

# Reproducibility
SEED = 42

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

# Two-phase fine-tuning
PHASE1_EPOCHS = 5      # frozen backbone, train new head only
PHASE2_EPOCHS = 20     # unfreeze, end-to-end at lower LR
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

PHASE1_LR = 1e-3       # head only -> can afford a larger LR
PHASE2_LR = 1e-4       # full fine-tune -> smaller to protect features
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1  # tolerates RAF-DB label noise/ambiguity
WARMUP_EPOCHS = 2      # linear warmup at the start of phase 1

# Reproducibility
SEED = 42

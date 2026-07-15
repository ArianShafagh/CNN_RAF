# CNN_RAF — RAF-DB Basic Emotion Classification (ResNet-18)

Fine-tunes an ImageNet-pretrained **ResNet-18** on the **RAF-DB Basic** 7-class
facial-emotion dataset, with a focus on doing well on the rare classes
(Fear, Disgust), not just overall accuracy.

**Pipeline:** two-phase fine-tuning (frozen head → full network), AdamW with
cosine-annealing + warmup, label smoothing, class-imbalance handling via a
`WeightedRandomSampler`, per-epoch macro-F1 / per-class recall / confusion
matrix, best-checkpoint selection on macro-F1, and ONNX export verified against
PyTorch with onnxruntime.

## Classes

7 emotions, folders `1..7` mapping to internal labels `0..6`:

| Folder | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------|---|---|---|---|---|---|---|
| Emotion | Surprise | Fear | Disgust | Happiness | Sadness | Anger | Neutral |

Fear and Disgust are the rare classes the rebalancing targets.

## Project layout

```
config.py      # paths, class mapping, all hyperparameters (single source of truth)
dataset.py     # ImageFolder datasets, transforms/augmentation, weighted sampler, device
model.py       # ResNet-18 + fresh 7-class head, freeze/unfreeze, label-smoothing loss
train.py       # two-phase fine-tuning loop, per-epoch metrics, checkpointing
evaluate.py    # full test-split report + confusion plots + JSON
export.py      # ONNX export with dynamic batch axis + onnxruntime verification
requirements.txt
```

## Dataset

The code expects the RAF-DB Basic split already in `ImageFolder` layout:

```
Dataset/DATASET/train/<1..7>/*.jpg
Dataset/DATASET/test/<1..7>/*.jpg
```

The **test** split is used as the validation set during training (RAF-DB Basic
has no separate val partition). Paths are defined in [config.py](config.py) —
edit `DATA_ROOT` there if your data lives elsewhere.

## Setup

Requires Python 3.11+ (developed on 3.14). On Apple Silicon the training uses
the **MPS** GPU automatically; otherwise it falls back to CUDA, then CPU.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Quick sanity check that the data and device are wired up correctly:

```bash
python dataset.py                 # prints sample counts, class mapping, a batch shape
python model.py                   # prints param counts + a forward-pass output shape
```

## Usage

Run the three steps in order. Each script reads all its settings from
[config.py](config.py).

### 1. Train

```bash
python train.py
```

- **Phase 1** (`PHASE1_EPOCHS=5`): backbone frozen, trains only the new 7-class
  head at `lr=1e-3` with a 2-epoch warmup.
- **Phase 2** (`PHASE2_EPOCHS=20`): whole network unfrozen, end-to-end fine-tune
  at `lr=1e-4` with cosine annealing.

Writes:
- `checkpoints/best_model.pth` — best validation **macro-F1** so far (the
  selection metric — balanced across classes, not raw accuracy).
- `checkpoints/last_model.pth` — latest epoch, overwritten each epoch.
- `outputs/cm_phaseN_epNN.png` — per-epoch confusion matrix.

### 2. Evaluate

```bash
python evaluate.py                # uses best_model.pth
python evaluate.py --last         # uses last_model.pth instead
```

Prints and saves a detailed report: overall/balanced accuracy, macro & weighted
F1, per-class precision/recall/F1, raw + row-normalised confusion matrices, top
cross-class confusions, a rare-class (Fear/Disgust) focus, and a calibration
glance. Artifacts land in `outputs/`: `eval_report.txt`, `eval_metrics.json`,
`eval_confusion.png`.

### 3. Export to ONNX

```bash
python export.py                  # exports best_model.pth
python export.py --last           # exports last_model.pth
python export.py --opset 18       # override ONNX opset (default 17)
```

Exports the model to `outputs/raf_resnet18.onnx` with a **dynamic batch axis**
(accepts any batch size, fixed `3x224x224` per image), validates the graph with
`onnx.checker`, and verifies PyTorch vs onnxruntime logits match (`rtol=1e-3,
atol=1e-5`) on batch sizes 1 and 8 plus a real test batch. Exits non-zero on
mismatch.

## Tuning

All hyperparameters live in [config.py](config.py): image size, batch size,
epochs per phase, learning rates, weight decay, label smoothing, warmup, and the
random seed. Change them there — every script picks the values up automatically.

## Notes

- Class imbalance is handled by the `WeightedRandomSampler` (inverse-frequency
  oversampling); label smoothing (`0.1`) tolerates RAF-DB's label noise. A
  weighted-loss alternative is available via `dataset.class_weights_for_loss`.
- `export.py` pins the legacy TorchScript exporter (`dynamo=False`) so it works
  without the extra `onnxscript` dependency. To use the newer torch.export-based
  exporter instead, `pip install onnxscript` and drop that flag.

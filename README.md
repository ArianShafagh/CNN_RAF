# CNN_RAF — RAF-DB Basic Emotion Classification (ResNet-18)

Fine-tunes an ImageNet-pretrained **ResNet-18** on the **RAF-DB Basic** 7-class
facial-emotion dataset, with a focus on doing well on the rare classes
(Fear, Disgust), not just overall accuracy.

**Pipeline:** two-phase fine-tuning (frozen head → full network) under a single
AdamW optimizer with layer-wise learning rates and one warmup + cosine schedule,
label smoothing, class-imbalance handling via a `WeightedRandomSampler`,
checkpoint selection on **validation** macro-F1, ONNX export verified against
PyTorch with onnxruntime, and a live webcam demo driven by the exported graph.

### Evaluation protocol

RAF-DB Basic ships only `train` and `test`. A stratified **10% validation split
is held out of `train`** and is the only thing checkpoint selection ever sees.
The official test partition is scored once, by `evaluate.py`, at the end. This
matters: selecting the best of N epochs on the test set and then reporting that
same set inflates the number.

## Results

ResNet-18, single run (seed 42), scored once on the official RAF-DB Basic test
partition. Model selection used a validation split held out of `train`, so the
test set was never seen during training or checkpoint selection.

| Metric | Value |
|---|---|
| Accuracy | **0.8611** |
| Macro-F1 | 0.7881 |
| Balanced accuracy | 0.7663 |
| Weighted-F1 | 0.8594 |

Per-class recall: Happiness 94.1%, Neutral 87.4%, Surprise 86.9%, Sadness
83.5%, Anger 72.8%, Fear 56.8%, Disgust 55.0%. Fear and Disgust remain the
weak classes; they have 74 and 160 test images respectively.

Figures include horizontal-flip test-time augmentation (`--tta`), worth about
0.7 points of accuracy over the plain forward pass (0.8543 without).

> **Caveat for reporting:** this is one seed. Fear has only 74 test images, so a
> single flipped prediction moves its recall by 1.35 points. Run three seeds and
> use `aggregate_seeds.py` before quoting these numbers in a paper:
> ```bash
> for s in 42 43 44; do
>   python train.py --seed $s --tag seed$s && python evaluate.py --tag seed$s
> done
> python aggregate_seeds.py
> ```

## Classes

7 emotions, folders `1..7` mapping to internal labels `0..6`:

| Folder | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------|---|---|---|---|---|---|---|
| Emotion | Surprise | Fear | Disgust | Happiness | Sadness | Anger | Neutral |

Fear and Disgust are the rare classes the rebalancing targets.

## Project layout

```
config.py           # paths, class mapping, all hyperparameters (single source of truth)
dataset.py          # datasets, stratified val split, augmentation, weighted sampler, device
model.py            # ResNet-18 + fresh 7-class head, freeze/unfreeze, BN-eval, param groups
train.py            # fine-tuning loop, per-epoch metrics, history CSV, checkpointing
evaluate.py         # full report on a chosen split + confusion plots + JSON
export.py           # ONNX export with dynamic batch axis + onnxruntime verification
aggregate_seeds.py  # mean +/- std across seeds, for reporting
webcam_demo.py      # live camera inference on the exported ONNX graph
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

- **Phase 1** (`PHASE1_EPOCHS=3`): backbone frozen — its learning rates are held
  at zero *and* its BatchNorm layers pinned to eval, so the running statistics
  cannot drift — while the fresh head settles.
- **Phase 2** (`PHASE2_EPOCHS=27`): whole network trains end to end.

Both phases share one optimizer and one warmup + cosine schedule, so no momentum
is discarded at the boundary. Learning rates are layer-wise: the stem runs at
`0.1x` the head's rate, rising through the blocks to `1.0x` at `fc`.

Useful flags: `--seed N`, `--tag NAME` (suffixes all artifacts), `--resume`.

Writes:
- `checkpoints/best_model.pth` — best **validation** macro-F1 (balanced across
  classes, not raw accuracy). Includes optimizer and scheduler state.
- `checkpoints/last_model.pth` — latest epoch, genuinely resumable.
- `outputs/history.csv` — per-epoch LR, train loss, val accuracy, val macro-F1.
- `outputs/val_confusion_best.png` — confusion matrix at the best epoch.

### 2. Evaluate

```bash
python evaluate.py                # best_model.pth on the held-out test split
python evaluate.py --tta          # + horizontal-flip test-time augmentation
python evaluate.py --split val    # score the validation split instead
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

### 4. Live webcam demo

First-time setup — fetch the BlazeFace detector (230 KB):

```bash
mkdir -p models && curl -L -o models/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

```bash
python webcam_demo.py                  # opens the camera, labels your face live
python webcam_demo.py --image face.jpg # one still image, no camera needed
python webcam_demo.py --detector haar  # fall back if mediapipe is unavailable
python webcam_demo.py --no-align       # raw box crop instead of eye alignment
```

Runs the exported ONNX graph through `onnxruntime` — no torch import — so it
tests the deployment artifact rather than the training-time model. Preprocessing
matches `dataset.eval_transforms` exactly: scoring the whole test split through
this path reproduces `evaluate.py` to within 0.06 points, the residual being PIL
vs OpenCV resize interpolation.

**Face alignment.** RAF-DB ships landmark-aligned crops — measured over 400
training images, the eyes sit at y = 0.34 with a standard deviation of 0.05 and
an inter-eye distance of 0.39 of the crop width. Webcam faces are not aligned,
which is a train/deploy mismatch. BlazeFace returns eye keypoints, so each frame
is similarity-warped onto that canonical geometry before classification. On
faces simulated with head roll and varied framing this is worth a lot:

| | Raw box crop | Eye-aligned |
|---|---|---|
| Accuracy | 0.7689 | **0.7980** |
| Macro-F1 | 0.6552 | **0.7150** |

On already-aligned stills it slightly *hurts* (0.8523 → 0.8321) because the warp
resamples and adds landmark noise, so `--no-align` exists for that case.

**Why BlazeFace over Haar.** Measured on the live camera: 2.1 ms vs 4.1 ms per
frame, faces found in 20/20 frames vs 0/20, plus the landmarks that make
alignment possible. Haar also cannot detect an already-tight crop at all. It is
kept as `--detector haar` for environments without mediapipe.

**Speed.** ~26 FPS steady state, which is camera-bound: capture is 22.7 ms of a
33.7 ms frame, against 1.7 ms detection and 8.7 ms align-plus-inference. The
backend matters enormously — CoreML runs the network in ~1.2 ms against ~49 ms
on CPU, and the demo defaults to it on Apple Silicon with automatic CPU
fallback (`--provider`). Before that change the loop ran at 4.4 FPS.

The overlay shows the face box, the top label, all seven class probabilities and
a frame rate. Press `q` or Esc to quit. Useful flags: `--margin`, `--smooth`
(temporal smoothing; raw per-frame softmax flickers), `--min-conf`, `--threads`,
`--camera`, `--detector`, `--no-align`, `--provider`.

Expect live accuracy below the test-set figure. Webcam frames differ from RAF-DB
in lighting, pose and framing, and Fear and Disgust are the weakest classes.

### 5. Report across seeds

```bash
for s in 42 43 44; do
  python train.py    --seed $s --tag seed$s
  python evaluate.py --tag seed$s
done
python aggregate_seeds.py
```

One run is not a result: Fear has only 74 test images, so a single flipped
prediction moves its recall by 1.35 points. Report mean ± std over seeds.

## Tuning

All hyperparameters live in [config.py](config.py): image size, batch size,
epochs per phase, learning rates, weight decay, label smoothing, warmup, and the
random seed. Change them there — every script picks the values up automatically.

## Notes

- Class imbalance is handled by the `WeightedRandomSampler`, at strength
  `IMBALANCE_POWER=0.5` (square-root inverse frequency). Full inverse frequency
  gives Fear roughly a 17x sampling edge over Happiness, which overcorrects: it
  cost majority-class accuracy without lifting the rare classes. Label smoothing
  (`0.1`) tolerates RAF-DB's label noise. A weighted-loss alternative is
  available via `dataset.class_weights_for_loss`.
- Augmentation is deliberately gentle because RAF-DB ships pre-aligned crops:
  `RandomResizedCrop` for scale jitter, 8-degree rotation, no hue jitter.
- `opencv-python` is pinned below 5.0 — OpenCV 5 removed the bundled Haar
  cascade XML files that `webcam_demo.py` relies on.
- `export.py` pins the legacy TorchScript exporter (`dynamo=False`) so it works
  without the extra `onnxscript` dependency. To use the newer torch.export-based
  exporter instead, `pip install onnxscript` and drop that flag.

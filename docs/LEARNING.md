# LEARNING.md — how this system was built, from scratch

A complete walkthrough of the RAF-DB facial expression recogniser in this
repository: what the data is, how the network is put together, how it was
trained, how it was measured, what was broken and how that was found, and how
the finished model is deployed to a live webcam.

Written to be read start to finish by someone who has not seen the code.

---

## 1. The problem

**Facial expression recognition (FER).** Given a photograph of a face, decide
which of seven emotions it shows.

The dataset is **RAF-DB Basic** (Real-world Affective Faces Database). Its
images are real photographs from the internet, not posed laboratory captures,
which makes it much harder and much more useful than older FER datasets. Each
image was labelled by around 40 independent annotators, and the "basic" subset
keeps images where annotators broadly agreed on one of seven categories.

On disk it is already in the layout `torchvision.datasets.ImageFolder` expects,
with numbered folders:

```
Dataset/DATASET/train/<1..7>/*.jpg
Dataset/DATASET/test/<1..7>/*.jpg
```

Folder `1..7` maps to internal label `0..6`:

| Folder | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Emotion | Surprise | Fear | Disgust | Happiness | Sadness | Anger | Neutral |

### The data is severely imbalanced

This single fact drives most of the design decisions that follow.

| Class | Train images | Share | Test images |
|---|---|---|---|
| Happiness | 4,772 | 38.9% | 1,185 |
| Neutral | 2,524 | 20.6% | 680 |
| Sadness | 1,982 | 16.2% | 478 |
| Surprise | 1,290 | 10.5% | 329 |
| Disgust | 717 | 5.8% | 160 |
| Anger | 705 | 5.7% | 162 |
| **Fear** | **281** | **2.3%** | **74** |

Happiness has **17 times** as many training images as Fear. A model that never
once predicted Fear would lose only 2.3% of its accuracy. So plain accuracy is a
misleading target here, and that shapes both the loss and the metrics.

The images themselves are **100x100 pixel aligned face crops** — already
cropped to the face and rotated so the eyes are level. Section 9 shows this
turns out to matter enormously for the webcam demo.

---

## 2. Architecture

### Why ResNet-18

A **convolutional neural network** learns visual features by sliding small
learned filters across the image. Early layers respond to edges and textures;
deeper layers combine those into parts (an eye, a curled lip) and finally whole
configurations (a smile).

**ResNet** ("residual network") solved a problem that had blocked deep networks:
stacking many layers made them *harder* to train, not easier, because the error
signal weakened as it propagated backwards. ResNet adds a **skip connection**
around every pair of convolutions:

```
output = F(x) + x
```

The `+ x` gives gradients a clean path straight back to earlier layers. It also
means a block only has to learn the *residual* — the difference from passing the
input through unchanged — which is an easier thing to learn than the full
mapping.

**ResNet-18** is the smallest member of the family: 18 weighted layers,
11.2 million parameters. It is a deliberate fit to this problem. RAF-DB has only
12,271 training images, and a larger network (ResNet-50 at 25M parameters) would
overfit faster without better features to show for it.

### The layer-by-layer structure

Input is a `3 x 224 x 224` tensor (RGB, 224 pixels square).

| Stage | What it does | Output shape |
|---|---|---|
| `conv1` | 7x7 convolution, 64 filters, stride 2 | 64 x 112 x 112 |
| `bn1` + ReLU | normalise, then non-linearity | 64 x 112 x 112 |
| `maxpool` | 3x3 max pool, stride 2 | 64 x 56 x 56 |
| `layer1` | 2 residual blocks, 64 channels | 64 x 56 x 56 |
| `layer2` | 2 residual blocks, 128 channels, stride 2 | 128 x 28 x 28 |
| `layer3` | 2 residual blocks, 256 channels, stride 2 | 256 x 14 x 14 |
| `layer4` | 2 residual blocks, 512 channels, stride 2 | 512 x 7 x 7 |
| `avgpool` | average each channel over space | 512 x 1 x 1 |
| `fc` | linear layer, 512 -> 7 | 7 |

The pattern is the standard CNN trade: spatial resolution halves at each stage
while channel count doubles. The network gives up *where* things are in exchange
for richer descriptions of *what* they are.

Each **residual block** is: 3x3 conv -> BatchNorm -> ReLU -> 3x3 conv ->
BatchNorm, then add the input and apply ReLU. Where the channel count changes, a
1x1 "downsample" convolution on the skip path matches the shapes so the addition
is valid.

**BatchNorm** normalises each channel to roughly zero mean and unit variance
using statistics from the current batch. It keeps activations in a healthy range
and lets training use larger learning rates. It also keeps a *running average* of
those statistics for use at inference time, when there is no batch to measure.
That running average causes a subtle bug, described in section 7.2.

**The final `avgpool` is worth pausing on.** It collapses each of the 512
channels to a single number by averaging over the 7x7 grid. This is why the
network accepts the image as a whole rather than needing a fixed arrangement of
parts, and it is why the parameter count stays small: the classifier sees only
512 numbers regardless of input size.

Total: **11,180,103 parameters** across 62 tensors.

### Transfer learning

Training 11 million parameters from scratch on 12,000 images would badly overfit.
Instead the network starts from weights trained on **ImageNet**, a 1.28 million
image, 1000 class classification dataset.

The insight is that the early and middle layers of an ImageNet network have
learned genuinely general visual machinery — edge detectors, texture and shape
descriptors — that is useful far beyond the original 1000 categories. Only the
final classifier is specific to ImageNet's classes, so that layer is discarded
and replaced:

```python
model.fc = nn.Linear(512, 7)     # fresh, random, 7-way
```

This is **transfer learning**, and it is the single reason a model this size can
be trained on this little data.

---

## 3. Data pipeline

### The validation split — the most important decision here

RAF-DB ships only `train` and `test`. There is no validation set.

Training needs a validation set because we must choose **which epoch's weights to
keep**. The model is evaluated after every epoch and the best-scoring one is
saved. If that choice is made by looking at the test set, the test score stops
being an honest estimate of performance on unseen data. You have effectively
fitted to the test set through the act of choosing, and the number you report is
optimistically biased.

So a **stratified 10% of the training data** is held out as validation.
"Stratified" means the split is done per class, so the validation set mirrors the
training class proportions exactly rather than by luck:

| Class | Train | Val | Train % | Val % |
|---|---|---|---|---|
| Surprise | 1,161 | 129 | 10.51% | 10.52% |
| Fear | 253 | 28 | 2.29% | 2.28% |
| Disgust | 645 | 72 | 5.84% | 5.87% |
| Happiness | 4,295 | 477 | 38.89% | 38.91% |
| Sadness | 1,784 | 198 | 16.15% | 16.15% |
| Anger | 635 | 70 | 5.75% | 5.71% |
| Neutral | 2,272 | 252 | 20.57% | 20.55% |

Final arrangement: **11,045 train / 1,226 validation / 3,068 test.**

The split uses its own fixed seed (`SPLIT_SEED = 1234`), deliberately separate
from the training seed, so that runs with different training seeds still share
an identical split and remain comparable.

One implementation subtlety: training images get random augmentation and
validation images must not. Since both come from the same directory, the code
opens **two** `ImageFolder` objects over it — one with augmentation, one without
— and indexes into them with `Subset`. Splitting a single dataset afterwards
would leak augmentation into validation.

### Augmentation

**Augmentation** applies random transformations to training images so the model
sees a slightly different version each epoch. It cannot memorise individual
images, and it learns that identity is preserved under these changes.

The augmentations here are deliberately *gentle*, because RAF-DB images are
already aligned crops:

| Transform | Setting | Why |
|---|---|---|
| `RandomResizedCrop` | 224, scale 0.8-1.0, ratio 0.9-1.1 | scale and position jitter, the variation a camera really introduces |
| `RandomHorizontalFlip` | p = 0.5 | a mirrored face is still the same expression |
| `RandomRotation` | 8 degrees | small tolerance to head tilt |
| `ColorJitter` | brightness/contrast/saturation 0.2, **hue 0** | lighting varies; skin hue does not signal emotion |
| `ToTensor` | — | to a float tensor, scaled to 0-1 |
| `Normalize` | ImageNet mean/std | match what the pretrained weights expect |
| `RandomErasing` | p = 0.25 | blank a random patch, forcing use of the whole face |

Two of these were changed from the original code, and the reasoning generalises:
rotation was 12 degrees and hue jitter was 0.05. Both *fight the data*. Rotating
an already-aligned face destroys the alignment the dataset provides and brings
black corners into frame, and hue shifts are not a cue any expression depends
on. Augmentation should simulate variation the model will genuinely meet, not
add noise for its own sake.

Validation and test use only `Resize` -> `ToTensor` -> `Normalize`, with no
randomness, so their scores are deterministic and comparable across epochs.

### Normalisation

```
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
```

Each channel has the mean subtracted and is divided by the standard deviation.
These specific constants are ImageNet's channel statistics. They must be used
because the pretrained weights were calibrated against inputs in that
distribution. **The same constants must be applied at inference**, which is why
the webcam demo imports them from `rafdb/core/config.py` rather than hardcoding them.

### Handling the imbalance

The chosen approach is a **`WeightedRandomSampler`**, which changes how often
each image is *drawn* during an epoch rather than how much it counts in the loss.
Each image gets a sampling weight

```
weight = 1 / count(its class) ** IMBALANCE_POWER
```

`IMBALANCE_POWER` controls the strength, and choosing it is a real trade-off:

- **1.0** — full inverse frequency. Every class appears equally often. This
  gives Fear a **17x** sampling edge over Happiness, meaning each of the 253 Fear
  images is revisited many times per epoch. That invites overfitting on those
  specific images while starving the majority classes.
- **0.5** — square-root inverse frequency. Used here.
- **0.0** — no rebalancing.

The original code used 1.0, and the evidence says it backfired: Fear recall was
50.0% and Disgust 48.75% while Happiness fell to 87.4%, below the ~93% a plain
baseline achieves. It paid for the rare classes and got nothing back.

At 0.5 an epoch actually draws:

| Class | Natural share | Drawn share |
|---|---|---|
| Fear | 2.29% | 6.22% |
| Disgust | 5.84% | 9.90% |
| Anger | 5.75% | 10.29% |
| Happiness | 38.89% | 25.41% |

The rare classes get a real, meaningful boost without the distortion.

---

## 4. The loss function

**Cross-entropy loss** is the standard choice for classification. The network's
seven output numbers (**logits**) are turned into probabilities by **softmax**:

```
p_i = exp(z_i) / sum_j exp(z_j)
```

and the loss is `-log(p_correct)`. Confident and correct gives a loss near zero;
confident and wrong is punished heavily.

**Label smoothing (0.1)** modifies the target. Instead of asking the model for
probability 1.0 on the true class and 0.0 elsewhere, it asks for 0.9 on the true
class with the remaining 0.1 spread across the others.

This matters especially for RAF-DB. The labels come from human annotators who
did not fully agree, and many real expressions genuinely sit between categories
— a face can be legitimately part sad and part neutral. Demanding total certainty
on a noisy, ambiguous label teaches overconfidence. Smoothing keeps the model
honest, and it improves **calibration**, meaning the confidence it reports is
closer to how often it is actually right.

Note that per-class loss weights are *not* used on top of the sampler. Doing both
compounds the correction and tends to overshoot.

---

## 5. Training

### Two-phase fine-tuning

The backbone carries useful pretrained features; the head is random. Training
them together immediately is harmful, because large gradients from the random
head flow backwards and damage the good features before they can help.

So training runs in two phases:

**Phase 1 (epochs 1-3): frozen backbone.** Every backbone parameter has
`requires_grad = False` and its learning rate set to zero. Only the 3,591
parameters of the new head train. The backbone acts as a fixed feature
extractor while the head learns to read its output.

**Phase 2 (epochs 4-30): end-to-end.** Everything unfreezes and the whole network
adapts to faces.

### One optimizer, one schedule

A natural but wrong implementation creates a fresh optimizer for each phase. That
throws away the head's accumulated momentum at the boundary and restarts the
learning rate schedule mid-run. Instead **one optimizer and one schedule span all
30 epochs**, and phase 1 is expressed simply by holding the backbone learning
rates at zero.

### AdamW

**AdamW** adapts a per-parameter learning rate from running estimates of the
gradient's mean and variance, so parameters with small or noisy gradients still
make progress. The "W" is **decoupled weight decay**: the decay is applied
directly to the weights rather than folded into the gradient, which is the
mathematically correct form for adaptive optimizers.

**Weight decay (1e-4)** shrinks weights slightly at each step, discouraging large
values and so reducing overfitting. But it is applied only to conv and linear
weight *matrices*. **BatchNorm parameters and all biases are excluded**, because
decaying a normalisation layer's scale and shift works against the normalisation
itself. The code detects these by `param.ndim <= 1`, which catches every bias and
every BatchNorm weight.

### Layer-wise learning rates

Not every layer should learn at the same speed. Early layers hold generic edge
and texture filters that transfer almost unchanged; the head is random and must
move a long way.

| Stage | Multiplier | Effective LR |
|---|---|---|
| `conv1` + `bn1` (stem) | 0.1x | 1e-4 |
| `layer1` | 0.1x | 1e-4 |
| `layer2` | 0.2x | 2e-4 |
| `layer3` | 0.4x | 4e-4 |
| `layer4` | 0.6x | 6e-4 |
| `fc` (head) | 1.0x | 1e-3 |

A single flat learning rate has to compromise: high enough for the head means
high enough to damage the stem.

### The schedule: warmup then cosine

The learning rate is not constant. It is stepped **per batch**, not per epoch.

**Warmup (first 2 epochs).** The rate rises linearly from near zero to the peak.
At initialisation the gradients are large and unreliable; taking full-size steps
immediately can knock the network into a bad region it never recovers from.

**Cosine annealing (remaining epochs).** The rate follows a half cosine from peak
down to zero:

```
lr = peak * 0.5 * (1 + cos(pi * progress))
```

Large steps early explore broadly; small steps late settle precisely into a
minimum. The smooth decay of cosine works better in practice than abrupt drops.

### Hyperparameters in full

| Setting | Value |
|---|---|
| Image size | 224 x 224 |
| Batch size | 64 |
| Epochs | 30 (3 frozen + 27 end-to-end) |
| Optimizer | AdamW |
| Peak head LR | 1e-3 |
| Weight decay | 1e-4, excluding norms and biases |
| Label smoothing | 0.1 |
| Warmup | 2 epochs |
| Schedule | linear warmup -> cosine to zero |
| Imbalance power | 0.5 |
| Seed | 42 |
| Device | MPS (Apple Silicon GPU) |

### What actually happened

| Epoch | Phase | Train loss | Val accuracy | Val macro-F1 |
|---|---|---|---|---|
| 1 | frozen | 1.8306 | 0.4413 | 0.2914 |
| 3 | frozen | 1.5942 | 0.4739 | 0.3902 |
| 4 | **unfrozen** | 1.2580 | 0.7341 | 0.6412 |
| **21** | **e2e (best)** | **0.4952** | **0.8744** | **0.8215** |
| 30 | e2e | 0.4761 | 0.8679 | 0.8099 |

The jump at epoch 4 is the backbone unfreezing: validation accuracy goes from
0.47 to 0.73 in one epoch. That is the clearest possible evidence that frozen
ImageNet features alone are not enough for faces, and that fine-tuning them is
where the performance comes from.

Validation peaks at epoch 21 and drifts slightly down afterwards — mild
overfitting, exactly what best-checkpoint selection exists to handle.

Total training time was 9.7 hours, though that is not a clean measurement: the
machine was running other work, and the slowest single epoch took 7,434 seconds
against a typical 130.

---

## 6. Metrics — what was measured and why

With 38.9% of the data in one class, **accuracy alone is close to useless**. A
model predicting only Happiness and Neutral scores about 59% while being blind to
five of seven emotions. So several metrics are reported together.

For a single class: **TP** true positives, **FP** false positives, **FN** false
negatives.

**Precision** = TP / (TP + FP). Of the images called Fear, how many were Fear?
Low precision means false alarms.

**Recall** = TP / (TP + FN). Of the actual Fear images, how many were found?
Low recall means misses.

These trade off. Predicting Fear for everything gives perfect recall and dreadful
precision.

**F1** is their harmonic mean:

```
F1 = 2 * precision * recall / (precision + recall)
```

The harmonic mean is used rather than the plain average because it punishes
imbalance: 1.0 precision with 0.1 recall averages 0.55 but has F1 of just 0.18.
You cannot score well by sacrificing one for the other.

**Macro-F1** averages the per-class F1 scores giving **every class equal weight**,
regardless of size. Fear's 74 test images count as much as Happiness's 1,185.
This is the **model selection metric** for exactly that reason: it is the number
that improves only if the rare classes genuinely improve.

**Weighted-F1** averages weighted by class size, so it tracks accuracy closely
and is dominated by the majority classes. Reported for comparison.

**Balanced accuracy** is the mean of per-class recall. The gap between accuracy
(0.8611) and balanced accuracy (0.7663) directly measures how much the headline
number is being carried by the easy, common classes.

**The confusion matrix** is the most informative single view. Rows are the true
class, columns the prediction, so cell (i, j) counts images of class i called
class j. The diagonal is correct; everything off it names a specific failure.
Both raw counts and row-normalised percentages are reported.

**Calibration** asks whether confidence is trustworthy. When the model says 90%,
is it right 90% of the time? The report gives mean confidence when correct
(0.8438) against when wrong (0.6093) — the model is appropriately less sure when
it is about to be wrong — and an **ECE-lite** score of 0.0499, a five-bin
approximation of Expected Calibration Error, the average gap between confidence
and actual accuracy. Under 0.05 is well calibrated, and the label smoothing
deserves much of the credit.

---

## 7. Five bugs, and how each was found

This section is the most useful part of the document, because the bugs are all
ones that fail *silently*. Nothing crashes. The numbers simply lie.

### 7.1 Test-set leakage in model selection

**The bug.** The original code used the official test partition as its validation
set, selected `best_model.pth` by test macro-F1 across 25 epochs, then reported
performance on that same partition.

**Why it matters.** Picking the best of 25 attempts on a set and then reporting
that set does not measure generalisation, it measures the best of 25 draws. The
reported 79.17% was optimistically biased.

**The fix.** The stratified validation split of section 3. Test is now read
exactly once, by `rafdb/pipeline/evaluate.py`, after all decisions are final.

### 7.2 BatchNorm was never actually frozen

**The bug.** `freeze_backbone` set `requires_grad = False` on every backbone
parameter. That correctly stops gradient updates. But BatchNorm's `running_mean`
and `running_var` are **buffers, not parameters**. They are refreshed on every
forward pass whenever the module is in training mode, entirely independently of
gradients.

So during the phase explicitly designed to hold the backbone fixed, every
BatchNorm layer was quietly rewriting the ImageNet statistics the frozen weights
were calibrated against.

**How it was found.** By direct measurement — capture `bn1.running_mean`, run one
forward pass, compare:

```
drift with set_bn_eval : 0.000e+00
drift without it       : 1.263e-02
```

**The fix.** `set_bn_eval()` puts every BatchNorm module in eval mode, called
after `model.train()` on frozen epochs.

### 7.3 Wrong initialisation on the classifier head

**The bug.** The new head was initialised with Kaiming normal, `mode="fan_out"`:

```python
nn.init.kaiming_normal_(model.fc.weight, mode="fan_out", nonlinearity="relu")
```

Kaiming initialisation is designed for convolutional layers feeding a ReLU. For a
`Linear(512, 7)`, `fan_out` is 7, so the standard deviation becomes
`sqrt(2/7) = 0.53` — roughly **20 times too large**.

**The consequence.** Logits came out with a standard deviation around 8, and
training started at a loss of **11.9**. For 7 classes a correctly initialised
model starts at `ln(7) = 1.95`. The early epochs were spent undoing the
initialisation rather than learning.

**How it was found.** The first smoke run printed `train_loss 8.7697` in epoch 1.
That number is impossible for a healthy 7-class problem, which prompted a direct
comparison of initialisation schemes.

**The fix.** `nn.init.normal_(std=0.01)`, giving a starting loss of 1.95.

### 7.4 `pin_memory` set for the wrong device

**The bug.** Pinned host memory exists to make asynchronous host-to-device copies
possible, which benefits **CUDA**. MPS shares memory with the host, so pinning
buys nothing there. The condition was inverted — enabled for MPS, disabled for
CUDA — so it was the wrong setting on both.

### 7.5 Layer-wise learning rates silently did nothing

This one was introduced *during this work* and caught before it cost a run.

**The bug.** `LambdaLR` snapshots each group's `initial_lr` into
`scheduler.base_lrs` when it is constructed, and multiplies **that list** by the
lambda on every step. Code that raised the backbone learning rate at unfreeze
time by editing the optimizer's param groups was therefore ignored.

The backbone would have sat at learning rate zero for all 30 epochs. Training
would have completed, printed plausible-looking numbers, and simply never
fine-tuned the backbone.

**How it was found.** A deliberate test of the transition before launching:

```
frozen phase       {'stem': 0.00e+00, ..., 'fc': 2.00e-04}
after unfreeze     {'stem': 0.00e+00, ..., 'fc': 3.00e-04}   <-- still zero
```

**The fix.** Rewrite `scheduler.base_lrs` alongside the param groups. Verified:

```
after unfreeze     {'stem': 7.00e-05, 'layer2': 1.40e-04, 'layer4': 4.20e-04, 'fc': 7.00e-04}
```

**The lesson.** Every one of these five is invisible without a targeted check.
Training "worked" in all five cases. If a mechanism matters, test that it does
what you believe it does.

---

## 8. Results

Evaluated **once** on the official 3,068 image test partition, using the
checkpoint selected on validation, with horizontal-flip test-time augmentation.

### Overall

| Metric | Before | After | Change |
|---|---|---|---|
| Accuracy | 0.7917 (leaked) | **0.8611** | +6.9 |
| Macro-F1 | 0.7028 | **0.7881** | +8.5 |
| Balanced accuracy | 0.7010 | **0.7663** | +6.5 |
| Weighted-F1 | 0.7925 | 0.8594 | +6.7 |
| Macro-precision | 0.7082 | 0.8182 | +11.0 |

The comparison flatters the new model slightly *less* than it appears: the old
number was inflated by test-set selection, so the true gap is wider than +6.9.

### Per class

| Class | Precision | Recall | F1 | Support | Old recall |
|---|---|---|---|---|---|
| Happiness | 0.9362 | **0.9409** | 0.9386 | 1,185 | 87.43% |
| Neutral | 0.8126 | 0.8735 | 0.8420 | 680 | 77.79% |
| Surprise | 0.8640 | 0.8693 | 0.8667 | 329 | 84.50% |
| Sadness | 0.8193 | 0.8347 | 0.8269 | 478 | 76.15% |
| Anger | 0.8741 | 0.7284 | 0.7946 | 162 | 66.05% |
| Fear | 0.7925 | **0.5676** | 0.6614 | 74 | 50.00% |
| Disgust | 0.6286 | **0.5500** | 0.5867 | 160 | 48.75% |

Every class improved. **The target of 60% recall on Fear and Disgust was not
met** — they reached 56.8% and 55.0%. They remain the weak classes and this is
stated plainly rather than buried.

Note the *shape* of Fear's result: precision 0.79 against recall 0.57. When the
model says Fear it is usually right; it simply misses many Fear faces. That is
the signature of a class the model is reluctant to predict, which is what 253
training images produces.

### What gets confused with what

| True | Predicted | Count |
|---|---|---|
| Sadness | Neutral | 43 |
| Neutral | Sadness | 41 |
| Happiness | Neutral | 36 |
| Neutral | Happiness | 31 |
| Disgust | Sadness | 24 |
| Disgust | Neutral | 23 |
| Fear | Surprise | 14 |

These are not random. Sadness and Neutral are mutually confused, as are
Happiness and Neutral, and Fear leaks into Surprise. Every one of these pairs is
genuinely visually similar — a mildly sad face and a neutral face differ subtly,
and fear and surprise share wide eyes and a raised brow. Human annotators
disagree on the same pairs. Some of this error is irreducible given the labels.

### Honest caveat

**This is one seed.** Fear has 74 test images, so a single flipped prediction
moves its recall by 1.35 points. Run-to-run variation can exceed the differences
often claimed between methods. `rafdb/reporting/aggregate_seeds.py` exists to report mean and
standard deviation across seeds, and any publication of these numbers should use
it.

---

## 9. Deployment

### ONNX export

Training runs in PyTorch, but shipping a PyTorch dependency to every consumer is
heavy. **ONNX** (Open Neural Network Exchange) is a portable graph format, so the
trained model is exported once and then runs anywhere with an ONNX runtime.

The export uses opset 17, with a **dynamic batch axis** so the graph accepts any
batch size, and runs on CPU regardless of the training device for portability.

Export is only trustworthy if verified, so `rafdb/pipeline/export.py` does two checks:

1. **Structural** — `onnx.checker` validates the graph.
2. **Numerical** — PyTorch and onnxruntime logits are compared on batch size 1,
   a random batch of 8, and a real batch of test images.

```
batch=1 (dummy)                  max|delta| = 1.550e-06  [OK]
batch=8 (random, dynamic axis)   max|delta| = 1.669e-06  [OK]
batch=8 (real test images)       max|delta| = 2.027e-06  [OK]
```

Differences of 1e-6 are floating-point noise. The exported 43 MB graph is the
same model.

### The webcam demo

`rafdb/deploy/webcam_demo.py` runs the ONNX graph on a live camera. It deliberately **does not
import torch**, so it exercises the real deployment artifact rather than the
training-time model.

Per frame: capture -> detect the face -> align it -> preprocess -> ONNX
inference -> softmax -> smooth over time -> draw.

**Preprocessing parity is the thing that can silently break.** The demo uses
OpenCV while training used PIL, and the two resize slightly differently. Rather
than assume, the entire test split was pushed through the demo's own code path:

```
demo path : accuracy 0.8523, macro-F1 0.7753
evaluate.py reference : 0.8543 / 0.7778
```

A 0.2 point gap purely from interpolation. The preprocessing is faithful.

### Face detection: Haar vs BlazeFace

The first version used OpenCV's **Haar cascade** — a classical, hand-designed
detector from 2001. It was replaced by **BlazeFace** (short range), a small
neural detector from Google's MediaPipe, tuned for faces within about 2 metres.

Measured on the live camera:

| | Haar | BlazeFace |
|---|---|---|
| Time per frame | 4.1 ms | **2.1 ms** |
| Faces found | 0 of 20 | **20 of 20** |
| Landmarks | none | 6 keypoints |

BlazeFace is faster *and* far more reliable. Haar also cannot detect an already
tight crop at all, at any parameter setting.

### Face alignment — matching the training distribution

This is the most interesting deployment finding.

RAF-DB images are **landmark-aligned**. Measuring how tightly, by running
BlazeFace over 400 training images:

| Keypoint | x | y |
|---|---|---|
| Right eye | 0.3032 +/- 0.059 | 0.3411 +/- 0.056 |
| Left eye | 0.6927 +/- 0.053 | 0.3382 +/- 0.053 |
| Nose | 0.4978 | 0.5523 |
| Mouth | 0.4972 | 0.7504 |

Inter-eye distance is 0.3895 of the crop width, and the midpoint is at
(0.498, 0.340). Standard deviations near 0.05 confirm the dataset really is
aligned to a canonical geometry.

**Webcam faces are not.** Heads tilt, sit off-centre, and fill different
fractions of the box. Feeding raw crops to a model trained only on aligned faces
is a train/deploy mismatch.

Since BlazeFace returns eye positions, each frame is **similarity-warped** so the
eyes land on exactly those canonical coordinates. A similarity transform —
rotation, uniform scale, translation — is the correct family: it cancels head
roll and normalises face size **without shearing or stretching**, which would
distort the very geometry the expression is read from. Two point pairs determine
it exactly.

To test whether this helps, test faces were placed into larger scenes with random
head roll, scale and offset, simulating webcam conditions:

| | Raw box crop | Eye-aligned |
|---|---|---|
| Accuracy | 0.7689 | **0.7980** |
| Macro-F1 | 0.6552 | **0.7150** |

Worth 2.9 points of accuracy and 6.0 of macro-F1.

**The honest counter-result:** on already-aligned still images alignment
slightly *hurts*, 0.8523 down to 0.8321, because the warp resamples the image and
adds landmark noise on top of the dataset's own better alignment. Hence
`--no-align`. Alignment helps exactly when the input is misaligned, which is the
live case and not the benchmark case.

### Performance

The demo first ran at **4.4 FPS**. Profiling each stage rather than guessing
found the cause immediately:

| Stage | Time |
|---|---|
| Camera capture | 22.70 ms |
| Detection (BlazeFace) | 1.69 ms |
| Align + inference (CoreML) | 8.71 ms |
| Drawing | 0.47 ms |
| Frame flip | 0.16 ms |
| **Total** | **33.74 ms** |

The dominant fix was the **execution backend**. ResNet-18 at 224x224 is about
1.8 GFLOPs per frame, and it was running on the CPU while the Mac's Neural Engine
sat idle. Switching to onnxruntime's CoreML provider:

| Backend | Inference |
|---|---|
| CPU (4 threads) | ~49 ms |
| **CoreML** | **~1.2 ms** |

Verified not to cost accuracy: predictions agree with CPU on **99.97%** of the
test set, the difference being fp16 rounding.

Result: **26.1 FPS steady state**, a 6x improvement, now limited by the camera
rather than the model.

Two things tried that did **not** work, recorded because negative results are
worth keeping:

- **Downscaling the frame before detection.** 24.2 ms against 23.4 ms full
  resolution — no gain, because shrinking `minSize` to compensate adds back the
  pyramid levels the smaller image saved. Reverted.
- **Blending only the overlay panel** instead of the whole frame. 12.9 to 13.5
  FPS, almost nothing. The overlay was never the bottleneck.

An earlier 13.5 FPS reading was also simply **wrong**: it counted CoreML graph
compilation and camera startup as loop time. Measuring steady state separately
gave the true 26.1.

---

## 9b. The figures, and what each one is for

Every figure is written to its own file. Multi-panel images are convenient on
screen and useless in a report, where each view needs its own caption and its own
place in the argument.

All of them share one palette, validated rather than eyeballed. The two-series
colours **#00A693** (Persian green) and **#C2410C** (burnt orange) pass the full
six-check test: lightness band, chroma floor, colour-vision-deficiency
separation, normal-vision floor, and contrast against the page. The worst pair
separation is Delta E 11.8 under simulated protanopia against a target of 8, and
25.7 under normal vision against a floor of 15. The confusion matrices use a
single-hue green ramp running light to dark, never a rainbow, because they encode
magnitude rather than identity.

Regenerate the training curves with `python -m rafdb.reporting.plot_history` and the evaluation
figures with `python -m rafdb.pipeline.evaluate --tta`.

### Training curves

**`fig_train_loss.png` — did the model fit the training data?**
Cross-entropy loss against epoch. The dashed line marks epoch 3, where the
backbone unfreezes. The curve falls from 1.83 to 0.48 and flattens, which is
what a healthy run looks like. It does not approach zero, and should not: label
smoothing at 0.1 puts a deliberate floor under the loss, because the model is
never asked to be fully certain. Use this figure to show training was stable and
converged. A loss that started near 11.9 instead of 1.95 is what exposed the
head-initialisation bug in section 7.3.

**`fig_val_metrics.png` — did it generalise, and which epoch was kept?**
Two series on one axis, accuracy and macro-F1, measured on the held-out
validation split. This is the most important figure of the three, for three
reasons. The step at epoch 4 is the backbone unfreezing, taking accuracy from
0.47 to 0.73 in a single epoch, which is the clearest evidence that frozen
ImageNet features alone are insufficient for faces. The persistent gap between
the two lines is the class imbalance made visible: accuracy is always higher
because it is carried by Happiness and Neutral. The marked point at epoch 21 is
the checkpoint that was kept, chosen on macro-F1, and the slight decline after it
is the mild overfitting that best-checkpoint selection exists to absorb.

**`fig_lr_schedule.png` — what did the learning rate actually do?**
The head's learning rate against epoch: linear warmup for two epochs, then cosine
decay to zero. Include this when explaining the optimisation recipe. Note it
shows the head only; backbone groups run at 0.1x to 0.6x of these values under
layer-wise scaling, and at exactly zero during the frozen phase. That last detail
is what the bug in section 7.5 silently broke.

### Evaluation figures

These exist in two variants, with and without flip test-time augmentation. The
`_tta` files are the ones behind the reported 0.8611 accuracy.

**`fig_confusion_raw_test_tta.png` — how many images landed where?**
Rows are the true class, columns the prediction, so the diagonal is correct and
every other cell is one specific kind of mistake. Raw counts are the honest view
of scale: the single largest cell is Happiness, correct 1,115 times, which is
more images than five of the seven classes contain in total. Use this figure when
the point is volume.

**`fig_confusion_normalised_test_tta.png` — what share of each true class went where?**
The same matrix with each row divided by its own total, so rows sum to 100%. This
is the figure to read for behaviour rather than volume, because it makes classes
of wildly different size comparable. It is where the structure of the errors
becomes visible: Sadness and Neutral leak into each other, Fear sends 18.9% of
its images to Surprise, and Disgust scatters into Sadness and Neutral. Those are
genuinely similar expressions that human annotators also disagree on, so part of
that error is irreducible given the labels.

**`fig_per_class_recall_test_tta.png` — which emotions does the model actually find?**
Recall per class, sorted, with the rare classes in orange and the value printed
on every bar so identity never rests on colour alone. The dashed rule is balanced
accuracy at 76.6%, the mean of the seven bars. This is the figure that makes the
central weakness of the model impossible to hide: Happiness at 94.1% sits beside
Fear at 56.8% and Disgust at 55.0%. If you show only one evaluation figure,
show this one, because overall accuracy of 0.8611 conceals exactly this spread.

### Training-time figure

**`val_confusion_best.png`** is written by `rafdb/pipeline/train.py` at the best epoch only, on
the validation split. It is a progress check during training rather than a
reportable result; the test-split figures above are the ones that belong in a
writeup.

---

## 10. File map

| Module | Role |
|---|---|
| `rafdb/core/config.py` | every path, class name and hyperparameter — one source of truth |
| `rafdb/core/dataset.py` | datasets, stratified split, augmentation, sampler, device selection |
| `rafdb/core/model.py` | ResNet-18 construction, freeze/unfreeze, BN-eval, parameter groups |
| `rafdb/pipeline/train.py` | training loop, phase handling, metrics, history CSV, checkpoints |
| `rafdb/pipeline/evaluate.py` | full report on a chosen split, figures, metrics JSON |
| `rafdb/pipeline/export.py` | ONNX export with verification against PyTorch |
| `rafdb/reporting/plotstyle.py` | shared palette and figure styling |
| `rafdb/reporting/plot_history.py` | training curves, one figure per file |
| `rafdb/reporting/aggregate_seeds.py` | mean and standard deviation across seeds |
| `rafdb/deploy/webcam_demo.py` | live camera inference on the exported graph |
| `run_all.py` | runs every stage in order, with checks and a summary |

Run a stage as a module from the project root, for example
`python -m rafdb.pipeline.train --seed 42`, or run everything with
`python3 run_all.py`.

Everything reads its settings from `rafdb/core/config.py`, so the training, evaluation,
export and deployment paths cannot drift apart. That is what made the
preprocessing parity check in section 9 possible.

---

## 11. What would come next

Honest assessment of where the remaining headroom is, in order of expected value:

1. **A face-pretrained backbone.** ImageNet features are generic; initialising
   from a face recognition network (VGGFace2, MS1M) is the single largest known
   lever in FER, typically worth several points. This is the first thing to try.
2. **Three seeds and variance.** Required before quoting any of these numbers.
   The machinery is already in `rafdb/reporting/aggregate_seeds.py`.
3. **The rare classes.** Fear and Disgust are still near 55%. Options include
   focal loss, targeted augmentation, or simply more Fear data from a compatible
   dataset such as AffectNet.
4. **A smaller or quantised model.** At 43 MB and 1.8 GFLOPs, ResNet-18 is
   comfortable on a laptop but heavy for a phone. Training at 112x112 or moving
   to MobileNetV3 would cut both substantially.

"""Live webcam emotion recognition using the exported ONNX model.

Runs `outputs/raf_resnet18.onnx` on the laptop camera in real time. Faces are
located with OpenCV's Haar cascade, cropped, preprocessed exactly as
`dataset.eval_transforms` does at evaluation time, and pushed through
onnxruntime.

This script deliberately does *not* import torch: it exercises the deployment
artifact (the ONNX graph) rather than the training-time model, so what you see
here is what a deployed service would produce.

Run:
    python webcam_demo.py                  # best available ONNX graph
    python webcam_demo.py --camera 1       # second camera
    python webcam_demo.py --no-detect      # skip detection, use centre crop
    python webcam_demo.py --margin 0.25    # looser crop around the face box

Press q or Esc to quit.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from rafdb.core import config as C

# --------------------------------------------------------------------------
# Drawing constants. BGR, because OpenCV.
# --------------------------------------------------------------------------
BOX_COLOR = (60, 200, 90)
BOX_COLOR_WEAK = (90, 140, 200)     # low-confidence prediction
TEXT_COLOR = (255, 255, 255)
PANEL_BG = (32, 32, 32)
BAR_COLOR = (200, 160, 60)
BAR_COLOR_TOP = (60, 200, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# Preprocessing — must mirror dataset.eval_transforms()
# --------------------------------------------------------------------------
# transforms.Normalize works on CHW tensors; we keep HWC until the transpose
# at the end, so mean/std are shaped to broadcast over the channel axis.
_MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
_STD = np.array(C.IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)


def preprocess(bgr_crop: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 crop -> normalised NCHW float32 batch of 1.

    Mirrors eval_transforms(): Resize((224, 224)) -> ToTensor() -> Normalize().
    ToTensor() converts RGB and scales to [0, 1], which is why the channel swap
    and the /255 both happen here.
    """
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    # INTER_AREA matches PIL's downscaling behaviour more closely than the
    # default bilinear when the crop is larger than 224.
    interp = cv2.INTER_AREA if rgb.shape[0] > C.IMAGE_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (C.IMAGE_SIZE, C.IMAGE_SIZE), interpolation=interp)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.transpose(arr, (2, 0, 1))[None, ...].copy()   # 1x3x224x224


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# RAF-DB canonical geometry
# --------------------------------------------------------------------------
# RAF-DB Basic ships landmark-aligned crops, and it is aligned tightly: running
# BlazeFace over 400 training images puts the eyes at y = 0.34 with a standard
# deviation of only 0.05, symmetric about the centre.
#
# Webcam frames have no such alignment — the head tilts, sits off-centre, and
# fills a different fraction of the box. Feeding a raw crop to a model trained
# only on aligned faces is a train/deploy mismatch. Given eye keypoints we can
# remove it: warp each frame so the eyes land exactly where RAF-DB puts them.
#
# Measured on the training split (fractions of the crop):
CANON_RIGHT_EYE = (0.3032, 0.3411)
CANON_LEFT_EYE = (0.6927, 0.3382)


def align_face(frame, right_eye, left_eye, size: int = C.IMAGE_SIZE):
    """Similarity-warp `frame` so the eyes land on the RAF-DB canonical points.

    Eye coordinates are in pixels. A similarity transform (rotation, uniform
    scale, translation) is the right family here: it cancels head roll and
    normalises face size without shearing or stretching the face, which would
    distort the very geometry the expression is read from.
    """
    src = np.float32([right_eye, left_eye])
    dst = np.float32([
        [CANON_RIGHT_EYE[0] * size, CANON_RIGHT_EYE[1] * size],
        [CANON_LEFT_EYE[0] * size, CANON_LEFT_EYE[1] * size],
    ])
    # Two point pairs determine a similarity transform exactly.
    (sx, sy), (dx, dy) = src, dst
    svec = sx - sy
    dvec = dx - dy
    scale = np.linalg.norm(dvec) / max(1e-6, np.linalg.norm(svec))
    angle = math.atan2(dvec[1], dvec[0]) - math.atan2(svec[1], svec[0])
    cos_a, sin_a = math.cos(angle) * scale, math.sin(angle) * scale
    M = np.float32([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]])
    # Translate so the source eye midpoint maps onto the destination midpoint.
    src_mid = src.mean(axis=0)
    dst_mid = dst.mean(axis=0)
    M[0, 2] = dst_mid[0] - (M[0, 0] * src_mid[0] + M[0, 1] * src_mid[1])
    M[1, 2] = dst_mid[1] - (M[1, 0] * src_mid[0] + M[1, 1] * src_mid[1])
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# Face handling
# --------------------------------------------------------------------------
def expand_box(x, y, w, h, margin, frame_w, frame_h):
    """Grow a detection box by `margin` (fraction of its size), clipped to frame.

    RAF-DB images are tight aligned face crops but the Haar box is tighter still
    around the eyes/mouth region; a little margin brings the framing closer to
    what the model saw in training.
    """
    dx, dy = int(w * margin), int(h * margin)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(frame_w, x + w + dx)
    y1 = min(frame_h, y + h + dy)
    return x0, y0, x1, y1


def centre_square(frame: np.ndarray):
    """Largest centred square of the frame — the --no-detect fallback."""
    h, w = frame.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return x0, y0, x0 + side, y0 + side


def largest_face(faces):
    """Pick the biggest detection; a demo only ever wants one face."""
    return max(faces, key=lambda f: f[2] * f[3])





# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------
def draw_probability_panel(frame, probs, origin=(10, 10), width=220, row_h=22):
    """Horizontal bars for all 7 class probabilities, top-left of the frame."""
    x0, y0 = origin
    top_idx = int(np.argmax(probs))
    panel_h = row_h * C.NUM_CLASSES + 12
    # Semi-transparent backing so text stays readable over any scene.
    # Blend only the panel rectangle: copying and blending the whole 1280x720
    # frame costs more than the network does, and the rest of it is unchanged.
    px0, py0 = max(0, x0 - 5), max(0, y0 - 5)
    px1 = min(frame.shape[1], x0 + width + 70)
    py1 = min(frame.shape[0], y0 + panel_h)
    roi = frame[py0:py1, px0:px1]
    if roi.size:
        roi[:] = cv2.addWeighted(roi, 0.45, np.full_like(roi, PANEL_BG, np.uint8),
                                 0.55, 0)

    for i, (name, p) in enumerate(zip(C.EMOTION_NAMES, probs)):
        row_y = y0 + 6 + i * row_h
        cv2.putText(frame, f"{name:<9}", (x0, row_y + 12), FONT, 0.42,
                    TEXT_COLOR, 1, cv2.LINE_AA)
        bar_x = x0 + 72
        bar_w = int(round(float(p) * (width - 72)))
        color = BAR_COLOR_TOP if i == top_idx else BAR_COLOR
        if bar_w > 0:
            cv2.rectangle(frame, (bar_x, row_y + 2), (bar_x + bar_w, row_y + 14),
                          color, thickness=-1)
        cv2.putText(frame, f"{float(p)*100:5.1f}%", (x0 + width - 2, row_y + 12),
                    FONT, 0.40, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_face_label(frame, box, label, conf, confident):
    """Face rectangle plus the top-1 label sitting above it."""
    x0, y0, x1, y1 = box
    color = BOX_COLOR if confident else BOX_COLOR_WEAK
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    text = f"{label} {conf*100:.0f}%" if confident else "uncertain"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    ty = max(th + 6, y0 - 8)
    cv2.rectangle(frame, (x0, ty - th - 6), (x0 + tw + 10, ty + 4), color, -1)
    cv2.putText(frame, text, (x0 + 5, ty), FONT, 0.7, (20, 20, 20), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Model / camera setup
# --------------------------------------------------------------------------
def load_session(onnx_path: Path, threads: int = 4, provider: str = "auto"):
    """Open an onnxruntime session and sanity-check the graph's shapes.

    The network, not face detection, dominates this loop: ResNet-18 at 224x224
    is ~1.8 GFLOPs per frame, against ~8 ms for the Haar cascade. Which backend
    runs it is therefore the single biggest lever on frame rate. Measured here:

        CPUExecutionProvider    (4 threads)   ~49 ms   ~20 inferences/s
        CoreMLExecutionProvider (ANE/GPU)      ~1.2 ms  ~800 inferences/s

    So CoreML is preferred on Apple Silicon and the loop becomes camera-bound
    instead of compute-bound. It falls back to CPU wherever CoreML is missing.
    Capping CPU intra-op threads at 4 beats using all 10 cores, which contend.
    """
    if not onnx_path.exists():
        raise SystemExit(
            f"ONNX model not found: {onnx_path}\n"
            "Run `python export.py` first to produce it."
        )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        opts.intra_op_num_threads = threads

    available = ort.get_available_providers()
    if provider == "auto":
        wanted = (["CoreMLExecutionProvider"]
                  if "CoreMLExecutionProvider" in available else [])
        wanted.append("CPUExecutionProvider")
    else:
        wanted = [provider] if provider == "CPUExecutionProvider" else \
                 [provider, "CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(str(onnx_path), opts, providers=wanted)
    except Exception as exc:                      # pragma: no cover
        print(f"  ({wanted[0]} unavailable: {exc}; falling back to CPU)")
        session = ort.InferenceSession(str(onnx_path), opts,
                                       providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"model   : {onnx_path}")
    print(f"backend : {session.get_providers()[0]}")
    print(f"input   : {inp.name} {inp.shape}")
    print(f"output  : {out.name} {out.shape}")
    n_out = out.shape[-1]
    if isinstance(n_out, int) and n_out != C.NUM_CLASSES:
        raise SystemExit(
            f"Model outputs {n_out} classes but config expects {C.NUM_CLASSES}."
        )
    return session, inp.name


def open_camera(index: int):
    """Open a camera, preferring AVFoundation on macOS for a faster handshake."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():             # non-macOS, or that backend unavailable
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {index}.\n"
            "On macOS, grant camera access to your terminal under\n"
            "System Settings > Privacy & Security > Camera, then retry.\n"
            "Try --camera 1 if you have an external or Continuity camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


class HaarDetector:
    """OpenCV's Haar frontal-face cascade. No extra download, but weak.

    It wants a whole head — forehead and chin included — and returns no
    landmarks, so faces cannot be aligned. It also fails outright on already
    tight crops such as the RAF-DB images themselves.
    """

    has_landmarks = False
    name = "haar"

    def __init__(self):
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not path.exists():
            raise SystemExit(
                f"Haar cascade not found at {path}\n"
                "Your OpenCV build does not bundle the cascades (5.0 removed "
                "them).\nInstall the 4.x line: pip install "
                "'opencv-python>=4.9,<5' — or use --detector mediapipe."
            )
        self.cascade = cv2.CascadeClassifier(str(path))
        if self.cascade.empty():
            raise SystemExit(f"Failed to load cascade from {path}")

    def detect(self, frame):
        """Return (box, eyes) with eyes always None — Haar gives no landmarks."""
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=6, minSize=(90, 90)
        )
        if len(faces) == 0:
            return None, None
        return largest_face(faces), None


class BlazeFaceDetector:
    """MediaPipe BlazeFace (short range), the better option for a webcam.

    Short range is the variant tuned for faces within ~2 m, which is where a
    laptop camera sits. Beyond being more reliable than Haar — it even fires on
    tight aligned crops, which Haar never does — it returns six keypoints, and
    the two eye points are what make RAF-DB-style alignment possible.
    """

    has_landmarks = True
    name = "mediapipe-blazeface"

    def __init__(self, model_path: Path, min_conf: float = 0.5):
        if not model_path.exists():
            raise SystemExit(
                f"BlazeFace model not found: {model_path}\n"
                "Download it with:\n"
                "  mkdir -p models && curl -L -o models/blaze_face_short_range.tflite \\\n"
                "    https://storage.googleapis.com/mediapipe-models/face_detector/"
                "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError:
            raise SystemExit(
                "mediapipe is not installed. Either:\n"
                "  pip install 'mediapipe==0.10.35'\n"
                "or fall back to --detector haar."
            )
        self._mp = mp
        self._detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=min_conf,
            )
        )

    def detect(self, frame):
        """Return (box, eyes); eyes is ((rx, ry), (lx, ly)) in pixels."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._detector.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        )
        if not result.detections:
            return None, None
        det = max(result.detections,
                  key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb = det.bounding_box
        box = (bb.origin_x, bb.origin_y, bb.width, bb.height)
        # Keypoint order is fixed: right eye, left eye, nose, mouth, ears.
        # Coordinates are normalised to the frame, so scale them to pixels.
        fh, fw = frame.shape[:2]
        kp = det.keypoints
        eyes = ((kp[0].x * fw, kp[0].y * fh), (kp[1].x * fw, kp[1].y * fh))
        return box, eyes


def build_detector(kind: str, model_path: Path):
    """Construct the requested detector, falling back to Haar if asked."""
    if kind == "haar":
        return HaarDetector()
    return BlazeFaceDetector(model_path)


# --------------------------------------------------------------------------
# One frame, end to end
# --------------------------------------------------------------------------
def infer_crop(crop, session, input_name):
    """Preprocess a BGR crop and return the 7-class softmax, or None if empty."""
    if crop is None or crop.size == 0:
        return None
    logits = session.run(None, {input_name: preprocess(crop)})[0]
    return softmax(logits)[0]


def infer_box(frame, box, session, input_name, eyes=None):
    """Classify the face at `box`; align on the eyes first when we have them."""
    if eyes is not None:
        return infer_crop(align_face(frame, eyes[0], eyes[1]), session, input_name)
    x0, y0, x1, y1 = box
    return infer_crop(frame[y0:y1, x0:x1], session, input_name)


def classify_frame(frame, session, input_name, detector, box, args, annotate=True):
    """Detect, classify, and annotate `frame` in place.

    Returns (box, eyes, probs). `box` is None when no face was found; `eyes` is
    None when the detector provides no landmarks or alignment is disabled.
    """
    fh, fw = frame.shape[:2]
    eyes = None
    if detector is None:
        box = centre_square(frame)
    else:
        found, eyes = detector.detect(frame)
        box = (expand_box(*found, args.margin, fw, fh)
               if found is not None else None)
        if args.no_align:
            eyes = None

    probs = None if box is None else infer_box(frame, box, session,
                                               input_name, eyes)
    if probs is not None and annotate:
        top = int(np.argmax(probs))
        conf = float(probs[top])
        draw_face_label(frame, box, C.EMOTION_NAMES[top], conf,
                        conf >= args.min_conf)
    return box, eyes, probs


def run_single_image(path: Path, session, input_name, detector, args) -> None:
    """Classify one image file and write an annotated copy beside it.

    Lets you verify the model and preprocessing without granting camera access.
    """
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    box, eyes, probs = classify_frame(frame, session, input_name, detector,
                                      None, args)
    if probs is None:
        print("no face found. retry with --no-detect to classify the whole image.")
        return
    order = np.argsort(probs)[::-1]
    print(f"\n{path.name}:")
    for i in order:
        print(f"  {C.EMOTION_NAMES[i]:<10} {float(probs[i])*100:6.2f}%")
    draw_probability_panel(frame, probs)
    out = path.with_name(path.stem + "_annotated.png")
    cv2.imwrite(str(out), frame)
    print(f"\nannotated image -> {out}")


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live webcam emotion recognition with the exported ONNX model."
    )
    ap.add_argument("--onnx", type=Path, default=C.ONNX_PATH,
                    help=f"path to the ONNX graph (default: {C.ONNX_PATH})")
    ap.add_argument("--camera", type=int, default=0,
                    help="camera device index (default: 0)")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="fraction to expand the face box by (default: 0.15)")
    ap.add_argument("--det-every", type=int, default=2,
                    help="run face detection every N frames; steadies the box "
                         "and leaves CPU for the network (default: 2)")
    ap.add_argument("--threads", type=int, default=4,
                    help="onnxruntime intra-op threads, 0 for the default (default: 4)")
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "CoreMLExecutionProvider", "CPUExecutionProvider"],
                    help="execution backend; auto prefers CoreML on Apple "
                         "Silicon, which is ~40x faster than CPU (default: auto)")
    ap.add_argument("--image", type=Path, default=None,
                    help="classify a single image file instead of opening the "
                         "camera; writes <name>_annotated.png next to it")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="temporal smoothing of probabilities, 0 disables; "
                         "raw per-frame softmax flickers badly (default: 0.6)")
    ap.add_argument("--min-conf", type=float, default=0.35,
                    help="hide the label below this softmax confidence (default: 0.35)")
    ap.add_argument("--detector", choices=["mediapipe", "haar"], default="mediapipe",
                    help="face detector; mediapipe (BlazeFace short range) is "
                         "more reliable and returns eye landmarks, which enable "
                         "RAF-DB-style alignment (default: mediapipe)")
    ap.add_argument("--face-model", type=Path,
                    default=C.ROOT / "models" / "blaze_face_short_range.tflite",
                    help="path to the BlazeFace .tflite")
    ap.add_argument("--no-align", action="store_true",
                    help="disable eye-based alignment and use the raw box crop")
    ap.add_argument("--no-detect", action="store_true",
                    help="skip face detection; classify the centre square of the frame")
    ap.add_argument("--mirror", action="store_true", default=True,
                    help="mirror the preview (default: on, more natural to use)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    args = ap.parse_args()

    session, input_name = load_session(args.onnx, args.threads, args.provider)
    detector = None if args.no_detect else build_detector(args.detector,
                                                          args.face_model)
    if detector is not None:
        aligned = detector.has_landmarks and not args.no_align
        print(f"detector: {detector.name}"
              f"   alignment: {'on (eyes -> RAF-DB geometry)' if aligned else 'off'}")

    if args.image is not None:
        run_single_image(args.image, session, input_name, detector, args)
        return

    cap = open_camera(args.camera)
    print("camera  : opened. press q or Esc to quit.\n")

    fps_window: deque[float] = deque(maxlen=30)
    last_box = None          # reused between detection frames
    last_eyes = None
    probs = np.full(C.NUM_CLASSES, 1.0 / C.NUM_CLASSES, dtype=np.float32)
    frame_i = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed; stopping.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            t0 = time.perf_counter()
            fh, fw = frame.shape[:2]

            # Detection runs on a subset of frames; on the others we reuse the
            # previous box, which also stops the rectangle jittering.
            redetect = args.no_detect or frame_i % max(1, args.det_every) == 0
            if redetect:
                last_box, last_eyes, raw = classify_frame(
                    frame, session, input_name, detector, last_box, args,
                    annotate=False,
                )
            elif last_box is not None:
                # Between detections the box and eyes are stale by a frame or
                # two, which is imperceptible at these frame rates.
                raw = infer_box(frame, last_box, session, input_name, last_eyes)
            else:
                raw = None

            if raw is not None:
                # Exponential moving average over frames. The per-frame softmax
                # jumps around on a live face; this keeps the label readable
                # without hiding genuine expression changes.
                a = float(np.clip(args.smooth, 0.0, 0.95))
                probs = raw if a == 0 else a * probs + (1.0 - a) * raw
                top = int(np.argmax(probs))
                conf = float(probs[top])
                draw_face_label(frame, last_box, C.EMOTION_NAMES[top], conf,
                                conf >= args.min_conf)

            if last_box is None:
                cv2.putText(frame, "no face detected", (10, fh - 20), FONT, 0.6,
                            BOX_COLOR_WEAK, 2, cv2.LINE_AA)

            # --- overlay ----------------------------------------------------
            draw_probability_panel(frame, probs)
            fps_window.append(time.perf_counter() - t0)
            fps = 1.0 / max(1e-6, sum(fps_window) / len(fps_window))
            cv2.putText(frame, f"{fps:5.1f} FPS", (fw - 110, 26), FONT, 0.6,
                        TEXT_COLOR, 2, cv2.LINE_AA)

            cv2.imshow("RAF-DB ResNet-18 (ONNX) — press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            frame_i += 1
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("camera released.")


if __name__ == "__main__":
    main()

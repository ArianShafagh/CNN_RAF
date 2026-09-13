"""Export the trained RAF-DB ResNet-18 to ONNX and verify it against PyTorch.

This is the final step of the pipeline. It:

  1. Loads the best checkpoint (by validation macro-F1) — or --last.
  2. Exports the model to ONNX with `torch.onnx.export`, using a *dynamic*
     batch axis so the graph accepts any batch size (fixed 3x224x224 image).
  3. Structurally validates the graph with `onnx.checker`.
  4. Numerically verifies with `onnxruntime`: PyTorch vs ORT logits must match
     (rtol=1e-3, atol=1e-5) on batch-size 1, a random batch of 8, and one real
     batch from the RAF-DB test split — the last two also prove the dynamic
     batch axis works.

Export runs on CPU regardless of the training device: the ONNX graph is more
portable that way and it sidesteps MPS-specific export quirks.

Run:
    python export.py            # uses best_model.pth
    python export.py --last     # use last_model.pth instead
    python export.py --opset 18 # override the ONNX opset (default 17)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import config as C
import dataset as D
from evaluate import load_checkpoint

import onnx
import onnxruntime as ort


# Numerical tolerance for the PyTorch-vs-ONNXRuntime comparison.
RTOL = 1e-3
ATOL = 1e-5


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_onnx(model: torch.nn.Module, onnx_path: Path, opset: int) -> None:
    """Export `model` (already on CPU, eval mode) to `onnx_path` with a dynamic
    batch axis on both the input and the output."""
    dummy = torch.randn(1, 3, C.IMAGE_SIZE, C.IMAGE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        # Use the legacy TorchScript exporter: the newer dynamo path pulls in
        # onnxscript, which isn't a project dependency.
        dynamo=False,
    )
    print(f"exported ONNX graph -> {onnx_path}  (opset {opset})")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def check_structure(onnx_path: Path) -> None:
    """Load the graph and run onnx.checker (raises on a malformed model)."""
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("onnx.checker: model is structurally valid")


def _torch_logits(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model(x).cpu().numpy()


def verify_numerically(model: torch.nn.Module, onnx_path: Path) -> bool:
    """Compare PyTorch vs onnxruntime logits across several input shapes.

    Returns True if every case is within tolerance, else False.
    """
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # Build the verification cases: (label, input tensor).
    cases: list[tuple[str, torch.Tensor]] = [
        ("batch=1 (dummy)", torch.randn(1, 3, C.IMAGE_SIZE, C.IMAGE_SIZE)),
        ("batch=8 (random, dynamic axis)",
         torch.randn(8, 3, C.IMAGE_SIZE, C.IMAGE_SIZE)),
    ]

    # One real batch from the RAF-DB test split (up to 8 images).
    real = _real_test_batch(n=8)
    if real is not None:
        cases.append(("batch=%d (real test images)" % real.shape[0], real))

    all_ok = True
    for label, x in cases:
        torch_out = _torch_logits(model, x)
        ort_out = session.run(None, {input_name: x.numpy()})[0]
        max_abs = float(np.max(np.abs(torch_out - ort_out)))
        try:
            np.testing.assert_allclose(torch_out, ort_out, rtol=RTOL, atol=ATOL)
            status = "OK"
        except AssertionError:
            status = "MISMATCH"
            all_ok = False
        print(f"  {label:<38} max|Δ|={max_abs:.3e}  [{status}]")
    return all_ok


def _real_test_batch(n: int) -> torch.Tensor | None:
    """Return a stacked tensor of up to `n` real test images, or None if the
    test set can't be loaded (verification still proceeds on synthetic inputs)."""
    try:
        *_, test_ds = D.build_datasets()
    except Exception as exc:  # pragma: no cover - dataset is optional here
        print(f"  (skipping real-image case: {exc})")
        return None
    imgs = [test_ds[i][0] for i in range(min(n, len(test_ds)))]
    return torch.stack(imgs) if imgs else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RAF-DB ResNet-18 to ONNX and verify it."
    )
    parser.add_argument("--last", action="store_true",
                        help="use last_model.pth instead of best_model.pth")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    ckpt_path = C.LAST_CKPT if args.last else C.BEST_CKPT

    # Load on CPU and export from CPU for a portable graph.
    device = torch.device("cpu")
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    print("=" * 70)
    print("ONNX EXPORT")
    print("=" * 70)
    export_onnx(model, C.ONNX_PATH, args.opset)

    print("\n" + "-" * 70)
    print("STRUCTURAL CHECK")
    print("-" * 70)
    check_structure(C.ONNX_PATH)

    print("\n" + "-" * 70)
    print("NUMERICAL VERIFICATION (PyTorch vs onnxruntime)")
    print("-" * 70)
    ok = verify_numerically(model, C.ONNX_PATH)

    print("\n" + "=" * 70)
    if ok:
        print(f"PASS — ONNX model verified and saved to {C.ONNX_PATH}")
        print("=" * 70)
    else:
        print("FAIL — ONNX outputs diverged from PyTorch beyond tolerance")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()

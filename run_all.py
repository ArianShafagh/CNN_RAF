#!/usr/bin/env python3
"""Run the whole pipeline end to end: check -> train -> evaluate -> figures -> export.

One entry point instead of running six scripts by hand in the right order with
the right flags.

    python3 run_all.py                  # full pipeline
    python3 run_all.py --skip train     # reuse the existing checkpoint
    python3 run_all.py --only figures   # just redraw the plots
    python3 run_all.py --seeds 42 43 44 # three seeds, then aggregate
    python3 run_all.py --list           # show the stages and exit

This file deliberately imports nothing outside the standard library, and runs
every stage through the project's own virtualenv interpreter if one is present.
So it works when launched with the system python, even if `venv/bin/activate`
carries stale paths from a moved project directory.

Training is long. If a checkpoint already exists the train stage is skipped with
a note; pass --force-train to retrain anyway.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "Dataset" / "DATASET"

STAGES = ["check", "train", "evaluate", "figures", "export", "aggregate"]

C_OK, C_WARN, C_ERR, C_DIM, C_HEAD, C_OFF = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)
if not sys.stdout.isatty():                 # keep logs clean when redirected
    C_OK = C_WARN = C_ERR = C_DIM = C_HEAD = C_OFF = ""


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------
def interpreter() -> str:
    """Prefer the project venv, whatever interpreter launched this script."""
    for candidate in (ROOT / "venv" / "bin" / "python",
                      ROOT / "venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


PY = interpreter()


def run(module: str, *args: str) -> None:
    """Run one pipeline module with `-m`, streaming its output.

    `-m` rather than a file path so the rafdb package imports resolve; cwd is
    the project root, which puts it on sys.path. Raises on failure.
    """
    cmd = [PY, "-m", module, *args]
    print(f"{C_DIM}$ python -m {module} {' '.join(args)}{C_OFF}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"{C_ERR}stage failed: {module} exited {result.returncode}{C_OFF}"
        )


def banner(n: int, total: int, name: str, detail: str = "") -> None:
    print(f"\n{C_HEAD}[{n}/{total}] {name}{C_OFF}"
          + (f" {C_DIM}{detail}{C_OFF}" if detail else ""))
    print(f"{C_DIM}{'-' * 66}{C_OFF}")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_check() -> None:
    """Fail early on the things that actually go wrong, with a fix for each."""
    print(f"interpreter : {PY}")
    if not (ROOT / "venv").exists():
        print(f"{C_WARN}  no venv/ found — using {sys.executable}{C_OFF}")

    missing = [m for m in ("torch", "torchvision", "sklearn", "matplotlib",
                           "onnx", "onnxruntime")
               if subprocess.run([PY, "-c", f"import {m}"],
                                 capture_output=True).returncode != 0]
    if missing:
        raise SystemExit(
            f"{C_ERR}missing packages: {', '.join(missing)}{C_OFF}\n"
            f"  fix: {PY} -m pip install -r requirements.txt"
        )
    print(f"{C_OK}  packages    : all present{C_OFF}")

    if not DATA_DIR.exists():
        raise SystemExit(
            f"{C_ERR}dataset not found at {DATA_DIR}{C_OFF}\n"
            "  expected Dataset/DATASET/train/<1..7>/ and .../test/<1..7>/\n"
            "  fix: place RAF-DB Basic there, or edit DATA_ROOT in config.py"
        )
    counts = {}
    for split in ("train", "test"):
        for cls in range(1, 8):
            d = DATA_DIR / split / str(cls)
            if not d.is_dir():
                raise SystemExit(f"{C_ERR}missing folder: {d}{C_OFF}")
            counts[split] = counts.get(split, 0) + len(list(d.glob("*.jpg")))
    print(f"{C_OK}  dataset     : {counts['train']:,} train / "
          f"{counts['test']:,} test images{C_OFF}")

    dev = subprocess.run(
        [PY, "-c", "from rafdb.core import dataset as D; print(D.select_device())"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(f"{C_OK}  device      : {dev.stdout.strip() or 'unknown'}{C_OFF}")


def stage_train(seed: int, tag: str, force: bool) -> None:
    best = CKPT_DIR / f"best_model{suffix(tag)}.pth"
    if best.exists() and not force:
        print(f"{C_WARN}  checkpoint exists: {best.name} — skipping training."
              f"{C_OFF}\n  {C_DIM}pass --force-train to retrain "
              f"(roughly 2-10 hours){C_OFF}")
        return
    args = ["--seed", str(seed)]
    if tag:
        args += ["--tag", tag]
    run("rafdb.pipeline.train", *args)


def stage_evaluate(tag: str) -> None:
    base = ["--tag", tag] if tag else []
    run("rafdb.pipeline.evaluate", *base)           # plain forward pass
    run("rafdb.pipeline.evaluate", *base, "--tta")  # flip test-time augmentation


def stage_figures(tag: str) -> None:
    run("rafdb.reporting.plot_history", *(["--tag", tag] if tag else []))


def stage_export(tag: str) -> None:
    # export reads config.BEST_CKPT, which has no tag; only export the
    # untagged run, otherwise the ONNX would not match the tag being processed.
    if tag:
        print(f"{C_DIM}  skipped for tagged run '{tag}' "
              f"(export uses the untagged checkpoint){C_OFF}")
        return
    run("rafdb.pipeline.export")


def stage_aggregate(tags: list[str]) -> None:
    if len(tags) < 2:
        print(f"{C_DIM}  only one run — nothing to aggregate. "
              f"Use --seeds 42 43 44 for mean and standard deviation.{C_OFF}")
        return
    run("rafdb.reporting.aggregate_seeds", "--glob",
        "eval_metrics_*_test_tta.json")


def suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarise(tags: list[str]) -> None:
    print(f"\n{C_HEAD}{'=' * 66}\nRESULTS\n{'=' * 66}{C_OFF}")
    any_found = False
    for tag in tags:
        path = OUT_DIR / f"eval_metrics{suffix(tag)}_test_tta.json"
        if not path.exists():
            continue
        any_found = True
        m = json.load(open(path))["overall"]
        label = tag or "seed 42"
        print(f"  {label:<12} accuracy {m['accuracy']:.4f}   "
              f"macro-F1 {m['macro_f1']:.4f}   "
              f"balanced {m['balanced_accuracy']:.4f}")
    if not any_found:
        print("  (no evaluation metrics found)")
        return
    print(f"\n{C_DIM}  Scored once on the held-out test split, with flip TTA.\n"
          f"  Checkpoint selected on validation macro-F1.{C_OFF}")

    figures = sorted(OUT_DIR.glob("fig_*.png"))
    if figures:
        print(f"\n{C_HEAD}FIGURES{C_OFF}")
        for f in figures:
            print(f"  {f.relative_to(ROOT)}")
    onnx = OUT_DIR / "raf_resnet18.onnx"
    if onnx.exists():
        mb = onnx.stat().st_size / 1e6
        print(f"\n{C_HEAD}DEPLOYABLE MODEL{C_OFF}\n"
              f"  {onnx.relative_to(ROOT)}  ({mb:.0f} MB)\n"
              f"{C_DIM}  try it live:  python3 -m rafdb.deploy.webcam_demo{C_OFF}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the whole RAF-DB pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages: " + ", ".join(STAGES),
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="one or more seeds; more than one tags each run and "
                         "aggregates at the end (default: 42)")
    ap.add_argument("--only", nargs="+", choices=STAGES, metavar="STAGE",
                    help="run only these stages")
    ap.add_argument("--skip", nargs="+", choices=STAGES, metavar="STAGE",
                    default=[], help="skip these stages")
    ap.add_argument("--force-train", action="store_true",
                    help="retrain even when a checkpoint already exists")
    ap.add_argument("--list", action="store_true",
                    help="print the stages and exit")
    args = ap.parse_args()

    if args.list:
        print("stages, in order:")
        for s in STAGES:
            print(f"  {s}")
        return

    wanted = [s for s in (args.only or STAGES) if s not in args.skip]
    multi = len(args.seeds) > 1
    tags = [f"seed{s}" for s in args.seeds] if multi else [""]

    print(f"{C_HEAD}RAF-DB pipeline{C_OFF}")
    print(f"{C_DIM}stages: {' -> '.join(wanted)}")
    print(f"seeds : {args.seeds}{C_OFF}")

    t_start = time.time()
    timings: list[tuple[str, float]] = []
    # "check" and "aggregate" run once; the rest run per seed.
    per_seed = [s for s in wanted if s not in ("check", "aggregate")]
    total = len(wanted) if not multi else (
        (1 if "check" in wanted else 0)
        + len(per_seed) * len(tags)
        + (1 if "aggregate" in wanted else 0)
    )
    step = 0

    if "check" in wanted:
        step += 1
        banner(step, total, "check")
        t0 = time.time()
        stage_check()
        timings.append(("check", time.time() - t0))

    for seed, tag in zip(args.seeds, tags):
        for name in per_seed:
            step += 1
            banner(step, total, name, f"seed {seed}" if multi else "")
            t0 = time.time()
            if name == "train":
                stage_train(seed, tag, args.force_train)
            elif name == "evaluate":
                stage_evaluate(tag)
            elif name == "figures":
                stage_figures(tag)
            elif name == "export":
                stage_export(tag)
            timings.append((f"{name}{f' ({tag})' if tag else ''}",
                            time.time() - t0))

    if "aggregate" in wanted:
        step += 1
        banner(step, total, "aggregate")
        t0 = time.time()
        stage_aggregate(tags)
        timings.append(("aggregate", time.time() - t0))

    summarise(tags)

    print(f"\n{C_HEAD}TIMING{C_OFF}")
    for name, secs in timings:
        print(f"  {name:<22} {secs / 60:6.1f} min")
    print(f"  {'total':<22} {(time.time() - t_start) / 60:6.1f} min")
    print(f"\n{C_OK}pipeline complete.{C_OFF}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(f"\n{C_WARN}interrupted.{C_OFF}")

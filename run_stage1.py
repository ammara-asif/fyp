#!/usr/bin/env python3
"""
Stage 1 concurrent runner for a single 16 GB GPU.

ReducedResNet18(nf=20) is ~1.1M params; at batch 32 on 32x32 inputs one run
uses well under 1 GB and leaves the GPU mostly idle. Running arms concurrently
is the entire speedup.

    python run_stage1.py                      # all arms, seeds 0-2
    python run_stage1.py --seeds 0 1 2 3 4    # final 5-seed comparison
    python run_stage1.py --arms nested_a nested_b flat_llrd
    python run_stage1.py --jobs 4             # override concurrency
    python run_stage1.py --dry-run

Place next to experiments/ and configs/. Resumable: a run whose summary.json
already exists is skipped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "experiments" / "run_control.py"
RESULTS = ROOT / "results"

ALL_ARMS = [
    "flat",
    "flat_llrd",        # THE matched control (3 slabs, ratios 1 / 1/4 / 1/16)
    "flat_llrd6",       # secondary control, per-block LLRD
    "flat_bigbatch",    # matched-effective-batch control
    "nested_a",
    "nested_b",
    "nested_c",
    "nested_d",
    "nested_2level",
    "nested_3_reversed",
    "nested_a_wide",
]

# The five that decide Gate 1. Run these at 5 seeds.
FINALISTS = ["flat_llrd", "flat_bigbatch", "nested_a", "nested_c", "nested_d"]

# nested_d runs a double backward; give it a lane of its own rather than
# letting several of them land at once.
HEAVY = {"nested_d"}


def gpu_plan(force_jobs: int | None) -> tuple[int, str]:
    try:
        import torch
    except ImportError:
        return (force_jobs or 1, "torch not importable")
    if not torch.cuda.is_available():
        return (force_jobs or 1, "no CUDA — CPU only, this will be very slow")

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cpus = os.cpu_count() or 4

    # Memory is not the binding constraint here; CPU and kernel-launch
    # overhead are. One job per ~2 physical cores is about right.
    jobs = max(1, min(8, cpus // 2))
    if vram < 8:
        jobs = min(jobs, 3)
    if force_jobs:
        jobs = force_jobs
    return jobs, f"{name}, {vram:.0f} GB, {cpus} vCPU"


def summary_path(arm: str, seed: int) -> Path:
    return RESULTS / arm / f"seed{seed}" / "summary.json"


def worker(q: "Queue[tuple[str,int]]", lock: threading.Lock,
           state: dict, extra: list[str], dry: bool) -> None:
    while True:
        try:
            arm, seed = q.get_nowait()
        except Exception:
            return
        try:
            cmd = [sys.executable, str(RUNNER),
                   f"arm={arm}", f"seed={seed}", *extra]
            if dry:
                print("  " + " ".join(cmd))
                continue

            logdir = RESULTS / arm / f"seed{seed}"
            logdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with (logdir / "stdout.log").open("w") as lf:
                lf.write("# " + " ".join(cmd) + "\n")
                lf.flush()
                r = subprocess.run(cmd, cwd=ROOT, stdout=lf,
                                   stderr=subprocess.STDOUT, text=True)
            mins = (time.time() - t0) / 60

            with lock:
                state["done"] += 1
                n, tot = state["done"], state["total"]
                elapsed = (time.time() - state["t0"]) / 60
                eta = (elapsed / n) * (tot - n) if n else 0
            ok = r.returncode == 0 and summary_path(arm, seed).exists()
            tag = "OK  " if ok else "FAIL"
            if not ok:
                with lock:
                    state["failed"].append(f"{arm}/seed{seed}")
            print(f"[{n}/{tot}] {tag} {arm}/seed{seed}  {mins:.0f}m  "
                  f"ETA {eta:.0f}m", flush=True)
        finally:
            q.task_done()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ALL_ARMS)
    ap.add_argument("--finalists", action="store_true",
                    help="only the five arms that decide Gate 1")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override train.epochs for every arm")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if summary.json exists")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if not RUNNER.exists():
        sys.exit(f"cannot find {RUNNER} — run this from the project root")

    if args.finalists:
        args.arms = FINALISTS

    extra = []
    if args.epochs:
        extra.append(f"train.epochs={args.epochs}")
    if args.wandb:
        extra.append("wandb.enabled=true")

    jobs, desc = gpu_plan(args.jobs)
    print(f"GPU: {desc}")
    print(f"Concurrency: {jobs} jobs")

    # Heavy arms first so they are not the long tail at the end.
    pairs = [(a, s) for a in args.arms for s in args.seeds]
    pairs.sort(key=lambda p: (p[0] not in HEAVY, p[0], p[1]))

    q: "Queue[tuple[str,int]]" = Queue()
    skipped = 0
    for arm, seed in pairs:
        if not args.force and summary_path(arm, seed).exists():
            skipped += 1
            continue
        q.put((arm, seed))

    total = q.qsize()
    print(f"{skipped} already complete, {total} to run\n")
    if total == 0:
        return

    state = {"done": 0, "total": total, "failed": [], "t0": time.time()}
    lock = threading.Lock()
    threads = [threading.Thread(target=worker,
                                args=(q, lock, state, extra, args.dry_run),
                                daemon=True)
               for _ in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nFinished in {(time.time()-state['t0'])/60:.0f}m")
    if state["failed"]:
        print(f"FAILED ({len(state['failed'])}): {', '.join(state['failed'])}")
        print("Check results/<arm>/seed<n>/stdout.log")


if __name__ == "__main__":
    main()

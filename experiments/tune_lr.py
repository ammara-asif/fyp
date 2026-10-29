#!/usr/bin/env python3
"""
Per-arm learning-rate selection on a validation slice carved from train.

Every arm gets the same grid and the same budget. Selection rule, fixed in
advance: highest average incremental accuracy on the validation stream, seed 0.
The test set is never touched here.

This exists because periodic stepping and per-level LR ratios both change an
arm's effective update rate. Comparing arms at one shared learning rate
compares tuning, not architecture — the Stage 1 note already showed Flat peaks
near 0.1 while LLRD peaks near 0.5, and that they tie once each is tuned.

    python experiments/tune_lr.py                      # all arms, default grid
    python experiments/tune_lr.py --arms nested_a nested_d
    python experiments/tune_lr.py --epochs 5 --n-exp 10   # cheaper pass
    python experiments/tune_lr.py --apply               # write results into configs/arm/*.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "experiments" / "run_control.py"
ARMS_DIR = ROOT / "configs" / "arm"
OUT = ROOT / "results" / "tuning"

DEFAULT_GRID = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

ALL_ARMS = ["flat", "flat_llrd", "flat_llrd6", "flat_bigbatch",
            "nested_a", "nested_b", "nested_c", "nested_d",
            "nested_2level", "nested_3_reversed", "nested_a_wide"]


def run_dir(arm: str, lr: float) -> Path:
    return OUT / arm / f"lr{lr}"


def worker(q, lock, state, args):
    while True:
        try:
            arm, lr = q.get_nowait()
        except Exception:
            return
        try:
            d = run_dir(arm, lr)
            summ = d / "summary.json"
            if summ.exists() and not args.force:
                with lock:
                    state["done"] += 1
                continue
            d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(RUNNER),
                   f"arm={arm}", f"arm.lr={lr}", "seed=0",
                   "tune.enabled=true",
                   f"train.epochs={args.epochs}",
                   f"benchmark.n_experiences={args.n_exp}",
                   "probe.enabled=false",          # tuning needs accuracy only
                   f"hydra.run.dir={d.as_posix()}"]
            if args.dry_run:
                print("  " + " ".join(cmd))
                continue
            t0 = time.time()
            with (d / "stdout.log").open("w") as lf:
                lf.write("# " + " ".join(cmd) + "\n")
                lf.flush()
                subprocess.run(cmd, cwd=ROOT, stdout=lf,
                               stderr=subprocess.STDOUT, text=True)
            with lock:
                state["done"] += 1
                n, tot = state["done"], state["total"]
            print(f"[{n}/{tot}] {arm} lr={lr}  {(time.time()-t0)/60:.0f}m", flush=True)
        finally:
            q.task_done()


def collect(arms, grid):
    best = {}
    table = {}
    for arm in arms:
        row = {}
        for lr in grid:
            s = run_dir(arm, lr) / "summary.json"
            if not s.exists():
                continue
            try:
                d = json.loads(s.read_text())
            except json.JSONDecodeError:
                continue
            v = d.get("avg_incremental_acc")
            if v is not None:
                row[lr] = v
        if row:
            table[arm] = row
            best[arm] = max(row, key=row.get)
    return table, best


def apply_to_configs(best):
    for arm, lr in best.items():
        p = ARMS_DIR / f"{arm}.yaml"
        if not p.exists():
            print(f"  skip {arm}: no config file")
            continue
        txt = p.read_text()
        new, n = re.subn(r"^lr:\s*[\d.eE+-]+.*$", f"lr: {lr}" +
                         "                      # selected by tune_lr.py",
                         txt, count=1, flags=re.MULTILINE)
        if n == 0:
            new = txt.rstrip() + f"\nlr: {lr}  # selected by tune_lr.py\n"
        p.write_text(new)
        print(f"  {arm}: lr -> {lr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ALL_ARMS)
    ap.add_argument("--grid", nargs="+", type=float, default=DEFAULT_GRID)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-exp", type=int, default=10,
                    help="experiences during tuning; fewer is cheaper and ranks the same")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--apply", action="store_true",
                    help="write the selected lr into configs/arm/*.yaml")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if not args.collect_only:
        jobs = args.jobs or max(1, min(8, (os.cpu_count() or 4) // 2))
        pairs = [(a, lr) for a in args.arms for lr in args.grid]
        q = Queue()
        for p in pairs:
            q.put(p)
        print(f"{len(pairs)} tuning runs, {jobs} concurrent, "
              f"{args.epochs} epochs x {args.n_exp} experiences\n")
        state, lock = {"done": 0, "total": len(pairs)}, threading.Lock()
        ts = [threading.Thread(target=worker, args=(q, lock, state, args), daemon=True)
              for _ in range(jobs)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    table, best = collect(args.arms, args.grid)

    print("\nAverage incremental accuracy on validation (%)\n")
    hdr = "arm".ljust(20) + "".join(f"{lr:>8}" for lr in args.grid) + "   best"
    print(hdr)
    print("-" * len(hdr))
    for arm, row in table.items():
        cells = "".join(f"{row[lr]:>8.2f}" if lr in row else f"{'-':>8}"
                        for lr in args.grid)
        print(arm.ljust(20) + cells + f"   {best[arm]}")

    (OUT / "tuned_lr.json").write_text(json.dumps(best, indent=2))
    print(f"\nWrote {OUT / 'tuned_lr.json'}")

    if args.apply:
        print("\nApplying to configs/arm/:")
        apply_to_configs(best)
    else:
        print("\nRe-run with --apply to write these into configs/arm/*.yaml")


if __name__ == "__main__":
    main()

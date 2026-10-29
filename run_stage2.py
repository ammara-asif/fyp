#!/usr/bin/env python3
"""
Launch the Stage 2 runs concurrently on one GPU.

    python run_stage2.py --smoke          # 1-2 min on fake data, run this FIRST
    python run_stage2.py                  # the 7 defense runs, seed 0
    python run_stage2.py --seeds 0 1 2    # more seeds if time allows
    python run_stage2.py --jobs 4

Resumable: runs with an existing summary.json are skipped.
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
SCRIPT = ROOT / "experiments" / "stage2.py"
OUT = ROOT / "results" / "stage2"

# (mode, arm, stream) — ordered by importance for the defense
RUNS = [
    ("plastic", "nested_a", "disjoint"),    # the proposal
    ("frozen",  "nested_a", "disjoint"),    # isolation baseline: plastic vs frozen
    ("single",  "nested_a", "disjoint"),    # memory-matched replay, one network
    ("plastic", "flat",     "disjoint"),    # does nesting matter inside the system?
    ("plastic", "nested_a", "recurring"),   # recurring domains: absorbed, not re-spawned
    ("frozen",  "nested_a", "recurring"),
    ("single",  "nested_a", "recurring"),
]


def run_dir(mode, arm, stream, seed):
    return OUT / f"{mode}_{arm}_{stream}" / f"seed{seed}"


def worker(q, lock, st, extra):
    while True:
        try:
            mode, arm, stream, seed = q.get_nowait()
        except Exception:
            return
        try:
            d = run_dir(mode, arm, stream, seed)
            d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(SCRIPT), "--mode", mode, "--arm", arm,
                   "--stream", stream, "--seed", str(seed), "--out", str(d), *extra]
            t0 = time.time()
            with (d / "stdout.log").open("w") as lf:
                lf.write("# " + " ".join(cmd) + "\n")
                lf.flush()
                r = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
            ok = r.returncode == 0 and (d / "summary.json").exists()
            with lock:
                st["done"] += 1
                if not ok:
                    st["failed"].append(str(d))
                print(f"[{st['done']}/{st['total']}] {'OK  ' if ok else 'FAIL'} "
                      f"{mode}_{arm}_{stream} seed{seed}  {(time.time()-t0)/60:.1f}m",
                      flush=True)
        finally:
            q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="fake data, tiny, all modes")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    extra = []
    if args.epochs:
        extra += ["--epochs", str(args.epochs)]
    if args.smoke:
        global OUT
        OUT = ROOT / "results" / "stage2_smoke"
        extra += ["--fake-data", "--epochs", "2", "--n-tasks", "6"]
        args.force = True

    jobs = args.jobs or max(1, min(7, (os.cpu_count() or 4) // 2))
    q = Queue()
    for seed in args.seeds:
        for mode, arm, stream in RUNS:
            if not args.force and (run_dir(mode, arm, stream, seed) / "summary.json").exists():
                continue
            q.put((mode, arm, stream, seed))
    total = q.qsize()
    print(f"{total} runs, {jobs} concurrent -> {OUT}")
    if not total:
        return
    st, lock = {"done": 0, "total": total, "failed": []}, threading.Lock()
    ts = [threading.Thread(target=worker, args=(q, lock, st, extra), daemon=True)
          for _ in range(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if st["failed"]:
        print("\nFAILED — read the stdout.log in each:")
        for f in st["failed"]:
            print("  ", f)
    else:
        print("\nAll runs succeeded. Next: python experiments/stage2_report.py"
              + (" --smoke" if args.smoke else ""))


if __name__ == "__main__":
    main()

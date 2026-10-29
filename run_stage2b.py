#!/usr/bin/env python3
"""
Launch the Stage 2b runs (interference-triggered splitting).

    python run_stage2b.py --smoke           # ~3 min on fake data, run FIRST
    python run_stage2b.py                   # exploratory runs, a-priori thresholds
    python run_stage2b.py --tuned           # FINAL runs, thresholds from tune_theta.py
    python run_stage2b.py --tuned --seeds 0 1 2

Exploratory results -> results/stage2b
Tuned (final) results -> results/stage2b_tuned
Resumable: runs with an existing summary.json are skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parent
SYSTEM = "c"                       # set from --system in main()


def script():
    return ROOT / "experiments" / f"stage2{SYSTEM}.py"


def tuned_json():
    return ROOT / "results" / f"stage2{SYSTEM}_tuning" / "tuned.json"

# Exploratory: (trigger, arm, stream, threshold or None), in priority order
RUNS = [
    ("predictive", "nested_a", "disjoint",  0.1),
    ("reactive",   "nested_a", "disjoint",  None),
    ("predictive", "flat",     "disjoint",  0.1),
    ("never",      "nested_a", "disjoint",  None),
    ("predictive", "nested_a", "recurring", 0.1),
    ("reactive",   "nested_a", "recurring", None),
    ("predictive", "nested_a", "disjoint",  0.0),
    ("predictive", "nested_a", "disjoint",  0.3),
    ("predictive", "nested_2level", "disjoint", 0.1),
]

# Final: thresholds filled in from tuned.json.
# (trigger, arm, stream[, extra cli]).  nf=90 gives ONE learner with ~22M
# parameters, the budget of 20 learners at nf=20: the parameter-matched baseline.
RUNS_TUNED = [
    ("predictive", "nested_a",      "disjoint"),
    ("reactive",   "nested_a",      "disjoint"),
    ("predictive", "flat",          "disjoint"),
    ("never",      "nested_a",      "disjoint"),
    ("never",      "nested_a",      "disjoint",  ["--nf", "90"]),
    ("predictive", "nested_a",      "recurring"),
    ("reactive",   "nested_a",      "recurring"),
    ("never",      "nested_a",      "recurring"),
    ("predictive", "nested_2level", "disjoint"),
    ("predictive", "avg_control",   "disjoint"),   # Stage 1's averaging control
    ("predictive", "nested_heads",  "disjoint"),   # big slow level: stem..layer3 | layer4 | heads
    # guilty until proven innocent: no foreign negatives, no referral loss, no refresh,
    # routing by the learner's own evidence only. No learner ever sees another's data.
    ("predictive", "nested_a",      "disjoint",  ["--neg", "0", "--lam-ref", "0",
                                                  "--refresh-scope", "none",
                                                  "--signals", "own,shared,lr", "--evidence", "own"]),
]


def name_of(trigger, arm, stream, flag, value, extra=()):
    tag = "" if value is None else f"_{'th' if flag == '--theta' else 'd'}{value}"
    extra = list(extra)
    if "--nf" in extra:
        tag += f"_nf{extra[extra.index('--nf') + 1]}"
    if "--refresh-scope" in extra and extra[extra.index("--refresh-scope") + 1] == "none":
        tag += "_isolated"
    return f"{trigger}_{arm}_{stream}{tag}"


def build_jobs(args):
    """Return (out_root, list of (name, cli_args))."""
    jobs = []
    if args.tuned:
        if not tuned_json().exists():
            raise SystemExit(f"{tuned_json()} not found. Run experiments/tune_theta.py "
                             f"--system {SYSTEM} first.")
        tuned = json.loads(tuned_json().read_text())
        print(f"Using tuned thresholds: {tuned}")
        out_root = ROOT / "results" / f"stage2{SYSTEM}_tuned"
        for spec in RUNS_TUNED:
            trigger, arm, stream = spec[:3]
            extra_cli = list(spec[3]) if len(spec) > 3 else []
            flag, value = None, None
            if trigger == "predictive":
                flag = "--theta"
                value = tuned.get(f"predictive_{arm}", tuned.get("predictive_nested_a"))
            elif trigger == "reactive":
                flag = "--delta"
                value = tuned.get("reactive_nested_a")
            cli = ["--trigger", trigger, "--arm", arm, "--stream", stream]
            if flag and value is not None:
                cli += [flag, str(value)]
            cli += extra_cli
            jobs.append((name_of(trigger, arm, stream, flag, value, extra_cli), cli))
    else:
        out_root = ROOT / "results" / (f"stage2{SYSTEM}_smoke" if args.smoke else f"stage2{SYSTEM}")
        for trigger, arm, stream, theta in RUNS:
            cli = ["--trigger", trigger, "--arm", arm, "--stream", stream]
            flag = "--theta" if theta is not None else None
            if flag:
                cli += [flag, str(theta)]
            jobs.append((name_of(trigger, arm, stream, flag, theta), cli))
    return out_root, jobs


def worker(q, lock, st, extra):
    while True:
        try:
            name, cli, seed, d = q.get_nowait()
        except Exception:
            return
        try:
            d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script()), *cli, "--seed", str(seed),
                   "--out", str(d), *extra]
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
                      f"{name} seed{seed}  {(time.time()-t0)/60:.1f}m", flush=True)
        finally:
            q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-modules", type=int, default=None,
                    help="learner cap for every run; 0 = no cap")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tuned", action="store_true",
                    help="final runs with thresholds selected on validation")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--system", choices=["b", "c"], default="c",
                    help="b = Stage 2b; c = Stage 2c (ownership + referral routing, per-class splits)")
    args = ap.parse_args()
    global SYSTEM
    SYSTEM = args.system

    extra = []
    if args.epochs:
        extra += ["--epochs", str(args.epochs)]
    if args.max_modules is not None:
        extra += ["--max-modules", str(args.max_modules)]
    if args.smoke:
        extra += ["--fake-data", "--epochs", "2", "--n-tasks", "6"]
        args.force = True

    out_root, jobs = build_jobs(args)
    q = Queue()
    for seed in args.seeds:
        for name, cli in jobs:
            d = out_root / name / f"seed{seed}"
            if not args.force and (d / "summary.json").exists():
                continue
            q.put((name, cli, seed, d))
    total = q.qsize()
    print(f"{total} runs, {args.jobs} concurrent -> {out_root}")
    if not total:
        return
    st, lock = {"done": 0, "total": total, "failed": []}, threading.Lock()
    ts = [threading.Thread(target=worker, args=(q, lock, st, extra), daemon=True)
          for _ in range(args.jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if st["failed"]:
        print("\nFAILED — read stdout.log in each:")
        for f in st["failed"]:
            print("  ", f)
    else:
        nxt = f"python experiments/stage2{SYSTEM}_report.py"
        if args.smoke:
            nxt += " --smoke"
        elif args.tuned:
            nxt += " --tuned"
        print(f"\nAll runs succeeded. Next: {nxt}")


if __name__ == "__main__":
    main()

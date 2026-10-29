#!/usr/bin/env python3
"""
Select the split thresholds on a VALIDATION split carved from the training data.
The test set is never used here.

Every trigger gets the same budget, so the comparison stays fair:
    predictive, nested_a   theta in THETAS
    predictive, flat       theta in THETAS
    reactive,   nested_a   delta in DELTAS

Selection rule, fixed in advance:
    highest final validation accuracy (conf router);
    if several are within --tie points of the best, take the one with fewer learners.

    python experiments/tune_theta.py                 # ~30-40 min at 3 jobs
    python experiments/tune_theta.py --collect-only  # re-read results, re-select
    python experiments/tune_theta.py --smoke         # fake data, ~3 min

Writes results/stage2b_tuning/tuned.json, which `run_stage2b.py --tuned` reads.
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

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = "c"
METRIC = {"b": ("final_avg_acc_conf", "forgetting_conf"),
          "c": ("final_avg_acc", "forgetting")}


def script():
    return ROOT / "experiments" / f"stage2{SYSTEM}.py"

THETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DELTAS = [0.05, 0.1, 0.2, 0.3]

FAMILIES = {
    # key in tuned.json : (trigger, arm, flag, grid)
    "predictive_nested_a": ("predictive", "nested_a", "--theta", THETAS),
    "predictive_flat":     ("predictive", "flat",     "--theta", THETAS),
    "predictive_nested_heads": ("predictive", "nested_heads", "--theta", THETAS),
    "reactive_nested_a":   ("reactive",   "nested_a", "--delta", DELTAS),
}


def run_dir(base, fam, value):
    return base / fam / f"v{value}"


def worker(q, lock, st, extra):
    while True:
        try:
            fam, value, d = q.get_nowait()
        except Exception:
            return
        try:
            trigger, arm, flag, _ = FAMILIES[fam]
            d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script()), "--trigger", trigger, "--arm", arm,
                   "--stream", "disjoint", flag, str(value), "--val",
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
                      f"{fam} {flag[2:]}={value}  {(time.time()-t0)/60:.1f}m", flush=True)
        finally:
            q.task_done()


def select(base, tie):
    tuned, report = {}, []
    for fam, (_, _, flag, grid) in FAMILIES.items():
        rows = []
        for v in grid:
            s = run_dir(base, fam, v) / "summary.json"
            if s.exists():
                d = json.loads(s.read_text())
                acc_key, fg_key = METRIC[SYSTEM]
                rows.append((v, d[acc_key], d["n_modules"], d[fg_key],
                             d.get("routing_soft"), d.get("final_acc_oracle")))
        if not rows:
            continue
        best_acc = max(r[1] for r in rows)
        near = [r for r in rows if r[1] >= best_acc - tie]
        pick = min(near, key=lambda r: (r[2], -r[1]))
        tuned[fam] = pick[0]
        report.append((fam, flag[2:], rows, pick[0]))
    return tuned, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--n-tasks", type=int, default=10,
                    help="tasks used for tuning; fewer is cheaper and ranks similarly")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--tie", type=float, default=0.5,
                    help="points within the best treated as a tie -> prefer fewer learners")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--system", choices=["b", "c"], default="c")
    args = ap.parse_args()
    global SYSTEM
    SYSTEM = args.system

    base = ROOT / "results" / (f"stage2{SYSTEM}_tuning_smoke" if args.smoke
                               else f"stage2{SYSTEM}_tuning")
    extra = ["--n-tasks", str(args.n_tasks), "--epochs", str(args.epochs)]
    if args.smoke:
        extra = ["--fake-data", "--n-tasks", "6", "--epochs", "2"]
        args.force = True

    if not args.collect_only:
        q = Queue()
        for fam, (_, _, _, grid) in FAMILIES.items():
            for v in grid:
                d = run_dir(base, fam, v)
                if args.force or not (d / "summary.json").exists():
                    q.put((fam, v, d))
        total = q.qsize()
        print(f"{total} tuning runs on VALIDATION data, {args.jobs} concurrent -> {base}")
        st, lock = {"done": 0, "total": total, "failed": []}, threading.Lock()
        ts = [threading.Thread(target=worker, args=(q, lock, st, extra), daemon=True)
              for _ in range(args.jobs)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if st["failed"]:
            print("\nFAILED:")
            for f in st["failed"]:
                print("  ", f)

    tuned, report = select(base, args.tie)
    print(f"\nValidation results (Stage 2{SYSTEM}, final accuracy)\n")
    for fam, name, rows, pick in report:
        print(f"{fam}")
        print(f"  {name:>6} {'val acc':>9} {'learners':>9} {'forgetting':>11} "
              f"{'routing':>8} {'oracle':>8}")
        for v, acc, nm, fg, rt, orc in rows:
            mark = "  <- selected" if v == pick else ""
            rts = f"{rt:8.2f}" if rt is not None else f"{'-':>8}"
            ors = f"{orc:8.2f}" if orc is not None else f"{'-':>8}"
            print(f"  {v:>6} {acc:>9.2f} {nm:>9} {fg:>11.2f} {rts} {ors}{mark}")
        print()

    if not tuned:
        raise SystemExit("nothing to select from")
    out = base / "tuned.json"
    out.write_text(json.dumps(tuned, indent=2))
    print(f"Selected: {tuned}")
    print(f"Wrote {out}")
    if not args.smoke:
        print("\nNext, the one test-set run with these values fixed:")
        print(f"  python run_stage2b.py --system {SYSTEM} --tuned --jobs 3")


if __name__ == "__main__":
    main()

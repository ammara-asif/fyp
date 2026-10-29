#!/usr/bin/env python3
"""
Aggregate Stage 1 results and test the Gate 1 comparison for significance.

Gate 1 is decided on per-level PROBE FORGETTING at matched new-task accuracy,
not on class-IL forgetting. Under plain sequential training every old task ends
at exactly 0% in every arm, so class-IL forgetting collapses to "how well did
this arm learn in the first place" and cannot separate methods.

Pairing: with `benchmark.fixed_order: true`, a given seed fixes the
initialisation and every arm sees an identical class order and data order. Runs
with the same seed are therefore matched, and a PAIRED test is both valid and
far more powerful than an unpaired one at these sample sizes. Both are
reported; the paired result is the one to quote.

    python experiments/collect.py
    python experiments/collect.py --baseline flat_llrd --metric probe_forgetting/L0
    python experiments/collect.py --power 1.0     # seeds needed to detect a 1-point effect
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

PRIMARY = "probe_forgetting/L0"      # slowest level: where nesting should pay off
GUARD = "new_task_acc"               # arms must learn new tasks equally well


def load() -> pd.DataFrame:
    rows = []
    for s in RESULTS.rglob("summary.json"):
        if "tuning" in s.parts:
            continue
        try:
            d = json.loads(s.read_text())
        except json.JSONDecodeError:
            continue
        d["path"] = str(s.parent)
        rows.append(d)
    if not rows:
        raise SystemExit(f"no summary.json found under {RESULTS}")
    df = pd.DataFrame(rows)
    df = df[~df.get("tune", pd.Series(False, index=df.index)).fillna(False).astype(bool)]
    return df


def cohens_d_paired(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("inf") if diff.mean() else 0.0


def seeds_needed(d: float, alpha=0.05, power=0.80) -> float:
    """Approximate n for a paired two-sided t-test at the given effect size."""
    if not HAVE_SCIPY or d == 0 or not math.isfinite(d):
        return float("inf")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return math.ceil(((z_a + z_b) / abs(d)) ** 2) + 1


def compare(df: pd.DataFrame, baseline: str, metric: str, alpha: float):
    if metric not in df.columns:
        print(f"  metric '{metric}' not present in results")
        return
    base = df[df.arm == baseline].set_index("seed")[metric].dropna()
    if base.empty:
        print(f"  baseline '{baseline}' has no values for {metric}")
        return

    print(f"\n### {metric}   (baseline: {baseline}, lower is better)\n")
    hdr = (f"{'arm':<20}{'n':>3}{'mean':>9}{'sd':>8}{'Δ vs base':>11}"
           f"{'d':>7}{'p (paired)':>12}{'p (Welch)':>11}  verdict")
    print(hdr)
    print("-" * len(hdr))

    b_mean, b_sd = base.mean(), base.std(ddof=1)
    print(f"{baseline:<20}{len(base):>3}{b_mean:>9.2f}{b_sd:>8.2f}"
          f"{'—':>11}{'—':>7}{'—':>12}{'—':>11}")

    for arm in sorted(a for a in df.arm.unique() if a != baseline):
        v = df[df.arm == arm].set_index("seed")[metric].dropna()
        if v.empty:
            continue
        shared = sorted(set(base.index) & set(v.index))
        delta = v.mean() - b_mean

        p_pair, d_pair = float("nan"), float("nan")
        if len(shared) >= 2:
            diff = (v.loc[shared] - base.loc[shared]).to_numpy()
            d_pair = cohens_d_paired(diff)
            if HAVE_SCIPY:
                p_pair = stats.ttest_rel(v.loc[shared], base.loc[shared]).pvalue

        p_welch = float("nan")
        if HAVE_SCIPY and len(v) >= 2 and len(base) >= 2:
            p_welch = stats.ttest_ind(v, base, equal_var=False).pvalue

        if not math.isnan(p_pair):
            verdict = ("BETTER" if (p_pair < alpha and delta < 0)
                       else "worse" if (p_pair < alpha and delta > 0)
                       else "ns")
        else:
            verdict = "?"

        print(f"{arm:<20}{len(v):>3}{v.mean():>9.2f}{v.std(ddof=1):>8.2f}"
              f"{delta:>+11.2f}{d_pair:>7.2f}{p_pair:>12.4f}{p_welch:>11.4f}  {verdict}")


def guard_check(df: pd.DataFrame, baseline: str, tol: float):
    """An arm that forgets less because it learned less is not a win."""
    if GUARD not in df.columns:
        return
    base = df[df.arm == baseline][GUARD].mean()
    print(f"\n### Plasticity guard: {GUARD} must stay within {tol} points of "
          f"{baseline} ({base:.2f}%)\n")
    for arm in sorted(df.arm.unique()):
        v = df[df.arm == arm][GUARD].dropna()
        if v.empty:
            continue
        gap = v.mean() - base
        flag = "ok" if abs(gap) <= tol else "FAILS GUARD — learned less, so lower forgetting is not a win"
        print(f"  {arm:<20}{v.mean():>8.2f}%{gap:>+9.2f}   {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="flat_llrd")
    ap.add_argument("--metric", default=None,
                    help="default: every probe_forgetting level, then accuracy metrics")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--guard-tol", type=float, default=2.0)
    ap.add_argument("--power", type=float, default=None,
                    help="report seeds needed to detect an effect of this many points")
    args = ap.parse_args()

    df = load()
    if not HAVE_SCIPY:
        print("scipy not installed — p-values unavailable. pip install scipy\n")

    print(f"{len(df)} runs, {df.arm.nunique()} arms, "
          f"seeds {sorted(df.seed.unique())}")
    counts = df.groupby("arm").seed.nunique().sort_index()
    thin = counts[counts < 5]
    if len(thin):
        print(f"\nArms with fewer than 5 seeds (underpowered for ~1-point effects): "
              f"{', '.join(thin.index)}")

    metrics = ([args.metric] if args.metric else
               sorted(c for c in df.columns if c.startswith("probe_forgetting/")) +
               [m for m in ["avg_incremental_acc", "new_task_acc"] if m in df.columns])

    for m in metrics:
        compare(df, args.baseline, m, args.alpha)

    guard_check(df, args.baseline, args.guard_tol)

    if args.power is not None:
        pm = metrics[0]
        sd = df[df.arm == args.baseline][pm].std(ddof=1)
        if sd and not math.isnan(sd):
            d = args.power / sd
            print(f"\n### Power\n")
            print(f"  metric            {pm}")
            print(f"  baseline sd       {sd:.2f} points")
            print(f"  target effect     {args.power:.2f} points  (d = {d:.2f})")
            print(f"  seeds for 80%     {seeds_needed(d)}")
            print(f"  seeds for 90%     {seeds_needed(d, power=0.90)}")
            print("\n  Paired testing is assumed. Without pairing, roughly double these.")

    out = RESULTS / "stage1_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stage 2b table and figures.

    python experiments/stage2b_report.py
    python experiments/stage2b_report.py --smoke

Writes to results/stage2b/figures/:
    table.md, spawn_curve.png, accuracy_curve.png, routing_curve.png, conflict_hist.png

If the Stage 2 single-network replay runs exist, they are added as a reference row.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

COLORS = {
    "predictive": "#2563eb", "reactive": "#dc2626", "never": "#94a3b8",
    "single": "#f59e0b",
}


def label(d):
    if d.get("stage") != "2b":
        return "Single net + replay (Stage 2)"
    t = d["trigger"]
    name = {"predictive": "Predictive split (ours)", "reactive": "Reactive split (DEN-style)",
            "never": "No splitting"}[t]
    extra = []
    if d["arm"] != "nested_a":
        extra.append(d["arm"])
    if t == "predictive":
        extra.append(f"θ={d['theta']}")
    elif t == "reactive":
        extra.append(f"δ={d['delta']}")
    return name + (f" [{', '.join(extra)}]" if extra else "")


def style(d):
    if d.get("stage") != "2b":
        return dict(color=COLORS["single"], ls=":", lw=2)
    s = dict(color=COLORS[d["trigger"]], lw=2.2)
    if d["arm"] == "flat":
        s.update(color="#10b981", ls="-.")
    elif d["arm"] != "nested_a":
        s.update(color="#7c3aed", ls="--")
    if d.get("_sweep"):
        s.update(alpha=0.55, lw=1.4, ls=(0, (2, 2)))
    return s


def load(base, with_reference=True):
    runs = {}
    for s in sorted(base.glob("*/seed*/summary.json")):
        d = json.loads(s.read_text())
        runs.setdefault(d["run"], []).append(d)
    # In an exploratory threshold sweep, grey out every theta except the a-priori 0.1.
    for stream in ("disjoint", "recurring"):
        pred = [ds for ds in runs.values() if ds[0].get("trigger") == "predictive"
                and ds[0]["arm"] == "nested_a" and ds[0]["stream"] == stream]
        if len({ds[0]["theta"] for ds in pred}) > 1:
            for ds in pred:
                if abs(ds[0]["theta"] - 0.1) > 1e-9:
                    for d in ds:
                        d["_sweep"] = True
    if with_reference:
        for s in sorted((ROOT / "results" / "stage2").glob("single_*/seed*/summary.json")):
            d = json.loads(s.read_text())
            d["final_avg_acc_conf"] = d["final_avg_acc"]
            d["avg_incremental_acc_conf"] = d["avg_incremental_acc"]
            d["forgetting_conf"] = d["forgetting"]
            d["n_splits"] = 0
            for r in d["per_task"]:
                r.setdefault("avg_acc_conf", r["avg_acc"])
            runs.setdefault("ref_" + d["run"], []).append(d)
    return runs


def fmt(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "—"
    m = float(np.mean(vals))
    return f"{m:.2f}" if len(vals) == 1 else f"{m:.2f} ± {np.std(vals, ddof=1):.2f}"


def table(runs):
    cols = [("final_avg_acc_conf", "Final acc, conf router (%)"),
            ("final_avg_acc", "Final acc, proto router (%)"),
            ("avg_incremental_acc_conf", "Avg incr. acc (%)"),
            ("forgetting_conf", "Forgetting (pts)"),
            ("n_modules", "Learners"), ("n_splits", "Splits"),
            ("routing_acc_conf", "Routing, conf (%)"),
            ("routing_acc_test", "Routing, proto (%)"),
            ("params", "Params (M)"), ("anchor_kb", "Raw memory (KB)")]
    out = []
    for stream in ("disjoint", "recurring"):
        sel = [(k, v) for k, v in runs.items() if v[0]["stream"] == stream]
        if not sel:
            continue
        out.append(f"\n### {stream.capitalize()} stream ({sel[0][1][0]['n_tasks']} tasks, class-IL)\n")
        out.append("| Method | n | " + " | ".join(c[1] for c in cols) + " |")
        out.append("|---|---|" + "---|" * len(cols))
        order = {"predictive": 0, "reactive": 1, "never": 2}
        sel.sort(key=lambda kv: (kv[1][0].get("stage") != "2b",
                                 order.get(kv[1][0].get("trigger"), 9), kv[0]))
        for _, ds in sel:
            cells = []
            for c, _ in cols:
                v = [d.get(c) for d in ds]
                if c == "params":
                    v = [x / 1e6 if x is not None else None for x in v]
                if c.startswith("routing") and ds[0].get("stage") != "2b":
                    cells.append("n/a")
                else:
                    cells.append(fmt(v))
            out.append(f"| {label(ds[0])} | {len(ds)} | " + " | ".join(cells) + " |")
    out.append("\nPreliminary, one seed per row unless n > 1. Training data is routed by "
               "label; both test-time routers are label-free. 'conf' picks the learner most "
               "confident over its own classes; 'proto' picks the nearest class centroid.")
    return "\n".join(out)


def curves(runs, field, ylabel, fname, out, diag=False, skip_ref=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    streams = [s for s in ("disjoint", "recurring")
               if any(v[0]["stream"] == s for v in runs.values())]
    fig, axes = plt.subplots(1, len(streams), figsize=(6.4 * len(streams), 4.4), squeeze=False)
    for ax, stream in zip(axes[0], streams):
        n = 0
        for _, ds in runs.items():
            d0 = ds[0]
            if d0["stream"] != stream or (skip_ref and d0.get("stage") != "2b"):
                continue
            ys = np.array([[r.get(field, np.nan) if r.get(field) is not None else np.nan
                            for r in d["per_task"]] for d in ds], dtype=float)
            x = np.arange(1, ys.shape[1] + 1)
            n = max(n, len(x))
            ax.plot(x, np.nanmean(ys, 0), label=label(d0), **style(d0))
        if diag and n:
            ax.plot(np.arange(1, n + 1), np.arange(1, n + 1), color="#ef4444", lw=1,
                    ls=(0, (4, 3)), label="One learner per task (PNN-style)")
        if stream == "recurring" and n:
            ax.axvline(n / 2 + 0.5, color="#64748b", lw=1, ls=":")
        ax.set_title(f"{stream.capitalize()} stream")
        ax.set_xlabel("Tasks seen")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / fname, dpi=180)
    plt.close(fig)


def conflict_hist(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    any_ = False
    for _, ds in runs.items():
        d = ds[0]
        if d.get("trigger") != "predictive":
            continue
        vals = [r["mean_conflict"] for r in d["per_task"] if r.get("mean_conflict") is not None]
        if vals:
            any_ = True
            ax.plot(np.arange(1, len(d["per_task"]) + 1),
                    [r["mean_conflict"] if r.get("mean_conflict") is not None else np.nan
                     for r in d["per_task"]], label=label(d), **style(d))
    if not any_:
        plt.close(fig)
        return
    ax.axhline(0.1, color="#64748b", lw=1, ls=":", label="θ = 0.1")
    ax.set_xlabel("Task")
    ax.set_ylabel("Mean timescale-weighted conflict")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "conflict_curve.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tuned", action="store_true", help="report the final tuned runs")
    args = ap.parse_args()
    base = ROOT / "results" / ("stage2b_smoke" if args.smoke else
                               "stage2b_tuned" if args.tuned else "stage2b")
    runs = load(base, with_reference=not args.smoke)
    if not any(v[0].get("stage") == "2b" for v in runs.values()):
        raise SystemExit(f"no Stage 2b summary.json under {base}")
    out = base / "figures"
    out.mkdir(parents=True, exist_ok=True)

    md = table(runs)
    (out / "table.md").write_text(md)
    print(md)

    curves(runs, "n_modules", "Learners", "spawn_curve.png", out, diag=True, skip_ref=True)
    curves(runs, "avg_acc_conf", "Average accuracy on seen tasks (%)", "accuracy_curve.png", out)
    curves(runs, "routing_acc_conf", "Routing accuracy, conf router (%)", "routing_curve.png",
           out, skip_ref=True)
    conflict_hist(runs, out)

    print("\nSplits and conflict levels:")
    for _, ds in sorted(runs.items()):
        d = ds[0]
        if d.get("stage") != "2b":
            continue
        p = d.get("conflict_pct", {})
        cp = (f"conflict p10/p50/p90 = {p.get('10', float('nan')):.3f}/"
              f"{p.get('50', float('nan')):.3f}/{p.get('90', float('nan')):.3f}"
              if d["trigger"] == "predictive" else "")
        print(f"  {d['run']:<42} splits {d['n_splits']:>2}  absorbs {d['n_absorbs']:>2}  {cp}")
    print(f"\nFigures and table in {out}")


if __name__ == "__main__":
    main()

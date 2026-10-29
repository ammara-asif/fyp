#!/usr/bin/env python3
"""
Turn Stage 2 results into a table and three figures for the defense.

    python experiments/stage2_report.py
    python experiments/stage2_report.py --smoke

Writes to results/stage2/figures/:
    table.md                  headline table (paste into slides)
    spawn_curve.png           modules vs tasks, against one-module-per-task
    accuracy_curve.png        average accuracy vs tasks
    routing_curve.png         routing accuracy vs tasks (the known bottleneck)
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

LABELS = {
    "plastic_nested_a": "Ours (plastic offspring)",
    "frozen_nested_a": "Frozen offspring",
    "single_nested_a": "Single net + replay (same memory)",
    "plastic_flat": "Ours, flat learner (no nesting)",
}
STYLE = {
    "plastic_nested_a": dict(color="#2563eb", lw=2.4),
    "frozen_nested_a": dict(color="#94a3b8", lw=2, ls="--"),
    "single_nested_a": dict(color="#f59e0b", lw=2, ls=":"),
    "plastic_flat": dict(color="#10b981", lw=2, ls="-."),
}


def load(base: Path):
    runs = {}
    for s in sorted(base.glob("*/seed*/summary.json")):
        d = json.loads(s.read_text())
        runs.setdefault(d["run"], []).append(d)
    return runs


def key_of(run):           # "plastic_nested_a_disjoint" -> ("plastic_nested_a", "disjoint")
    stream = run.rsplit("_", 1)[1]
    return run[: -len(stream) - 1], stream


def mean_sd(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "—"
    m = float(np.mean(vals))
    return f"{m:.2f}" if len(vals) == 1 else f"{m:.2f} ± {np.std(vals, ddof=1):.2f}"


def table(runs):
    cols = [("final_avg_acc", "Final acc (%)"), ("avg_incremental_acc", "Avg incr. acc (%)"),
            ("forgetting", "Forgetting (pts)"), ("n_modules", "Modules"),
            ("routing_acc_test", "Routing acc (%)"), ("params", "Params (M)"),
            ("anchor_kb", "Raw memory (KB)")]
    lines = []
    for stream in ("disjoint", "recurring"):
        sel = {k: v for k, v in runs.items() if key_of(k)[1] == stream}
        if not sel:
            continue
        n_tasks = next(iter(sel.values()))[0]["n_tasks"]
        lines.append(f"\n### {stream.capitalize()} stream ({n_tasks} tasks, class-IL, "
                     f"seeds per row shown as n)\n")
        lines.append("| Method | n | " + " | ".join(c[1] for c in cols) + " |")
        lines.append("|---|---|" + "---|" * len(cols))
        for run, ds in sorted(sel.items(), key=lambda kv: list(LABELS).index(key_of(kv[0])[0])
                              if key_of(kv[0])[0] in LABELS else 99):
            k = key_of(run)[0]
            cells = []
            for c, _ in cols:
                vals = [d.get(c) for d in ds]
                if c == "params":
                    vals = [v / 1e6 for v in vals]
                if c == "routing_acc_test" and k.startswith("single"):
                    cells.append("n/a")
                else:
                    cells.append(mean_sd(vals))
            lines.append(f"| {LABELS.get(k, k)} | {len(ds)} | " + " | ".join(cells) + " |")
    lines.append("\nPreliminary: Stage 2 minimal system, no consolidation. "
                 "Every method uses the same nested backbone, data order and epochs; "
                 "the single-network baseline replays from a buffer holding the same "
                 "number of raw images per class as the anchor sets.")
    return "\n".join(lines)


def curves(runs, field, ylabel, fname, out, ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    streams = sorted({key_of(r)[1] for r in runs})
    fig, axes = plt.subplots(1, len(streams), figsize=(6.2 * len(streams), 4.2), squeeze=False)
    for ax, stream in zip(axes[0], streams):
        for run, ds in runs.items():
            k, s = key_of(run)
            if s != stream:
                continue
            if field == "routing_acc_test" and k.startswith("single"):
                continue
            ys = np.array([[r[field] for r in d["per_task"]] for d in ds], dtype=float)
            x = np.arange(1, ys.shape[1] + 1)
            m = np.nanmean(ys, 0)
            st = STYLE.get(k, {})
            ax.plot(x, m, label=LABELS.get(k, k), **st)
            if ys.shape[0] > 1:
                sd = np.nanstd(ys, 0)
                ax.fill_between(x, m - sd, m + sd, alpha=0.15, color=st.get("color"))
        if ref == "diag":
            n = max(len(d[0]["per_task"]) for d in runs.values())
            ax.plot(np.arange(1, n + 1), np.arange(1, n + 1), color="#ef4444", lw=1.2,
                    ls=(0, (4, 3)), label="One module per task (PNN-style)")
        if stream == "recurring":
            n = max(len(d[0]["per_task"]) for d in runs.values())
            ax.axvline(n / 2 + 0.5, color="#64748b", lw=1, ls=":")
            ax.text(n / 2 + 0.8, ax.get_ylim()[1] * 0.95, "domains recur →",
                    fontsize=9, color="#64748b", va="top")
        ax.set_title(f"{stream.capitalize()} stream", fontsize=12)
        ax.set_xlabel("Tasks seen")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][-1].legend(fontsize=8.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / fname, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    base = ROOT / "results" / ("stage2_smoke" if args.smoke else "stage2")
    runs = load(base)
    if not runs:
        raise SystemExit(f"no summary.json under {base}")
    out = base / "figures"
    out.mkdir(parents=True, exist_ok=True)

    md = table(runs)
    (out / "table.md").write_text(md)
    print(md)

    curves(runs, "n_modules", "Modules", "spawn_curve.png", out, ref="diag")
    curves(runs, "avg_acc", "Average accuracy on seen tasks (%)", "accuracy_curve.png", out)
    curves(runs, "routing_acc_test", "Routing accuracy (%)", "routing_curve.png", out)

    print("\nSpawn triggers:")
    for run, ds in sorted(runs.items()):
        for d in ds:
            if d["mode"] != "single":
                print(f"  {run} seed{d['seed']}: {d['spawn_reasons']}")
    print(f"\nFigures and table in {out}")


if __name__ == "__main__":
    main()

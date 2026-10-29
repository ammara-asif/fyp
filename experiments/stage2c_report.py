#!/usr/bin/env python3
"""
Stage 2c table, figures, and the groups each learner ended up owning.

    python experiments/stage2c_report.py            # exploratory runs
    python experiments/stage2c_report.py --tuned    # final runs (thresholds from validation)
    python experiments/stage2c_report.py --smoke

Writes to results/stage2c[_tuned|_smoke]/figures/:
    table.md, groups.md, spawn_curve.png, accuracy_curve.png, routing_curve.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CIFAR100 = (
    "apple aquarium_fish baby bear beaver bed bee beetle bicycle bottle bowl boy bridge bus "
    "butterfly camel can castle caterpillar cattle chair chimpanzee clock cloud cockroach couch "
    "crab crocodile cup dinosaur dolphin elephant flatfish forest fox girl hamster house kangaroo "
    "keyboard lamp lawn_mower leopard lion lizard lobster man maple_tree motorcycle mountain mouse "
    "mushroom oak_tree orange orchid otter palm_tree pear pickup_truck pine_tree plain plate poppy "
    "porcupine possum rabbit raccoon ray road rocket rose sea seal shark shrew skunk skyscraper "
    "snail snake spider squirrel streetcar sunflower sweet_pepper table tank telephone television "
    "tiger tractor train trout tulip turtle wardrobe whale willow_tree wolf woman worm"
).split()

COLORS = {"predictive": "#2563eb", "reactive": "#dc2626", "never": "#94a3b8"}


def label(d):
    name = {"predictive": "Predictive split (ours)", "reactive": "Reactive split (DEN-style)",
            "never": "No splitting"}[d["trigger"]]
    extra = []
    if d["arm"] != "nested_a":
        extra.append(d["arm"])
    if d.get("refresh_scope") == "none":
        extra.append("isolated")
    if d.get("nf", 20) != 20:
        extra.append(f"nf={d['nf']}, parameter-matched")
    if d["trigger"] == "predictive":
        extra.append(f"θ={d['theta']}")
    elif d["trigger"] == "reactive":
        extra.append(f"δ={d['delta']}")
    return name + (f" [{', '.join(extra)}]" if extra else "")


def style(d):
    s = dict(color=COLORS[d["trigger"]], lw=2.2)
    if d["arm"] == "flat":
        s.update(color="#10b981", ls="-.")
    elif d["arm"] == "avg_control":
        s.update(color="#f59e0b", ls="--")
    elif d["arm"] != "nested_a":
        s.update(color="#7c3aed", ls="--")
    return s


def load(base):
    runs = {}
    for s in sorted(base.glob("*/seed*/summary.json")):
        d = json.loads(s.read_text())
        if d.get("stage") == "2c":
            runs.setdefault(d["run"], []).append(d)
    return runs


def fmt(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "—"
    m = float(np.mean(vals))
    return f"{m:.2f}" if len(vals) == 1 else f"{m:.2f} ± {np.std(vals, ddof=1):.2f}"


def table(runs):
    cols = [("final_avg_acc", "Final acc (%)"),
            ("new_task_acc", "New-task acc (%)"),
            ("forgetting", "Forgetting (pts)"),
            ("final_acc_taskil", "Task-IL acc (%)"),
            ("new_task_acc_taskil", "Task-IL new-task (%)"),
            ("forgetting_taskil", "Task-IL forgetting (pts)"),
            ("final_acc_oracle", "Oracle acc (%)"),
            ("forgetting_oracle", "Forgetting inside learners (pts)"),
            ("avg_incremental_acc", "Avg incr. acc (%)"),
            ("n_modules", "Learners"),
            ("cap_hit_at_task", "Cap hit at task"),
            ("routing_soft", "Routing, all signals (%)"),
            ("routing_own", "'Mine' only"),
            ("routing_consensus", "Referrals only"),
            ("routing_shared", "Shared space only"),
            ("routing_lr", "Likelihood ratio only"),
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
        sel.sort(key=lambda kv: (order[kv[1][0]["trigger"]], kv[0]))
        for _, ds in sel:
            cells = []
            for c, _ in cols:
                v = [d.get(c) for d in ds]
                if c == "params":
                    v = [x / 1e6 for x in v]
                cells.append(fmt(v))
            out.append(f"| {label(ds[0])} | {len(ds)} | " + " | ".join(cells) + " |")
    out.append(
        "\n**Class-IL (the real setting):** 'Final acc' uses the soft router, which multiplies "
        "four label-free signals ('mine', referrals, a shared-space model in the root's slow "
        "levels, and a likelihood ratio), then P(class) = Σ_learner P(learner) × P(class | learner). "
        "'New-task acc' = accuracy on each task right after it was learned (plasticity). "
        "Forgetting = mean over tasks of (best accuracy ever − final accuracy); it can only be "
        "read together with new-task acc, because a system that learns less also forgets less.\n\n"
        "**Task-IL (same quiz for every system):** the task's 5 classes are given, each class is "
        "scored by its owner. This removes both routing and the single learner's bias towards "
        "recent classes, so 'Task-IL forgetting' is the fair measure of how much knowledge "
        "survives inside the system, and the column to compare splitting against no splitting.\n\n"
        "**Oracle** sends every image to its true owner, which answers among its own classes: "
        "the ceiling routing is chasing. 'Forgetting inside learners' is oracle forgetting: "
        "how much the owner itself lost. Not comparable with the single learner (5-way vs "
        "100-way). Routing = how often the most likely learner is the true owner; the "
        "single-signal columns show each signal alone.")
    return "\n".join(out)


def groups_md(runs):
    out = ["# Classes owned by each learner at the end of the run\n",
           "Coherent groups (similar classes together) are what make routing easy.\n"]
    for _, ds in sorted(runs.items()):
        d = ds[0]
        if d["trigger"] == "never":
            continue
        out.append(f"\n## {label(d)} — {d['stream']}, seed {d['seed']}\n")
        for i, g in enumerate(d.get("groups", []), 1):
            if g:
                out.append(f"- **Learner {i}** ({len(g)}): " + ", ".join(CIFAR100[c] for c in g))
    return "\n".join(out)


def curves(runs, field, ylabel, fname, out, diag=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    streams = [s for s in ("disjoint", "recurring") if any(v[0]["stream"] == s for v in runs.values())]
    fig, axes = plt.subplots(1, len(streams), figsize=(6.4 * len(streams), 4.4), squeeze=False)
    for ax, stream in zip(axes[0], streams):
        n = 0
        for _, ds in runs.items():
            d0 = ds[0]
            if d0["stream"] != stream:
                continue
            ys = np.array([[r.get(field, np.nan) for r in d["per_task"]] for d in ds], dtype=float)
            x = np.arange(1, ys.shape[1] + 1)
            n = max(n, len(x))
            ax.plot(x, np.nanmean(ys, 0), label=label(d0), **style(d0))
            if field == "acc_soft":
                yo = np.array([[r.get("acc_oracle", np.nan) for r in d["per_task"]] for d in ds], float)
                ax.plot(x, np.nanmean(yo, 0), color=style(d0)["color"], lw=1, ls=":", alpha=0.7)
        if diag and n:
            ax.plot(np.arange(1, n + 1), np.arange(1, n + 1), color="#ef4444", lw=1,
                    ls=(0, (4, 3)), label="One learner per task")
        if stream == "recurring" and n:
            ax.axvline(n / 2 + 0.5, color="#64748b", lw=1, ls=":")
        ax.set_title(f"{stream.capitalize()} stream")
        ax.set_xlabel("Tasks seen")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False, loc="best")
    if field == "acc_soft":
        fig.text(0.01, 0.01, "dotted = oracle routing (ceiling)", fontsize=8, color="#64748b")
    fig.tight_layout()
    fig.savefig(out / fname, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tuned", action="store_true")
    args = ap.parse_args()
    base = ROOT / "results" / ("stage2c_smoke" if args.smoke else
                               "stage2c_tuned" if args.tuned else "stage2c")
    runs = load(base)
    if not runs:
        raise SystemExit(f"no Stage 2c summary.json under {base}")
    out = base / "figures"
    out.mkdir(parents=True, exist_ok=True)

    md = table(runs)
    (out / "table.md").write_text(md)
    print(md)
    (out / "groups.md").write_text(groups_md(runs))

    curves(runs, "n_modules", "Learners", "spawn_curve.png", out, diag=True)
    curves(runs, "acc_soft", "Average accuracy on seen tasks (%)", "accuracy_curve.png", out)
    curves(runs, "route_soft", "Routing accuracy (%)", "routing_curve.png", out)
    curves(runs, "acc_taskil", "Task-IL accuracy on seen tasks (%)", "taskil_curve.png", out)

    print("\nSplits:")
    for _, ds in sorted(runs.items()):
        d = ds[0]
        print(f"  {d['run']:<44} learners {d['n_modules']:>2}  splits {d['n_splits']:>2}  "
              f"new-on-arrival {d['n_spawn_on_arrival']:>2}  absorbs {d['n_absorbs']:>3}")
    print(f"\nFigures, table and learner groups in {out}")


if __name__ == "__main__":
    main()

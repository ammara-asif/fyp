"""Experiment infrastructure: paths, stable run IDs, resumable sweeps.

Design notes (for the defence):
- Every experiment is a plain config dict. Its run_id is a hash of the dict,
  so the same config always maps to the same result file. That file is both
  the permanent record and the "already done" marker for resuming.
- A result file is only written when a run *finishes*, and it is written
  atomically (temp file + os.replace), so a Colab disconnect mid-run can never
  leave a half-written result that looks finished. Crashed/interrupted runs
  simply have no result file and are retried on the next call.
"""
import hashlib
import json
import os
import time
import traceback

DRIVE_ROOT = "/content/drive/MyDrive/fyp_nested_cl"


def setup_paths(root=None):
    """Return the folder layout and create it.

    Priority: explicit `root` > $FYP_ROOT > Google Drive (if mounted) > ./local_runs.
    This function does NOT mount Drive: mounting only works from the notebook
    kernel, so the notebook calls drive.mount() first.
    """
    if root is None:
        root = os.environ.get("FYP_ROOT")
    if root is None:
        root = DRIVE_ROOT if os.path.isdir("/content/drive/MyDrive") else os.path.abspath("local_runs")
    paths = {
        "root": root,
        "data": os.path.join(root, "data"),
        "results": os.path.join(root, "results"),
        "logs": os.path.join(root, "logs"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def run_id(cfg):
    """Stable 12-char ID: sha1 of the config serialised with sorted keys."""
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def result_path(cfg, paths):
    return os.path.join(paths["results"], f"{run_id(cfg)}.json")


def atomic_write_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_result(cfg, paths):
    """Return the finished result for cfg, or None."""
    p = result_path(cfg, paths)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        res = json.load(f)
    return res if res.get("status") == "finished" else None


def run_sweep(configs, run_fn, paths, force=False):
    """Run `run_fn(cfg, paths) -> dict` for each config, skipping finished ones.

    A crash in one run is logged to logs/{run_id}.error.txt and the sweep moves
    on; KeyboardInterrupt stops the sweep. Returns {run_id: result-or-None}.
    """
    out = {}
    for i, cfg in enumerate(configs):
        rid = run_id(cfg)
        prev = None if force else load_result(cfg, paths)
        if prev is not None:
            print(f"[{i + 1}/{len(configs)}] {rid} already finished, skipping")
            out[rid] = prev
            continue
        print(f"[{i + 1}/{len(configs)}] {rid} running")
        t0 = time.time()
        try:
            metrics = run_fn(cfg, paths)
        except KeyboardInterrupt:
            raise
        except Exception:
            err = traceback.format_exc()
            with open(os.path.join(paths["logs"], f"{rid}.error.txt"), "w") as f:
                f.write(err)
            print(f"  {rid} CRASHED (will be retried next call):\n{err}")
            out[rid] = None
            continue
        res = {
            "status": "finished",
            "run_id": rid,
            "config": cfg,
            "metrics": metrics,
            "wall_clock_s": time.time() - t0,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(res, result_path(cfg, paths))
        out[rid] = res
    return out

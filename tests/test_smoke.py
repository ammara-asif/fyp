"""CPU smoke tests. No Drive, no wandb, no CIFAR download.

Run:  WANDB_MODE=disabled python -m pytest tests -q
"""
import json
import os
import sys

os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from fyp_infra import run_id, run_sweep, setup_paths  # noqa: E402
import stage1_runner as sr  # noqa: E402

TINY = dict(benchmark="synthetic", n_experiences=2, epochs=1, model="simple_cnn", nf=4,
            batch_size=8, num_workers=0, stage="test")


@pytest.fixture
def paths(tmp_path):
    return setup_paths(str(tmp_path))


def test_run_id_stable_and_sensitive():
    a, b = sr.make_cfg(**TINY), sr.make_cfg(**TINY)
    assert run_id(a) == run_id(b)
    assert run_id(a) != run_id(sr.make_cfg(**dict(TINY, seed=1)))
    # irrelevant LLRD fields must not change a flat run's id
    assert run_id(a) == run_id(sr.make_cfg(**dict(TINY, level_ratios=(1, 16, 256))))


@pytest.mark.parametrize("extra", [
    dict(strategy="naive"),
    dict(strategy="ewc"),
    dict(strategy="naive", lr_schedule="llrd"),
])
def test_end_to_end(paths, extra):
    cfg = sr.make_cfg(**TINY, **extra)
    out = run_sweep([cfg], sr.run_experiment, paths)
    res = out[run_id(cfg)]
    m = res["metrics"]
    assert len(m["acc_matrix"]) == 2 and len(m["acc_matrix"][0]) == 2
    assert m["optimizer_steps"] > 0
    assert os.path.exists(os.path.join(paths["results"], f"{run_id(cfg)}.json"))


def test_llrd_matches_flat_params_and_steps(paths):
    flat = sr.make_cfg(**TINY)
    llrd = sr.make_cfg(**TINY, lr_schedule="llrd")
    out = run_sweep([flat, llrd], sr.run_experiment, paths)
    mf, ml = out[run_id(flat)]["metrics"], out[run_id(llrd)]["metrics"]
    assert mf["n_params"] == ml["n_params"]
    assert mf["optimizer_steps"] == ml["optimizer_steps"]
    lrs = ml["lr_per_group"]
    assert lrs["0"] == pytest.approx(0.03) and lrs["2"] == pytest.approx(0.03 / 16)


def test_resume_skips_finished_and_retries_crashed(paths):
    cfgs = [sr.make_cfg(**dict(TINY, seed=s)) for s in (0, 1)]
    calls = []

    def flaky(cfg, p):
        calls.append(cfg["seed"])
        if cfg["seed"] == 1 and calls.count(1) == 1:
            raise RuntimeError("simulated crash")
        return {"ok": True}

    out = run_sweep(cfgs, flaky, paths)
    assert out[run_id(cfgs[1])] is None
    assert os.path.exists(os.path.join(paths["logs"], f"{run_id(cfgs[1])}.error.txt"))
    out = run_sweep(cfgs, flaky, paths)
    assert calls == [0, 1, 1]  # seed 0 skipped on 2nd call, seed 1 retried
    with open(os.path.join(paths["results"], f"{run_id(cfgs[1])}.json")) as f:
        assert json.load(f)["status"] == "finished"


def test_summarize_known_values():
    s = sr.summarize([[80, 0], [50, 70]])
    assert s["final_avg_acc"] == 60
    assert s["avg_forgetting"] == 30
    assert s["avg_incremental_acc"] == pytest.approx((80 + 60) / 2)

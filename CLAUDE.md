# CLAUDE.md — FYP: Autonomous Dynamic-Nesting Networks

## Read these first (source of truth)
- `documents/plan.md` — the staged plan (Stages 0–4, gates, benchmarks, metrics, ablations).
- `documents/proposal.md` — the research proposal (hypotheses, novelty claims, positioning).
- `README.md` — current status, results so far, open supervisor decisions.

If anything in this file conflicts with those two documents, **the documents win**. Tell me about the conflict instead of picking silently.

## Who I am and how to work with me
- I'm a final-year student. I have to defend every design choice, so explain non-obvious decisions briefly in plain language (comments or a short note), not just the code.
- Work one stage at a time and respect the gates in the plan. **Do not build ahead of a gate** (e.g. no routing/spawning/Dream Phase code while Stage 1 is open).
- When the plan leaves a decision open, **ask me** instead of assuming.
  - Decided: reduced ResNet-18 (nf=20), 20 tasks × 5 classes (class-IL), 10 epochs per task (epoch sweep), lr Flat 0.1 / LLRD 0.5 (validation lr grid).
  - Still open (supervisor): Gate 1 criterion (proposed: per-level linear-probe forgetting at matched new-task accuracy), meaning of "matched effective lr" (currently: tune each arm's base lr separately), head placement / head-bias fix, ratios (1,4,16) only vs also (1,16,256), baseline scope.
- Before writing a big chunk of code, give me a short breakdown of what you'll build and in what order, ask me anything that is ambiguous/unclear instead of making assumptions, then wait for a go-ahead.
- Never invent results or numbers. If you couldn't run something, say so plainly. Negative results are acceptable and must be reported as such (the plan explicitly allows this).

## Compute and environment (important constraints)
- My laptop has **no dedicated GPU**. Don't install torch/avalanche locally (avalanche's `proxsuite` dependency fails to compile on Windows); local checks are syntax/JSON only, and I run everything on Colab and report results back.
- **All real experiments run on Google Colab (GPU)**, Python 3.13, with wandb. Code must work when imported from a notebook cell. Code reaches Colab via GitHub: I push from my laptop, and the notebook setup cell clones/pulls it.
- Colab runtimes are ephemeral: the disk is wiped on reset and sessions can disconnect. Everything that must survive (dataset, per-run result JSONs, logs, code) lives on **Google Drive** under `MyDrive/fyp_nested_cl/`.
- Drive mounting only works from the notebook kernel, never from a `!python script.py` subprocess.
- Libraries: `torch`, `torchvision`, `avalanche-lib`, `hydra-core`, `wandb`, `scikit-learn`, `torchmetrics`. Avoid adding heavy dependencies without asking (native-compile packages already caused trouble on my Windows machine).

## Known pitfalls (already hit, don't rediscover)
- **Do not use Avalanche's `WandBLogger`.** With current wandb it crashes (`module 'wandb' has no attribute 'viz'`). Log to wandb manually with `wandb.log(...)`.
- `SplitCIFAR100(n_experiences=N)` requires 100 to be divisible by N (e.g. 5, 10, 20, not 3).
- CIFAR-100 download on Colab was extremely slow once (~47 min). Always pass `dataset_root` pointing at the Drive data folder.
- Never hardcode the wandb API key; use `wandb.login()` / the `WANDB_API_KEY` env var.
- Avalanche's per-experience accuracy metric keys look like `Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{j:03d}`.
- EWC with λ=100 at lr 0.03 diverged to NaN (and silently logged 0% everywhere). Hence the NaN guard + grad clipping in the runner.
- After `git pull` in Colab, a plain `import` keeps the OLD module; the setup cell pops cached modules. Check the printed commit hash.
- Colab can silently give a CPU runtime; training notebooks assert a GPU (`REQUIRE_GPU`).
- With Naive training in class-IL every old task ends at exactly 0%, so class-IL forgetting = peak accuracy; it can't separate methods. Always report `acc_just_learned` and the probe metrics alongside it.
- In GNU sed, `\'` is an end-of-buffer anchor, not a quote. Edit `_build_notebooks.py` with Python/Edit, not sed.

## Existing code (build on it, don't rewrite it)
- `src/fyp_infra.py`: `setup_paths()`, `run_id(cfg)` (stable hash of the config), `run_sweep(configs, run_fn, paths, force=False)`. Finished runs are skipped on re-run, crashed runs are retried, result files are written atomically to `results/{run_id}.json`. This was tested with a simulated-crash test.
- `src/stage1_runner.py`: `DEFAULTS`, `make_cfg(**overrides)`, `run_experiment(cfg, paths)`, plus registries `MODEL_BUILDERS` and `STRATEGY_BUILDERS`. New models and methods are added to the registries; the training loop, wandb logging, accuracy matrix and forgetting calculation should not need to change. Has run on Colab (Stage 0 and Stage 1). Also has: `eval_split` ("val" for tuning, "test" for final), LLRD param groups via `model.depth_groups()` (the same 3-level split must be used by nesting), `save_checkpoints`, NaN guard, grad clipping.
- `src/stage1_analysis.py`: post-hoc task-IL accuracy and per-level linear probes from saved checkpoints; runs via `run_sweep` (CPU is fine).
- `notebooks/_build_notebooks.py` generates all notebooks; edit it, not the `.ipynb` files. Executed notebooks with outputs are copied into `runs/`.
- Stage 0 is done: Gate 0 passed, and the second seed reproduced the first (within ~0.5 points).
- Stage 1 status: epoch sweep, lr grid, Flat vs Flat+LLRD control (3 seeds) and checkpoint analysis are done. Pending: re-run Flat at lr 0.1 (the control used 0.3 by mistake), then build Nested A–D (needs my go-ahead). Results summary is in `README.md`.

## Conventions
- Every experiment is a config dict; the same config must always give the same `run_id`. Any new config field that changes behaviour must be part of the config (so it changes the ID).
- Seeds are explicit, and results are reported over multiple seeds, never a single run.
- wandb project `fyp-nested-cl`; set `group` per stage and `name` to the `run_id`. Per-run JSON results always get saved to Drive in addition to wandb (wandb is the dashboard, the JSON is the record and the resume marker).
- Comparisons against baselines must be fair: matched parameter count and total optimisation steps unless the plan says otherwise. The deflationary control (nesting vs flat + layer-wise LR decay) is the most important comparison in Stage 1, so keep it honest.
- Keep metric definitions stable. If you change how a metric is computed, tell me, because it invalidates earlier runs.
- Local tests must run on CPU with a tiny config (2 experiences, 1 epoch, a small data subset) and must not need Drive or wandb (use `WANDB_MODE=disabled`).

## Definition of done for any task
1. It compiles and the notebooks parse locally; tiny-config smoke tests exist in `tests/` (run them on Colab if needed, since nothing is installed locally).
2. It integrates with `run_sweep` (resumable, config-driven).
3. You've told me exactly what you ran, what you did **not** run, and what I should check on Colab.
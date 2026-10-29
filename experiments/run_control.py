import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics
from avalanche.logging import CSVLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from src.models.resnet import ReducedResNet18
from src.nested.learner import NestedLearner
from src.nested.update_rules import build_rule
from src.strategies.nested_strategy import NestedStrategy
from src.metrics.structural import StructuralProbePlugin
from src.utils import seed_everything, count_params


def build_benchmark(cfg):
    """Split-CIFAR-100, class-IL.

    `fixed_class_order` and `seed` are mutually exclusive in Avalanche's
    nc_benchmark: passing a fixed order makes the seed meaningless for the
    split, so only one is sent.
    """
    kw = dict(n_experiences=cfg.benchmark.n_experiences, return_task_id=False)
    if cfg.benchmark.fixed_order:
        kw["fixed_class_order"] = list(range(cfg.model.num_classes))
    else:
        kw["seed"] = cfg.seed
    bm = SplitCIFAR100(**kw)

    if cfg.get("tune", {}).get("enabled", False):
        # Carve a validation stream out of train so nothing is selected on test.
        from avalanche.benchmarks.generators import benchmark_with_validation_stream
        bm = benchmark_with_validation_stream(
            bm, validation_size=cfg.tune.val_fraction, shuffle=True
        )
    return bm


def eval_stream_of(benchmark, cfg):
    if cfg.get("tune", {}).get("enabled", False):
        return benchmark.valid_stream
    return benchmark.test_stream


@hydra.main(config_path="../configs", config_name="control", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))

    benchmark = build_benchmark(cfg)
    eval_stream = eval_stream_of(benchmark, cfg)

    # Per-arm base learning rate. Arms are tuned separately with the same budget,
    # because periodic stepping and per-level ratios both change the effective
    # update rate; a shared lr would compare tuning rather than architecture.
    lr = cfg.arm.get("lr", None) or cfg.optim.lr

    backbone = ReducedResNet18(num_classes=cfg.model.num_classes, nf=cfg.model.nf)
    learner = NestedLearner(backbone, **OmegaConf.to_container(cfg.arm.nesting)).to(device)
    rule = build_rule(cfg.arm.rule, learner, lr=lr, momentum=cfg.optim.momentum,
                      weight_decay=cfg.optim.weight_decay, grad_clip=cfg.optim.grad_clip,
                      **OmegaConf.to_container(cfg.arm.rule_kwargs))
    if hasattr(rule, "shadow"):
        rule.shadow.to(device)

    print(f"arm={cfg.arm.name} lr={lr} clip={cfg.optim.grad_clip} "
          f"params={count_params(learner)}")
    for d in learner.describe():
        print("   ", d)

    loggers = [InteractiveLogger(), CSVLogger(os.path.join(out_dir, "avalanche_logs"))]
    if cfg.wandb.enabled:
        import wandb
        from avalanche.logging import WandBLogger
        run_name = f"{cfg.arm.name}-lr{lr}-s{cfg.seed}"
        wandb.init(project=cfg.wandb.project, name=run_name,
                   config=OmegaConf.to_container(cfg, resolve=True), dir=out_dir)
        loggers.append(WandBLogger(project_name=cfg.wandb.project, run_name=run_name))

    evaluator = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(stream=True),
        loggers=loggers,
    )

    probe = None
    if cfg.probe.enabled:
        probe = StructuralProbePlugin(
            benchmark, out_dir, device, n_cka=cfg.probe.n_cka,
            n_probe_train=cfg.probe.n_probe_train, n_probe_test=cfg.probe.n_probe_test,
            use_wandb=cfg.wandb.enabled, seed=cfg.seed)

    strategy = NestedStrategy(
        learner, rule,
        reset_rule_per_exp=cfg.optim.reset_per_exp,
        train_mb_size=cfg.train.mb_size, train_epochs=cfg.train.epochs,
        eval_mb_size=cfg.train.eval_mb_size, device=device,
        plugins=[probe] if probe else [], evaluator=evaluator, eval_every=-1,
    )

    # Average incremental accuracy: the mean, over the stream, of average
    # accuracy on all tasks seen so far. More sensitive than the final number.
    incremental, new_task = [], []
    for exp in benchmark.train_stream:
        t = exp.current_experience
        print(f"--- experience {t} classes={exp.classes_in_this_experience}")
        strategy.train(exp)
        res = strategy.eval(eval_stream[: t + 1])
        accs = [v for k, v in res.items() if "Top1_Acc_Exp" in k]
        if accs:
            incremental.append(100.0 * sum(accs) / len(accs))
            new_task.append(100.0 * accs[-1])

    final_res = strategy.eval(eval_stream)
    final = {k: v for k, v in final_res.items()
             if "Top1_Acc_Stream" in k or "StreamForgetting" in k}
    final.update(
        arm=cfg.arm.name, seed=cfg.seed, lr=lr, epochs=cfg.train.epochs,
        params=count_params(learner),
        avg_incremental_acc=float(sum(incremental) / len(incremental)) if incremental else None,
        new_task_acc=float(sum(new_task) / len(new_task)) if new_task else None,
        tune=bool(cfg.get("tune", {}).get("enabled", False)),
    )
    if probe is not None:
        final.update(probe.probe_forgetting())

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

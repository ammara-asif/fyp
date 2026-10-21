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


@hydra.main(config_path="../configs", config_name="control", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))

    benchmark = SplitCIFAR100(
        n_experiences=cfg.benchmark.n_experiences,
        seed=cfg.seed,
        return_task_id=False,                         # class-IL
        fixed_class_order=list(range(100)) if cfg.benchmark.fixed_order else None,
    )

    backbone = ReducedResNet18(num_classes=cfg.model.num_classes, nf=cfg.model.nf)
    learner = NestedLearner(backbone, **OmegaConf.to_container(cfg.arm.nesting)).to(device)
    rule = build_rule(cfg.arm.rule, learner, lr=cfg.optim.lr, momentum=cfg.optim.momentum,
                      weight_decay=cfg.optim.weight_decay, **OmegaConf.to_container(cfg.arm.rule_kwargs))
    if hasattr(rule, "shadow"):
        rule.shadow.to(device)

    print(f"arm={cfg.arm.name} params={count_params(learner)} levels={learner.describe()}")

    loggers = [InteractiveLogger(), CSVLogger(os.path.join(out_dir, "avalanche_logs"))]
    if cfg.wandb.enabled:
        import wandb
        from avalanche.logging import WandBLogger
        wandb.init(project=cfg.wandb.project, name=f"{cfg.arm.name}-s{cfg.seed}",
                   config=OmegaConf.to_container(cfg, resolve=True), dir=out_dir)
        loggers.append(WandBLogger(project_name=cfg.wandb.project, run_name=f"{cfg.arm.name}-s{cfg.seed}"))

    evaluator = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(stream=True),
        loggers=loggers,
    )

    plugins = []
    if cfg.probe.enabled:
        plugins.append(StructuralProbePlugin(
            benchmark, out_dir, device, n_cka=cfg.probe.n_cka, n_probe_train=cfg.probe.n_probe_train,
            n_probe_test=cfg.probe.n_probe_test, use_wandb=cfg.wandb.enabled, seed=cfg.seed))

    strategy = NestedStrategy(
        learner, rule,
        reset_rule_per_exp=cfg.optim.reset_per_exp,
        train_mb_size=cfg.train.mb_size, train_epochs=cfg.train.epochs,
        eval_mb_size=cfg.train.eval_mb_size, device=device,
        plugins=plugins, evaluator=evaluator, eval_every=-1,
    )

    for exp in benchmark.train_stream:
        print(f"--- experience {exp.current_experience} classes={exp.classes_in_this_experience}")
        strategy.train(exp)
        strategy.eval(benchmark.test_stream)

    final = {k: v for k, v in evaluator.get_last_metrics().items()
             if "Top1_Acc_Stream" in k or "StreamForgetting" in k}
    final.update(arm=cfg.arm.name, seed=cfg.seed, params=count_params(learner))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
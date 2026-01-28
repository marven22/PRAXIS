from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from praxis.utils import DEVICE, set_seed

from ..dro.data import build_test_loader, build_train_loader
from ..dro.eval import eval_accuracy, eval_over_universes
from ..dro.models import build_resnet34
from ..dro.universe import Universe, assert_disjoint, build_universe_grid
from .exp3 import EXP3


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _init_logs(
    args: argparse.Namespace,
    seed: int,
    universes: list[Universe],
    heldout_universes: list[Universe],
) -> Dict[str, Any]:
    return {
        "args": vars(args),
        "seed": seed,
        "universes": [u.to_dict() for u in universes],
        "heldout_universes": [u.to_dict() for u in heldout_universes],
        "generator": {"entropy": [], "max_prob": [], "q": []},
        "fixed_mean": [],
        "robustness": {
            "min_acc": [],
            "mean_acc": [],
            "std_acc": [],
            "cvar_acc": [],
            "accs": [],
        },
        "heldout": {
            "min_acc": [],
            "mean_acc": [],
            "std_acc": [],
            "cvar_acc": [],
            "accs": [],
        },
    }


def run(args: argparse.Namespace, seed: int) -> str:
    set_seed(seed)

    sizes = np.linspace(args.size_min, args.size_max, args.size_grid).astype(int).tolist()
    augs = np.linspace(args.aug_min, args.aug_max, args.aug_grid).tolist()
    severities = [0.0, 0.3, 0.6]

    train_corruptions = ["none", "blur", "jpeg", "noise"]
    heldout_corruptions = [
        c.strip() for c in args.heldout_corruptions.split(",") if c.strip()
    ]

    universes = build_universe_grid(sizes, augs, train_corruptions, severities)
    heldout_universes = build_universe_grid(
        args.heldout_sizes,
        args.heldout_augs,
        heldout_corruptions,
        args.heldout_severities,
    )

    if not args.allow_heldout_overlap:
        assert_disjoint(universes, heldout_universes)

    bandit = EXP3(len(universes), gamma=args.bandit_gamma, seed=seed)
    model = build_resnet34(args.num_classes).to(DEVICE)

    fixed_tests = {
        s: build_test_loader(
            args.test_dir,
            size=args.size_test,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            seed=int(s),
        )
        for s in args.test_seeds
    }

    logs = _init_logs(args, seed, universes, heldout_universes)

    for iteration in range(args.iters):
        idxs, probs = bandit.sample(args.group_size)
        rewards = []

        for universe_idx in idxs:
            universe = universes[universe_idx]
            loader = build_train_loader(
                args.train_dir,
                size=universe.size,
                aug=universe.aug,
                corruption=universe.corruption,
                severity=universe.severity,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
                seed=seed + universe_idx,
            )

            opt = optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            iterator = iter(loader)
            for _ in range(args.solver_steps):
                try:
                    x, y = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    x, y = next(iterator)
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            accs = [eval_accuracy(model, tl, args.max_test_batches) for tl in fixed_tests.values()]
            rewards.append(float(np.mean(accs)))

        bandit.update(idxs, rewards)

        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        logs["generator"]["entropy"].append(entropy)
        logs["generator"]["max_prob"].append(float(probs.max()))
        logs["generator"]["q"].append(probs.tolist())
        logs["fixed_mean"].append(float(np.mean(rewards)))

        if (iteration + 1) % args.robust_eval_every == 0:
            accs = eval_over_universes(
                model,
                universes,
                args.test_dir,
                args.batch_size,
                args.num_classes,
                seed,
                args.size_test,
                args.robust_max_batches,
            )
            k = max(1, int(args.cvar_alpha * len(accs)))
            logs["robustness"]["min_acc"].append(float(accs.min()))
            logs["robustness"]["mean_acc"].append(float(accs.mean()))
            logs["robustness"]["std_acc"].append(float(accs.std()))
            logs["robustness"]["cvar_acc"].append(float(np.sort(accs)[:k].mean()))
            logs["robustness"]["accs"].append(accs.tolist())
        else:
            logs["robustness"]["accs"].append(None)
            for key in logs["robustness"]:
                if key != "accs":
                    logs["robustness"][key].append(None)

        if (iteration + 1) % args.heldout_eval_every == 0:
            accs = eval_over_universes(
                model,
                heldout_universes,
                args.test_dir,
                args.batch_size,
                args.num_classes,
                seed + 999,
                args.size_test,
                args.heldout_max_batches,
            )
            k = max(1, int(args.cvar_alpha * len(accs)))
            logs["heldout"]["min_acc"].append(float(accs.min()))
            logs["heldout"]["mean_acc"].append(float(accs.mean()))
            logs["heldout"]["std_acc"].append(float(accs.std()))
            logs["heldout"]["cvar_acc"].append(float(np.sort(accs)[:k].mean()))
            logs["heldout"]["accs"].append(accs.tolist())
        else:
            logs["heldout"]["accs"].append(None)
            for key in logs["heldout"]:
                if key != "accs":
                    logs["heldout"][key].append(None)

        print(
            f"[Iter {iteration + 1}/{args.iters}] "
            f"fixed_mean={logs['fixed_mean'][-1]:.3f} "
            f"ent={entropy:.3f} max_p={probs.max():.3f}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"logs_bandit_exp3_seed{seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(json_safe(logs), f, indent=2)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MIT Indoor - EXP3 Bandit (universe-aligned)")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out-dir", default="./logs_bandit")

    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--solver-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--size-min", type=int, default=800)
    parser.add_argument("--size-max", type=int, default=2400)
    parser.add_argument("--size-grid", type=int, default=3)
    parser.add_argument("--aug-min", type=float, default=0.2)
    parser.add_argument("--aug-max", type=float, default=0.6)
    parser.add_argument("--aug-grid", type=int, default=3)

    parser.add_argument("--size-test", type=int, default=800)
    parser.add_argument("--max-test-batches", type=int, default=10)
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    parser.add_argument("--robust-eval-every", type=int, default=10)
    parser.add_argument("--robust-max-batches", type=int, default=3)
    parser.add_argument("--cvar-alpha", type=float, default=0.10)

    parser.add_argument("--heldout-eval-every", type=int, default=10)
    parser.add_argument("--heldout-max-batches", type=int, default=3)
    parser.add_argument("--heldout-sizes", type=int, nargs="+", default=[2000])
    parser.add_argument("--heldout-augs", type=float, nargs="+", default=[0.5])
    parser.add_argument("--heldout-severities", type=float, nargs="+", default=[0.15, 0.45, 0.75])
    parser.add_argument("--heldout-corruptions", type=str, default="blur,jpeg,noise")
    parser.add_argument(
        "--allow-heldout-overlap",
        action="store_true",
        help="Allow heldout universes to overlap with training grid.",
    )

    parser.add_argument("--bandit-gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[Device]", DEVICE)
    run(args, args.seed)


if __name__ == "__main__":
    main()

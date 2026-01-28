from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from praxis.utils import DEVICE, set_seed

from .data import build_test_loader, build_train_loader
from .dro import GroupDRO
from .eval import eval_accuracy, eval_over_universes
from .models import build_resnet34
from .universe import Universe, assert_disjoint, build_universe_grid


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
        "fixed_mean": [],
        "generator": {"entropy": [], "max_prob": [], "q": []},
        "heldout": {"mean_acc": [], "min_acc": [], "cvar_acc": []},
    }


def run(args: argparse.Namespace, seed: int) -> str:
    set_seed(seed)

    sizes = np.linspace(args.size_min, args.size_max, args.size_grid).astype(int).tolist()
    augs = np.linspace(args.aug_min, args.aug_max, args.aug_grid).tolist()
    severities = [0.0, 0.3, 0.6]

    train_corruptions = ["none", "blur", "jpeg", "noise"]
    heldout_corruptions = [
        c.strip()
        for c in args.heldout_corruptions.split(",")
        if c.strip()
    ]

    universes = build_universe_grid(sizes, augs, train_corruptions, severities)
    heldout_universes = build_universe_grid(
        args.heldout_sizes,
        args.heldout_augs,
        heldout_corruptions,
        args.heldout_severities,
    )

    assert_disjoint(universes, heldout_universes)

    dro = GroupDRO(len(universes), eta=args.dro_eta, seed=seed)
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
        idxs, q = dro.sample(args.group_size)
        logs["generator"]["q"].append(q.tolist())
        logs["generator"]["entropy"].append(float(-(q * np.log(q + 1e-12)).sum()))
        logs["generator"]["max_prob"].append(float(q.max()))

        losses = []
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
            fixed_mean = float(np.mean(accs))
            losses.append(1.0 - fixed_mean)

        dro.update(idxs, losses)
        logs["fixed_mean"].append(1.0 - float(np.mean(losses)))

        if (iteration + 1) % args.heldout_eval_every == 0:
            accs = eval_over_universes(
                model,
                heldout_universes,
                args.test_dir,
                args.batch_size,
                args.num_classes,
                seed + 999,
                args.heldout_size_test,
                args.heldout_max_batches,
            )
            k = max(1, int(args.cvar_alpha * len(accs)))
            logs["heldout"]["mean_acc"].append(float(accs.mean()))
            logs["heldout"]["min_acc"].append(float(accs.min()))
            logs["heldout"]["cvar_acc"].append(float(np.sort(accs)[:k].mean()))
        else:
            logs["heldout"]["mean_acc"].append(None)
            logs["heldout"]["min_acc"].append(None)
            logs["heldout"]["cvar_acc"].append(None)

        print(
            f"[Iter {iteration + 1}/{args.iters}] "
            f"fixed_mean={logs['fixed_mean'][-1]:.3f} "
            f"q_ent={logs['generator']['entropy'][-1]:.3f}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"logs_dro_seed{seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MIT Indoor - Group DRO (universe-aligned)")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out-dir", default="./logs_dro")

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

    parser.add_argument("--dro-eta", type=float, default=0.5)

    parser.add_argument("--heldout-sizes", type=int, nargs="+", default=[2000])
    parser.add_argument("--heldout-augs", type=float, nargs="+", default=[0.5])
    parser.add_argument("--heldout-severities", type=float, nargs="+", default=[0.15, 0.45, 0.75])
    parser.add_argument("--heldout-corruptions", type=str, default="blur,jpeg,noise")
    parser.add_argument("--heldout-eval-every", type=int, default=10)
    parser.add_argument("--heldout-size-test", type=int, default=800)
    parser.add_argument("--heldout-max-batches", type=int, default=3)

    parser.add_argument("--cvar-alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[Device]", DEVICE)
    run(args, args.seed)


if __name__ == "__main__":
    main()

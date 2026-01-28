from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from praxis.data import build_test_loader, build_train_loader
from praxis.models import build_resnet34
from praxis.training import eval_accuracy, eval_over_universes
from praxis.universe import Universe, assert_no_overlap
from praxis.utils import DEVICE, set_seed

from .metrics import entropy_from_counts, json_safe, kl_from_counts
from .sampler import CurriculumSampler


def _compute_difficulties(
    model,
    universes: List[Universe],
    test_dir: str,
    batch_size: int,
    num_classes: int,
    seed: int,
    size_test: int,
    max_batches: Optional[int],
) -> np.ndarray:
    diffs = []
    for i, _ in enumerate(universes):
        loader = build_test_loader(
            test_dir,
            size_test=size_test,
            batch_size=batch_size,
            num_classes=num_classes,
            seed=seed + i,
        )
        acc = eval_accuracy(model, loader, max_batches)
        diffs.append(1.0 - acc)
    return np.array(diffs, dtype=np.float64)


def _init_logs(
    args: argparse.Namespace,
    seed: int,
    universes: List[Universe],
    heldout_universes: List[Universe],
    difficulties: np.ndarray,
) -> Dict[str, Any]:
    return {
        "args": vars(args),
        "seed": int(seed),
        "universes": [u.to_dict() for u in universes],
        "heldout_universes": [u.to_dict() for u in heldout_universes],
        "difficulty": difficulties.tolist(),
        "fixed_mean": [],
        "generator": {
            "entropy": [],
            "kl_t1_t": [],
            "drift_proxy": [],
            "U": [],
            "counts": [],
        },
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
        "timing": {
            "iter_total": [],
            "robust_eval": [],
            "heldout_eval": [],
        },
    }


def run(args: argparse.Namespace, seed: int) -> str:
    set_seed(seed)

    sizes = np.linspace(args.size_min, args.size_max, args.size_grid).astype(int).tolist()
    augs = np.linspace(args.aug_min, args.aug_max, args.aug_grid).tolist()
    corruptions = ["none", "blur", "noise"]
    severities = [0.0, 0.3, 0.6]

    universes = [
        Universe(size=int(s), aug_strength=float(a), corruption=str(c), severity=float(v))
        for s in sizes
        for a in augs
        for c in corruptions
        for v in severities
    ]

    heldout_universes = [
        Universe(size=int(s), aug_strength=float(a), corruption=str(c), severity=float(v))
        for s in args.heldout_sizes
        for a in args.heldout_augs
        for c in ["blur", "jpeg", "noise"]
        for v in args.heldout_severities
    ]

    assert_no_overlap(universes, heldout_universes)

    model = build_resnet34(args.num_classes).to(DEVICE)

    fixed_tests = {
        int(s): build_test_loader(
            args.test_dir,
            size_test=args.size_test,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            seed=int(s),
        )
        for s in args.test_seeds
    }

    print("[Curriculum] Computing universe difficulties...")
    difficulties = _compute_difficulties(
        model=model,
        universes=universes,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        seed=seed,
        size_test=args.size_test,
        max_batches=args.max_test_batches,
    )

    sampler = CurriculumSampler(
        difficulties=difficulties,
        iters=args.iters,
        group_size=args.group_size,
    )

    logs = _init_logs(args, seed, universes, heldout_universes, difficulties)

    prev_counts = None
    prev_fixed_mean = None

    for t in range(args.iters):
        t_iter0 = time.perf_counter()
        idxs = sampler.sample(t)

        counts = np.zeros(len(universes), dtype=np.int64)
        for ui in idxs:
            counts[ui] += 1

        logs["generator"]["U"].append(int((counts > 0).sum()))
        logs["generator"]["counts"].append(counts.tolist())
        logs["generator"]["entropy"].append(entropy_from_counts(counts))
        if prev_counts is None:
            logs["generator"]["kl_t1_t"].append(None)
        else:
            logs["generator"]["kl_t1_t"].append(kl_from_counts(counts, prev_counts))
        prev_counts = counts

        for ui in idxs:
            universe = universes[ui]
            loader = build_train_loader(
                args.train_dir,
                size=universe.size,
                aug_strength=universe.aug_strength,
                corruption=universe.corruption,
                severity=universe.severity,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
                seed=seed + 10 * t + ui,
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

        accs_fixed = [
            eval_accuracy(model, tl, max_batches=args.max_test_batches)
            for tl in fixed_tests.values()
        ]
        logs["fixed_mean"].append(float(np.mean(accs_fixed)))
        if prev_fixed_mean is None:
            logs["generator"]["drift_proxy"].append(None)
        else:
            logs["generator"]["drift_proxy"].append(
                float(abs(logs["fixed_mean"][-1] - prev_fixed_mean))
            )
        prev_fixed_mean = logs["fixed_mean"][-1]

        if (t + 1) % args.robust_eval_every == 0:
            t0 = time.perf_counter()
            accs = eval_over_universes(
                model=model,
                universes=universes,
                test_dir=args.test_dir,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
                seed=seed,
                size_test=args.robust_size_test,
                max_batches=args.robust_max_batches,
            )
            k = max(1, int(args.cvar_alpha * len(accs)))
            worst = np.sort(accs)[:k]
            logs["robustness"]["min_acc"].append(float(accs.min()))
            logs["robustness"]["mean_acc"].append(float(accs.mean()))
            logs["robustness"]["std_acc"].append(float(accs.std()))
            logs["robustness"]["cvar_acc"].append(float(worst.mean()))
            logs["robustness"]["accs"].append(accs.tolist())
            logs["timing"]["robust_eval"].append(float(time.perf_counter() - t0))
        else:
            logs["robustness"]["min_acc"].append(None)
            logs["robustness"]["mean_acc"].append(None)
            logs["robustness"]["std_acc"].append(None)
            logs["robustness"]["cvar_acc"].append(None)
            logs["robustness"]["accs"].append(None)
            logs["timing"]["robust_eval"].append(0.0)

        if (t + 1) % args.heldout_eval_every == 0:
            t0 = time.perf_counter()
            accs = eval_over_universes(
                model=model,
                universes=heldout_universes,
                test_dir=args.test_dir,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
                seed=seed + 999,
                size_test=args.heldout_size_test,
                max_batches=args.heldout_max_batches,
            )
            k = max(1, int(args.cvar_alpha * len(accs)))
            worst = np.sort(accs)[:k]
            logs["heldout"]["min_acc"].append(float(accs.min()))
            logs["heldout"]["mean_acc"].append(float(accs.mean()))
            logs["heldout"]["std_acc"].append(float(accs.std()))
            logs["heldout"]["cvar_acc"].append(float(worst.mean()))
            logs["heldout"]["accs"].append(accs.tolist())
            logs["timing"]["heldout_eval"].append(float(time.perf_counter() - t0))
        else:
            logs["heldout"]["min_acc"].append(None)
            logs["heldout"]["mean_acc"].append(None)
            logs["heldout"]["std_acc"].append(None)
            logs["heldout"]["cvar_acc"].append(None)
            logs["heldout"]["accs"].append(None)
            logs["timing"]["heldout_eval"].append(0.0)

        logs["timing"]["iter_total"].append(float(time.perf_counter() - t_iter0))

        print(
            f"[Iter {t + 1}/{args.iters}] "
            f"fixed_mean={logs['fixed_mean'][-1]:.3f} "
            f"U={logs['generator']['U'][-1]} "
            f"ent={logs['generator']['entropy'][-1]:.3f}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"logs_curriculum_seed{seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(json_safe(logs), f, indent=2)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MIT Indoor - Curriculum Baseline (universe-aligned)")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out-dir", default="./logs_curriculum_full_log")

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
    parser.add_argument("--robust-size-test", type=int, default=800)
    parser.add_argument("--robust-max-batches", type=int, default=3)
    parser.add_argument("--cvar-alpha", type=float, default=0.10)

    parser.add_argument("--heldout-sizes", type=int, nargs="+", default=[2000])
    parser.add_argument("--heldout-augs", type=float, nargs="+", default=[0.5])
    parser.add_argument("--heldout-severities", type=float, nargs="+", default=[0.15, 0.45, 0.75])
    parser.add_argument("--heldout-eval-every", type=int, default=10)
    parser.add_argument("--heldout-size-test", type=int, default=800)
    parser.add_argument("--heldout-max-batches", type=int, default=3)

    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[Device]", DEVICE)
    run(args, args.seed)


if __name__ == "__main__":
    main()

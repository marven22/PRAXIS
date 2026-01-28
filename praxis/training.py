"""Training and evaluation utilities."""
from __future__ import annotations

import math
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .archive import PRAXISArchive
from .data import build_test_loader
from .utils import DEVICE
from .universe import Universe


@torch.no_grad()
def eval_accuracy(model: nn.Module, loader: DataLoader, max_batches: Optional[int] = None) -> float:
    model.eval()
    correct = 0
    total = 0
    for bi, (x, y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        if max_batches is not None and bi >= max_batches:
            break
    return correct / max(1, total)


@torch.no_grad()
def batch_output_distributions(model: nn.Module, x: torch.Tensor) -> List[Dict[int, float]]:
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    num_classes = probs.size(1)
    outs = []
    for i in range(probs.size(0)):
        d = {j: float(probs[i, j].item()) for j in range(num_classes)}
        outs.append(d)
    return outs


def train_solver_one_universe(
    model: nn.Module,
    loader: DataLoader,
    archive: Optional[PRAXISArchive],
    alpha_t: float,
    steps: int,
    args: argparse.Namespace,
    opt: torch.optim.Optimizer,
) -> Dict[str, float]:
    model.train()

    ce_vals: List[float] = []
    reg_vals: List[float] = []
    total_vals: List[float] = []

    gate_total = 0
    gate_steps_nonzero = 0
    gate_total_steps = 0

    it = iter(loader)
    for step in range(steps):
        gate_total_steps += 1

        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        ce = F.cross_entropy(logits, y)

        outs = batch_output_distributions(model, x)
        probs = F.softmax(logits, dim=1)
        pmax, yhat = probs.max(dim=1)

        reg = torch.tensor(0.0, device=DEVICE)
        step_gate_count = 0

        if archive is not None and alpha_t > 0:
            for j, cand in enumerate(archive.programs):
                yprog = []
                for i in range(len(outs)):
                    yi = cand.fn(0, step, [outs[i]])
                    yi = int(max(0, min(yi, logits.size(1) - 1)))
                    yprog.append(yi)
                yprog_t = torch.tensor(yprog, device=DEVICE)

                disagree = (yprog_t != yhat).float()

                gate = torch.ones_like(disagree)
                if args.psi_gate in ["low_conf", "both"]:
                    gate = gate * (pmax < args.psi_conf_thresh).float()
                if args.psi_gate in ["high_disagree", "both"]:
                    gate = gate * (disagree > args.psi_disagree_thresh).float()

                gsum = int(gate.sum().item())
                if gsum > 0:
                    reg_j = F.cross_entropy(logits[gate.bool()], yprog_t[gate.bool()])
                    reg = reg + float(archive.w[j]) * reg_j
                    step_gate_count += gsum

            if step_gate_count > 0:
                gate_steps_nonzero += 1
            gate_total += step_gate_count

        loss = ce + alpha_t * reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        ce_vals.append(float(ce.item()))
        reg_vals.append(float(reg.item()))
        total_vals.append(float(loss.item()))

    return {
        "train_ce": float(np.mean(ce_vals)) if ce_vals else 0.0,
        "train_reg": float(np.mean(reg_vals)) if reg_vals else 0.0,
        "total_loss": float(np.mean(total_vals)) if total_vals else 0.0,
        "gate_total": int(gate_total),
        "gate_steps_nonzero": int(gate_steps_nonzero),
        "gate_total_steps": int(gate_total_steps),
        "gate_avg_per_step": float(gate_total / max(1, gate_total_steps)),
    }


@torch.no_grad()
def archive_metrics_on_probe(
    model: nn.Module,
    probe_loader: DataLoader,
    archive: PRAXISArchive,
    max_batches: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    num_programs = len(archive.programs)
    F_counts = np.zeros(num_programs, dtype=np.float64)
    F_total = np.zeros(num_programs, dtype=np.float64)

    B_counts = np.zeros(num_programs, dtype=np.float64)
    B_total = np.zeros(num_programs, dtype=np.float64)

    for bi, (x, y) in enumerate(probe_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)

        outs = batch_output_distributions(model, x)

        solver_correct = (pred == y).detach().cpu().numpy().astype(np.float64)

        for j, cand in enumerate(archive.programs):
            p = []
            for i in range(len(outs)):
                pi = cand.fn(0, bi, [outs[i]])
                pi = int(max(0, min(pi, logits.size(1) - 1)))
                p.append(pi)
            p_t = torch.tensor(p, device=DEVICE)

            acc = (p_t == y).detach().cpu().numpy().astype(np.float64)
            F_counts[j] += acc.sum()
            F_total[j] += acc.shape[0]

            agree = (p_t == pred).detach().cpu().numpy().astype(np.float64)
            disc = (1.0 - agree) * solver_correct
            B_counts[j] += disc.sum()
            B_total[j] += solver_correct.sum()

        if bi >= max_batches - 1:
            break

    Fv = np.divide(F_counts, np.maximum(F_total, 1.0))
    Bv = np.divide(B_counts, np.maximum(B_total, 1.0))
    return Fv, Bv


def objective_drift(q_new: np.ndarray, q_old: np.ndarray, obj_per_u: np.ndarray) -> Tuple[float, float]:
    q_new = np.asarray(q_new, dtype=np.float64)
    q_old = np.asarray(q_old, dtype=np.float64)
    obj = np.asarray(obj_per_u, dtype=np.float64)
    e_new = float((q_new * obj).sum())
    e_old = float((q_old * obj).sum())
    tv = float(0.5 * np.abs(q_new - q_old).sum())
    return e_new - e_old, tv


@torch.no_grad()
def mixture_eval(
    model: nn.Module,
    universes: List[Universe],
    q_np: np.ndarray,
    test_dir: str,
    batch_size: int,
    num_classes: int,
    seed: int,
    max_batches: int = 5,
) -> float:
    model.eval()
    total = 0.0
    for i, u in enumerate(universes):
        if q_np[i] < 1e-6:
            continue
        loader = build_test_loader(
            test_dir,
            size_test=u.size,
            batch_size=batch_size,
            num_classes=num_classes,
            seed=seed + i,
        )
        acc = eval_accuracy(model, loader, max_batches=max_batches)
        total += q_np[i] * acc
    return float(total)


def symbolic_reward_term(archive: PRAXISArchive) -> float:
    w = np.asarray(archive.w, dtype=np.float64)
    if w.size == 0:
        return 0.0
    max_w = float(w.max())
    ent = PRAXISArchive.entropy(w)
    num_programs = max(1, int(w.size))
    ent_norm = float(ent / max(math.log(float(num_programs) + 1e-12), 1e-12))
    return float(max_w - ent_norm)


@torch.no_grad()
def eval_over_universes(
    model: nn.Module,
    universes: List[Universe],
    test_dir: str,
    batch_size: int,
    num_classes: int,
    seed: int,
    size_test: int,
    max_batches: int,
) -> np.ndarray:
    accs = []
    for i, u in enumerate(universes):
        loader = build_test_loader(
            test_dir,
            size_test=size_test,
            batch_size=batch_size,
            num_classes=num_classes,
            seed=seed + i,
        )
        acc = eval_accuracy(model, loader, max_batches=max_batches)
        accs.append(acc)
    return np.array(accs, dtype=np.float64)

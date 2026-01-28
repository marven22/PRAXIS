from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import build_test_loader
from .universe import Universe
from praxis.utils import DEVICE


@torch.no_grad()
def eval_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        if max_batches is not None and batch_idx >= max_batches:
            break
    return correct / max(1, total)


@torch.no_grad()
def eval_over_universes(
    model: torch.nn.Module,
    universes: Iterable[Universe],
    test_dir: str,
    batch_size: int,
    num_classes: int,
    seed: int,
    size_test: int,
    max_batches: Optional[int],
) -> np.ndarray:
    accs = []
    for i, _ in enumerate(universes):
        loader = build_test_loader(
            test_dir,
            size=size_test,
            batch_size=batch_size,
            num_classes=num_classes,
            seed=seed + i,
        )
        accs.append(eval_accuracy(model, loader, max_batches))
    return np.array(accs, dtype=np.float64)

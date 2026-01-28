from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


class GroupDRO:
    def __init__(self, num_groups: int, eta: float = 0.1, seed: int = 0) -> None:
        self.weights = np.ones(num_groups, dtype=np.float64)
        self.eta = eta
        self.rng = np.random.RandomState(seed)

    def probs(self) -> np.ndarray:
        return self.weights / (self.weights.sum() + 1e-12)

    def sample(self, n: int) -> Tuple[list[int], np.ndarray]:
        q = self.probs()
        idxs = self.rng.choice(len(q), size=n, p=q)
        return idxs.tolist(), q

    def update(self, idxs: Iterable[int], losses: Iterable[float]) -> None:
        for i, loss in zip(idxs, losses):
            self.weights[i] *= math.exp(self.eta * float(loss))

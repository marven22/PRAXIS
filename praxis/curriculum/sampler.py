from __future__ import annotations

import numpy as np


class CurriculumSampler:
    def __init__(self, difficulties: np.ndarray, iters: int, group_size: int) -> None:
        self.difficulties = np.asarray(difficulties, dtype=np.float64)
        self.order = np.argsort(self.difficulties)
        self.iters = int(iters)
        self.group_size = int(group_size)

    def sample(self, step: int) -> list[int]:
        frac = min(1.0, (step + 1) / max(1, self.iters))
        k = max(self.group_size, int(frac * len(self.order)))
        eligible = self.order[:k]
        return np.random.choice(eligible, size=self.group_size, replace=True).tolist()

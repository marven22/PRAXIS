from __future__ import annotations

import numpy as np


class EXP3:
    def __init__(self, num_arms: int, gamma: float = 0.1, seed: int = 0) -> None:
        self.num_arms = num_arms
        self.gamma = gamma
        self.weights = np.ones(num_arms, dtype=np.float64)
        self.rng = np.random.RandomState(seed)

    def probs(self) -> np.ndarray:
        weight_sum = self.weights.sum()
        return (1 - self.gamma) * (self.weights / weight_sum) + self.gamma / self.num_arms

    def sample(self, count: int) -> tuple[list[int], np.ndarray]:
        probs = self.probs()
        idxs = self.rng.choice(self.num_arms, size=count, p=probs)
        return idxs.tolist(), probs

    def update(self, idxs: list[int], rewards: list[float]) -> None:
        probs = self.probs()
        for idx, reward in zip(idxs, rewards):
            reward_hat = reward / max(probs[idx], 1e-12)
            self.weights[idx] *= np.exp(self.gamma * reward_hat / self.num_arms)

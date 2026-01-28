"""Generator model over discrete universes."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteGenerator(nn.Module):
    def __init__(self, num_universes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_universes))

    def q(self, temperature: float = 1.0) -> torch.Tensor:
        return F.softmax(self.logits / max(temperature, 1e-6), dim=0)

    def sample(self, K: int, temperature: float = 1.0) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        q = self.q(temperature)
        dist = torch.distributions.Categorical(probs=q)
        idx = dist.sample((K,))
        logp = dist.log_prob(idx)
        return idx.tolist(), logp, q

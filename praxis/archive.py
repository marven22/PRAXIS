"""Archive data structures for PRAXIS programs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class ProgramCandidate:
    name: str
    fn: Callable
    family: str = "base"
    meta: Optional[Dict[str, Any]] = None


class PRAXISArchive:
    def __init__(
        self,
        programs: List[ProgramCandidate],
        beta: float,
        eta_p: float,
        gamma: float,
        max_programs: int = 64,
    ) -> None:
        self.programs = list(programs)
        self.beta = beta
        self.eta_p = eta_p
        self.gamma = gamma
        self.max_programs = int(max_programs)

        self.s = np.zeros(len(self.programs), dtype=np.float64)
        self.w = self._softmax(self.beta * self.s)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-12)

    @staticmethod
    def entropy(p: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float64)
        return float(-(p * np.log(p + 1e-12)).sum())

    def _recompute_weights(self) -> None:
        self.w = self._softmax(self.beta * self.s)

    def maybe_add_programs(self, new_programs: List[ProgramCandidate]) -> Dict[str, Any]:
        added = []
        skipped = []
        name_set = {p.name for p in self.programs}

        for p in new_programs:
            if len(self.programs) >= self.max_programs:
                skipped.append({"name": p.name, "reason": "max_programs_cap"})
                continue
            if p.name in name_set:
                skipped.append({"name": p.name, "reason": "duplicate_name"})
                continue
            self.programs.append(p)
            name_set.add(p.name)
            self.s = np.concatenate([self.s, np.zeros(1, dtype=np.float64)], axis=0)
            added.append(p.name)

        self._recompute_weights()
        return {"added": added, "skipped": skipped, "total": len(self.programs)}

    def update(self, F_vec: np.ndarray, B_vec: np.ndarray) -> Dict[str, Any]:
        U = F_vec - self.gamma * B_vec
        self.s = self.s + self.eta_p * U
        self._recompute_weights()

        j_star = int(np.argmax(self.w))
        margins = [float(U[j_star] - U[j]) for j in range(len(U)) if j != j_star]
        margin = float(min(margins)) if margins else 0.0

        return {
            "F": {self.programs[i].name: float(F_vec[i]) for i in range(len(self.programs))},
            "B": {self.programs[i].name: float(B_vec[i]) for i in range(len(self.programs))},
            "U": {self.programs[i].name: float(U[i]) for i in range(len(self.programs))},
            "s": {self.programs[i].name: float(self.s[i]) for i in range(len(self.programs))},
            "w": {self.programs[i].name: float(self.w[i]) for i in range(len(self.programs))},
            "entropy": self.entropy(self.w),
            "max_w": float(np.max(self.w)),
            "u_star": self.programs[j_star].name,
            "margin": margin,
        }

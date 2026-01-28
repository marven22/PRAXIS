from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Universe:
    size: int
    aug: float
    corruption: str
    severity: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "size": self.size,
            "aug": self.aug,
            "corruption": self.corruption,
            "severity": self.severity,
        }


def universe_key(u: Universe) -> Tuple[int, float, str, float]:
    return (u.size, round(u.aug, 6), u.corruption, round(u.severity, 6))


def build_universe_grid(
    sizes: Iterable[int],
    augs: Iterable[float],
    corruptions: Iterable[str],
    severities: Iterable[float],
) -> List[Universe]:
    return [
        Universe(int(s), float(a), str(c), float(v))
        for s in sizes
        for a in augs
        for c in corruptions
        for v in severities
    ]


def assert_disjoint(train: List[Universe], heldout: List[Universe]) -> None:
    train_keys = {universe_key(u) for u in train}
    heldout_keys = {universe_key(u) for u in heldout}
    overlap = train_keys & heldout_keys
    if overlap:
        raise ValueError(
            "Held-out universes overlap with training universes: "
            f"{list(overlap)[:5]}"
        )

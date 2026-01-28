"""Universe grid definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class Universe:
    size: int
    aug_strength: float
    corruption: str
    severity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "aug_strength": self.aug_strength,
            "corruption": self.corruption,
            "severity": self.severity,
        }


def universe_key(u: Universe) -> Tuple[int, float, str, float]:
    return (int(u.size), float(u.aug_strength), str(u.corruption), float(u.severity))


def assert_no_overlap(train_universes: Iterable[Universe], heldout_universes: Iterable[Universe]) -> None:
    train_keys = {universe_key(u) for u in train_universes}
    held_keys = {universe_key(u) for u in heldout_universes}
    inter = train_keys.intersection(held_keys)
    if inter:
        raise ValueError(
            f"[UniverseOverlap] {len(inter)} overlapping universes between train and heldout. "
            f"Examples: {list(inter)[:5]}"
        )

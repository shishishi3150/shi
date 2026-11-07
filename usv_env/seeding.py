from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EpisodeSeeds:
    master_seed: int
    scenario_seed: int
    wave_seed: int
    communication_seed: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def derive_episode_seeds(master_seed: int) -> EpisodeSeeds:
    master = int(master_seed)
    if master < 0:
        raise ValueError("master_seed must be non-negative")
    children = np.random.SeedSequence(master).spawn(3)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return EpisodeSeeds(master, values[0], values[1], values[2])


def episode_seed_records(master_seeds: Iterable[int]) -> list[dict[str, int]]:
    return [derive_episode_seeds(seed).as_dict() for seed in master_seeds]

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class WaveEpisode:
    height_m: float
    wavelength_m: float
    direction_rad: float
    initial_phase_rad: float
    seed: int


class RegularLinearWave:
    """Regular linear artificial-lake wave; it supplies water velocity only."""

    def __init__(self, episode: WaveEpisode, gravity: float = 9.81) -> None:
        self.episode = episode
        self.amplitude = episode.height_m / 2.0
        self.wavenumber = 2.0 * math.pi / episode.wavelength_m
        self.omega = math.sqrt(gravity * self.wavenumber)
        self.direction = np.asarray(
            (math.cos(episode.direction_rad), math.sin(episode.direction_rad)), dtype=np.float64
        )

    @classmethod
    def sample(cls, config: Mapping[str, Any], seed: int) -> "RegularLinearWave":
        rng = np.random.default_rng(int(seed))
        height = config.get("height_range_m", (0.0, 0.04))
        wavelength = config.get("wavelength_range_m", (0.94125, 2.51))
        direction = config.get("direction_range_degrees", (0., 360.))
        phase = config.get("initial_phase_range_rad", (0., 2. * math.pi))
        for name, bounds in (("height", height), ("wavelength", wavelength), ("direction", direction), ("phase", phase)):
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"invalid wave {name} range")
        if height[0] < 0 or wavelength[0] <= 0: raise ValueError("invalid wave dimensions")
        if config.get("submarine_direct_wave_coupling", 0.) != 0.:
            raise ValueError("nonzero AUV direct wave coupling is not implemented")
        if config.get("model", "regular_linear") != "regular_linear": raise ValueError("unknown wave model")
        return cls(WaveEpisode(float(rng.uniform(*height)), float(rng.uniform(*wavelength)),
                               math.radians(float(rng.uniform(*direction))),
                               float(rng.uniform(*phase)), int(seed)))

    def phase(self, position: np.ndarray, time_s: float) -> float:
        return float(self.wavenumber * (self.direction @ np.asarray(position, dtype=np.float64))
                     - self.omega * time_s + self.episode.initial_phase_rad)

    def horizontal_velocity(self, position: np.ndarray, time_s: float) -> np.ndarray:
        return (self.amplitude * self.omega * math.cos(self.phase(position, time_s)) * self.direction).astype(np.float64)

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.omega

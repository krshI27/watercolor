from dataclasses import dataclass
from typing import Tuple


@dataclass
class WatercolorConfig:
    num_colors: int = 5
    output_size: Tuple[int, int] = (200, 200)
    diffusion_rate: float = 0.1
    iterations: int = 10
    num_points_factor: int = 300
    max_distance_factor: int = 20
    falloff_base: float = 0.2
    branch_probability: float = 0.15
    branch_length_factor: float = 0.3
    jitter_sigma: float = 0.2
    debug: bool = False
    num_workers: int = -1  # -1 means use all available cores

#!/usr/bin/env python3
"""
Watercolor Simulation Package

This package implements the watercolor simulation based on the paper:
'Computer-Generated Watercolor' by Curtis et al.

The simulation is organized according to the paper's structure:
1. Properties of watercolor
2. Computer-generated watercolor
3. The fluid simulation
4. Rendering the pigmented layers
5. Applications
"""

from .paper import PaperModel
from .fluid_simulation import WaterSimulation
from .pigment import Pigment
from .kubelka_munk import KubelkaMunk
from .renderer import WatercolorRenderer

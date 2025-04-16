#!/usr/bin/env python3
"""
Watercolor simulation package.
Based on 'Computer-Generated Watercolor' by Curtis et al.
"""
from .main import *
from .paper import Paper
from .fluid_simulation import FluidSimulation
from .pigment import Pigment
from .kubelka_munk import KubelkaMunk
from .renderer import WatercolorRenderer
from .watercolor_simulation import WatercolorSimulation

__all__ = [
    'Paper',
    'FluidSimulation',
    'Pigment',
    'KubelkaMunk',
    'WatercolorRenderer',
    'WatercolorSimulation'
]

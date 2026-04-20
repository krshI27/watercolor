"""Watercolor simulation package.

Production API: ``watercolorize(image, **params)`` — pragmatic pigment-cluster
bleed + edge darkening + paper texture. Runs in seconds.

Research physics code (Navier-Stokes + Kubelka-Munk) lives in
``watercolor.simulation`` for slower, higher-fidelity runs.
"""

from .watercolorize import watercolorize, watercolorize_array

__all__ = ["watercolorize", "watercolorize_array"]

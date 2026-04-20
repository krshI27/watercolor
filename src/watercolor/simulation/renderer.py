#!/usr/bin/env python3
"""
Watercolor rendering module.
Implements Kubelka-Munk optical model from Section 5 of the paper.
"""

import numpy as np
import cv2
from .fluid_simulation import FluidSimulation
from .kubelka_munk import KubelkaMunk


class Renderer:
    """Base renderer: holds output buffer and optional paper texture."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.paper_texture: np.ndarray | None = None
        self.output_buffer = np.ones((height, width, 3), dtype=np.float32)

    def load_texture(self, texture_path, texture_type: str = "canvas") -> None:
        """Load a subtle paper texture from an image path.

        Passing ``None`` clears the texture. Loaded textures are converted to
        grayscale, resized to the renderer grid, and normalized to [0.95, 1.0]
        so they modulate the final render gently.
        """
        if texture_path is None:
            self.paper_texture = None
            return

        img = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load texture: {texture_path}")

        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        # Map to [0.95, 1.0] so the texture is subtle regardless of source contrast.
        lo, hi = float(img.min()), float(img.max())
        if hi - lo < 1e-6:
            normalized = np.full_like(img, 0.975, dtype=np.float32)
        else:
            normalized = 0.95 + 0.05 * (img - lo) / (hi - lo)

        self.paper_texture = normalized


class WatercolorRenderer:
    """Renders watercolor simulation results using Kubelka-Munk model."""

    def __init__(self, simulation):
        self.simulation = simulation
        self.km = KubelkaMunk()

    def render_pigment(self, pigment_idx: int) -> np.ndarray:
        """
        Render a single pigment using Kubelka-Munk equations.
        Returns RGB color array.
        """
        # Get pigment properties
        km_params = self.simulation.pigment_properties[pigment_idx][
            "kubelka_munk_params"
        ]

        # Get total pigment thickness (water + paper)
        thickness = (
            self.simulation.pigment_water[pigment_idx]
            + self.simulation.pigment_paper[pigment_idx]
        )

        # Calculate reflectance using K-M model
        R, T = self.km.compute_layer_optics(km_params["K"], km_params["S"], thickness)

        return R

    def render_all_pigments(self) -> np.ndarray:
        """
        Render all pigments using K-M optical compositing.
        Returns final RGB image.
        """
        background = np.ones(3)  # White background
        height, width = self.simulation.pigment_paper[0].shape
        result = np.ones((height, width, 3))  # Initialize with white

        # Process each pixel individually to avoid broadcasting issues
        for i in range(height):
            for j in range(width):
                # Check if there's any pigment at this pixel
                has_pigment = False

                # Prepare glazes for this pixel
                glazes = []

                for idx in range(len(self.simulation.pigment_properties)):
                    km_params = self.simulation.pigment_properties[idx][
                        "kubelka_munk_params"
                    ]
                    thickness_value = (
                        self.simulation.pigment_water[idx][i, j]
                        + self.simulation.pigment_paper[idx][i, j]
                    )

                    # Only add glazes with visible pigment
                    if thickness_value > 0.001:
                        has_pigment = True
                        glazes.append(
                            {
                                "K": km_params["K"],
                                "S": km_params["S"],
                                "thickness": thickness_value,
                            }
                        )

                # If there's pigment at this pixel, compute the color
                if has_pigment:
                    # Start with white background
                    R_total = background
                    T_total = np.ones_like(background)  # Background transmittance is 1

                    # Process glazes from bottom to top (same as test)
                    for (
                        glaze
                    ) in glazes:  # Not reversed - matches test's bottom-up order
                        # Get R, T for current glaze
                        R_glaze, T_glaze = self.km.get_reflectance_transmittance(
                            glaze["K"], glaze["S"], glaze["thickness"]
                        )

                        # Use same compositing formula as test
                        denom = 1.0 - R_glaze * R_total + 1e-10  # Same epsilon as test
                        R_total = R_glaze + (T_glaze * T_glaze * R_total) / denom

                    # Store result for this pixel
                    result[i, j] = R_total

        return result

#!/usr/bin/env python3
"""
Watercolor rendering module.
Implements Kubelka-Munk optical model from Section 5 of the paper.
"""

import numpy as np
from .fluid_simulation import FluidSimulation
from .kubelka_munk import KubelkaMunk

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
        km_params = self.simulation.pigment_properties[pigment_idx]['kubelka_munk_params']
        
        # Get total pigment thickness (water + paper)
        thickness = (
            self.simulation.pigment_water[pigment_idx] +
            self.simulation.pigment_paper[pigment_idx]
        )
        
        # Calculate reflectance using K-M model
        R, T = self.km.compute_layer_optics(
            km_params['K'],
            km_params['S'],
            thickness
        )
        
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
                    km_params = self.simulation.pigment_properties[idx]['kubelka_munk_params']
                    thickness_value = (self.simulation.pigment_water[idx][i, j] +
                                      self.simulation.pigment_paper[idx][i, j])
                    
                    # Only add glazes with visible pigment
                    if thickness_value > 0.001:
                        has_pigment = True
                        glazes.append({
                            'K': km_params['K'],
                            'S': km_params['S'],
                            'thickness': thickness_value
                        })
                
                # If there's pigment at this pixel, compute the color
                if has_pigment:
                    # Use render_glazes to correctly handle the compositing
                    R_total = background
                    T_total = np.zeros_like(background)
                    
                    # Composite each glaze, from bottom to top
                    for glaze in reversed(glazes):
                        K = glaze['K']
                        S = glaze['S']
                        thickness = glaze['thickness']
                        
                        # Get reflectance and transmittance of this glaze
                        R, T = self.km.get_reflectance_transmittance(K, S, thickness)
                        
                        # Composite with already computed layers
                        if np.all(T_total == 0):  # First layer (on background)
                            R_total = R + (T**2 * R_total) / (1.0 - R * R_total + 1e-10)
                            T_total = T
                        else:
                            R_new, T_new = self.km.composite_layers(R, T, R_total, T_total)
                            R_total, T_total = R_new, T_new
                    
                    # Store result for this pixel
                    result[i, j] = R_total
        
        return result

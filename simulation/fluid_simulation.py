#!/usr/bin/env python3
"""
Fluid simulation module based on shallow water equations.
From Section 4.3 of 'Computer-Generated Watercolor'.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from .paper import Paper

class FluidSimulation:
    """
    Implements shallow water equations discretized on a staggered grid.
    Handles:
    - Velocity field updates based on water pressure and paper slope
    - Divergence relaxation for fluid conservation
    - Edge darkening effects at boundaries
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # Initialize fields
        self.u = np.zeros((height, width+1))  # x velocity at cell boundaries
        self.v = np.zeros((height+1, width))  # y velocity at cell boundaries
        self.p = np.zeros((height, width))    # pressure at cell centers
        
        # Parameters from paper
        self.viscosity = 0.1      # μ
        self.viscous_drag = 0.01  # κ
        self.edge_darkening = 0.03 # η
        
    def update_velocities(self, paper: Paper, wet_mask: np.ndarray, dt: float = 0.1):
        """
        Update water velocities based on pressure gradients and paper slope.
        Equations (1) and (2) from Section 4.3.
        """
        # Get paper slope
        dy, dx = paper.slope
        
        # Add slope effect
        self.u -= dx
        self.v -= dy
        
        # Advection and viscosity terms
        u_grad = np.gradient(self.u**2)
        v_grad = np.gradient(self.v**2)
        uv_grad = np.gradient(self.u * self.v)
        
        laplace_u = self._laplacian(self.u)
        laplace_v = self._laplacian(self.v)
        
        # Update velocities
        self.u += dt * (
            -u_grad[0] - uv_grad[1] +  # Advection
            self.viscosity * laplace_u - # Viscosity
            np.gradient(self.p)[1] -    # Pressure
            self.viscous_drag * self.u   # Drag
        )
        
        self.v += dt * (
            -v_grad[1] - uv_grad[0] +
            self.viscosity * laplace_v -
            np.gradient(self.p)[0] -
            self.viscous_drag * self.v
        )
        
        # Enforce boundary conditions
        self._enforce_boundaries(wet_mask)
        
    def relax_divergence(self, wet_mask: np.ndarray, iterations: int = 50, tolerance: float = 0.01):
        """
        Relax divergence of velocity field to ensure fluid conservation.
        Section 4.3.2 of the paper.
        """
        for _ in range(iterations):
            # Calculate divergence
            div = self._divergence()
            
            if np.abs(div).max() < tolerance:
                break
                
            # Update pressure to reduce divergence
            self.p += 0.1 * div * wet_mask
            
            # Update velocities
            self.u -= np.gradient(self.p)[1]
            self.v -= np.gradient(self.p)[0]
            
            self._enforce_boundaries(wet_mask)
            
    def flow_outward(self, wet_mask: np.ndarray, kernel_size: int = 10):
        """
        Create outward flow near edges for edge darkening effect.
        Equation (3) from Section 4.3.3.
        """
        # Gaussian blur of wet mask
        blurred = gaussian_filter(wet_mask.astype(float), sigma=kernel_size/6)
        
        # Decrease pressure near edges
        self.p -= self.edge_darkening * (1 - blurred) * wet_mask
        
    def _enforce_boundaries(self, wet_mask: np.ndarray):
        """Set velocity to zero at boundaries of wet regions."""
        # Pad mask for boundary checks
        mask_padded = np.pad(wet_mask, 1)
        
        # Zero velocities where either adjacent cell is dry
        self.u *= (mask_padded[1:-1, :-1] & mask_padded[1:-1, 1:])
        self.v *= (mask_padded[:-1, 1:-1] & mask_padded[1:, 1:-1])
        
    def _divergence(self) -> np.ndarray:
        """Calculate divergence of velocity field."""
        return (
            np.gradient(self.u)[1] +
            np.gradient(self.v)[0]
        )
        
    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Calculate Laplacian of a field using finite differences."""
        return (
            np.roll(field, 1, axis=0) +
            np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) +
            np.roll(field, -1, axis=1) -
            4 * field
        )

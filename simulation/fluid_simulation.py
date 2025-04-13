#!/usr/bin/env python3
"""
Fluid Simulation Module

This module implements the fluid simulation for watercolor as described in
Section 4 (The Fluid Simulation) of the 'Computer-Generated Watercolor' paper.

It implements the three-layer model:
1. Shallow-water layer - where water and pigment flow above the surface of the paper
2. Pigment-deposition layer - where pigment is deposited onto and lifted from the paper
3. Capillary layer - where water absorbed into the paper is diffused by capillary action
"""

import numpy as np
import scipy.ndimage as ndimage
from typing import List, Optional, Tuple, Dict, Any

from .paper import PaperModel
from .pigment import PigmentLayer


class WaterSimulation:
    """
    Main class for watercolor fluid simulation based on the paper:
    'Computer-Generated Watercolor' by Curtis et al.
    
    This class implements the fluid simulation described in Section 4 of the paper,
    including the shallow-water model, pigment movement, and capillary flow.
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize the watercolor fluid simulation.
        
        Parameters:
        -----------
        width : int
            Width of the simulation canvas
        height : int
            Height of the simulation canvas
        """
        self.width = width
        self.height = height
        
        # Initialize paper model
        self.paper = PaperModel(width, height)
        
        # Default physical parameter values (Section 4)
        self.viscosity = 0.1  # μ (mu) - Section 4.3
        self.viscous_drag = 0.01  # κ (kappa) - Section 4.3
        self.edge_darkening_factor = 0.03  # η (eta) - Section 4.3.3
        self.edge_darkening_kernel_size = 10  # K - Section 4.3.3
        
        # Capillary flow parameters (Section 4.6)
        self.absorption_rate = 0.1  # α (alpha)
        self.min_saturation_for_diffusion = 0.05  # ε (epsilon)
        self.min_saturation_to_receive = 0.01  # δ (delta)
        self.saturation_threshold = 0.5  # σ (sigma)
        
        # Initialize simulation layers
        self.pigment_layers = []
        self.reset()
    
    def reset(self):
        """Reset the simulation to its initial state."""
        # Shallow water layer (Section 4.3)
        self.wet_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.velocity_u = np.zeros((self.height, self.width + 1), dtype=np.float32)  # staggered grid
        self.velocity_v = np.zeros((self.height + 1, self.width), dtype=np.float32)  # staggered grid
        self.pressure = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Capillary layer (Section 4.6)
        self.water_saturation = np.zeros((self.height, self.width), dtype=np.float32)  # s
        
        # Reset pigment layers
        for layer in self.pigment_layers:
            layer.water_concentration.fill(0.0)
            layer.paper_concentration.fill(0.0)
    
    def add_pigment_layer(self, pigment_layer: PigmentLayer) -> int:
        """
        Add a new pigment layer to the simulation.
        
        Parameters:
        -----------
        pigment_layer : PigmentLayer
            The pigment layer to add
            
        Returns:
        --------
        int
            Index of the newly added pigment layer
        """
        self.pigment_layers.append(pigment_layer)
        return len(self.pigment_layers) - 1
    
    def set_wet_mask(self, mask: np.ndarray) -> None:
        """
        Set the wet area mask.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask defining wet areas (1 for wet, 0 for dry)
        """
        self.wet_mask = mask.astype(np.float32)
    
    def set_pressure(self, mask: np.ndarray, value: float) -> None:
        """
        Set water pressure in specified areas.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask defining areas to set pressure
        value : float
            Pressure value to set
        """
        self.pressure[mask > 0] = value
    
    def apply_drybrush(self, threshold: float = 0.7) -> None:
        """
        Apply drybrush effect by restricting wet areas to only high points on the paper.
        Section 4.7 of the paper.
        
        Parameters:
        -----------
        threshold : float
            Height threshold below which the paper is considered dry
        """
        if self.paper.height_field is None:
            raise ValueError("Paper height field must be generated before applying drybrush")
            
        self.wet_mask[self.paper.height_field < threshold] = 0.0
    
    def enforce_boundary_conditions(self) -> None:
        """
        Enforce boundary conditions on velocity at the edges of the wet mask.
        Sets velocity to zero at boundaries of the wet area.
        Part of Section 4.3.1.
        """
        # Create a padded version of the wet mask for checking neighbors
        padded_mask = np.pad(self.wet_mask, 1, mode='constant', constant_values=0)
        
        # Check for u velocity (horizontal) boundaries
        for i in range(self.height):
            for j in range(self.width + 1):
                # Check if this is a boundary velocity
                left_idx = max(0, j-1)
                right_idx = min(self.width-1, j)
                
                if padded_mask[i+1, left_idx+1] == 0 or padded_mask[i+1, right_idx+1] == 0:
                    self.velocity_u[i, j] = 0.0
        
        # Check for v velocity (vertical) boundaries
        for i in range(self.height + 1):
            for j in range(self.width):
                # Check if this is a boundary velocity
                top_idx = max(0, i-1)
                bottom_idx = min(self.height-1, i)
                
                if padded_mask[top_idx+1, j+1] == 0 or padded_mask[bottom_idx+1, j+1] == 0:
                    self.velocity_v[i, j] = 0.0
    
    def update_velocities(self) -> None:
        """
        Update water velocities based on the shallow water equations.
        This is an implementation of the UpdateVelocities() procedure from Section 4.3.1.
        """
        # Get paper slope (gradient of height field)
        dx, dy = self.paper.compute_slope()
        
        # Adjust velocity by paper slope
        for i in range(self.height):
            for j in range(self.width):
                if j < self.width - 1:
                    self.velocity_u[i, j+1] -= 0.5 * (dx[i, j] + dx[i, min(j+1, self.width-1)])
                if i < self.height - 1:
                    self.velocity_v[i+1, j] -= 0.5 * (dy[i, j] + dy[min(i+1, self.height-1), j])
        
        # Enforce boundary conditions after adjusting for slope
        self.enforce_boundary_conditions()
        
        # Calculate adaptive time step to ensure stability
        max_velocity = max(
            np.max(np.abs(self.velocity_u)),
            np.max(np.abs(self.velocity_v))
        )
        dt = 1.0 / max(1, np.ceil(max_velocity))
        
        # Use a staggered grid for the update as in the paper
        # Initialize temporary arrays for the update
        new_u = self.velocity_u.copy()
        new_v = self.velocity_v.copy()
        
        # Update velocities according to discretized version of equations (1) and (2) from the paper
        for t in np.arange(0, 1.0, dt):
            for i in range(self.height):
                for j in range(1, self.width):
                    # Only update velocities in wet areas
                    left_wet = j > 0 and self.wet_mask[i, j-1] > 0
                    right_wet = j < self.width and self.wet_mask[i, j] > 0
                    
                    if left_wet or right_wet:
                        # Get u at cell centers
                        u_i_j = 0.5 * (self.velocity_u[i, j-1] + self.velocity_u[i, j])
                        u_ip1_j = 0.5 * (self.velocity_u[i, j] + self.velocity_u[i, min(j+1, self.width)])
                        
                        # Get v at this u-velocity position
                        v_ip5_jm5 = 0.25 * (self.velocity_v[i, j-1] + self.velocity_v[i+1, j-1] + 
                                           self.velocity_v[i, j] + self.velocity_v[i+1, j]) if i < self.height - 1 else 0
                        v_ip5_jp5 = 0.25 * (self.velocity_v[i, j] + self.velocity_v[i+1, j] + 
                                           self.velocity_v[i, min(j+1, self.width-1)] + 
                                           self.velocity_v[i+1, min(j+1, self.width-1)]) if i < self.height - 1 else 0
                        
                        # Advection term
                        A = (u_i_j**2 - u_ip1_j**2) + ((u_i_j * v_ip5_jm5) - (u_ip1_j * v_ip5_jp5))
                        
                        # Diffusion (viscosity) term
                        B = 0
                        if j > 0 and j < self.width - 1 and i > 0 and i < self.height - 1:
                            B = (self.velocity_u[i, j+1] + self.velocity_u[i, j-1] + 
                                 self.velocity_u[i+1, j] + self.velocity_u[i-1, j] - 
                                 4.0 * self.velocity_u[i, j])
                        
                        # Pressure term
                        p_left = self.pressure[i, j-1] if j > 0 else 0
                        p_right = self.pressure[i, j] if j < self.width else 0
                        
                        # Update velocity
                        new_u[i, j] = self.velocity_u[i, j] + dt * (
                            A - 
                            self.viscosity * B + 
                            p_left - p_right - 
                            self.viscous_drag * self.velocity_u[i, j]
                        )
            
            for i in range(1, self.height):
                for j in range(self.width):
                    # Only update velocities in wet areas
                    top_wet = i > 0 and self.wet_mask[i-1, j] > 0
                    bottom_wet = i < self.height and self.wet_mask[i, j] > 0
                    
                    if top_wet or bottom_wet:
                        # Get v at cell centers
                        v_i_j = 0.5 * (self.velocity_v[i-1, j] + self.velocity_v[i, j])
                        v_i_jp1 = 0.5 * (self.velocity_v[i, j] + self.velocity_v[i, min(j+1, self.width-1)])
                        
                        # Get u at this v-velocity position
                        u_im5_jp5 = 0.25 * (self.velocity_u[i-1, j] + self.velocity_u[i-1, j+1] + 
                                           self.velocity_u[i, j] + self.velocity_u[i, j+1]) if j < self.width - 1 else 0
                        u_ip5_jp5 = 0.25 * (self.velocity_u[i, j] + self.velocity_u[i, j+1] + 
                                           self.velocity_u[min(i+1, self.height-1), j] + 
                                           self.velocity_u[min(i+1, self.height-1), j+1]) if j < self.width - 1 else 0
                        
                        # Advection term
                        A = (v_i_j**2 - v_i_jp1**2) + ((v_i_j * u_im5_jp5) - (v_i_jp1 * u_ip5_jp5))
                        
                        # Diffusion (viscosity) term
                        B = 0
                        if i > 0 and i < self.height - 1 and j > 0 and j < self.width - 1:
                            B = (self.velocity_v[i+1, j] + self.velocity_v[i-1, j] + 
                                 self.velocity_v[i, j+1] + self.velocity_v[i, j-1] - 
                                 4.0 * self.velocity_v[i, j])
                        
                        # Pressure term
                        p_top = self.pressure[i-1, j] if i > 0 else 0
                        p_bottom = self.pressure[i, j] if i < self.height else 0
                        
                        # Update velocity
                        new_v[i, j] = self.velocity_v[i, j] + dt * (
                            A - 
                            self.viscosity * B + 
                            p_top - p_bottom - 
                            self.viscous_drag * self.velocity_v[i, j]
                        )
            
            # Update velocity fields
            self.velocity_u = new_u.copy()
            self.velocity_v = new_v.copy()
            
            # Enforce boundary conditions
            self.enforce_boundary_conditions()
    
    def relax_divergence(self, max_iterations: int = 50, tolerance: float = 0.01, relaxation_factor: float = 0.1) -> None:
        """
        Relax the divergence of the velocity field.
        This is an implementation of the RelaxDivergence() procedure from Section 4.3.2.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of relaxation iterations
        tolerance : float
            Tolerance for convergence
        relaxation_factor : float
            Relaxation factor (ξ - xi)
        """
        for iteration in range(max_iterations):
            max_divergence = 0.0
            
            # Calculate divergence at each cell and redistribute fluid
            for i in range(self.height):
                for j in range(self.width):
                    if self.wet_mask[i, j] > 0:
                        # Calculate divergence (inflow - outflow)
                        divergence = (
                            self.velocity_u[i, j+1] - self.velocity_u[i, j] +
                            self.velocity_v[i+1, j] - self.velocity_v[i, j]
                        )
                        
                        max_divergence = max(max_divergence, abs(divergence))
                        
                        # Adjust pressure based on divergence
                        self.pressure[i, j] += relaxation_factor * divergence
                        
                        # Adjust velocities to reduce divergence
                        if j > 0:
                            self.velocity_u[i, j] -= 0.25 * relaxation_factor * divergence
                        if j < self.width:
                            self.velocity_u[i, j+1] += 0.25 * relaxation_factor * divergence
                        if i > 0:
                            self.velocity_v[i, j] -= 0.25 * relaxation_factor * divergence
                        if i < self.height:
                            self.velocity_v[i+1, j] += 0.25 * relaxation_factor * divergence
            
            # Enforce boundary conditions
            self.enforce_boundary_conditions()
            
            # Check for convergence
            if max_divergence < tolerance:
                break
    
    def flow_outward(self) -> None:
        """
        Create outward flow of fluid towards the edges to produce the edge-darkening effect.
        This is an implementation of the FlowOutward() procedure from Section 4.3.3.
        """
        # Create a Gaussian-blurred version of the wet mask
        kernel_size = self.edge_darkening_kernel_size
        blurred_mask = ndimage.gaussian_filter(self.wet_mask, sigma=kernel_size/6.0)
        
        # Create mask for edge darkening effect (eq. 3 in the paper)
        # p ← p - η(1-M')M
        edge_effect = self.edge_darkening_factor * (1.0 - blurred_mask) * self.wet_mask
        
        # Apply edge darkening by decreasing pressure near edges
        self.pressure -= edge_effect
    
    def move_water(self) -> None:
        """
        Move water in the shallow-water layer.
        This is an implementation of the MoveWater() procedure from Section 4.3.
        """
        # Update water velocities based on shallow water equations
        self.update_velocities()
        
        # Relax divergence of velocity field
        self.relax_divergence()
        
        # Flow outward for edge-darkening effect
        self.flow_outward()
    
    def move_pigment(self) -> None:
        """
        Move pigments within the shallow-water layer.
        This is an implementation of the MovePigment() procedure from Section 4.4.
        """
        # Calculate adaptive time step
        max_velocity = max(
            np.max(np.abs(self.velocity_u)),
            np.max(np.abs(self.velocity_v))
        )
        dt = 1.0 / max(1, np.ceil(max_velocity))
        
        # Move each pigment
        for pigment_layer in self.pigment_layers:
            g = pigment_layer.water_concentration
            
            for t in np.arange(0, 1.0, dt):
                new_g = g.copy()
                
                # Distribute pigment according to velocity field
                for i in range(self.height):
                    for j in range(self.width):
                        if self.wet_mask[i, j] > 0:
                            # Right flow (positive u)
                            if j < self.width - 1 and self.velocity_u[i, j+1] > 0:
                                flow = self.velocity_u[i, j+1] * g[i, j]
                                new_g[i, j] -= flow
                                new_g[i, j+1] += flow
                            
                            # Left flow (negative u)
                            if j > 0 and self.velocity_u[i, j] < 0:
                                flow = -self.velocity_u[i, j] * g[i, j]
                                new_g[i, j] -= flow
                                new_g[i, j-1] += flow
                            
                            # Bottom flow (positive v)
                            if i < self.height - 1 and self.velocity_v[i+1, j] > 0:
                                flow = self.velocity_v[i+1, j] * g[i, j]
                                new_g[i, j] -= flow
                                new_g[i+1, j] += flow
                            
                            # Top flow (negative v)
                            if i > 0 and self.velocity_v[i, j] < 0:
                                flow = -self.velocity_v[i, j] * g[i, j]
                                new_g[i, j] -= flow
                                new_g[i-1, j] += flow
                
                g = new_g
            
            # Update pigment concentration
            pigment_layer.water_concentration = g
    
    def transfer_pigment(self) -> None:
        """
        Transfer pigment between the shallow-water layer and the pigment-deposition layer.
        This is an implementation of the TransferPigment() procedure from Section 4.5.
        """
        if self.paper.height_field is None:
            raise ValueError("Paper height field must be generated before transferring pigment")
            
        for pigment_layer in self.pigment_layers:
            pigment_layer.transfer_pigment(self.paper.height_field, self.wet_mask)
    
    def simulate_capillary_flow(self) -> None:
        """
        Simulate capillary flow of water through the paper.
        This is an implementation of the SimulateCapillaryFlow() procedure from Section 4.6.
        """
        if self.paper.capacity_field is None:
            raise ValueError("Paper capacity field must be generated before simulating capillary flow")
            
        # Absorb water from shallow-water layer into paper
        for i in range(self.height):
            for j in range(self.width):
                if self.wet_mask[i, j] > 0:
                    # Add water to saturation (limited by capacity)
                    self.water_saturation[i, j] += max(
                        0,
                        min(self.absorption_rate, self.paper.capacity_field[i, j] - self.water_saturation[i, j])
                    )
        
        # Create a copy for updating
        new_saturation = self.water_saturation.copy()
        
        # Simulate diffusion through capillary layer
        for i in range(self.height):
            for j in range(self.width):
                # Skip cells with insufficient saturation
                if self.water_saturation[i, j] <= self.min_saturation_for_diffusion:
                    continue
                
                # Check each neighbor
                for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                    # Skip out-of-bounds neighbors
                    if ni < 0 or ni >= self.height or nj < 0 or nj >= self.width:
                        continue
                    
                    # Skip neighbors with too little saturation to receive water
                    if self.water_saturation[ni, nj] <= self.min_saturation_to_receive:
                        continue
                    
                    # Only diffuse if saturation gradient exists
                    if self.water_saturation[i, j] > self.water_saturation[ni, nj]:
                        # Calculate diffusion amount
                        delta_s = max(0, min(
                            self.water_saturation[i, j] - self.water_saturation[ni, nj],
                            self.paper.capacity_field[ni, nj] - self.water_saturation[ni, nj]
                        )) / 4.0
                        
                        # Update saturations
                        new_saturation[i, j] -= delta_s
                        new_saturation[ni, nj] += delta_s
        
        # Apply updates
        self.water_saturation = new_saturation
        
        # Expand wet-area mask where saturation exceeds threshold
        for i in range(self.height):
            for j in range(self.width):
                if self.water_saturation[i, j] > self.saturation_threshold:
                    self.wet_mask[i, j] = 1.0
    
    def main_loop(self, num_steps: int = 100) -> None:
        """
        Main simulation loop.
        This is an implementation of the MainLoop() procedure from Section 4.2.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps to simulate
        """
        if self.paper.height_field is None:
            raise ValueError("Paper must be generated before running simulation")
            
        for step in range(num_steps):
            # Move water in shallow-water layer
            self.move_water()
            
            # Move pigment within water
            self.move_pigment()
            
            # Transfer pigment between water and paper
            self.transfer_pigment()
            
            # Simulate capillary flow
            self.simulate_capillary_flow()

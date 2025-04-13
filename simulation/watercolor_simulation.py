#!/usr/bin/env python3
"""
Watercolor Simulation

This module implements the watercolor simulation based on the paper:
'Computer-Generated Watercolor' by Curtis et al.

The simulation is based on a three-layer model:
1. Shallow-water layer - where water and pigment flow above the surface of the paper
2. Pigment-deposition layer - where pigment is deposited onto and lifted from the paper
3. Capillary layer - where water absorbed into the paper is diffused by capillary action

The rendering uses the Kubelka-Munk model for optical compositing of glazes.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import scipy.ndimage as ndimage

class WatercolorSimulation:
    """
    Main class for watercolor simulation based on the paper:
    'Computer-Generated Watercolor' by Curtis et al.
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize the watercolor simulation with a given width and height.
        
        Parameters:
        -----------
        width : int
            Width of the simulation canvas
        height : int
            Height of the simulation canvas
        """
        self.width = width
        self.height = height
        
        # Default physical parameter values
        self.viscosity = 0.1  # μ (mu)
        self.viscous_drag = 0.01  # κ (kappa)
        self.edge_darkening_factor = 0.03  # η (eta)
        self.edge_darkening_kernel_size = 10
        
        # Capillary flow parameters
        self.absorption_rate = 0.1  # α (alpha)
        self.min_saturation_for_diffusion = 0.05  # ε (epsilon)
        self.min_saturation_to_receive = 0.01  # δ (delta)
        self.saturation_threshold = 0.5  # σ (sigma)
        
        # Initialize paper
        from simulation.paper import Paper
        self.paper = Paper(width, height, c_min=0.3, c_max=0.8)
        # For backwards compatibility
        self.paper_min_capacity = self.paper.c_min
        self.paper_max_capacity = self.paper.c_max
        
        # Initialize simulation layers
        self.reset()
    
    def reset(self):
        """Reset the simulation to its initial state."""
        # Shallow water layer
        self.wet_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.velocity_u = np.zeros((self.height, self.width + 1), dtype=np.float32)  # staggered grid
        self.velocity_v = np.zeros((self.height + 1, self.width), dtype=np.float32)  # staggered grid
        self.pressure = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Pigment layers
        self.pigment_water = []  # g^k - pigment in water
        self.pigment_paper = []  # d^k - pigment on paper
        self.pigment_properties = []  # density, staining power, granularity
        
        # Capillary layer
        self.water_saturation = np.zeros((self.height, self.width), dtype=np.float32)  # s
        
        # Paper is already initialized in __init__, no need to check/regenerate it
    
    def generate_paper(self, method: str = 'perlin', seed: Optional[int] = None):
        """
        Generate a paper texture as a height field.
        
        Parameters:
        -----------
        method : str
            Method to use for generating paper texture ('perlin', 'random', 'fractal')
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate paper height field
        if method == 'perlin':
            # A simple approximation of Perlin noise for this example
            # In a real implementation, would use a proper Perlin noise function
            scale = 8
            octaves = 4
            self.paper_height = np.zeros((self.height, self.width), dtype=np.float32)
            
            for octave in range(octaves):
                s = scale * (2 ** octave)
                h_samples = max(2, self.height // s)
                w_samples = max(2, self.width // s)
                
                noise = np.random.rand(h_samples, w_samples).astype(np.float32)
                noise = ndimage.zoom(noise, (self.height / h_samples, self.width / w_samples), order=1)
                self.paper_height += noise / (2 ** octave)
            
            # Normalize to [0, 1]
            self.paper_height -= np.min(self.paper_height)
            self.paper_height /= np.max(self.paper_height)
            
        elif method == 'random':
            # Simple random texture
            self.paper_height = np.random.rand(self.height, self.width).astype(np.float32)
            
        elif method == 'fractal':
            # Approximate fractal noise
            scale = 4
            octaves = 6
            self.paper_height = np.zeros((self.height, self.width), dtype=np.float32)
            
            for octave in range(octaves):
                s = scale * (2 ** octave)
                h_samples = max(2, self.height // s)
                w_samples = max(2, self.width // s)
                
                noise = np.random.rand(h_samples, w_samples).astype(np.float32)
                noise = ndimage.zoom(noise, (self.height / h_samples, self.width / w_samples), order=1)
                self.paper_height += noise / (2 ** octave)
            
            # Normalize to [0, 1]
            self.paper_height -= np.min(self.paper_height)
            self.paper_height /= np.max(self.paper_height)
        
        # Compute paper capacity from height
        self.paper_capacity = (self.paper_height * 
                              (self.paper_max_capacity - self.paper_min_capacity) + 
                              self.paper_min_capacity)
    
    def add_pigment(self, density: float = 1.0, staining_power: float = 0.5, 
                    granularity: float = 0.5, kubelka_munk_params: Dict = None):
        """
        Add a new pigment to the simulation.
        
        Parameters:
        -----------
        density : float
            Density of the pigment (ρ - rho)
        staining_power : float
            Staining power of the pigment (ω - omega)
        granularity : float
            Granularity of the pigment (γ - gamma)
        kubelka_munk_params : Dict
            Parameters for the Kubelka-Munk model (K and S coefficients)
        
        Returns:
        --------
        int
            Index of the newly added pigment
        """
        # Initialize pigment in water and on paper
        water_layer = np.zeros((self.height, self.width), dtype=np.float32)
        paper_layer = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Store pigment properties
        props = {
            'density': density,
            'staining_power': staining_power,
            'granularity': granularity,
            'kubelka_munk_params': kubelka_munk_params or {}
        }
        
        self.pigment_water.append(water_layer)
        self.pigment_paper.append(paper_layer)
        self.pigment_properties.append(props)
        
        return len(self.pigment_properties) - 1
    
    def set_pigment_water(self, pigment_idx: int, mask: np.ndarray, concentration: float = 1.0):
        """
        Set the concentration of pigment in water.
        
        Parameters:
        -----------
        pigment_idx : int
            Index of the pigment
        mask : np.ndarray
            Binary mask where to add the pigment
        concentration : float
            Concentration of the pigment
        """
        if 0 <= pigment_idx < len(self.pigment_water):
            self.pigment_water[pigment_idx][mask > 0] = concentration
            # Update wet mask where pigment is added
            self.wet_mask[mask > 0] = 1.0
    
    def set_wet_mask(self, mask: np.ndarray):
        """
        Set the wet area mask.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask defining wet areas (1 for wet, 0 for dry)
        """
        self.wet_mask = mask.astype(np.float32)
    
    def set_pressure(self, mask: np.ndarray, value: float):
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
    
    def apply_drybrush(self, threshold: float = 0.7):
        """
        Apply drybrush effect by restricting wet areas to only high points on the paper.
        
        Parameters:
        -----------
        threshold : float
            Height threshold below which the paper is considered dry
        """
        self.wet_mask[self.paper_height < threshold] = 0.0
    
    def compute_paper_slope(self):
        """
        Compute the gradient of the paper height field.
        
        Returns:
        --------
        tuple
            (dx, dy) gradient arrays
        """
        # Use Sobel filter to compute gradient
        dx = ndimage.sobel(self.paper_height, axis=1)
        dy = ndimage.sobel(self.paper_height, axis=0)
        
        # Normalize to reasonable values
        dx = dx / 8.0
        dy = dy / 8.0
        
        return dx, dy
    
    def enforce_boundary_conditions(self):
        """
        Enforce boundary conditions on velocity at the edges of the wet mask.
        Sets velocity to zero at boundaries of the wet area.
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

    def update_velocities(self):
        """
        Update water velocities based on the shallow water equations.
        This is an implementation of the UpdateVelocities() procedure from the paper.
        """
        # Get paper slope (gradient of height field)
        dx, dy = self.compute_paper_slope()
        
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
                    left_wet = self.wet_mask[i, j-1] > 0
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
                    top_wet = self.wet_mask[i-1, j] > 0
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

    def relax_divergence(self, max_iterations: int = 50, tolerance: float = 0.01, relaxation_factor: float = 0.1):
        """
        Relax the divergence of the velocity field.
        This is an implementation of the RelaxDivergence() procedure from the paper.
        
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

    def flow_outward(self):
        """
        Create outward flow of fluid towards the edges to produce the edge-darkening effect.
        This is an implementation of the FlowOutward() procedure from the paper.
        """
        # Create a Gaussian-blurred version of the wet mask
        kernel_size = self.edge_darkening_kernel_size
        blurred_mask = ndimage.gaussian_filter(self.wet_mask, sigma=kernel_size/6.0)
        
        # Create mask for edge darkening effect (eq. 3 in the paper)
        # p ← p - η(1-M')M
        edge_effect = self.edge_darkening_factor * (1.0 - blurred_mask) * self.wet_mask
        
        # Apply edge darkening by decreasing pressure near edges
        self.pressure -= edge_effect

    def move_water(self):
        """
        Move water in the shallow-water layer.
        This is an implementation of the MoveWater() procedure from the paper.
        """
        # Update water velocities based on shallow water equations
        self.update_velocities()
        
        # Relax divergence of velocity field
        self.relax_divergence()
        
        # Flow outward for edge-darkening effect
        self.flow_outward()

    def move_pigment(self):
        """
        Move pigments within the shallow-water layer.
        This is an implementation of the MovePigment() procedure from the paper.
        """
        # Calculate adaptive time step
        max_velocity = max(
            np.max(np.abs(self.velocity_u)),
            np.max(np.abs(self.velocity_v))
        )
        dt = 1.0 / max(1, np.ceil(max_velocity))
        
        # Move each pigment
        for pigment_idx in range(len(self.pigment_water)):
            g = self.pigment_water[pigment_idx]
            
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
            self.pigment_water[pigment_idx] = g

    def transfer_pigment(self):
        """
        Transfer pigment between the shallow-water layer and the pigment-deposition layer.
        This is an implementation of the TransferPigment() procedure from the paper.
        """
        for pigment_idx in range(len(self.pigment_water)):
            g = self.pigment_water[pigment_idx]
            d = self.pigment_paper[pigment_idx]
            props = self.pigment_properties[pigment_idx]
            
            density = props['density']
            staining_power = props['staining_power']
            granularity = props['granularity']
            
            for i in range(self.height):
                for j in range(self.width):
                    if self.wet_mask[i, j] > 0:
                        # Calculate pigment adsorption (water to paper)
                        delta_down = g[i, j] * (1.0 - self.paper_height[i, j] * granularity) * density
                        
                        # Calculate pigment desorption (paper to water)
                        delta_up = d[i, j] * (1.0 + (self.paper_height[i, j] - 1.0) * granularity) * density / staining_power
                        
                        # Limit transfers to keep values in [0, 1]
                        if d[i, j] + delta_down > 1.0:
                            delta_down = max(0, 1.0 - d[i, j])
                        
                        if g[i, j] + delta_up > 1.0:
                            delta_up = max(0, 1.0 - g[i, j])
                        
                        # Transfer pigment
                        d[i, j] += delta_down - delta_up
                        g[i, j] += delta_up - delta_down
            
            # Update pigment layers
            self.pigment_water[pigment_idx] = g
            self.pigment_paper[pigment_idx] = d

    def simulate_capillary_flow(self):
        """
        Simulate capillary flow of water through the paper.
        This is an implementation of the SimulateCapillaryFlow() procedure from the paper.
        """
        # Absorb water from shallow-water layer into paper
        for i in range(self.height):
            for j in range(self.width):
                if self.wet_mask[i, j] > 0:
                    # Add water to saturation (limited by capacity)
                    self.water_saturation[i, j] += max(
                        0,
                        min(self.absorption_rate, self.paper_capacity[i, j] - self.water_saturation[i, j])
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
                            self.paper_capacity[ni, nj] - self.water_saturation[ni, nj]
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

    def main_loop(self, num_steps: int = 100):
        """
        Main simulation loop.
        This is an implementation of the MainLoop() procedure from the paper.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps to simulate
        """
        for step in range(num_steps):
            # Move water in shallow-water layer
            self.move_water()
            
            # Move pigment within water
            self.move_pigment()
            
            # Transfer pigment between water and paper
            self.transfer_pigment()
            
            # Simulate capillary flow
            self.simulate_capillary_flow()
    
    def get_result(self):
        """
        Get the final pigment distribution on paper.
        
        Returns:
        --------
        list
            List of numpy arrays representing pigment distributions
        """
        return self.pigment_paper


class KubelkaMunk:
    """
    Implementation of the Kubelka-Munk model for optical compositing of pigment layers.
    """
    
    @staticmethod
    def get_coefficients_from_colors(white_color: np.ndarray, black_color: np.ndarray):
        """
        Calculate Kubelka-Munk coefficients from specified colors on white and black backgrounds.
        
        Parameters:
        -----------
        white_color : np.ndarray
            RGB color on white background (R_w), values in [0, 1]
        black_color : np.ndarray
            RGB color on black background (R_b), values in [0, 1]
        
        Returns:
        --------
        tuple
            (K, S) Kubelka-Munk absorption and scattering coefficients
        """
        # Make sure colors have valid values
        white_color = np.clip(white_color, 0.001, 0.999)
        black_color = np.clip(black_color, 0.001, np.minimum(0.999, white_color))
        
        # Calculate a and b
        a = 0.5 * (white_color + (black_color - white_color + 1.0) / black_color)
        b = np.sqrt(a**2 - 1.0)
        
        # Calculate scattering coefficient S
        S = np.arctanh((b**2 - (a - white_color) * (a - 1.0)) / (b * (1.0 - white_color))) / b
        
        # Calculate absorption coefficient K
        K = S * (a - 1.0)
        
        return K, S
    
    @staticmethod
    def get_reflectance_transmittance(K: np.ndarray, S: np.ndarray, thickness: float):
        """
        Calculate reflectance and transmittance for a pigment layer.
        
        Parameters:
        -----------
        K : np.ndarray
            Absorption coefficients
        S : np.ndarray
            Scattering coefficients
        thickness : float
            Layer thickness
        
        Returns:
        --------
        tuple
            (R, T) Reflectance and transmittance of the layer
        """
        a = 1.0 + K / S
        b = np.sqrt(a**2 - 1.0)
        bSx = b * S * thickness
        
        # Calculate sinh and cosh using exponential forms to avoid numerical issues
        sinh_bSx = (np.exp(bSx) - np.exp(-bSx)) / 2.0
        cosh_bSx = (np.exp(bSx) + np.exp(-bSx)) / 2.0
        
        c = a * sinh_bSx + b * cosh_bSx
        
        # Calculate reflectance and transmittance
        R = sinh_bSx / c
        T = b / c
        
        return R, T
    
    @staticmethod
    def composite_layers(R1: np.ndarray, T1: np.ndarray, R2: np.ndarray, T2: np.ndarray):
        """
        Composite two layers using Kubelka's optical compositing equations.
        
        Parameters:
        -----------
        R1, T1 : np.ndarray
            Reflectance and transmittance of layer 1 (top)
        R2, T2 : np.ndarray
            Reflectance and transmittance of layer 2 (bottom)
        
        Returns:
        --------
        tuple
            (R, T) Reflectance and transmittance of the composite
        """
        # Avoid division by zero
        denominator = 1.0 - R1 * R2
        denominator = np.where(denominator == 0, 1e-6, denominator)
        
        # Calculate composite reflectance and transmittance
        R = R1 + (T1**2 * R2) / denominator
        T = (T1 * T2) / denominator
        
        return R, T
    
    @staticmethod
    def render_glazes(glazes: List[Dict], background_color: np.ndarray = np.array([1.0, 1.0, 1.0])):
        """
        Render a stack of glazes on a background.
        
        Parameters:
        -----------
        glazes : List[Dict]
            List of glazes, each with K, S, thickness properties
        background_color : np.ndarray
            RGB color of the background
        
        Returns:
        --------
        np.ndarray
            Final RGB color after compositing all glazes
        """
        # Start with a perfect reflector (background)
        R_total = background_color
        T_total = np.zeros_like(background_color)
        
        # Composite each glaze, from bottom to top
        for glaze in reversed(glazes):
            K = glaze['K']
            S = glaze['S']
            thickness = glaze['thickness']
            
            # Get reflectance and transmittance of this glaze
            R, T = KubelkaMunk.get_reflectance_transmittance(K, S, thickness)
            
            # Composite with already computed layers
            if np.all(T_total == 0):  # First layer (on background)
                R_total = R + (T**2 * R_total) / (1.0 - R * R_total)
                T_total = T
            else:
                R_new, T_new = KubelkaMunk.composite_layers(R, T, R_total, T_total)
                R_total, T_total = R_new, T_new
        
        return R_total


class WatercolorRenderer:
    """
    Renderer for watercolor simulations using the Kubelka-Munk model.
    """
    
    def __init__(self, simulation: WatercolorSimulation):
        """
        Initialize renderer with a watercolor simulation.
        
        Parameters:
        -----------
        simulation : WatercolorSimulation
            The watercolor simulation to render
        """
        self.simulation = simulation
    
    def render_pigment(self, pigment_idx: int, background_color: np.ndarray = np.array([1.0, 1.0, 1.0])):
        """
        Render a single pigment layer.
        
        Parameters:
        -----------
        pigment_idx : int
            Index of the pigment to render
        background_color : np.ndarray
            RGB color of the background
        
        Returns:
        --------
        np.ndarray
            RGB image of rendered pigment
        """
        if pigment_idx >= len(self.simulation.pigment_paper):
            raise ValueError(f"Invalid pigment index: {pigment_idx}")
        
        # Get pigment properties
        props = self.simulation.pigment_properties[pigment_idx]
        kubelka_munk_params = props['kubelka_munk_params']
        
        # Check if K and S are provided
        if 'K' not in kubelka_munk_params or 'S' not in kubelka_munk_params:
            raise ValueError("Kubelka-Munk parameters K and S must be provided for rendering")
        
        K = kubelka_munk_params['K']
        S = kubelka_munk_params['S']
        
        # Get pigment thickness (combined water and paper concentrations)
        g = self.simulation.pigment_water[pigment_idx]
        d = self.simulation.pigment_paper[pigment_idx]
        thickness = g + d
        
        # Create output image
        height, width = self.simulation.height, self.simulation.width
        output = np.zeros((height, width, 3), dtype=np.float32)
        
        # For each pixel
        for i in range(height):
            for j in range(width):
                # If there's pigment at this pixel
                if thickness[i, j] > 0:
                    # Create glaze with thickness at this point
                    glaze = {
                        'K': K,
                        'S': S,
                        'thickness': thickness[i, j]
                    }
                    
                    # Render using Kubelka-Munk
                    output[i, j] = KubelkaMunk.render_glazes([glaze], background_color)
                else:
                    output[i, j] = background_color
        
        return output
    
    def render_all_pigments(self, background_color: np.ndarray = np.array([1.0, 1.0, 1.0])):
        """
        Render all pigment layers composited together.
        
        Parameters:
        -----------
        background_color : np.ndarray
            RGB color of the background
        
        Returns:
        --------
        np.ndarray
            RGB image of all rendered pigments
        """
        # Create list of glazes
        glazes = []
        
        for pigment_idx in range(len(self.simulation.pigment_paper)):
            # Get pigment properties
            props = self.simulation.pigment_properties[pigment_idx]
            kubelka_munk_params = props['kubelka_munk_params']
            
            # Check if K and S are provided
            if 'K' not in kubelka_munk_params or 'S' not in kubelka_munk_params:
                raise ValueError(f"Kubelka-Munk parameters K and S must be provided for pigment {pigment_idx}")
            
            K = kubelka_munk_params['K']
            S = kubelka_munk_params['S']
            
            # Get pigment thickness (combined water and paper concentrations)
            g = self.simulation.pigment_water[pigment_idx]
            d = self.simulation.pigment_paper[pigment_idx]
            thickness = g + d
            
            # Add to glazes list
            glazes.append({
                'K': K,
                'S': S,
                'thickness': thickness
            })
        
        # Create output image
        height, width = self.simulation.height, self.simulation.width
        output = np.zeros((height, width, 3), dtype=np.float32)
        
        # For each pixel
        for i in range(height):
            for j in range(width):
                # Create list of glazes at this point
                pixel_glazes = []
                
                for idx, glaze in enumerate(glazes):
                    if np.any(glaze['thickness'][i, j] > 0):
                        pixel_glazes.append({
                            'K': glaze['K'],
                            'S': glaze['S'],
                            'thickness': glaze['thickness'][i, j]
                        })
                
                # If there are glazes at this point, render them
                if pixel_glazes:
                    output[i, j] = KubelkaMunk.render_glazes(pixel_glazes, background_color)
                else:
                    output[i, j] = background_color
        
        return output


# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create simulation
    sim = WatercolorSimulation(256, 256)
    
    # Generate paper
    sim.generate_paper(method='perlin', seed=42)
    
    # Create a sample KM parameters for a blue pigment
    blue_km = {
        'K': np.array([0.8, 0.2, 0.1]),  # High absorption in red, low in blue
        'S': np.array([0.1, 0.2, 0.9])   # High scattering in blue
    }
    
    # Add a blue pigment
    blue_idx = sim.add_pigment(
        density=1.0,
        staining_power=0.6,
        granularity=0.4,
        kubelka_munk_params=blue_km
    )
    
    # Create a circular mask for the wet area
    y, x = np.ogrid[-128:128, -128:128]
    mask = x*x + y*y <= 80*80
    
    # Set wet mask and pigment
    sim.set_wet_mask(mask)
    sim.set_pigment_water(blue_idx, mask, concentration=0.8)
    
    # Run simulation
    sim.main_loop(50)
    
    # Render result
    renderer = WatercolorRenderer(sim)
    result = renderer.render_all_pigments()
    
    # Display result
    plt.figure(figsize=(8, 8))
    plt.imshow(np.clip(result, 0, 1))
    plt.axis('off')
    plt.title('Watercolor Simulation')
    plt.tight_layout()
    plt.savefig("watercolor_output.png", dpi=150)
    plt.show()

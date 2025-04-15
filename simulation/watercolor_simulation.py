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

import numba
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import scipy.ndimage as ndimage
import tqdm


@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_transfer_pigment_loop(
    g, d, paper_height, wet_mask, density, staining_power, granularity, height, width
):
    g_new = g.copy()
    d_new = d.copy()
    staining_power_safe = max(1e-6, staining_power)
    for i in numba.prange(height):
        for j in range(width):
            if wet_mask[i, j] > 0:
                height_factor = 1.0 - paper_height[i, j] * granularity
                delta_down = g[i, j] * height_factor * density
                height_effect = 1.0 + (paper_height[i, j] - 1.0) * granularity
                delta_up = d[i, j] * height_effect * density / staining_power_safe
                if d[i, j] + delta_down > 1.0:
                    delta_down = max(0.0, 1.0 - d[i, j])
                if g[i, j] + delta_up > 1.0:
                    delta_up = max(0.0, 1.0 - g[i, j])
                d_new[i, j] = min(1.0, max(0.0, d[i, j] + delta_down - delta_up))
                g_new[i, j] = min(1.0, max(0.0, g[i, j] + delta_up - delta_down))
    return g_new, d_new


@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_capillary_absorption_loop(
    water_saturation, wet_mask, paper_capacity, absorption_rate, height, width
):
    new_saturation = water_saturation.copy()
    for i in numba.prange(height):
        for j in range(width):
            if wet_mask[i, j] > 0:
                potential_absorption = paper_capacity[i, j] - water_saturation[i, j]
                if potential_absorption > 0:
                    absorbed = min(potential_absorption, absorption_rate)
                    new_saturation[i, j] += absorbed
    return new_saturation


@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_capillary_diffusion_loop(
    water_saturation,
    paper_capacity,
    min_saturation_for_diffusion,
    min_saturation_to_receive,
    height,
    width,
):
    new_saturation = water_saturation.copy()
    delta_saturation = np.zeros_like(water_saturation)
    for i in numba.prange(height):
        for j in range(width):
            current_saturation = water_saturation[i, j]
            if current_saturation > min_saturation_for_diffusion:
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                total_flow_request = 0.0
                neighbor_requests = np.zeros(4)
                valid_neighbor_indices = []
                for idx, (ni, nj) in enumerate(neighbors):
                    # Ensure integer indices for array access
                    ni_int = int(ni)
                    nj_int = int(nj)
                    if 0 <= ni_int < height and 0 <= nj_int < width:
                        neighbor_saturation = water_saturation[ni_int, nj_int]
                        if (
                            neighbor_saturation < paper_capacity[ni_int, nj_int]
                            and neighbor_saturation >= min_saturation_to_receive
                        ):
                            if current_saturation > neighbor_saturation:
                                diff = current_saturation - neighbor_saturation
                                capacity_left = (
                                    paper_capacity[ni_int, nj_int] - neighbor_saturation
                                )
                                potential_flow = (
                                    max(0.0, min(diff, capacity_left)) / 4.0
                                )
                                neighbor_requests[idx] = potential_flow
                                total_flow_request += potential_flow
                                valid_neighbor_indices.append(idx)
                available_to_diffuse = current_saturation - min_saturation_for_diffusion
                scale_factor = 1.0
                if (
                    total_flow_request > available_to_diffuse
                    and total_flow_request > 1e-6
                ):
                    scale_factor = available_to_diffuse / total_flow_request
                actual_total_flow_out = 0.0
                for idx in valid_neighbor_indices:
                    ni, nj = neighbors[idx]
                    ni_int = int(ni)
                    nj_int = int(nj)
                    actual_flow = neighbor_requests[idx] * scale_factor
                    delta_saturation[ni_int, nj_int] += actual_flow
                    actual_total_flow_out += actual_flow
                delta_saturation[i, j] -= actual_total_flow_out
    new_saturation += delta_saturation
    return np.clip(new_saturation, 0.0, 1.0)


@numba.jit(nopython=True, cache=True)
def _numba_get_reflectance_transmittance(K, S, thickness):
    S_safe = np.maximum(S, 1e-10)
    a = 1.0 + K / S_safe
    b = np.sqrt(np.maximum(0.0, a**2 - 1.0))
    safe_thickness = min(thickness, 10.0)
    bSx = b * S * safe_thickness
    max_exp = 20.0
    safe_bsx = np.clip(bSx, -max_exp, max_exp)
    sinh_bSx = np.sinh(safe_bsx)
    cosh_bSx = np.cosh(safe_bsx)
    c = a * sinh_bSx + b * cosh_bSx
    c_safe = np.maximum(c, 1e-10)
    R = np.clip(sinh_bSx / c_safe, 0.0, 1.0)
    T = np.clip(b / c_safe, 0.0, 1.0)
    return R, T


@numba.jit(nopython=True, cache=True)
def _numba_composite_layers(R1, T1, R2, T2):
    denominator = 1.0 - R1 * R2
    for i in range(len(denominator)):
        if denominator[i] == 0:
            denominator[i] = 1e-6
    R = R1 + (T1**2 * R2) / denominator
    T = (T1 * T2) / denominator
    return R, T


@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_render_all_pigments_loop(
    height,
    width,
    num_pigments,
    pigment_water_list,
    pigment_paper_list,
    pigment_properties_K_list,
    pigment_properties_S_list,
    background_color,
):
    result = np.ones((height, width, 3), dtype=np.float32) * background_color
    for i in numba.prange(height):
        for j in range(width):
            glazes_K = []
            glazes_S = []
            glazes_thickness = []
            has_pigment = False
            for idx in range(num_pigments):
                thickness_value = (
                    pigment_water_list[idx][i, j] + pigment_paper_list[idx][i, j]
                )
                if thickness_value > 0.001:
                    has_pigment = True
                    glazes_K.append(pigment_properties_K_list[idx])
                    glazes_S.append(pigment_properties_S_list[idx])
                    glazes_thickness.append(thickness_value)
            if has_pigment:
                R_pixel = background_color.copy()
                T_pixel = np.zeros_like(background_color)
                for k in range(len(glazes_K) - 1, -1, -1):
                    K = glazes_K[k]
                    S = glazes_S[k]
                    thickness = glazes_thickness[k]
                    R_glaze, T_glaze = _numba_get_reflectance_transmittance(
                        K, S, thickness
                    )
                    if np.all(T_pixel == 0):
                        denominator = 1.0 - R_glaze * R_pixel
                        for channel in range(len(denominator)):
                            if denominator[channel] == 0:
                                denominator[channel] = 1e-6
                        R_pixel = R_glaze + (T_glaze**2 * R_pixel) / denominator
                        T_pixel = T_glaze
                    else:
                        R_pixel, T_pixel = _numba_composite_layers(
                            R_glaze, T_glaze, R_pixel, T_pixel
                        )
                result[i, j] = R_pixel
    return result


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
        self.executor = ThreadPoolExecutor()  # Initialize thread pool

    def reset(self):
        """Reset the simulation to its initial state."""
        # Shallow water layer
        self.wet_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.velocity_u = np.zeros(
            (self.height, self.width + 1), dtype=np.float32
        )  # staggered grid
        self.velocity_v = np.zeros(
            (self.height + 1, self.width), dtype=np.float32
        )  # staggered grid
        self.pressure = np.zeros((self.height, self.width), dtype=np.float32)

        # Pigment layers
        self.pigment_water = []  # g^k - pigment in water
        self.pigment_paper = []  # d^k - pigment on paper
        self.pigment_properties = []  # density, staining power, granularity

        # Capillary layer
        self.water_saturation = np.zeros(
            (self.height, self.width), dtype=np.float32
        )  # s

        # Paper is already initialized in __init__, no need to check/regenerate it

    def generate_paper(self, method: str = "perlin", seed: Optional[int] = None):
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
        if method == "perlin":
            # A simple approximation of Perlin noise for this example
            # In a real implementation, would use a proper Perlin noise function
            scale = 8
            octaves = 4
            self.paper_height = np.zeros((self.height, self.width), dtype=np.float32)

            for octave in range(octaves):
                s = scale * (2**octave)
                h_samples = max(2, self.height // s)
                w_samples = max(2, self.width // s)

                noise = np.random.rand(h_samples, w_samples).astype(np.float32)
                noise = ndimage.zoom(
                    noise, (self.height / h_samples, self.width / w_samples), order=1
                )
                self.paper_height += noise / (2**octave)

            # Normalize to [0, 1]
            self.paper_height -= np.min(self.paper_height)
            self.paper_height /= np.max(self.paper_height)

        elif method == "random":
            # Simple random texture
            self.paper_height = np.random.rand(self.height, self.width).astype(
                np.float32
            )

        elif method == "fractal":
            # Approximate fractal noise
            scale = 4
            octaves = 6
            self.paper_height = np.zeros((self.height, self.width), dtype=np.float32)

            for octave in range(octaves):
                s = scale * (2**octave)
                h_samples = max(2, self.height // s)
                w_samples = max(2, self.width // s)

                noise = np.random.rand(h_samples, w_samples).astype(np.float32)
                noise = ndimage.zoom(
                    noise, (self.height / h_samples, self.width / w_samples), order=1
                )
                self.paper_height += noise / (2**octave)

            # Normalize to [0, 1]
            self.paper_height -= np.min(self.paper_height)
            self.paper_height /= np.max(self.paper_height)

        # Compute paper capacity from height
        self.paper_capacity = (
            self.paper_height * (self.paper_max_capacity - self.paper_min_capacity)
            + self.paper_min_capacity
        )

    def add_pigment(
        self,
        density: float = 1.0,
        staining_power: float = 0.5,
        granularity: float = 0.5,
        kubelka_munk_params: Dict = None,
    ):
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
            "density": density,
            "staining_power": staining_power,
            "granularity": granularity,
            "kubelka_munk_params": kubelka_munk_params or {},
        }

        self.pigment_water.append(water_layer)
        self.pigment_paper.append(paper_layer)
        self.pigment_properties.append(props)

        return len(self.pigment_properties) - 1

    def set_pigment_water(
        self, pigment_idx: int, mask: np.ndarray, concentration: float = 1.0
    ):
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
        padded_mask = np.pad(self.wet_mask, 1, mode="constant", constant_values=0)

        # Check for u velocity (horizontal) boundaries
        for i in range(self.height):
            for j in range(self.width + 1):
                # Check if this is a boundary velocity
                left_idx = max(0, j - 1)
                right_idx = min(self.width - 1, j)

                if (
                    padded_mask[i + 1, left_idx + 1] == 0
                    or padded_mask[i + 1, right_idx + 1] == 0
                ):
                    self.velocity_u[i, j] = 0.0

        # Check for v velocity (vertical) boundaries
        for i in range(self.height + 1):
            for j in range(self.width):
                # Check if this is a boundary velocity
                top_idx = max(0, i - 1)
                bottom_idx = min(self.height - 1, i)

                if (
                    padded_mask[top_idx + 1, j + 1] == 0
                    or padded_mask[bottom_idx + 1, j + 1] == 0
                ):
                    self.velocity_v[i, j] = 0.0

    def update_velocities(self):
        """
        Update water velocities based on the shallow water equations.
        This is an optimized implementation of the UpdateVelocities() procedure using NumPy vectorization.
        """
        # Get paper slope (gradient of height field)
        dx, dy = self.compute_paper_slope()

        # Create masks for wet regions
        wet_mask = self.wet_mask > 0

        # Adjust velocity by paper slope (vectorized)
        u_adjust = np.zeros_like(self.velocity_u)
        v_adjust = np.zeros_like(self.velocity_v)

        # Adjust u velocity (horizontal)
        u_adjust[:, 1:-1] = -0.5 * (dx[:, :-1] + dx[:, 1:])
        self.velocity_u[:, 1:-1] += u_adjust[:, 1:-1]

        # Adjust v velocity (vertical)
        v_adjust[1:-1, :] = -0.5 * (dy[:-1, :] + dy[1:, :])
        self.velocity_v[1:-1, :] += v_adjust[1:-1, :]

        # Enforce boundary conditions after adjusting for slope
        self.enforce_boundary_conditions()

        # Calculate adaptive time step to ensure stability
        max_velocity = max(
            np.max(np.abs(self.velocity_u)), np.max(np.abs(self.velocity_v))
        )
        dt = 1.0 / max(1, np.ceil(max_velocity))

        # Use a staggered grid for the update as in the paper
        # Initialize temporary arrays for the update
        new_u = self.velocity_u.copy()
        new_v = self.velocity_v.copy()

        # Create padded versions of arrays for easier boundary handling
        padded_wet_mask = np.pad(
            wet_mask, ((1, 1), (1, 1)), mode="constant", constant_values=0
        )
        padded_velocity_u = np.pad(
            self.velocity_u, ((0, 0), (0, 0)), mode="constant", constant_values=0
        )
        padded_velocity_v = np.pad(
            self.velocity_v, ((0, 0), (0, 0)), mode="constant", constant_values=0
        )
        padded_pressure = np.pad(
            self.pressure, ((0, 0), (0, 0)), mode="constant", constant_values=0
        )

        # Update velocities using time stepping
        t = 0.0
        while t < 1.0:
            # Create intermediate arrays for u and v velocities at cell centers
            u_centers = np.zeros((self.height, self.width + 1), dtype=np.float32)
            v_centers = np.zeros((self.height + 1, self.width), dtype=np.float32)

            # Calculate u at cell centers
            # Ensure proper shapes match for broadcasting
            u_centers[:, 1:-1] = 0.5 * (
                self.velocity_u[:, 1:-1] + self.velocity_u[:, 2:]
            )

            # Calculate v at cell centers
            # Ensure proper shapes match for broadcasting
            v_centers[1:-1, :] = 0.5 * (
                self.velocity_v[1:-1, :] + self.velocity_v[2:, :]
            )

            # --- Update horizontal velocity (u) ---
            # Create mask for valid u velocity cells (those with wet cells on either side)
            valid_u_mask = np.zeros_like(new_u, dtype=bool)
            for i in range(self.height):
                for j in range(1, self.width):
                    left_wet = j > 0 and self.wet_mask[i, j - 1]
                    right_wet = j < self.width and self.wet_mask[i, j]
                    valid_u_mask[i, j] = left_wet or right_wet

            # Process only valid u velocities
            for i in range(self.height):
                for j in range(1, self.width):
                    if valid_u_mask[i, j]:
                        # Get u at cell centers safely
                        u_i_j = 0.5 * (
                            self.velocity_u[i, max(0, j - 1)] + self.velocity_u[i, j]
                        )
                        u_ip1_j = 0.5 * (
                            self.velocity_u[i, j]
                            + self.velocity_u[i, min(self.width, j + 1)]
                        )

                        # Get v at this u-velocity position
                        v_ip5_jm5 = 0
                        v_ip5_jp5 = 0

                        if i < self.height - 1:
                            v_ip5_jm5 = 0.25 * (
                                self.velocity_v[i, max(0, j - 1)]
                                + self.velocity_v[i + 1, max(0, j - 1)]
                                + self.velocity_v[i, j]
                                + self.velocity_v[i + 1, j]
                            )

                            v_ip5_jp5 = 0.25 * (
                                self.velocity_v[i, j]
                                + self.velocity_v[i + 1, j]
                                + self.velocity_v[i, min(j + 1, self.width - 1)]
                                + self.velocity_v[i + 1, min(j + 1, self.width - 1)]
                            )

                        # Safety clips
                        u_i_j_safe = np.clip(u_i_j, -1e6, 1e6)
                        u_ip1_j_safe = np.clip(u_ip1_j, -1e6, 1e6)
                        v_ip5_jm5_safe = np.clip(v_ip5_jm5, -1e6, 1e6)
                        v_ip5_jp5_safe = np.clip(v_ip5_jp5, -1e6, 1e6)

                        # Advection term
                        A = ((u_i_j_safe**2) - (u_ip1_j_safe**2)) + (
                            (u_i_j_safe * v_ip5_jm5_safe)
                            - (u_ip1_j_safe * v_ip5_jp5_safe)
                        )
                        A = np.clip(A, -1e6, 1e6)

                        # Diffusion term
                        B = 0
                        if (
                            j > 0
                            and j < self.width - 1
                            and i > 0
                            and i < self.height - 1
                        ):
                            B = (
                                self.velocity_u[i, j + 1]
                                + self.velocity_u[i, j - 1]
                                + self.velocity_u[i + 1, j]
                                + self.velocity_u[i - 1, j]
                                - 4.0 * self.velocity_u[i, j]
                            )
                        B = np.clip(B, -1e6, 1e6)

                        # Pressure term
                        p_left = self.pressure[i, j - 1] if j > 0 else 0
                        p_right = self.pressure[i, j] if j < self.width else 0

                        # Update velocity
                        new_u[i, j] = self.velocity_u[i, j] + dt * (
                            A
                            - self.viscosity * B
                            + p_left
                            - p_right
                            - self.viscous_drag * self.velocity_u[i, j]
                        )
                        new_u[i, j] = np.clip(new_u[i, j], -1e6, 1e6)

            # --- Update vertical velocity (v) ---
            # Create mask for valid v velocity cells
            valid_v_mask = np.zeros_like(new_v, dtype=bool)
            for i in range(1, self.height):
                for j in range(self.width):
                    top_wet = i > 0 and self.wet_mask[i - 1, j]
                    bottom_wet = i < self.height and self.wet_mask[i, j]
                    valid_v_mask[i, j] = top_wet or bottom_wet

            # Process only valid v velocities
            for i in range(1, self.height):
                for j in range(self.width):
                    if valid_v_mask[i, j]:
                        # Get v at cell centers safely
                        v_i_j = 0.5 * (
                            self.velocity_v[max(0, i - 1), j] + self.velocity_v[i, j]
                        )
                        v_i_jp1 = 0.5 * (
                            self.velocity_v[i, j]
                            + self.velocity_v[i, min(j + 1, self.width - 1)]
                        )

                        # Get u at this v-velocity position
                        u_im5_jp5 = 0
                        u_ip5_jp5 = 0

                        if j < self.width - 1:
                            u_im5_jp5 = 0.25 * (
                                self.velocity_u[max(0, i - 1), j]
                                + self.velocity_u[max(0, i - 1), j + 1]
                                + self.velocity_u[i, j]
                                + self.velocity_u[i, j + 1]
                            )

                            u_ip5_jp5 = 0.25 * (
                                self.velocity_u[i, j]
                                + self.velocity_u[i, j + 1]
                                + self.velocity_u[min(i + 1, self.height - 1), j]
                                + self.velocity_u[min(i + 1, self.height - 1), j + 1]
                            )

                        # Safety clips
                        v_i_j_safe = np.clip(v_i_j, -1e6, 1e6)
                        v_i_jp1_safe = np.clip(v_i_jp1, -1e6, 1e6)
                        u_im5_jp5_safe = np.clip(u_im5_jp5, -1e6, 1e6)
                        u_ip5_jp5_safe = np.clip(u_ip5_jp5, -1e6, 1e6)

                        # Advection term
                        A = ((v_i_j_safe**2) - (v_i_jp1_safe**2)) + (
                            (v_i_j_safe * u_im5_jp5_safe)
                            - (v_i_jp1_safe * u_ip5_jp5_safe)
                        )
                        A = np.clip(A, -1e6, 1e6)

                        # Diffusion term
                        B = 0
                        if (
                            i > 0
                            and i < self.height - 1
                            and j > 0
                            and j < self.width - 1
                        ):
                            B = (
                                self.velocity_v[i + 1, j]
                                + self.velocity_v[i - 1, j]
                                + self.velocity_v[i, j + 1]
                                + self.velocity_v[i, j - 1]
                                - 4.0 * self.velocity_v[i, j]
                            )
                        B = np.clip(B, -1e6, 1e6)

                        # Pressure term
                        p_top = self.pressure[i - 1, j] if i > 0 else 0
                        p_bottom = self.pressure[i, j] if i < self.height else 0

                        # Update velocity
                        new_v[i, j] = self.velocity_v[i, j] + dt * (
                            A
                            - self.viscosity * B
                            + p_top
                            - p_bottom
                            - self.viscous_drag * self.velocity_v[i, j]
                        )
                        new_v[i, j] = np.clip(new_v[i, j], -1e6, 1e6)

            # Update velocity fields
            self.velocity_u = new_u.copy()
            self.velocity_v = new_v.copy()

            # Enforce boundary conditions
            self.enforce_boundary_conditions()

    def move_water(self):
        """
        Move water in the shallow-water layer.
        Efficient and strictly aligned with the source paper.
        """
        self.update_velocities()
        self.relax_divergence()
        self.flow_outward()  # Add flow_outward here as required by the paper

    def relax_divergence(
        self,
        max_iterations: int = 50,
        tolerance: float = 0.01,
        relaxation_factor: float = 0.1,
    ):
        """
        Relax divergence of velocity field to ensure fluid conservation (Section 4.3.2).
        This is an implementation of pressure-velocity coupling to enforce incompressibility.

        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations for the pressure solver
        tolerance : float
            Convergence tolerance for the pressure solver
        relaxation_factor : float
            Relaxation factor for SOR (Successive Over-Relaxation)
        """
        # Allocate arrays for the calculation
        pressure = np.zeros((self.height, self.width), dtype=np.float32)
        new_u = np.copy(self.velocity_u)
        new_v = np.copy(self.velocity_v)

        # Iterative pressure correction
        for iter_idx in range(max_iterations):
            max_div = 0.0

            # Calculate divergence of velocity field
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    if self.wet_mask[i, j] > 0:
                        div = (new_u[i, j + 1] - new_u[i, j]) + (
                            new_v[i + 1, j] - new_v[i, j]
                        )

                        # Update pressure using SOR
                        pressure[i, j] += relaxation_factor * div

                        # Apply pressure correction to velocities
                        new_u[i, j] -= (
                            pressure[i, j] - pressure[i, j - 1]
                            if j > 0
                            else pressure[i, j]
                        )
                        new_u[i, j + 1] -= (
                            pressure[i, j + 1] - pressure[i, j]
                            if j < self.width - 2
                            else pressure[i, j]
                        )
                        new_v[i, j] -= (
                            pressure[i, j] - pressure[i - 1, j]
                            if i > 0
                            else pressure[i, j]
                        )
                        new_v[i + 1, j] -= (
                            pressure[i + 1, j] - pressure[i, j]
                            if i < self.height - 2
                            else pressure[i, j]
                        )

                        max_div = max(max_div, abs(div))

            # Check convergence
            if max_div < tolerance:
                break

        # Update velocity fields
        self.velocity_u = new_u
        self.velocity_v = new_v
        self.pressure = pressure

        # Apply boundary conditions
        self.velocity_u[:, 0] = 0
        self.velocity_u[:, -1] = 0
        self.velocity_v[0, :] = 0
        self.velocity_v[-1, :] = 0

    def move_pigment(self):
        """
        Move pigment within the shallow-water layer based on water velocity.
        Advects pigment particles along the water flow.
        """
        # Skip if no pigments
        if not self.pigment_water:
            return

        # For each pigment layer, advect it according to velocity field
        for idx in range(len(self.pigment_water)):
            # Only advect pigment where there's water
            pigment_water = self.pigment_water[idx] * self.wet_mask

            # Simple first-order advection scheme
            # For a more accurate implementation, this could be replaced with a
            # semi-Lagrangian advection scheme as described in the paper
            pigment_new = np.zeros_like(pigment_water)

            # Apply advection based on velocity fields
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    if self.wet_mask[i, j] > 0:
                        # Get average velocity at cell center
                        u_avg = 0.5 * (
                            self.velocity_u[i, j] + self.velocity_u[i, j + 1]
                        )
                        v_avg = 0.5 * (
                            self.velocity_v[i, j] + self.velocity_v[i + 1, j]
                        )

                        # Calculate source position (backward tracing)
                        src_i = max(0, min(self.height - 1, i - v_avg))
                        src_j = max(0, min(self.width - 1, j - u_avg))

                        # Bilinear interpolation for smooth sampling
                        i_floor, j_floor = int(src_i), int(src_j)
                        i_ceil = min(i_floor + 1, self.height - 1)
                        j_ceil = min(j_floor + 1, self.width - 1)

                        # Interpolation weights
                        t = src_i - i_floor
                        s = src_j - j_floor

                        # Bilinear interpolation
                        pigment_new[i, j] = (
                            (1 - t) * (1 - s) * pigment_water[i_floor, j_floor]
                            + t * (1 - s) * pigment_water[i_ceil, j_floor]
                            + (1 - t) * s * pigment_water[i_floor, j_ceil]
                            + t * s * pigment_water[i_ceil, j_ceil]
                        )

            # Update the pigment water layer
            self.pigment_water[idx] = pigment_new

            # Ensure pigment remains only in wet areas
            self.pigment_water[idx] *= self.wet_mask

    def _transfer_single_pigment(self, pigment_idx):
        g = self.pigment_water[pigment_idx]
        d = self.pigment_paper[pigment_idx]
        props = self.pigment_properties[pigment_idx]

        # Use self.paper_height and self.wet_mask directly
        g_new, d_new = _numba_transfer_pigment_loop(
            g,
            d,
            self.paper_height,
            self.wet_mask,  # Use simulation's paper_height
            props["density"],
            props["staining_power"],
            props["granularity"],
            self.height,
            self.width,
        )
        return pigment_idx, g_new, d_new

    def transfer_pigment(self):
        """
        Transfer pigment between the shallow-water layer and the pigment-deposition layer.
        Uses Numba JIT for the core loop and ThreadPoolExecutor for parallel pigment processing.
        """
        futures = []
        for pigment_idx in range(len(self.pigment_water)):
            futures.append(
                self.executor.submit(self._transfer_single_pigment, pigment_idx)
            )

        # Use a temporary list to store results before updating self
        results = {}
        for future in futures:
            idx, g_new, d_new = future.result()
            results[idx] = (g_new, d_new)

        # Update the actual simulation arrays
        for idx in range(len(self.pigment_water)):
            if idx in results:
                self.pigment_water[idx], self.pigment_paper[idx] = results[idx]

    def simulate_capillary_flow(self):
        """
        Simulate capillary flow of water through the paper using Numba JIT loops.
        """
        # Absorb water from shallow-water layer into paper (using Numba)
        # This function now returns the updated saturation directly
        self.water_saturation = _numba_capillary_absorption_loop(
            self.water_saturation,
            self.wet_mask,
            self.paper_capacity,  # Use simulation's paper_capacity
            self.absorption_rate,
            self.height,
            self.width,
        )
        # Optional: Reduce water/pressure in the shallow layer based on absorbed_water
        # This requires the absorption loop to return the absorbed amount separately if needed.

        # Simulate diffusion through capillary layer (using Numba)
        self.water_saturation = _numba_capillary_diffusion_loop(
            self.water_saturation,
            self.paper_capacity,  # Use simulation's paper_capacity
            self.min_saturation_for_diffusion,
            self.min_saturation_to_receive,
            self.height,
            self.width,
        )

        # Expand wet-area mask where saturation exceeds threshold (vectorized)
        newly_wet = self.water_saturation > self.saturation_threshold
        self.wet_mask = np.maximum(self.wet_mask, newly_wet.astype(np.float32))

    def flow_outward(self):
        """
        Create outward flow near edges for edge darkening effect (Section 4.3.3).
        This simulates pigment migration to the edge by decreasing pressure near the wet mask boundary.
        """
        from scipy.ndimage import gaussian_filter

        blurred = gaussian_filter(
            self.wet_mask.astype(float), sigma=self.edge_darkening_kernel_size / 6
        )
        self.pressure -= self.edge_darkening_factor * (1 - blurred) * self.wet_mask

    def main_loop(self, num_steps: int = 100):
        """
        Main simulation loop. Uses efficient Numba-accelerated and thread-parallel steps.
        """
        try:
            from tqdm import trange

            iterator = trange(num_steps, desc="Simulating", ncols=80, leave=True)
        except ImportError:
            print(f"Running simulation for {num_steps} steps...")
            iterator = range(num_steps)
        except Exception:
            print(f"Running simulation for {num_steps} steps...")
            iterator = range(num_steps)

        # Pre-bind methods for speed
        move_water = self.move_water
        move_pigment = getattr(self, "move_pigment", None)
        transfer_pigment = self.transfer_pigment
        simulate_capillary_flow = self.simulate_capillary_flow

        for _ in iterator:
            move_water()  # Numba-optimized
            if move_pigment:
                move_pigment()  # If implemented, should be Numba/threaded
            transfer_pigment()  # Numba + ThreadPoolExecutor
            simulate_capillary_flow()  # Numba

        # Optionally, clean up thread pool
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

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
        S = (
            np.arctanh(
                (b**2 - (a - white_color) * (a - 1.0)) / (b * (1.0 - white_color))
            )
            / b
        )

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
        # Use np.clip to avoid division by very small values
        S_safe = np.maximum(S, 1e-10)
        a = 1.0 + K / S_safe

        # Avoid nan values with max(0, ...)
        b = np.sqrt(np.maximum(0, a**2 - 1.0))

        # Limit thickness to avoid overflow in exponential
        safe_thickness = min(thickness, 10.0)
        bSx = b * S * safe_thickness

        # Use numpy's hyperbolic functions with clipping to prevent overflow
        # Limit exponents to prevent overflow
        max_exp = 20.0  # Limit to avoid overflow
        safe_bsx = np.clip(bSx, -max_exp, max_exp)

        # Use numpy's more stable implementation of sinh/cosh
        sinh_bSx = np.sinh(safe_bsx)
        cosh_bSx = np.cosh(safe_bsx)

        # Avoid division by zero or very small values
        c = a * sinh_bSx + b * cosh_bSx
        c_safe = np.maximum(c, 1e-10)

        # Calculate reflectance and transmittance with safety bounds
        R = np.clip(sinh_bSx / c_safe, 0.0, 1.0)
        T = np.clip(b / c_safe, 0.0, 1.0)

        return R, T

    @staticmethod
    def composite_layers(
        R1: np.ndarray, T1: np.ndarray, R2: np.ndarray, T2: np.ndarray
    ):
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
    def render_glazes(
        glazes: List[Dict], background_color: np.ndarray = np.array([1.0, 1.0, 1.0])
    ):
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
            K = glaze["K"]
            S = glaze["S"]
            thickness = glaze["thickness"]

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
        # No KM instance needed if using static Numba functions

    def render_pigment(
        self, pigment_idx: int, background_color: np.ndarray = np.array([1.0, 1.0, 1.0])
    ):
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
        kubelka_munk_params = props["kubelka_munk_params"]

        # Check if K and S are provided
        if "K" not in kubelka_munk_params or "S" not in kubelka_munk_params:
            raise ValueError(
                "Kubelka-Munk parameters K and S must be provided for rendering"
            )

        K = np.asarray(kubelka_munk_params["K"], dtype=np.float32)
        S = np.asarray(kubelka_munk_params["S"], dtype=np.float32)

        # Get pigment thickness (combined water and paper concentrations)
        g = self.simulation.pigment_water[pigment_idx]
        d = self.simulation.pigment_paper[pigment_idx]
        thickness = g + d

        # Create output image
        height, width = self.simulation.height, self.simulation.width
        output = np.ones(
            (height, width, 3), dtype=np.float32
        ) * background_color.astype(np.float32)

        # Simple loop for single pigment rendering (can be numba-fied too)
        for i in range(height):
            for j in range(width):
                if thickness[i, j] > 0.001:
                    R_glaze, T_glaze = _numba_get_reflectance_transmittance(
                        K, S, thickness[i, j]
                    )
                    # Composite single glaze onto background
                    denominator = 1.0 - R_glaze * background_color
                    for channel in range(len(denominator)):
                        if denominator[channel] == 0:
                            denominator[channel] = 1e-6
                    output[i, j] = (
                        R_glaze + (T_glaze**2 * background_color) / denominator
                    )
        return output

    def render_all_pigments(
        self, background_color: np.ndarray = np.array([1.0, 1.0, 1.0])
    ):
        """
        Render all pigment layers composited together using Numba JIT parallel loop.
        """
        height, width = self.simulation.height, self.simulation.width
        num_pigments = len(self.simulation.pigment_properties)

        if num_pigments == 0:
            return np.ones(
                (height, width, 3), dtype=np.float32
            ) * background_color.astype(np.float32)

        # Prepare data for Numba loop
        pigment_properties_K_list = []
        pigment_properties_S_list = []
        for props in self.simulation.pigment_properties:
            km_params = props.get("kubelka_munk_params", {})
            K = km_params.get("K", np.zeros(3))
            S = km_params.get("S", np.zeros(3))
            pigment_properties_K_list.append(np.asarray(K, dtype=np.float32))
            pigment_properties_S_list.append(np.asarray(S, dtype=np.float32))

        # Numba works best with lists of arrays of the same type/shape if possible.
        # Ensure pigment_water and pigment_paper are lists of float32 arrays.
        pigment_water_list = [
            arr.astype(np.float32) for arr in self.simulation.pigment_water
        ]
        pigment_paper_list = [
            arr.astype(np.float32) for arr in self.simulation.pigment_paper
        ]

        # Call the Numba JIT compiled loop
        result = _numba_render_all_pigments_loop(
            height,
            width,
            num_pigments,
            pigment_water_list,  # Pass the list of arrays
            pigment_paper_list,  # Pass the list of arrays
            pigment_properties_K_list,  # Pass the list of arrays
            pigment_properties_S_list,  # Pass the list of arrays
            background_color.astype(np.float32),
        )

        return result


# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create simulation
    sim = WatercolorSimulation(256, 256)

    # Generate paper
    sim.generate_paper(method="perlin", seed=42)

    # Create a sample KM parameters for a blue pigment
    blue_km = {
        "K": np.array([0.8, 0.2, 0.1]),  # High absorption in red, low in blue
        "S": np.array([0.1, 0.2, 0.9]),  # High scattering in blue
    }

    # Add a blue pigment
    blue_idx = sim.add_pigment(
        density=1.0, staining_power=0.6, granularity=0.4, kubelka_munk_params=blue_km
    )

    # Create a circular mask for the wet area
    y, x = np.ogrid[-128:128, -128:128]
    mask = x * x + y * y <= 80 * 80

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
    plt.axis("off")
    plt.title("Watercolor Simulation")
    plt.tight_layout()
    plt.savefig("watercolor_output.png", dpi=150)
    plt.show()

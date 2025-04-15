#!/usr/bin/env python3
"""
Optimized computation kernels for watercolor simulation.
Uses Numba for CPU acceleration and CUDA when available.
"""

import numpy as np
import numba
from numba import prange

# Try to import CUDA support - fallback gracefully if not available
try:
    from numba import cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA support not available. Using CPU acceleration only.")

# Configuration for JIT compilation
PARALLEL = True
FASTMATH = True
CACHE = True

# ------------------------------------------------------------------------------
# Fluid simulation kernels
# ------------------------------------------------------------------------------


@numba.jit(nopython=True, parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def update_velocities_kernel(
    u, v, p, height_dx, height_dy, viscosity, viscous_drag, dt, height, width
):
    """
    Update velocity fields based on pressure gradients and paper slope.
    Implementation of equations (1) and (2) from Section 4.3.
    """
    u_new = np.empty_like(u)
    v_new = np.empty_like(v)

    # Compute Laplacian of velocity fields
    lap_u = np.zeros_like(u)
    lap_v = np.zeros_like(v)

    for i in prange(1, height - 1):
        for j in range(1, width):
            lap_u[i, j] = (
                u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j]
            )

    for i in prange(1, height):
        for j in range(1, width - 1):
            lap_v[i, j] = (
                v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1] - 4 * v[i, j]
            )

    # Compute pressure gradient
    p_grad_x = np.zeros_like(u)
    p_grad_y = np.zeros_like(v)

    for i in prange(height):
        for j in range(1, width):
            p_grad_x[i, j] = p[i, j] - p[i, j - 1]

    for i in prange(1, height):
        for j in range(width):
            p_grad_y[i, j] = p[i, j] - p[i - 1, j]

    # Update velocity fields
    for i in prange(height):
        for j in range(width + 1):
            if 0 < j < width:
                # Add slope effect, viscosity, pressure, and drag
                u_new[i, j] = u[i, j] - dt * (
                    height_dx[i, j]  # Paper slope
                    + p_grad_x[i, j]  # Pressure gradient
                    - viscosity * lap_u[i, j]  # Viscous diffusion
                    + viscous_drag * u[i, j]  # Drag
                )
            else:
                u_new[i, j] = 0.0  # Boundary condition

    for i in prange(height + 1):
        for j in range(width):
            if 0 < i < height:
                # Add slope effect, viscosity, pressure, and drag
                v_new[i, j] = v[i, j] - dt * (
                    height_dy[i, j]  # Paper slope
                    + p_grad_y[i, j]  # Pressure gradient
                    - viscosity * lap_v[i, j]  # Viscous diffusion
                    + viscous_drag * v[i, j]  # Drag
                )
            else:
                v_new[i, j] = 0.0  # Boundary condition

    return u_new, v_new


@numba.jit(nopython=True, parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def compute_divergence_kernel(u, v, height, width):
    """
    Compute the divergence of the velocity field.
    Used for pressure calculation in Section 4.3.2.
    """
    div = np.zeros((height, width))

    for i in prange(height):
        for j in range(width):
            # Compute divergence using forward differences
            u_diff = u[i, j + 1] - u[i, j]
            v_diff = v[i + 1, j] - v[i, j]
            div[i, j] = u_diff + v_diff

    return div


@numba.jit(nopython=True, parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def relax_pressure_kernel(p, div, wet_mask, omega, height, width):
    """
    Relax pressure using Successive Over-Relaxation (SOR) method.
    Much faster than Jacobi iteration for pressure relaxation.
    """
    p_new = p.copy()

    for i in prange(1, height - 1):
        for j in range(1, width - 1):
            if wet_mask[i, j] > 0:
                # SOR update formula
                p_new[i, j] = (1.0 - omega) * p[i, j] + omega / 4.0 * (
                    p_new[i - 1, j]
                    + p[i + 1, j]
                    + p_new[i, j - 1]
                    + p[i, j + 1]
                    - div[i, j]
                )

    return p_new


# ------------------------------------------------------------------------------
# Pigment transfer kernels
# ------------------------------------------------------------------------------


@numba.jit(nopython=True, parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def transfer_pigment_kernel(
    g, d, paper_height, wet_mask, density, staining_power, granularity, height, width
):
    """
    Transfer pigment between water and paper based on paper properties.
    Implementation of the pigment transfer mechanism from Section 4.5.
    """
    g_new = g.copy()
    d_new = d.copy()
    staining_power_safe = max(1e-6, staining_power)

    for i in prange(height):
        for j in range(width):
            if wet_mask[i, j] > 0:
                # Calculate pigment adsorption (water to paper)
                # Equation related to granularity in Section 4.5
                height_factor = 1.0 - paper_height[i, j] * granularity
                delta_down = g[i, j] * height_factor * density

                # Calculate pigment desorption (paper to water)
                height_effect = 1.0 + (paper_height[i, j] - 1.0) * granularity
                delta_up = d[i, j] * height_effect * density / staining_power_safe

                # Ensure physical constraints
                if d[i, j] + delta_down > 1.0:
                    delta_down = max(0.0, 1.0 - d[i, j])

                if g[i, j] + delta_up > 1.0:
                    delta_up = max(0.0, 1.0 - g[i, j])

                # Transfer pigment
                d_new[i, j] = min(1.0, max(0.0, d[i, j] + delta_down - delta_up))
                g_new[i, j] = min(1.0, max(0.0, g[i, j] + delta_up - delta_down))

    return g_new, d_new


@numba.jit(nopython=True, parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def advect_pigment_kernel(g, u, v, dt, height, width):
    """
    Advect pigment in water using Semi-Lagrangian advection.
    More stable than simple Eulerian advection.
    """
    g_new = np.zeros_like(g)

    for i in prange(height):
        for j in range(width):
            # Compute average velocities at cell centers
            u_avg = 0.5 * (u[i, j] + u[i, j + 1])
            v_avg = 0.5 * (v[i, j] + v[i + 1, j])

            # Compute source position using backward tracing
            src_i = i - dt * v_avg
            src_j = j - dt * u_avg

            # Clamp to grid boundaries
            src_i = max(0.0, min(height - 1.01, src_i))
            src_j = max(0.0, min(width - 1.01, src_j))

            # Bilinear interpolation
            i0 = int(src_i)
            j0 = int(src_j)
            i1 = i0 + 1
            j1 = j0 + 1

            s1 = src_i - i0
            s0 = 1.0 - s1
            t1 = src_j - j0
            t0 = 1.0 - t1

            g_new[i, j] = (
                g[i0, j0] * s0 * t0
                + g[i0, j1] * s0 * t1
                + g[i1, j0] * s1 * t0
                + g[i1, j1] * s1 * t1
            )

    return g_new


# ------------------------------------------------------------------------------
# Capillary flow kernels
# ------------------------------------------------------------------------------


@numba.jit(nopython=True, parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def capillary_absorption_kernel(
    water_saturation, wet_mask, paper_capacity, absorption_rate, height, width
):
    """
    Simulate water absorption from surface into paper.
    Implementation from Section 4.4 (Capillary Flow).
    """
    new_saturation = water_saturation.copy()

    for i in prange(height):
        for j in range(width):
            if wet_mask[i, j] > 0:
                # Calculate potential absorption based on remaining capacity
                potential_absorption = paper_capacity[i, j] - water_saturation[i, j]

                # Absorb water at controlled rate (α parameter)
                if potential_absorption > 0:
                    absorbed = min(potential_absorption, absorption_rate)
                    new_saturation[i, j] += absorbed

    return new_saturation


@numba.jit(nopython=True, parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def capillary_diffusion_kernel(
    water_saturation,
    paper_capacity,
    min_saturation_for_diffusion,
    min_saturation_to_receive,
    height,
    width,
):
    """
    Simulate water diffusion within the paper structure.
    Implementation from Section 4.4 (Capillary Flow).
    """
    # Use a more efficient approach than copying the whole array at once
    delta_saturation = np.zeros_like(water_saturation)

    for i in prange(height):
        for j in range(width):
            current_saturation = water_saturation[i, j]

            # Only diffuse if saturation is above threshold (ε parameter)
            if current_saturation > min_saturation_for_diffusion:
                # Check 4-connected neighbors
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                total_flow_request = 0.0
                neighbor_requests = np.zeros(4)
                valid_neighbor_indices = []

                # Calculate potential flow to each neighbor
                for idx, (ni, nj) in enumerate(neighbors):
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_saturation = water_saturation[ni, nj]

                        # Only flow to neighbors that are above receiving threshold (δ parameter)
                        if (
                            neighbor_saturation < paper_capacity[ni, nj]
                            and neighbor_saturation >= min_saturation_to_receive
                        ):

                            # Flow from higher to lower saturation
                            if current_saturation > neighbor_saturation:
                                diff = current_saturation - neighbor_saturation
                                capacity_left = (
                                    paper_capacity[ni, nj] - neighbor_saturation
                                )
                                potential_flow = (
                                    max(0.0, min(diff, capacity_left)) / 4.0
                                )
                                neighbor_requests[idx] = potential_flow
                                total_flow_request += potential_flow
                                valid_neighbor_indices.append(idx)

                # Calculate how much water is available to diffuse
                available_to_diffuse = current_saturation - min_saturation_for_diffusion
                scale_factor = 1.0

                # Scale down the flow if requests exceed available water
                if (
                    total_flow_request > available_to_diffuse
                    and total_flow_request > 1e-6
                ):
                    scale_factor = available_to_diffuse / total_flow_request

                # Distribute water to valid neighbors
                actual_total_flow_out = 0.0
                for idx in valid_neighbor_indices:
                    ni, nj = neighbors[idx]
                    actual_flow = neighbor_requests[idx] * scale_factor
                    delta_saturation[ni, nj] += actual_flow
                    actual_total_flow_out += actual_flow

                # Remove water from current cell
                delta_saturation[i, j] -= actual_total_flow_out

    # Apply all flow changes and clip to valid range
    return np.clip(water_saturation + delta_saturation, 0.0, 1.0)


# ------------------------------------------------------------------------------
# Rendering kernels
# ------------------------------------------------------------------------------


@numba.jit(nopython=True, fastmath=FASTMATH, cache=CACHE)
def get_reflectance_transmittance(K, S, thickness):
    """
    Calculate reflectance and transmittance for a pigment layer.
    Implementation of the Kubelka-Munk equations from Section 5.2.
    """
    # Ensure minimum values to prevent division by zero
    S_safe = np.maximum(S, 1e-10)

    # Calculate a and b parameters
    a = 1.0 + K / S_safe
    b = np.sqrt(np.maximum(0.0, a**2 - 1.0))

    # Limit thickness to prevent numerical overflow
    safe_thickness = min(thickness, 10.0)
    bSx = b * S * safe_thickness

    # Calculate sinh and cosh using exponential forms for numerical stability
    exp_bSx = np.exp(np.clip(bSx, -20.0, 20.0))
    exp_neg_bSx = 1.0 / exp_bSx

    sinh_bSx = 0.5 * (exp_bSx - exp_neg_bSx)
    cosh_bSx = 0.5 * (exp_bSx + exp_neg_bSx)

    # Calculate reflectance and transmittance
    c = a * sinh_bSx + b * cosh_bSx
    c_safe = np.maximum(c, 1e-10)

    R = np.clip(sinh_bSx / c_safe, 0.0, 1.0)
    T = np.clip(b / c_safe, 0.0, 1.0)

    return R, T


@numba.jit(nopython=True, fastmath=FASTMATH, cache=CACHE)
def composite_layers(R1, T1, R2, T2):
    """
    Composite two layers using Kubelka's optical compositing equations.
    Implementation from Section 5.2.
    """
    # Avoid division by zero
    denominator = 1.0 - R1 * R2
    denominator = np.maximum(denominator, 1e-10)

    # Calculate composite reflectance and transmittance
    R = R1 + (T1**2 * R2) / denominator
    T = (T1 * T2) / denominator

    return R, T


@numba.jit(nopython=True, parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def render_all_pigments(
    height,
    width,
    num_pigments,
    pigment_water_list,
    pigment_paper_list,
    pigment_properties_K_list,
    pigment_properties_S_list,
    background_color,
):
    """
    Render all pigments using Kubelka-Munk optical model.
    Implementation from Section 5 (Rendering the Pigmented Layers).
    """
    result = np.ones((height, width, 3), dtype=np.float32) * background_color

    for i in prange(height):
        for j in range(width):
            # Check if there's any pigment at this pixel
            has_pigment = False

            # Use arrays instead of list of dicts for better performance
            K_values = np.zeros((num_pigments, 3), dtype=np.float32)
            S_values = np.zeros((num_pigments, 3), dtype=np.float32)
            thickness_values = np.zeros(num_pigments, dtype=np.float32)
            pigment_count = 0

            # Gather all pigment data for this pixel
            for idx in range(num_pigments):
                thickness_value = (
                    pigment_water_list[idx][i, j] + pigment_paper_list[idx][i, j]
                )

                if thickness_value > 0.001:
                    has_pigment = True
                    K_values[pigment_count] = pigment_properties_K_list[idx]
                    S_values[pigment_count] = pigment_properties_S_list[idx]
                    thickness_values[pigment_count] = thickness_value
                    pigment_count += 1

            # If there's pigment at this pixel, compute the color
            if has_pigment:
                # Start with background color
                R_total = background_color
                T_total = np.zeros_like(background_color)

                # Composite each glaze (from bottom to top)
                for idx in range(pigment_count - 1, -1, -1):
                    # Calculate reflectance and transmittance for this layer
                    R, T = get_reflectance_transmittance(
                        K_values[idx], S_values[idx], thickness_values[idx]
                    )

                    # Composite with already computed layers
                    if np.all(T_total == 0):  # First layer on background
                        denom = 1.0 - R * R_total
                        denom = np.maximum(denom, 1e-10)
                        R_total = R + (T**2 * R_total) / denom
                        T_total = T
                    else:
                        R_new, T_new = composite_layers(R, T, R_total, T_total)
                        R_total, T_total = R_new, T_new

                # Store result for this pixel
                result[i, j] = np.clip(R_total, 0.0, 1.0)

    return result


# ------------------------------------------------------------------------------
# CUDA kernels (when available)
# ------------------------------------------------------------------------------

if CUDA_AVAILABLE:

    @cuda.jit
    def cuda_transfer_pigment(
        g, d, paper_height, wet_mask, density, staining_power, granularity, g_out, d_out
    ):
        """CUDA implementation of pigment transfer between water and paper."""
        i, j = cuda.grid(2)
        height, width = g.shape

        if i < height and j < width and wet_mask[i, j] > 0:
            # Calculate pigment adsorption (water to paper)
            height_factor = 1.0 - paper_height[i, j] * granularity
            delta_down = g[i, j] * height_factor * density

            # Calculate pigment desorption (paper to water)
            staining_power_safe = max(1e-6, staining_power)
            height_effect = 1.0 + (paper_height[i, j] - 1.0) * granularity
            delta_up = d[i, j] * height_effect * density / staining_power_safe

            # Ensure physical constraints
            if d[i, j] + delta_down > 1.0:
                delta_down = max(0.0, 1.0 - d[i, j])

            if g[i, j] + delta_up > 1.0:
                delta_up = max(0.0, 1.0 - g[i, j])

            # Transfer pigment
            d_out[i, j] = min(1.0, max(0.0, d[i, j] + delta_down - delta_up))
            g_out[i, j] = min(1.0, max(0.0, g[i, j] + delta_up - delta_down))

    @cuda.jit
    def cuda_capillary_diffusion(
        water_saturation,
        paper_capacity,
        min_saturation_for_diffusion,
        min_saturation_to_receive,
        delta_saturation,
    ):
        """CUDA implementation of water diffusion within paper."""
        i, j = cuda.grid(2)
        height, width = water_saturation.shape

        if i < height and j < width:
            current_saturation = water_saturation[i, j]

            # Only diffuse if saturation is above threshold
            if current_saturation > min_saturation_for_diffusion:
                # Check 4-connected neighbors
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                total_flow_request = 0.0
                neighbor_requests = cuda.local.array(4, dtype=numba.float32)
                valid_neighbors = cuda.local.array(4, dtype=numba.int32)
                valid_count = 0

                # Calculate potential flow to each neighbor
                for idx in range(4):
                    ni, nj = neighbors[idx]
                    neighbor_requests[idx] = 0.0

                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_saturation = water_saturation[ni, nj]

                        # Only flow to neighbors that meet criteria
                        if (
                            neighbor_saturation < paper_capacity[ni, nj]
                            and neighbor_saturation >= min_saturation_to_receive
                        ):

                            # Flow from higher to lower saturation
                            if current_saturation > neighbor_saturation:
                                diff = current_saturation - neighbor_saturation
                                capacity_left = (
                                    paper_capacity[ni, nj] - neighbor_saturation
                                )
                                potential_flow = (
                                    max(0.0, min(diff, capacity_left)) / 4.0
                                )
                                neighbor_requests[idx] = potential_flow
                                total_flow_request += potential_flow
                                valid_neighbors[valid_count] = idx
                                valid_count += 1

                # Calculate how much water is available to diffuse
                available_to_diffuse = current_saturation - min_saturation_for_diffusion
                scale_factor = 1.0

                # Scale down the flow if requests exceed available water
                if (
                    total_flow_request > available_to_diffuse
                    and total_flow_request > 1e-6
                ):
                    scale_factor = available_to_diffuse / total_flow_request

                # Distribute water to valid neighbors using atomic operations
                actual_total_flow_out = 0.0
                for idx_count in range(valid_count):
                    idx = valid_neighbors[idx_count]
                    ni, nj = neighbors[idx]
                    actual_flow = neighbor_requests[idx] * scale_factor
                    cuda.atomic.add(delta_saturation, (ni, nj), actual_flow)
                    actual_total_flow_out += actual_flow

                # Remove water from current cell
                cuda.atomic.add(delta_saturation, (i, j), -actual_total_flow_out)

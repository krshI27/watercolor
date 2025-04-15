#!/usr/bin/env python3
"""
Fluid simulation module based on shallow water equations.
From Section 4.3 of 'Computer-Generated Watercolor'.
"""

import numpy as np
import numba
from scipy.ndimage import gaussian_filter


@numba.jit(nopython=True, cache=True)
def _laplacian_numba(field):
    """Calculate Laplacian using Numba."""
    lap = np.zeros_like(field)
    height, width = field.shape
    for i in range(height):
        for j in range(width):
            up = field[i - 1 if i > 0 else height - 1, j]
            down = field[i + 1 if i < height - 1 else 0, j]
            left = field[i, j - 1 if j > 0 else width - 1]
            right = field[i, j + 1 if j < width - 1 else 0]
            lap[i, j] = up + down + left + right - 4 * field[i, j]
    return lap


@numba.jit(nopython=True, parallel=True, cache=True)
def _update_velocities_numba(
    u, v, p, slope_x, slope_y, wet_mask, viscosity, viscous_drag, dt, height, width
):
    """Numba optimized velocity update."""
    u_new = u.copy()
    v_new = v.copy()
    grad_p_x = np.zeros_like(u)
    grad_p_y = np.zeros_like(v)
    for i in numba.prange(height):
        for j in range(1, width):
            grad_p_x[i, j] = p[i, j] - p[i, j - 1]
    for i in numba.prange(1, height):
        for j in range(width):
            grad_p_y[i, j] = p[i, j] - p[i - 1, j]
    laplace_u = _laplacian_numba(u)
    laplace_v = _laplacian_numba(v)
    for i in numba.prange(height):
        for j in range(1, width):
            dudx = (u[i, j + 1] - u[i, j - 1]) / 2.0 if j > 0 and j < width else 0.0
            v_at_u = 0.25 * (v[i, j - 1] + v[i + 1, j - 1] + v[i, j] + v[i + 1, j])
            dudy = (
                (u[i + 1, j] - u[i - 1, j]) / 2.0 if i > 0 and i < height - 1 else 0.0
            )
            adv_u = u[i, j] * dudx + v_at_u * dudy
            u_new[i, j] += dt * (
                -adv_u
                + viscosity * laplace_u[i, j]
                - grad_p_x[i, j]
                - viscous_drag * u[i, j]
                - slope_x[i, j]
            )
    for i in numba.prange(1, height):
        for j in range(width):
            u_at_v = 0.25 * (u[i - 1, j] + u[i - 1, j + 1] + u[i, j] + u[i, j + 1])
            dvdx = (v[i, j + 1] - v[i, j - 1]) / 2.0 if j > 0 and j < width - 1 else 0.0
            dvdy = (v[i + 1, j] - v[i - 1, j]) / 2.0 if i > 0 and i < height else 0.0
            adv_v = u_at_v * dvdx + v[i, j] * dvdy
            v_new[i, j] += dt * (
                -adv_v
                + viscosity * laplace_v[i, j]
                - grad_p_y[i, j]
                - viscous_drag * v[i, j]
                - slope_y[i, j]
            )
    for i in numba.prange(height):
        for j in range(width + 1):
            is_boundary_u = False
            if j == 0 or j == width:
                is_boundary_u = True
            elif j > 0 and j < width:
                if not (wet_mask[i, j - 1] > 0 and wet_mask[i, j] > 0):
                    is_boundary_u = True
            if is_boundary_u:
                u_new[i, j] = 0.0
    for i in numba.prange(height + 1):
        for j in range(width):
            is_boundary_v = False
            if i == 0 or i == height:
                is_boundary_v = True
            elif i > 0 and i < height:
                if not (wet_mask[i - 1, j] > 0 and wet_mask[i, j] > 0):
                    is_boundary_v = True
            if is_boundary_v:
                v_new[i, j] = 0.0
    return u_new, v_new


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
        self.u = np.zeros((height, width + 1))  # x velocity at cell boundaries
        self.v = np.zeros((height + 1, width))  # y velocity at cell boundaries
        self.p = np.zeros((height, width))  # pressure at cell centers

        # Parameters from paper
        self.viscosity = 0.1  # μ
        self.viscous_drag = 0.01  # κ
        self.edge_darkening = 0.03  # η

    def update_velocities(self, paper, wet_mask: np.ndarray, dt: float = 0.1):
        """
        Update water velocities using Numba optimized function.
        Implements equations (1) and (2) from Section 4.3 of the source paper.
        """
        dy_full, dx_full = paper.slope
        slope_x = dx_full
        slope_y = dy_full
        u = np.ascontiguousarray(self.u, dtype=np.float32)
        v = np.ascontiguousarray(self.v, dtype=np.float32)
        p = np.ascontiguousarray(self.p, dtype=np.float32)
        wet_mask_f32 = np.ascontiguousarray(wet_mask, dtype=np.float32)
        slope_x_f32 = np.ascontiguousarray(slope_x, dtype=np.float32)
        slope_y_f32 = np.ascontiguousarray(slope_y, dtype=np.float32)
        u_new, v_new = _update_velocities_numba(
            u,
            v,
            p,
            slope_x_f32,
            slope_y_f32,
            wet_mask_f32,
            self.viscosity,
            self.viscous_drag,
            dt,
            self.height,
            self.width,
        )
        self.u = u_new
        self.v = v_new
        # Boundary conditions handled within Numba function

    def relax_divergence(
        self, wet_mask: np.ndarray, iterations: int = 50, tolerance: float = 0.01
    ):
        """
        Relax divergence of velocity field to ensure fluid conservation.
        Section 4.3.2 of the paper. (Uses NumPy for now, consider Numba if needed)
        """
        for _ in range(iterations):
            div = self._divergence()
            if np.abs(div).max() < tolerance:
                break
            self.p += 0.1 * div * wet_mask
            grad_p = np.gradient(self.p)
            self.u[:, 1:-1] -= self.p[:, 1:] - self.p[:, :-1]
            self.v[1:-1, :] -= self.p[1:, :] - self.p[:-1, :]
            self._enforce_boundaries(wet_mask)

    def flow_outward(self, wet_mask: np.ndarray, kernel_size: int = 10):
        """
        Create outward flow near edges for edge darkening effect.
        Equation (3) from Section 4.3.3.
        """
        # Gaussian blur of wet mask
        blurred = gaussian_filter(wet_mask.astype(float), sigma=kernel_size / 6)

        # Decrease pressure near edges
        self.p -= self.edge_darkening * (1 - blurred) * wet_mask

    def _divergence(self) -> np.ndarray:
        """Calculate divergence of velocity field on staggered grid."""
        div = np.zeros((self.height, self.width), dtype=np.float32)
        div += self.u[:, 1:] - self.u[:, :-1]
        div += self.v[1:, :] - self.v[:-1, :]
        return div

    def _enforce_boundaries(self, wet_mask: np.ndarray):
        """Set velocity to zero at boundaries of wet regions (NumPy version)."""
        mask_padded_h = np.pad(wet_mask, ((0, 0), (1, 1)), constant_values=0)
        mask_padded_v = np.pad(wet_mask, ((1, 1), (0, 0)), constant_values=0)
        self.u *= (mask_padded_h[:, :-1] > 0) & (mask_padded_h[:, 1:] > 0)
        self.v *= (mask_padded_v[:-1, :] > 0) & (mask_padded_v[1:, :] > 0)
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0

    def reset(self):
        """Reset the fluid simulation to its initial state."""
        self.u = np.zeros((self.height, self.width + 1))
        self.v = np.zeros((self.height + 1, self.width))
        self.p = np.zeros((self.height, self.width))

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

    def __init__(
        self,
        width: int,
        height: int,
        viscosity: float = 0.1,
        viscous_drag: float = 0.01,
        edge_darkening: float = 0.03,
    ):
        self.width = width
        self.height = height

        # Initialize fields on the staggered (MAC) grid from Section 4.3:
        # u lives on vertical cell boundaries, v on horizontal, pressure at cell centers.
        self.u = np.zeros((height, width + 1))
        self.v = np.zeros((height + 1, width))
        self.p = np.zeros((height, width))

        # Paper-physical parameters (μ, κ, η in Section 4.3).
        self.viscosity = viscosity
        self.viscous_drag = viscous_drag
        self.edge_darkening = edge_darkening

    def update(self, paper, wet_mask: np.ndarray, dt: float = 0.1):
        """Run one MoveWater step (§4.3): UpdateVelocities → RelaxDivergence → FlowOutward.

        Derives slopes from ``paper.height_field`` directly (rather than via a
        ``paper.slope`` property) so the method also works with lightweight
        paper test doubles that only expose ``height_field`` and
        ``update_capacity``.
        """
        paper.update_capacity()
        height_field = np.asarray(paper.height_field, dtype=np.float32)
        slope_y, slope_x = np.gradient(height_field)
        self._update_velocities(slope_x, slope_y, wet_mask, dt)
        self.relax_divergence(wet_mask)
        self.flow_outward(wet_mask)

    def update_velocities(self, paper, wet_mask: np.ndarray, dt: float = 0.1):
        """UpdateVelocities step from §4.3 — equations (1) and (2).

        Accepts any paper object exposing either a ``.slope`` property (the
        production :class:`Paper`) or just a ``.height_field`` attribute.
        """
        slope_y, slope_x = self._slopes_from_paper(paper)
        self._update_velocities(slope_x, slope_y, wet_mask, dt)

    @staticmethod
    def _slopes_from_paper(paper):
        slope_attr = getattr(paper, "slope", None)
        if isinstance(slope_attr, tuple) and len(slope_attr) == 2:
            return slope_attr
        height_field = np.asarray(paper.height_field, dtype=np.float32)
        return np.gradient(height_field)

    def _update_velocities(
        self, slope_x: np.ndarray, slope_y: np.ndarray, wet_mask: np.ndarray, dt: float
    ) -> None:
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

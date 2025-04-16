# filepath: /app/tests/test_fluid_refactored.py
"""
Tests for the fluid simulation components of the watercolor simulation.

This file contains tests for the FluidSimulation class, which handles the
physics of fluid movement in the watercolor rendering system.
"""
import pytest
import numpy as np
import json
from pathlib import Path

from src.simulation.fluid_simulation import FluidSimulation
from src.simulation.watercolor_simulation import WatercolorSimulation

# Import our test utilities
from tests.test_utils import assert_numpy_arrays_almost_equal, TEST_TIME_STEPS


class TestFluidSimulation:
    """Tests for the FluidSimulation class.

    This class groups all tests related to the fluid simulation component,
    allowing for shared setup and teardown operations.
    """

    @pytest.fixture(autouse=True)
    def setup(self, sim_size):
        """Setup run before each test method.

        Args:
            sim_size: Standard simulation size from conftest.py
        """
        self.width, self.height = sim_size

    def test_fluid_init(self, sim_size):
        """Test proper initialization of FluidSimulation with enhanced errors."""
        width, height = sim_size
        sim = FluidSimulation(width, height)

        # Test with enhanced error messages
        assert sim.width == width, f"Expected width {width}, got {sim.width}"
        assert sim.height == height, f"Expected height {height}, got {sim.height}"
        assert sim.u.shape == (height, width + 1), f"Invalid u shape: {sim.u.shape}"
        assert sim.v.shape == (height + 1, width), f"Invalid v shape: {sim.v.shape}"
        assert sim.p.shape == (height, width), f"Invalid pressure shape: {sim.p.shape}"
        assert sim.viscosity == 0.1, f"Wrong default viscosity: {sim.viscosity}"
        assert sim.viscous_drag == 0.01, f"Wrong default drag: {sim.viscous_drag}"
        assert (
            sim.edge_darkening == 0.03
        ), f"Wrong default edge darkening: {sim.edge_darkening}"

    @pytest.mark.parametrize("dt", TEST_TIME_STEPS[:2])  # Using shared test parameters
    def test_update_velocities(self, fluid_sim, paper_with_slope, wet_mask_all, dt):
        """Test velocity update based on pressure and slope with different time steps."""
        initial_u = fluid_sim.u.copy()
        initial_v = fluid_sim.v.copy()

        # Add pressure gradient
        h, w = fluid_sim.height, fluid_sim.width
        center_h, center_w = h // 2, w // 2
        fluid_sim.p[center_h - 1 : center_h + 1, center_w - 1 : center_w + 1] = 0.1

        fluid_sim.update_velocities(paper_with_slope, wet_mask_all, dt=dt)

        # Enhanced assertions
        assert not np.allclose(initial_u, fluid_sim.u), (
            "Velocities (u) did not change after update_velocities, but should have "
            f"with dt={dt}, pressure gradient and slope"
        )
        assert not np.allclose(initial_v, fluid_sim.v), (
            "Velocities (v) did not change after update_velocities, but should have "
            f"with dt={dt}, pressure gradient and slope"
        )

    def test_update_velocities_flat(self, fluid_sim, paper, wet_mask_all):
        """Test velocity update with flat paper (drag only)."""
        # Test with zero pressure and zero slope
        fluid_sim.p[:, :] = 0.0
        paper.height_field[:, :] = 0.5  # Flat paper
        paper.update_capacity()
        fluid_sim.u[:, :] = 0.1  # Initial velocity
        fluid_sim.v[:, :] = 0.1
        initial_u_flat = fluid_sim.u.copy()
        initial_v_flat = fluid_sim.v.copy()

        fluid_sim.update_velocities(paper, wet_mask_all, dt=0.1)

        # Enhanced assertions for drag effect
        assert np.all(
            np.abs(fluid_sim.u) < np.abs(initial_u_flat)
        ), "Velocity magnitude (u) did not decrease due to drag on flat surface"
        assert np.all(
            np.abs(fluid_sim.v) < np.abs(initial_v_flat)
        ), "Velocity magnitude (v) did not decrease due to drag on flat surface"

    @pytest.mark.parametrize("mask_type", ["all", "partial"])
    def test_enforce_boundaries(self, fluid_sim, request, mask_type):
        """Test enforcing boundary conditions with different wet masks."""
        # Get the appropriate mask fixture
        wet_mask = request.getfixturevalue(f"wet_mask_{mask_type}")

        # Set some velocity everywhere
        fluid_sim.u[:, :] = 0.1
        fluid_sim.v[:, :] = 0.1

        fluid_sim._enforce_boundaries(wet_mask)

        # Test boundary conditions with enhanced error reporting
        errors_u = []
        errors_v = []

        # For u velocities
        for i in range(fluid_sim.height):
            for j in range(fluid_sim.width + 1):
                # Check if this u-velocity component is outside wet area
                left_cell = (
                    j - 1 >= 0 and j - 1 < fluid_sim.width and wet_mask[i, j - 1]
                )
                right_cell = j < fluid_sim.width and wet_mask[i, j]
                if not (left_cell or right_cell) and fluid_sim.u[i, j] != 0:
                    errors_u.append(
                        f"u[{i},{j}] = {fluid_sim.u[i,j]} should be 0 (outside wet area)"
                    )
                    if len(errors_u) >= 5:  # Report up to 5 errors for clarity
                        break

        # For v velocities
        for i in range(fluid_sim.height + 1):
            for j in range(fluid_sim.width):
                # Check if this v-velocity component is outside wet area
                top_cell = (
                    i - 1 >= 0 and i - 1 < fluid_sim.height and wet_mask[i - 1, j]
                )
                bottom_cell = i < fluid_sim.height and wet_mask[i, j]
                if not (top_cell or bottom_cell) and fluid_sim.v[i, j] != 0:
                    errors_v.append(
                        f"v[{i},{j}] = {fluid_sim.v[i,j]} should be 0 (outside wet area)"
                    )
                    if len(errors_v) >= 5:  # Report up to 5 errors for clarity
                        break

        # Assertions with detailed error messages
        assert (
            len(errors_u) == 0
        ), f"Boundary enforcement failed for u: {', '.join(errors_u)}"
        assert (
            len(errors_v) == 0
        ), f"Boundary enforcement failed for v: {', '.join(errors_v)}"


class TestFluidIntegration:
    """Tests for integration between fluid and watercolor simulation components.

    These tests verify that the fluid simulation correctly interacts with
    other components of the watercolor renderer.
    """

    def test_integration_with_watercolor(self, sim, paper_with_slope):
        """Test fluid simulation integration with WatercolorSimulation."""
        # Initialize pigment layer in watercolor sim
        h, w = sim.height, sim.width
        sim.pigment[:, :, :] = 0.5  # Add some pigment

        # Create wet areas with gradient
        sim.wet_mask = np.zeros((h, w), dtype=bool)
        sim.wet_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        sim.fluid_amount = np.zeros((h, w))
        sim.fluid_amount[sim.wet_mask] = 0.8

        # Keep copy of initial state
        initial_pigment = sim.pigment.copy()

        # Run fluid simulation step
        sim.update(paper_with_slope, dt=0.1)

        # Verify pigment has moved with enhanced error message
        assert not np.allclose(
            initial_pigment, sim.pigment
        ), "Pigment distribution did not change after fluid simulation update"

        # Verify wet edges have darkening effect (if enabled)
        if sim.fluid_sim.edge_darkening > 0:
            # Find edge pixels - cells that are wet but have at least one dry neighbor
            edge_indices = np.where(
                (sim.wet_mask)
                & (
                    ~np.pad(sim.wet_mask, ((1, 1), (1, 1)), constant_values=True)[
                        1:-1, 2:
                    ]
                )
                | (
                    ~np.pad(sim.wet_mask, ((1, 1), (1, 1)), constant_values=True)[
                        1:-1, :-2
                    ]
                )
                | (
                    ~np.pad(sim.wet_mask, ((1, 1), (1, 1)), constant_values=True)[
                        2:, 1:-1
                    ]
                )
                | (
                    ~np.pad(sim.wet_mask, ((1, 1), (1, 1)), constant_values=True)[
                        :-2, 1:-1
                    ]
                )
            )

            # Sample a few edge points to verify darkening
            if edge_indices[0].size > 0:
                sample_size = min(10, edge_indices[0].size)
                sample_indices = np.random.choice(
                    edge_indices[0].size, sample_size, replace=False
                )

                darkening_failures = []
                for idx in sample_indices:
                    i, j = edge_indices[0][idx], edge_indices[1][idx]
                    # Edge points should be darker than initial pigment due to edge darkening
                    if not np.all(sim.pigment[i, j] <= initial_pigment[i, j]):
                        darkening_failures.append(
                            f"Edge pixel at [{i},{j}] not darkened"
                        )
                        if len(darkening_failures) >= 3:
                            break

                assert (
                    len(darkening_failures) == 0
                ), f"Edge darkening effect not applied correctly: {', '.join(darkening_failures)}"

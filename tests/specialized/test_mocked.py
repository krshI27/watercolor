# filepath: /app/tests/test_mocked.py
"""
Tests using mock objects and advanced fixture parametrization.

This module demonstrates testing components in isolation by mocking their dependencies.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from watercolor.simulation.fluid_simulation import FluidSimulation
from watercolor.simulation.watercolor_simulation import WatercolorSimulation


# --- Advanced Parametrized Fixtures ---


@pytest.fixture(params=[(20, 15), (50, 30), (100, 75)])
def variable_sim_size(request):
    """Fixture providing different simulation sizes."""
    return request.param


@pytest.fixture(params=[1, 3, 5])
def glaze_iterations(request):
    """Fixture providing different numbers of glaze iterations."""
    return request.param


class TestWithMocks:
    """Tests that use mocking to isolate components."""

    def test_isolated_fluid_update(self, mocker):
        """Test fluid simulation update with mocked paper."""
        # Create real fluid simulation
        fluid_sim = FluidSimulation(20, 15)

        # Create a mock Paper object
        mock_paper = mocker.MagicMock()

        # Configure the mock
        mock_paper.height_field = np.zeros((15, 20))  # Flat paper
        mock_paper.width = 20
        mock_paper.height = 15

        # Create a wet mask
        wet_mask = np.ones((15, 20), dtype=bool)

        # Add some velocity
        fluid_sim.u[:, :] = 0.1
        fluid_sim.v[:, :] = 0.1

        # Call the method under test
        fluid_sim.update(mock_paper, wet_mask, dt=0.1)

        # Verify the mock was used correctly
        mock_paper.update_capacity.assert_called_once()

    def test_watercolor_sim_with_mock_fluid(self, mocker):
        """Test WatercolorSimulation with mocked FluidSimulation."""
        # Patch the FluidSimulation class
        with patch(
            "src.simulation.watercolor_simulation.FluidSimulation"
        ) as MockFluidSim:
            # Configure the mock
            mock_fluid_sim_instance = MagicMock()
            MockFluidSim.return_value = mock_fluid_sim_instance

            # Create the watercolor simulation (which will use our mock)
            sim = WatercolorSimulation(20, 15)

            # Verify FluidSimulation was instantiated with correct params
            MockFluidSim.assert_called_once_with(
                20,
                15,
                viscosity=mocker.ANY,
                viscous_drag=mocker.ANY,
                edge_darkening=mocker.ANY,
            )

            # Create a mock paper
            mock_paper = mocker.MagicMock()
            mock_paper.height_field = np.zeros((15, 20))
            mock_paper.fluid_capacity = np.ones((15, 20)) * 0.5

            # Set up the simulation
            sim.wet_mask = np.ones((15, 20), dtype=bool)
            sim.fluid_amount = np.ones((15, 20)) * 0.5
            sim.pigment = np.ones((15, 20, 3)) * 0.5

            # Update the simulation
            sim.update(mock_paper, dt=0.1)

            # Verify fluid_sim.update was called
            mock_fluid_sim_instance.update.assert_called_once()


class TestWithParametrizedFixtures:
    """Tests using advanced parametrized fixtures."""

    def test_varying_sim_sizes(self, variable_sim_size):
        """Test that simulation works with different sizes."""
        width, height = variable_sim_size

        # Create objects with parametrized size
        fluid_sim = FluidSimulation(width, height)

        # Verify dimensions
        assert fluid_sim.width == width
        assert fluid_sim.height == height
        assert fluid_sim.u.shape == (height, width + 1)
        assert fluid_sim.v.shape == (height + 1, width)

        # Run a basic update to verify functionality
        paper = MagicMock()
        paper.height_field = np.zeros((height, width))
        paper.fluid_capacity = np.ones((height, width)) * 0.5

        wet_mask = np.ones((height, width), dtype=bool)

        # This should run without errors for all sizes
        fluid_sim.update(paper, wet_mask, dt=0.1)

        # If we got here without errors, the test passed
        assert True

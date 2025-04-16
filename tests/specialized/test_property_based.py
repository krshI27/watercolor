# filepath: /app/tests/test_property_based.py
"""
Property-based tests for the watercolor simulation.

This module uses Hypothesis to define properties that should hold true across
a range of possible inputs, allowing us to test behaviors rather than just specific cases.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from src.simulation.fluid_simulation import FluidSimulation
from src.simulation.paper import Paper
from tests.test_utils import assert_numpy_arrays_almost_equal


class TestFluidSimulationProperties:
    """Property-based tests for fluid simulation."""

    @given(
        width=st.integers(min_value=5, max_value=50),
        height=st.integers(min_value=5, max_value=50),
    )
    def test_fluid_sim_initialization(self, width, height):
        """Test that fluid simulation initializes correctly with various dimensions."""
        fluid_sim = FluidSimulation(width, height)

        # Check dimensions
        assert fluid_sim.width == width
        assert fluid_sim.height == height
        assert fluid_sim.u.shape == (height, width + 1)
        assert fluid_sim.v.shape == (height + 1, width)
        assert fluid_sim.p.shape == (height, width)

        # Check initialized values
        assert np.all(fluid_sim.u == 0)
        assert np.all(fluid_sim.v == 0)
        assert np.all(fluid_sim.p == 0)

    @given(
        viscosity=st.floats(min_value=0.01, max_value=1.0),
        drag=st.floats(min_value=0.001, max_value=0.1),
    )
    def test_fluid_parameters(self, viscosity, drag):
        """Test that fluid simulation accepts various parameter values."""
        fluid_sim = FluidSimulation(20, 15, viscosity=viscosity, viscous_drag=drag)

        assert fluid_sim.viscosity == viscosity
        assert fluid_sim.viscous_drag == drag

    @given(dt=st.floats(min_value=0.01, max_value=0.2))
    def test_velocity_decay(self, dt, fluid_sim, paper, wet_mask_all):
        """Test that velocity decays with drag over time."""
        # Set initial velocity
        fluid_sim.u[:, :] = 0.1
        fluid_sim.v[:, :] = 0.1

        # Ensure flat paper to isolate drag effect
        paper.height_field[:, :] = 0.5
        paper.update_capacity()

        # Zero out pressure
        fluid_sim.p[:, :] = 0

        initial_energy = np.sum(fluid_sim.u**2 + fluid_sim.v**2)

        # Update step
        fluid_sim.update_velocities(paper, wet_mask_all, dt=dt)

        # Energy should decrease due to drag
        final_energy = np.sum(fluid_sim.u**2 + fluid_sim.v**2)
        assert (
            final_energy < initial_energy
        ), f"Energy should decrease: {initial_energy} -> {final_energy}"


class TestPaperProperties:
    """Property-based tests for paper simulation."""

    @given(
        width=st.integers(min_value=5, max_value=50),
        height=st.integers(min_value=5, max_value=50),
        c_min=st.floats(min_value=0.1, max_value=0.4),
        c_max=st.floats(min_value=0.5, max_value=0.9),
    )
    def test_paper_capacity_range(self, width, height, c_min, c_max):
        """Test that paper capacity is within expected range."""
        paper = Paper(width, height, c_min=c_min, c_max=c_max)

        # Check capacity values are within range
        assert np.all(paper.fluid_capacity >= c_min)
        assert np.all(paper.fluid_capacity <= c_max)

        # Check height field values are normalized
        assert np.all(paper.height_field >= 0)
        assert np.all(paper.height_field <= 1)


class TestKubelkaMunkProperties:
    """Property-based tests for Kubelka-Munk color model."""

    @given(
        k=arrays(
            np.float32, shape=3, elements=st.floats(min_value=0.05, max_value=0.95)
        ),
        s=arrays(
            np.float32, shape=3, elements=st.floats(min_value=0.05, max_value=0.95)
        ),
    )
    def test_color_conservation(self, k, s):
        """Test that Kubelka-Munk reflectance is physically plausible."""
        from src.simulation.kubelka_munk import km_reflectance

        # Calculate reflectance
        reflectance = km_reflectance(k, s)

        # Reflectance should be in [0, 1] for physically plausible colors
        assert np.all(reflectance >= 0)
        assert np.all(reflectance <= 1)

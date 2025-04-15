# filepath: /app/test_fluid.py
import pytest
import numpy as np
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.fluid_simulation import FluidSimulation
from simulation.paper import Paper  # Needed for slope input
from simulation.watercolor_simulation import (
    WatercolorSimulation,
)  # For integration context


# Fixtures
@pytest.fixture
def fluid_sim():
    """Fixture for a basic FluidSimulation instance."""
    return FluidSimulation(10, 10)


@pytest.fixture
def paper():
    """Fixture for a Paper instance."""
    p = Paper(10, 10)
    # Create a simple slope for testing
    p.height_field = np.linspace(0, 0.1, 100).reshape(10, 10)
    p.update_capacity()
    return p


@pytest.fixture
def wet_mask_all():
    """Fixture for a fully wet mask."""
    return np.ones((10, 10), dtype=bool)


@pytest.fixture
def wet_mask_partial():
    """Fixture for a partially wet mask."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    return mask


# --- FluidSimulation Tests ---


def test_fluid_init():
    sim = FluidSimulation(20, 30)
    assert sim.width == 20
    assert sim.height == 30
    assert sim.u.shape == (30, 21)
    assert sim.v.shape == (31, 20)
    assert sim.p.shape == (30, 20)
    assert sim.viscosity == 0.1  # Default
    assert sim.viscous_drag == 0.01  # Default
    assert sim.edge_darkening == 0.03  # Default


@pytest.mark.timeout(60)
def test_update_velocities(fluid_sim, paper, wet_mask_all):
    """Test velocity update based on pressure and slope."""
    initial_u = fluid_sim.u.copy()
    initial_v = fluid_sim.v.copy()

    # Add pressure gradient
    fluid_sim.p[4:6, 4:6] = 0.1

    fluid_sim.update_velocities(paper, wet_mask_all, dt=0.1)

    assert fluid_sim.u.shape == (10, 11)
    assert fluid_sim.v.shape == (11, 10)
    # Velocities should change due to pressure and slope
    assert not np.allclose(initial_u, fluid_sim.u)
    assert not np.allclose(initial_v, fluid_sim.v)

    # Test with zero pressure and zero slope
    fluid_sim.p[:, :] = 0.0
    paper.height_field[:, :] = 0.5  # Flat paper
    paper.update_capacity()
    fluid_sim.u[:, :] = 0.1  # Initial velocity
    fluid_sim.v[:, :] = 0.1
    initial_u_flat = fluid_sim.u.copy()
    initial_v_flat = fluid_sim.v.copy()
    fluid_sim.update_velocities(paper, wet_mask_all, dt=0.1)
    # Velocity should decrease due to drag, but not pressure/slope
    assert np.all(np.abs(fluid_sim.u) < np.abs(initial_u_flat))
    assert np.all(np.abs(fluid_sim.v) < np.abs(initial_v_flat))


def test_divergence(fluid_sim):
    """Test the divergence calculation."""
    # Test zero divergence for zero velocity
    div_zero = fluid_sim._divergence()
    assert np.allclose(div_zero, 0.0)

    # Test known divergence: flow expanding from center
    fluid_sim.u[:, 5] = 0.1
    fluid_sim.u[:, 6] = 0.1
    fluid_sim.v[5, :] = 0.1
    fluid_sim.v[6, :] = 0.1
    div_expand = fluid_sim._divergence()
    # Expect positive divergence around the center outflow boundaries
    assert div_expand[5, 5] > 0  # (u[:,6]-u[:,5]) + (v[6,:]-v[5,:]) at [5,5]
    # Expect negative divergence where flow enters
    # (Difficult to set up perfectly on staggered grid without boundary effects)

    # Test known divergence: flow converging to center
    fluid_sim.u[:, :] = 0.0
    fluid_sim.v[:, :] = 0.0
    fluid_sim.u[:, 4] = 0.1
    fluid_sim.u[:, 5] = -0.1
    fluid_sim.v[4, :] = 0.1
    fluid_sim.v[5, :] = -0.1
    div_converge = fluid_sim._divergence()
    # Expect negative divergence near the center
    assert div_converge[4, 4] < 0


@pytest.mark.timeout(60)
def test_relax_divergence(fluid_sim, wet_mask_all):
    """Test divergence relaxation process."""
    # Introduce divergence
    fluid_sim.u[:, 5] = 0.1
    fluid_sim.v[5, :] = -0.1
    initial_divergence = fluid_sim._divergence()
    max_initial_div = np.max(np.abs(initial_divergence))
    assert max_initial_div > 1e-6

    # Relax divergence
    fluid_sim.relax_divergence(wet_mask_all, iterations=50, tolerance=0.001)

    final_divergence = fluid_sim._divergence()
    max_final_div = np.max(np.abs(final_divergence))

    # Divergence should be significantly reduced
    assert max_final_div < max_initial_div
    assert max_final_div < 0.01  # Check against a threshold


def test_enforce_boundaries(fluid_sim, wet_mask_partial):
    """Test enforcing boundary conditions (zero velocity outside wet mask)."""
    # Set some velocity everywhere
    fluid_sim.u[:, :] = 0.1
    fluid_sim.v[:, :] = 0.1

    fluid_sim._enforce_boundaries(wet_mask_partial)

    # Check velocities inside the wet mask (should be mostly unchanged, except near edge)
    # Note: _enforce_boundaries zeros out velocities *at* the boundary edge cells
    # Check a cell clearly inside
    assert fluid_sim.u[4, 5] != 0.0
    assert fluid_sim.v[5, 4] != 0.0

    # Check velocities outside the wet mask (should be zero)
    assert np.allclose(fluid_sim.u[0, :], 0.0)
    assert np.allclose(fluid_sim.u[:, 0], 0.0)
    assert np.allclose(fluid_sim.u[9, :], 0.0)
    assert np.allclose(fluid_sim.u[:, 10], 0.0)
    assert np.allclose(fluid_sim.v[0, :], 0.0)
    assert np.allclose(fluid_sim.v[:, 0], 0.0)
    assert np.allclose(fluid_sim.v[10, :], 0.0)
    assert np.allclose(fluid_sim.v[:, 9], 0.0)

    # Check specific cells outside the mask[2:8, 2:8]
    assert np.allclose(fluid_sim.u[1, 1], 0.0)
    assert np.allclose(fluid_sim.v[1, 1], 0.0)
    assert np.allclose(fluid_sim.u[8, 8], 0.0)
    assert np.allclose(fluid_sim.v[8, 8], 0.0)

    # Check boundary cells (should be zeroed)
    assert np.allclose(
        fluid_sim.u[2, 2], 0.0
    )  # u[i,j] depends on mask[i,j-1] and mask[i,j]
    assert np.allclose(
        fluid_sim.v[2, 2], 0.0
    )  # v[i,j] depends on mask[i-1,j] and mask[i,j]


@pytest.mark.timeout(60)
def test_flow_outward(fluid_sim, wet_mask_partial):
    """Test the flow_outward method for edge darkening pressure adjustment."""
    initial_pressure = fluid_sim.p.copy()
    fluid_sim.edge_darkening = 0.05  # Set noticeable factor

    fluid_sim.flow_outward(
        wet_mask_partial, kernel_size=5
    )  # Smaller kernel for small grid

    # Pressure should decrease near the edges of the wet mask
    assert fluid_sim.p[2, 2] < initial_pressure[2, 2]  # Corner
    assert fluid_sim.p[4, 2] < initial_pressure[4, 2]  # Edge
    # Pressure decrease should be less further inside
    assert fluid_sim.p[3, 3] > fluid_sim.p[2, 2]
    # Center pressure should be least affected
    assert np.isclose(fluid_sim.p[4, 4], initial_pressure[4, 4], atol=1e-3)
    # Pressure outside the mask should be unchanged
    assert np.allclose(
        fluid_sim.p[~wet_mask_partial], initial_pressure[~wet_mask_partial]
    )


# --- Performance Tests (from test_performance.py) ---


@pytest.mark.timeout(60)
def test_update_velocities_performance():
    """Test the performance of the update_velocities method with different grid sizes."""
    print("\\nTesting update_velocities performance:")
    results = {}
    for size in [10, 30, 50]:  # Reduced sizes for faster CI
        print(f"\\nGrid size: {size}x{size}")
        # Use WatercolorSimulation to easily set up paper and mask
        sim = WatercolorSimulation(size, size)
        sim.generate_paper(method="perlin", seed=1)
        wet_mask = np.zeros((size, size), dtype=bool)
        wet_mask[: size // 2, : size // 2] = True
        sim.set_wet_mask(wet_mask)

        start_time = time.time()
        # Call the fluid_sim method directly
        sim.fluid_sim.update_velocities(sim.paper, sim.wet_mask, dt=0.1)
        end_time = time.time()
        duration = end_time - start_time
        print(f"update_velocities completed in {duration:.4f} seconds")
        results[size] = duration
    # Basic check: larger sizes should generally take longer
    if 50 in results and 10 in results:
        assert results[50] > results[10]


@pytest.mark.timeout(60)
def test_relax_divergence_performance():
    """Test the performance of the relax_divergence method with different settings"""
    print("\\nTesting relax_divergence performance:")
    results = {}
    size = 30  # Use a slightly larger size for performance test

    # Use WatercolorSimulation for setup
    sim = WatercolorSimulation(size, size)
    sim.generate_paper(method="perlin", seed=1)
    sim.set_wet_mask(np.ones((size, size), dtype=bool))
    # Add some velocity to create divergence
    sim.velocity_u = np.random.rand(size, size + 1) * 0.1
    sim.velocity_v = np.random.rand(size + 1, size) * 0.1
    sim.fluid_sim.u = sim.velocity_u
    sim.fluid_sim.v = sim.velocity_v

    # Test with different max_iterations
    for max_iter in [10, 20, 40]:  # Reduced iterations
        print(f"\\nMax iterations: {max_iter}")
        # Reset pressure before each run
        sim.pressure[:, :] = 0.0
        sim.fluid_sim.p = sim.pressure

        start_time = time.time()
        # Call the fluid_sim method
        sim.fluid_sim.relax_divergence(
            sim.wet_mask, iterations=max_iter, tolerance=0.01
        )
        end_time = time.time()
        duration = end_time - start_time
        print(
            f"relax_divergence with {max_iter} iterations completed in {duration:.4f} seconds"
        )
        results[max_iter] = duration
    # Basic check: more iterations should take longer
    if 40 in results and 10 in results:
        assert results[40] > results[10]

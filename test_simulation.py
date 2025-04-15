import pytest
import numpy as np
import cv2
import os
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.watercolor_simulation import WatercolorSimulation
from simulation.renderer import WatercolorRenderer  # Needed for glazing test
from simulation.paper import Paper  # Needed for some tests
from simulation.fluid_simulation import (
    FluidSimulation,
)  # Needed for relax_divergence test


# Fixtures (consider moving common fixtures to conftest.py later)
@pytest.fixture
def sim():
    """Fixture for a basic WatercolorSimulation instance."""
    return WatercolorSimulation(10, 10)


@pytest.fixture
def pigment_km():
    """Fixture for sample Kubelka-Munk parameters."""
    return {"K": np.array([0.7, 0.2, 0.1]), "S": np.array([0.1, 0.2, 0.8])}


# --- WatercolorSimulation Basic Tests ---


def test_simulation_init():
    sim = WatercolorSimulation(20, 30)
    assert sim.width == 20
    assert sim.height == 30
    assert sim.water_saturation.shape == (30, 20)
    assert sim.velocity_u.shape == (30, 21)
    assert sim.velocity_v.shape == (31, 20)
    assert sim.wet_mask.shape == (30, 20)
    assert sim.paper is not None
    assert sim.fluid_sim is not None


def test_reset(sim, pigment_km):
    """Test resetting the simulation state."""
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.ones((10, 10), dtype=bool)
    sim.set_wet_mask(mask)
    sim.set_pigment_water(idx, mask, 0.5)
    sim.water_saturation[:] = 0.5
    sim.velocity_u[:] = 0.1
    sim.velocity_v[:] = 0.1
    sim.pigment_paper[idx][:] = 0.2

    sim.reset()

    assert np.all(sim.water_saturation == 0)
    assert np.all(sim.velocity_u == 0)
    assert np.all(sim.velocity_v == 0)
    assert np.all(sim.wet_mask == 0)
    assert np.all(sim.pressure == 0)
    assert np.all(sim.divergence == 0)
    assert len(sim.pigment_water) == 0
    assert len(sim.pigment_paper) == 0
    assert len(sim.pigment_properties) == 0
    # Paper should likely persist or be regenerated, depending on desired reset behavior.
    # Assuming paper structure remains after reset for now.
    assert sim.paper is not None


def test_add_pigment(sim, pigment_km):
    idx = sim.add_pigment(
        density=1.0, staining_power=0.5, granularity=0.5, kubelka_munk_params=pigment_km
    )
    assert idx == 0
    assert len(sim.pigment_water) == 1
    assert len(sim.pigment_paper) == 1
    assert len(sim.pigment_properties) == 1
    assert sim.pigment_water[0].shape == (10, 10)
    assert sim.pigment_paper[0].shape == (10, 10)
    assert sim.pigment_properties[0]["kubelka_munk_params"] == pigment_km
    assert sim.pigment_properties[0]["density"] == 1.0
    assert sim.pigment_properties[0]["staining_power"] == 0.5
    assert sim.pigment_properties[0]["granularity"] == 0.5

    idx2 = sim.add_pigment(kubelka_munk_params=pigment_km)
    assert idx2 == 1
    assert len(sim.pigment_properties) == 2


def test_set_pigment_water(sim, pigment_km):
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.8)
    assert np.allclose(sim.pigment_water[idx][mask], 0.8)
    assert np.allclose(sim.pigment_water[idx][~mask], 0.0)
    # Setting pigment should also update the wet mask and saturation
    assert np.all(sim.wet_mask[mask] > 0)
    assert np.all(sim.water_saturation[mask] > 0)


def test_set_wet_mask(sim):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True  # Use boolean mask
    sim.set_wet_mask(mask)
    # Check wet_mask itself
    assert np.all(sim.wet_mask[2:5, 2:5] > 0)
    assert np.all(sim.wet_mask[:2, :] == 0.0)
    assert np.all(sim.wet_mask[5:, :] == 0.0)
    assert np.all(sim.wet_mask[:, :2] == 0.0)
    assert np.all(sim.wet_mask[:, 5:] == 0.0)
    # Check that water saturation is also set
    assert np.all(sim.water_saturation[2:5, 2:5] > 0)
    assert np.all(sim.water_saturation[:2, :] == 0.0)


def test_set_pressure(sim):
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    sim.set_pressure(mask, 0.5)
    assert np.allclose(sim.pressure[mask], 0.5)
    assert np.allclose(sim.pressure[~mask], 0.0)
    # Check fluid_sim pressure too
    assert np.allclose(sim.fluid_sim.p[mask], 0.5)


# --- Simulation Step Tests ---


@pytest.mark.timeout(60)
def test_move_water(sim):
    sim.set_wet_mask(np.ones((10, 10), dtype=bool))
    initial_saturation = sim.water_saturation.copy()
    initial_u = sim.velocity_u.copy()
    initial_v = sim.velocity_v.copy()
    # Add some initial water and pressure gradient to cause movement
    sim.water_saturation[4:6, 4:6] = 0.8
    sim.pressure[3:5, 4:6] = 0.1  # Pressure gradient to push water right/down

    sim.move_water()  # Includes velocity update, advection, boundary enforcement

    # Water should have moved/diffused
    assert not np.allclose(initial_saturation, sim.water_saturation)
    # Velocities should have changed
    assert not np.allclose(initial_u, sim.velocity_u)
    assert not np.allclose(initial_v, sim.velocity_v)
    # Check shapes remain consistent
    assert sim.water_saturation.shape == (10, 10)
    assert sim.velocity_u.shape == (10, 11)
    assert sim.velocity_v.shape == (11, 10)
    # Check saturation bounds
    assert np.all(sim.water_saturation >= 0) and np.all(sim.water_saturation <= 1)


@pytest.mark.timeout(60)
def test_move_pigment(sim, pigment_km):
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)  # Ensure area is wet
    initial_pigment_water = sim.pigment_water[idx].copy()

    # Set some velocity to ensure movement (e.g., flow to the right)
    sim.velocity_u[:, 5:] = 0.1  # Positive u velocity in the right half
    sim.enforce_boundary_conditions()  # Apply boundary conditions after setting velocity

    sim.move_pigment()

    assert sim.pigment_water[idx].shape == (10, 10)
    # Pigment should have moved
    assert not np.allclose(initial_pigment_water, sim.pigment_water[idx])
    # Check if pigment moved generally to the right
    left_sum_before = np.sum(initial_pigment_water[:, :5])
    right_sum_before = np.sum(initial_pigment_water[:, 5:])
    left_sum_after = np.sum(sim.pigment_water[idx][:, :5])
    right_sum_after = np.sum(sim.pigment_water[idx][:, 5:])
    if (
        right_sum_before > 1e-6
    ):  # Avoid division by zero if no pigment initially on right
        assert right_sum_after > right_sum_before or left_sum_after < left_sum_before


@pytest.mark.timeout(60)
def test_transfer_pigment(sim, pigment_km):
    # Use pigment properties relevant to transfer
    idx = sim.add_pigment(
        kubelka_munk_params=pigment_km,
        density=1.2,  # Higher density -> more settling
        staining_power=0.3,  # Lower staining -> more lifting
        granularity=0.8,  # High granularity -> settles in valleys
    )
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    initial_pigment_water = sim.pigment_water[idx].copy()
    initial_pigment_paper = sim.pigment_paper[idx].copy()

    # Ensure paper has some height variation for granularity effect
    sim.paper.height_field[5, 5] = 0.1  # Low spot (valley)
    sim.paper.height_field[4, 4] = 0.9  # High spot (peak)
    sim.paper.update_capacity()  # Update capacity based on new height

    sim.transfer_pigment()

    assert sim.pigment_water[idx].shape == (10, 10)
    assert sim.pigment_paper[idx].shape == (10, 10)

    # Pigment should have transferred from water to paper (deposition)
    assert np.sum(sim.pigment_water[idx][mask]) < np.sum(initial_pigment_water[mask])
    assert np.sum(sim.pigment_paper[idx][mask]) > np.sum(initial_pigment_paper[mask])

    # Check granularity effect (more pigment in low spots)
    # Need to ensure both spots were wet
    if mask[5, 5] and mask[4, 4]:
        assert sim.pigment_paper[idx][5, 5] > sim.pigment_paper[idx][4, 4]

    # Test lifting (desorption) - add pigment to paper first
    sim.reset()
    idx = sim.add_pigment(kubelka_munk_params=pigment_km, staining_power=0.3)
    sim.pigment_paper[idx][mask] = 0.6
    sim.set_wet_mask(mask)  # Make the area wet
    sim.water_saturation[mask] = 0.5  # Ensure there's water to lift into
    initial_pigment_water = sim.pigment_water[idx].copy()
    initial_pigment_paper = sim.pigment_paper[idx].copy()

    sim.transfer_pigment()
    # Pigment should have transferred from paper to water (lifting)
    assert np.sum(sim.pigment_water[idx][mask]) > np.sum(initial_pigment_water[mask])
    assert np.sum(sim.pigment_paper[idx][mask]) < np.sum(initial_pigment_paper[mask])


@pytest.mark.timeout(60)
def test_simulate_capillary_flow(sim):
    # Setup: Wet area surrounded by dry, absorptive paper
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True
    sim.set_wet_mask(mask)
    sim.water_saturation[mask] = 0.8  # Initial water
    sim.paper.fluid_capacity[:, :] = 0.9  # High capacity everywhere
    sim.paper.sizing[:, :] = 0.5  # Moderate sizing (allows absorption)
    # Set simulation parameters relevant to capillary flow
    sim.absorption_rate = 0.1  # alpha
    sim.diffusion_threshold = 0.7  # sigma (threshold for wet expansion)
    sim.min_saturation_for_diffusion = 0.1  # epsilon_d
    sim.min_saturation_to_receive = 0.05  # epsilon_r

    initial_saturation = sim.water_saturation.copy()
    initial_wet_mask = sim.wet_mask.copy()

    sim.simulate_capillary_flow()

    assert sim.water_saturation.shape == (10, 10)
    # Saturation should decrease in the initially wet area due to absorption
    assert np.all(sim.water_saturation[mask] < initial_saturation[mask])
    # Saturation should increase in areas adjacent to the initial wet area due to diffusion
    assert sim.water_saturation[3, 4] > initial_saturation[3, 4]
    assert sim.water_saturation[4, 3] > initial_saturation[4, 3]
    assert sim.water_saturation[6, 5] > initial_saturation[6, 5]
    assert sim.water_saturation[5, 6] > initial_saturation[5, 6]
    # Wet mask should expand because diffusion occurred into areas above threshold
    assert np.sum(sim.wet_mask) > np.sum(initial_wet_mask)
    assert sim.wet_mask[3, 4] > 0
    # Check bounds
    assert np.all(sim.water_saturation >= 0) and np.all(sim.water_saturation <= 1)


@pytest.mark.timeout(60)
def test_flow_outward(sim):
    """Tests the edge darkening effect setup by checking pressure changes."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    sim.set_wet_mask(mask)
    initial_pressure = sim.pressure.copy()
    sim.edge_darkening_factor = 0.05  # Set a noticeable factor

    sim.flow_outward()  # Uses sim.edge_darkening_factor

    assert sim.pressure.shape == (10, 10)
    # Pressure should decrease near the edges of the wet mask
    # Check corners and edges specifically
    assert sim.pressure[2, 2] < initial_pressure[2, 2]
    assert sim.pressure[2, 7] < initial_pressure[2, 7]
    assert sim.pressure[7, 2] < initial_pressure[7, 2]
    assert sim.pressure[7, 7] < initial_pressure[7, 7]
    assert sim.pressure[4, 2] < initial_pressure[4, 2]  # Mid-edge
    assert sim.pressure[2, 4] < initial_pressure[2, 4]  # Mid-edge
    # Pressure should decrease more at the edge than slightly inside
    assert sim.pressure[2, 2] < sim.pressure[3, 3]
    # Pressure in the center should be less affected (or unaffected)
    assert np.isclose(sim.pressure[4, 4], initial_pressure[4, 4], atol=1e-3)
    # No change outside the original wet mask
    assert np.allclose(sim.pressure[~mask], initial_pressure[~mask])


def test_apply_drybrush(sim):
    """Tests the drybrush effect by modifying the wet mask based on saturation."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    sim.set_wet_mask(mask)  # Sets wet_mask and initial saturation

    # Case 1: Saturation above threshold
    sim.water_saturation[:, :] = 0.8
    sim.apply_drybrush(threshold=0.7)
    # Wet mask should be unchanged where it was initially wet
    assert np.all(sim.wet_mask[mask] > 0)
    assert np.all(sim.wet_mask[~mask] == 0)

    # Case 2: Saturation below threshold
    sim.water_saturation[:, :] = 0.5
    sim.apply_drybrush(threshold=0.7)
    # Wet mask should become zero everywhere
    assert np.all(sim.wet_mask == 0)

    # Case 3: Mixed saturation
    sim.set_wet_mask(mask)  # Reset wet mask
    sim.water_saturation[:, :] = 0.0  # Reset saturation
    sim.water_saturation[2:5, 2:5] = 0.9  # High saturation area
    sim.water_saturation[5:8, 5:8] = 0.4  # Low saturation area
    sim.apply_drybrush(threshold=0.6)
    # Wet mask should remain only in the high saturation area
    assert np.all(sim.wet_mask[2:5, 2:5] > 0)
    assert np.all(sim.wet_mask[5:8, 5:8] == 0)
    assert np.all(sim.wet_mask[:2, :] == 0)  # Check outside original mask


# --- Simulation Effects Tests ---


@pytest.mark.timeout(60)
def test_backruns(sim):
    """Test the backrun effect: water diffusing back into a drying area.
    Focus on capillary flow behavior leading to potential backruns.
    """
    # Simulate a drying edge: high saturation inside, low outside, but all wet
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:9, 1:9] = True  # Large wet area
    sim.set_wet_mask(mask)
    sim.water_saturation[:, :] = 0.1  # Low base saturation
    sim.water_saturation[3:7, 3:7] = 0.9  # High saturation core
    sim.paper.fluid_capacity[:, :] = 0.95  # High capacity
    sim.absorption_rate = 0.01  # Slow absorption
    sim.diffusion_threshold = 0.05  # Low threshold for expansion
    sim.min_saturation_for_diffusion = 0.05
    sim.min_saturation_to_receive = 0.01

    initial_saturation = sim.water_saturation.copy()

    # Run capillary flow multiple times to simulate drying and diffusion
    for _ in range(5):
        sim.simulate_capillary_flow()

    # Check that water has diffused outward from the core
    assert (
        sim.water_saturation[2, 5] > initial_saturation[2, 5]
    )  # Check area outside core
    assert sim.water_saturation[5, 2] > initial_saturation[5, 2]
    # Check that the core saturation decreased
    assert np.mean(sim.water_saturation[3:7, 3:7]) < np.mean(
        initial_saturation[3:7, 3:7]
    )
    # Check that saturation increased in the initially low-saturation wet area
    assert np.mean(sim.water_saturation[1:3, 1:3]) > np.mean(
        initial_saturation[1:3, 1:3]
    )

    # A true backrun visualization would require pigment and rendering,
    # but this tests the underlying fluid dynamics (diffusion into less saturated areas).


@pytest.mark.timeout(60)
def test_granulation(sim, pigment_km):
    """Test pigment granulation: settling in paper valleys during transfer."""
    idx = sim.add_pigment(
        kubelka_munk_params=pigment_km, granularity=0.9, density=1.5
    )  # High granularity & density
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.8)
    sim.set_wet_mask(mask)

    # Create distinct valleys and peaks
    sim.paper.height_field[:, :] = 0.5  # Flat baseline
    sim.paper.height_field[5, 5] = 0.1  # Valley
    sim.paper.height_field[4, 4] = 0.9  # Peak
    sim.paper.update_capacity()

    # Run transfer multiple times to allow settling
    for _ in range(3):
        # Need some water movement to keep pigment suspended initially
        sim.velocity_u[:, 5] = 0.01
        sim.move_water()
        sim.transfer_pigment()

    # Check that more pigment is deposited in the valley (low height)
    assert sim.pigment_paper[idx][5, 5] > sim.pigment_paper[idx][4, 4]
    # Check that valley has more pigment than average deposition in the wet area
    avg_deposition = np.mean(sim.pigment_paper[idx][mask])
    assert sim.pigment_paper[idx][5, 5] > avg_deposition


@pytest.mark.timeout(60)
def test_glazing(sim, pigment_km):
    """Test the glazing effect: optical compositing of multiple pigment layers."""
    # Pigment 1 (e.g., Blue)
    km1 = {"K": np.array([0.1, 0.1, 0.8]), "S": np.array([0.1, 0.1, 0.2])}
    idx1 = sim.add_pigment(kubelka_munk_params=km1)

    # Pigment 2 (e.g., Yellow, more transparent)
    km2 = {"K": np.array([0.1, 0.8, 0.1]), "S": np.array([0.2, 0.2, 0.1])}
    idx2 = sim.add_pigment(kubelka_munk_params=km2)

    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    # Apply first glaze (Blue)
    sim.set_pigment_water(idx1, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    sim.transfer_pigment()  # Deposit some blue

    # Reset water/wetness for second glaze, keep deposited pigment
    sim.water_saturation[:] = 0.0
    sim.wet_mask[:] = 0.0
    sim.pigment_water[idx1][:] = 0.0

    # Apply second glaze (Yellow) over the first
    sim.set_pigment_water(idx2, mask, concentration=0.4)
    sim.set_wet_mask(mask)  # Rewet the area
    sim.transfer_pigment()  # Deposit some yellow

    # Render the result
    # renderer = WatercolorRenderer(sim) # Use the one integrated in sim
    result = sim.get_result()

    # Check the rendered output
    assert result.shape == (10, 10, 3)
    assert np.all(result >= 0) and np.all(result <= 1)

    # Check color in the glazed area - should be a mix (e.g., greenish)
    glaze_center_color = result[5, 5, :]
    assert not np.allclose(glaze_center_color, [1.0, 1.0, 1.0])  # Not white
    # Check relative components - expect lower blue, lower yellow, higher green compared to pure
    # This is tricky without exact color math, but check general direction
    # Kinda greenish: R low, G high, B low
    assert glaze_center_color[1] > glaze_center_color[0]  # Green > Red
    assert glaze_center_color[1] > glaze_center_color[2]  # Green > Blue

    # Check area with only first glaze (if we simulated drying/partial overlap)
    # (Current test setup fully overlaps)

    # Check area outside glazes is white
    assert np.allclose(result[0, 0, :], [1.0, 1.0, 1.0])


# --- Main Loop/Integration ---


@pytest.mark.timeout(60)
def test_main_loop(sim, pigment_km):
    """Test running the main simulation loop for several steps."""
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)

    state_before = {
        "sat": sim.water_saturation.copy(),
        "u": sim.velocity_u.copy(),
        "v": sim.velocity_v.copy(),
        "pw": sim.pigment_water[idx].copy(),
        "pp": sim.pigment_paper[idx].copy(),
    }

    num_steps = 5
    sim.main_loop(num_steps)

    # Check that state has changed significantly after multiple steps
    assert not np.allclose(state_before["sat"], sim.water_saturation)
    assert not np.allclose(state_before["u"], sim.velocity_u)
    assert not np.allclose(state_before["v"], sim.velocity_v)
    assert not np.allclose(state_before["pw"], sim.pigment_water[idx])
    assert not np.allclose(state_before["pp"], sim.pigment_paper[idx])
    # Check bounds
    assert np.all(sim.water_saturation >= 0) and np.all(sim.water_saturation <= 1)
    assert np.all(sim.pigment_water[idx] >= 0)
    assert np.all(sim.pigment_paper[idx] >= 0)


@pytest.mark.timeout(60)
def test_get_result(sim, pigment_km):
    """Test the final rendering output via get_result."""
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    sim.transfer_pigment()  # Ensure some pigment is on paper

    img = sim.get_result()

    assert img.shape == (10, 10, 3)
    assert img.dtype == np.float32  # Or float64 depending on Numba output
    assert np.all(img >= 0) and np.all(img <= 1)
    # Check that pigmented area is not white
    assert not np.allclose(img[4, 4, :], [1.0, 1.0, 1.0])
    # Check that non-pigmented area is white
    assert np.allclose(img[0, 0, :], [1.0, 1.0, 1.0])


# --- Tests Moved from test_watercolor.py ---


@pytest.mark.timeout(60)  # Added timeout
def test_update_velocities(sim):
    """Test the update_velocities method of WatercolorSimulation."""
    sim.set_wet_mask(np.ones((10, 10), dtype=bool))
    initial_u = sim.velocity_u.copy()
    initial_v = sim.velocity_v.copy()
    # Add some pressure gradient to ensure velocity changes
    sim.pressure[4:6, 4:6] = 0.1
    # Ensure fluid_sim pressure is updated if necessary (depends on implementation)
    # sim.fluid_sim.p = sim.pressure # Example if direct update is needed

    sim.update_velocities()  # This calls fluid_sim.update_velocities internally

    assert sim.velocity_u.shape == (10, 11)
    assert sim.velocity_v.shape == (11, 10)
    # Velocities should change due to pressure and slope (even if slope is 0)
    assert not np.allclose(initial_u, sim.velocity_u)
    assert not np.allclose(initial_v, sim.velocity_v)
    # Check that the fluid_sim velocities were also updated
    assert np.allclose(sim.velocity_u, sim.fluid_sim.u)
    assert np.allclose(sim.velocity_v, sim.fluid_sim.v)


@pytest.mark.timeout(60)  # Added timeout
def test_compute_paper_slope(sim):
    """Test calculation of paper slope within WatercolorSimulation."""
    # Create a simple ramp height field directly on the simulation's paper object
    sim.paper.height_field = np.linspace(0, 1, sim.width * sim.height).reshape(
        sim.height, sim.width
    )
    # Update the paper's internal slope calculation
    sim.paper.update_slope()
    # Update the simulation's reference to the slope (if needed, depends on init)
    sim.fluid_sim.slope_x = sim.paper.slope_x
    sim.fluid_sim.slope_y = sim.paper.slope_y

    # Call the simulation's method (which might just return paper's slope)
    dx, dy = (
        sim.compute_paper_slope()
    )  # This method might be redundant if paper handles it

    assert dy.shape == (sim.height, sim.width)
    assert dx.shape == (sim.height, sim.width)
    # Slope should be roughly constant for a linear ramp
    assert np.std(dy) < 0.1
    assert np.std(dx) < 0.1
    # dy should be larger than dx for this ramp shape (changes faster vertically)
    assert np.mean(np.abs(dy)) > np.mean(np.abs(dx))
    # Check consistency with paper object
    assert np.allclose(dx, sim.paper.slope_x)
    assert np.allclose(dy, sim.paper.slope_y)


@pytest.mark.timeout(60)  # Added timeout
def test_relax_divergence(sim):
    """Test divergence relaxation called via WatercolorSimulation."""
    sim.set_wet_mask(np.ones((10, 10), dtype=bool))
    # Introduce some divergence directly into fluid_sim
    sim.fluid_sim.u[:, 5] = 0.1
    sim.fluid_sim.v[5, :] = -0.1
    # Ensure sim velocities match if they are separate copies
    sim.velocity_u = sim.fluid_sim.u
    sim.velocity_v = sim.fluid_sim.v

    initial_divergence = sim.fluid_sim._divergence()
    assert np.mean(np.abs(initial_divergence)) > 1e-6

    # Call the simulation's relax_divergence method
    sim.relax_divergence(
        max_iterations=50, tolerance=1e-4
    )  # Use more iterations/tighter tolerance

    final_divergence = sim.fluid_sim._divergence()
    # Divergence should be significantly reduced
    assert np.mean(np.abs(final_divergence)) < np.mean(np.abs(initial_divergence))
    assert (
        np.mean(np.abs(final_divergence)) < 5e-3
    )  # Check against a reasonable threshold
    # Check that simulation's pressure was updated
    assert not np.allclose(sim.pressure, 0.0)
    # Check that velocities were adjusted by pressure gradient
    # (This might require comparing velocities before/after relaxation,
    # but relax_divergence primarily updates pressure)

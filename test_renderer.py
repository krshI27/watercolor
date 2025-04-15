# filepath: /app/test_renderer.py
import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.watercolor_simulation import WatercolorSimulation
from simulation.renderer import WatercolorRenderer
from simulation.kubelka_munk import KubelkaMunk  # For manual checks


# Fixtures
@pytest.fixture
def sim():
    """Fixture for a basic WatercolorSimulation instance."""
    return WatercolorSimulation(10, 10)


@pytest.fixture
def pigment_km():
    """Fixture for sample Kubelka-Munk parameters."""
    return {"K": np.array([0.7, 0.2, 0.1]), "S": np.array([0.1, 0.2, 0.8])}


@pytest.fixture
def renderer(sim):
    """Fixture for a WatercolorRenderer instance."""
    return WatercolorRenderer(sim)


# --- Renderer Tests ---


@pytest.mark.timeout(60)
def test_renderer_init(renderer, sim):
    assert renderer.simulation == sim
    assert isinstance(renderer.km, KubelkaMunk)


@pytest.mark.timeout(60)
def test_renderer_render_single_pigment(renderer, sim, pigment_km):
    """Test rendering a single pigment layer using the renderer method."""
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    sim.transfer_pigment()  # Deposit some pigment

    # Render the single pigment
    img_rendered = renderer.render_pigment(idx)

    # Manually calculate expected reflectance (R) of the pigment layer itself
    thickness = sim.pigment_water[idx] + sim.pigment_paper[idx]
    R_manual, T_manual = KubelkaMunk.get_reflectance_transmittance(
        pigment_km["K"], pigment_km["S"], thickness
    )

    # The render_pigment method returns the reflectance (R) of the layer
    assert img_rendered.shape == (10, 10, 3)
    assert np.allclose(img_rendered, R_manual)
    assert np.all(img_rendered >= 0) and np.all(img_rendered <= 1)
    # Check that pigmented area has non-zero reflectance
    assert not np.allclose(img_rendered[4, 4, :], 0.0)
    # Check that non-pigmented area has zero reflectance
    assert np.allclose(img_rendered[0, 0, :], 0.0)


@pytest.mark.timeout(60)
def test_renderer_render_all_pigments(renderer, sim, pigment_km):
    """Test rendering multiple pigments composited together."""
    # Pigment 1 (Blueish)
    km1 = pigment_km
    idx1 = sim.add_pigment(kubelka_munk_params=km1)
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[2:6, 2:6] = True  # Top-left square
    sim.set_pigment_water(idx1, mask1, concentration=0.6)

    # Pigment 2 (Reddish)
    km2 = {"K": np.array([0.2, 0.8, 0.8]), "S": np.array([0.8, 0.2, 0.2])}
    idx2 = sim.add_pigment(kubelka_munk_params=km2)
    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[4:8, 4:8] = True  # Bottom-right square (overlapping)
    sim.set_pigment_water(idx2, mask2, concentration=0.5)

    # Set wet mask covering both areas
    wet_mask = mask1 | mask2
    sim.set_wet_mask(wet_mask)
    sim.transfer_pigment()  # Deposit pigments

    # Render all pigments
    img = renderer.render_all_pigments()  # Uses white background by default

    # --- Verification ---
    assert img.shape == (10, 10, 3)
    assert np.all(img >= 0) and np.all(img <= 1)

    # Area outside pigments should be white
    assert np.allclose(img[0, 0, :], [1.0, 1.0, 1.0])
    assert np.allclose(img[9, 9, :], [1.0, 1.0, 1.0])

    # Area with only pigment 1 (Blueish) should not be white
    assert not np.allclose(img[3, 3, :], [1.0, 1.0, 1.0])
    # Should be blueish (higher B component)
    assert img[3, 3, 2] > img[3, 3, 0] and img[3, 3, 2] > img[3, 3, 1]

    # Area with only pigment 2 (Reddish) should not be white
    assert not np.allclose(img[6, 6, :], [1.0, 1.0, 1.0])
    # Should be reddish (higher R component)
    assert img[6, 6, 0] > img[6, 6, 1] and img[6, 6, 0] > img[6, 6, 2]

    # Area with both pigments (overlap) should not be white and different from single pigment areas
    assert not np.allclose(img[5, 5, :], [1.0, 1.0, 1.0])
    assert not np.allclose(img[5, 5, :], img[3, 3, :])
    assert not np.allclose(img[5, 5, :], img[6, 6, :])
    # Expect a mixed color (purplish/darker)

    # --- Manual Check for Overlap Area (Pixel 5,5) ---
    # Get thickness for both pigments at (5,5)
    thick1 = sim.pigment_water[idx1][5, 5] + sim.pigment_paper[idx1][5, 5]
    thick2 = sim.pigment_water[idx2][5, 5] + sim.pigment_paper[idx2][5, 5]

    # Get R, T for each layer at this thickness
    R1, T1 = KubelkaMunk.get_reflectance_transmittance(km1["K"], km1["S"], thick1)
    R2, T2 = KubelkaMunk.get_reflectance_transmittance(km2["K"], km2["S"], thick2)

    # Composite manually (Pigment 1 is bottom, Pigment 2 is top in sim list)
    # Render_all_pigments iterates through sim.pigment_properties, applying them bottom-up.
    # So, glaze for idx=0 (Pigment 1) is applied first, then idx=1 (Pigment 2) on top.
    background = np.ones(3)
    # Composite Pigment 1 onto background
    R_1_bg = R1 + (T1**2 * background) / (1.0 - R1 * background + 1e-10)
    T_1_bg = (T1 * 1.0) / (1.0 - R1 * background + 1e-10)
    # Composite Pigment 2 onto (Pigment 1 + background)
    R_manual_overlap = R2 + (T2**2 * R_1_bg) / (1.0 - R2 * R_1_bg + 1e-10)

    # Compare with rendered image at overlap
    assert np.allclose(img[5, 5, :], R_manual_overlap, atol=1e-6)

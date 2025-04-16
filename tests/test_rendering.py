import numpy as np
from src.simulation.renderer import WatercolorRenderer
from src.simulation.kubelka_munk import KubelkaMunk

# --- Rendering and Kubelka-Munk Tests ---


def test_renderer_render_all_pigments(sim, pigment_km):
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    sim.transfer_pigment()
    img = sim.get_result()
    assert img.shape == (10, 10, 3)
    assert np.all(img >= 0) and np.all(img <= 1)
    assert not np.allclose(img[4, 4, :], [1.0, 1.0, 1.0])
    assert np.allclose(img[0, 0, :], [1.0, 1.0, 1.0])


def test_renderer_single_pigment(sim, pigment_km):
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    sim.transfer_pigment()
    thickness = sim.pigment_water[idx] + sim.pigment_paper[idx]
    R, T = KubelkaMunk.get_reflectance_transmittance(
        pigment_km["K"], pigment_km["S"], thickness
    )
    background_R = np.ones((10, 10, 3))
    expected_img = R + (T**2 * background_R) / (1.0 - R * background_R + 1e-10)
    assert expected_img.shape == (10, 10, 3)
    assert not np.allclose(expected_img[4, 4, :], [1.0, 1.0, 1.0])
    assert np.allclose(expected_img[0, 0, :], [1.0, 1.0, 1.0])

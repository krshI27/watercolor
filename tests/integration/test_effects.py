import pytest
import numpy as np

# --- Effects: Edge Darkening, Backruns, Drybrush, Glazing ---


def test_flow_outward(sim):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    sim.set_wet_mask(mask)
    initial_pressure = sim.pressure.copy()
    sim.edge_darkening_factor = 0.05
    sim.flow_outward()
    assert sim.pressure.shape == (10, 10)
    assert sim.pressure[2, 2] < initial_pressure[2, 2]
    assert sim.pressure[4, 2] < initial_pressure[4, 2]
    assert sim.pressure[2, 2] < sim.pressure[3, 3]
    assert np.isclose(sim.pressure[4, 4], initial_pressure[4, 4], atol=1e-3)
    assert np.allclose(sim.pressure[~mask], initial_pressure[~mask])


def test_apply_drybrush(sim):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    sim.set_wet_mask(mask)
    sim.water_saturation[:, :] = 0.8
    sim.apply_drybrush(threshold=0.7)
    assert np.all(sim.wet_mask[mask] > 0)
    assert np.all(sim.wet_mask[~mask] == 0)
    sim.water_saturation[:, :] = 0.5
    sim.apply_drybrush(threshold=0.7)
    assert np.all(sim.wet_mask == 0)
    sim.set_wet_mask(mask)
    sim.water_saturation[:, :] = 0.0
    sim.water_saturation[2:5, 2:5] = 0.9
    sim.water_saturation[5:8, 5:8] = 0.4
    sim.apply_drybrush(threshold=0.6)
    assert np.all(sim.wet_mask[2:5, 2:5] > 0)
    assert np.all(sim.wet_mask[5:8, 5:8] == 0)
    assert np.all(sim.wet_mask[:2, :] == 0)


def test_backruns(sim):
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:9, 1:9] = True
    sim.set_wet_mask(mask)
    sim.water_saturation[:, :] = 0.1
    sim.water_saturation[3:7, 3:7] = 0.9
    sim.paper.fluid_capacity[:, :] = 0.95
    sim.absorption_rate = 0.01
    sim.diffusion_threshold = 0.05
    sim.min_saturation_for_diffusion = 0.05
    sim.min_saturation_to_receive = 0.01
    initial_saturation = sim.water_saturation.copy()
    for _ in range(5):
        sim.simulate_capillary_flow()
    assert sim.water_saturation[2, 5] > initial_saturation[2, 5]
    assert sim.water_saturation[5, 2] > initial_saturation[5, 2]
    assert np.mean(sim.water_saturation[3:7, 3:7]) < np.mean(
        initial_saturation[3:7, 3:7]
    )
    assert np.mean(sim.water_saturation[1:3, 1:3]) > np.mean(
        initial_saturation[1:3, 1:3]
    )


def test_glazing(sim, pigment_km):
    km1 = {"K": np.array([0.1, 0.1, 0.8]), "S": np.array([0.1, 0.1, 0.2])}
    idx1 = sim.add_pigment(kubelka_munk_params=km1)
    km2 = {"K": np.array([0.1, 0.8, 0.1]), "S": np.array([0.2, 0.2, 0.1])}
    idx2 = sim.add_pigment(kubelka_munk_params=km2)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx1, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    sim.transfer_pigment()
    sim.water_saturation[:] = 0.0
    sim.wet_mask[:] = 0.0
    sim.pigment_water[idx1][:] = 0.0
    sim.set_pigment_water(idx2, mask, concentration=0.4)
    sim.set_wet_mask(mask)
    sim.transfer_pigment()
    result = sim.get_result()
    assert result.shape == (10, 10, 3)
    assert np.all(result >= 0) and np.all(result <= 1)
    glaze_center_color = result[5, 5, :]
    assert not np.allclose(glaze_center_color, [1.0, 1.0, 1.0])
    assert glaze_center_color[1] > glaze_center_color[0]
    assert glaze_center_color[1] > glaze_center_color[2]
    assert np.allclose(result[0, 0, :], [1.0, 1.0, 1.0])

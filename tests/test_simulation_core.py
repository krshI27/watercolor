import pytest
import numpy as np

from src.simulation.watercolor_simulation import WatercolorSimulation

# --- Simulation Core Tests ---


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
    assert np.all(sim.wet_mask[mask] > 0)


def test_set_wet_mask(sim):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    sim.set_wet_mask(mask)
    assert np.all(sim.wet_mask[2:5, 2:5] > 0)
    assert np.all(sim.wet_mask[:2, :] == 0.0)
    assert np.all(sim.wet_mask[5:, :] == 0.0)
    assert np.all(sim.wet_mask[:, :2] == 0.0)
    assert np.all(sim.wet_mask[:, 5:] == 0.0)
    assert np.all(sim.water_saturation[2:5, 2:5] > 0)
    assert np.all(sim.water_saturation[:2, :] == 0.0)


def test_set_pressure(sim):
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    sim.set_pressure(mask, 0.5)
    assert np.allclose(sim.pressure[mask], 0.5)
    assert np.allclose(sim.pressure[~mask], 0.0)
    assert np.allclose(sim.fluid_sim.p[mask], 0.5)

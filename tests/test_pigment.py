import pytest
import numpy as np

# --- Pigment and Transfer Tests ---


def test_transfer_pigment(sim, pigment_km):
    idx = sim.add_pigment(kubelka_munk_params=pigment_km, granularity=0.8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    initial_pigment_water = sim.pigment_water[idx].copy()
    initial_pigment_paper = sim.pigment_paper[idx].copy()
    sim.paper.height_field[5, 5] = 0.1
    sim.transfer_pigment()
    assert sim.pigment_water[idx].shape == (10, 10)
    assert sim.pigment_paper[idx].shape == (10, 10)
    assert np.sum(sim.pigment_water[idx]) < np.sum(initial_pigment_water)
    assert np.sum(sim.pigment_paper[idx]) > np.sum(initial_pigment_paper)
    assert sim.pigment_paper[idx][5, 5] > sim.pigment_paper[idx][4, 4]


def test_granulation(sim, pigment_km):
    idx = sim.add_pigment(kubelka_munk_params=pigment_km, granularity=0.9, density=1.5)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.8)
    sim.set_wet_mask(mask)
    sim.paper.height_field[:, :] = 0.5
    sim.paper.height_field[5, 5] = 0.1
    sim.paper.height_field[4, 4] = 0.9
    sim.paper.update_capacity()
    for _ in range(3):
        sim.velocity_u[:, 5] = 0.01
        sim.move_water()
        sim.transfer_pigment()
    assert sim.pigment_paper[idx][5, 5] > sim.pigment_paper[idx][4, 4]
    avg_deposition = np.mean(sim.pigment_paper[idx][mask])
    assert sim.pigment_paper[idx][5, 5] > avg_deposition

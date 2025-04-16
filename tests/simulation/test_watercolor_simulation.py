#!/usr/bin/env python3
"""
Tests for the WatercolorSimulation class.

This file contains all tests related to the watercolor simulation core functionality,
including initialization, pigment handling, and simulation steps.
"""
import pytest
import numpy as np

from src.simulation.watercolor_simulation import WatercolorSimulation


# --- Simulation Initialization Tests ---


def test_simulation_init():
    """Test proper initialization of WatercolorSimulation."""
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
    assert sim.paper is not None


# --- Pigment Handling Tests ---


def test_add_pigment(sim, pigment_km):
    """Test adding a pigment to the simulation."""
    idx = sim.add_pigment(
        density=1.0, staining_power=0.5, granularity=0.5, kubelka_munk_params=pigment_km
    )
    assert idx == 0
    assert len(sim.pigment_water) == 1
    assert len(sim.pigment_paper) == 1
    assert len(sim.pigment_properties) == 1
    assert sim.pigment_water[0].shape == (10, 10)

    # Add another pigment
    idx2 = sim.add_pigment(
        density=0.8, staining_power=0.3, granularity=0.2, kubelka_munk_params=pigment_km
    )
    assert idx2 == 1
    assert len(sim.pigment_water) == 2
    assert len(sim.pigment_paper) == 2
    assert len(sim.pigment_properties) == 2

    # Verify properties were stored
    assert sim.pigment_properties[0]["density"] == 1.0
    assert sim.pigment_properties[0]["staining_power"] == 0.5
    assert sim.pigment_properties[0]["granularity"] == 0.5
    assert sim.pigment_properties[1]["density"] == 0.8
    assert sim.pigment_properties[1]["staining_power"] == 0.3
    assert sim.pigment_properties[1]["granularity"] == 0.2


def test_set_pigment_water(sim, pigment_km):
    """Test setting pigment in water layer."""
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True

    # Set pigment in water
    sim.set_pigment_water(idx, mask, 0.5)
    assert np.all(sim.pigment_water[idx][3:7, 3:7] == 0.5)
    assert np.all(sim.pigment_water[idx][~mask] == 0.0)

    # Modify existing pigment
    sim.set_pigment_water(idx, mask, 0.8)
    assert np.all(sim.pigment_water[idx][3:7, 3:7] == 0.8)


def test_set_wet_mask(sim):
    """Test setting wet mask."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True

    sim.set_wet_mask(mask)
    assert np.all(sim.wet_mask[3:7, 3:7] > 0)
    assert np.all(sim.wet_mask[~mask] == 0)


def test_transfer_pigment(sim, pigment_km):
    """Test transferring pigment from water to paper."""
    idx = sim.add_pigment(
        density=1.0, staining_power=0.5, granularity=0.0, kubelka_munk_params=pigment_km
    )
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True

    # Set up wet region and pigment in water
    sim.set_wet_mask(mask)
    sim.set_pigment_water(idx, mask, 0.5)

    # Before transfer, paper should have no pigment
    assert np.all(sim.pigment_paper[idx] == 0.0)

    # Run transfer
    sim.transfer_pigment()

    # After transfer, both water and paper should have pigment
    assert np.any(
        sim.pigment_paper[idx][3:7, 3:7] > 0.0
    )  # Some pigment transferred to paper
    assert np.any(
        sim.pigment_water[idx][3:7, 3:7] < 0.5
    )  # Some pigment removed from water

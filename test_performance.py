#!/usr/bin/env python3
"""
Performance tests for watercolor simulation - Keep specific benchmarks here.
Individual component performance tests moved to respective test files (e.g., test_fluid.py).
"""

import time
import numpy as np
from simulation.watercolor_simulation import WatercolorSimulation
import pytest

# Fixtures specific to performance tests if needed


# --- Full Cycle Performance ---


@pytest.mark.timeout(120)  # Allow more time for full cycle tests
def test_full_cycle_performance():
    """Test the performance of a full simulation cycle with small grid"""
    print("\nTesting full simulation cycle performance:")

    size = 20  # Slightly larger size for performance
    sim = WatercolorSimulation(size, size)
    sim.generate_paper(method="perlin", seed=1)

    # Set wet area
    wet_mask = np.zeros((size, size), dtype=bool)
    wet_mask[2 : size - 2, 2 : size - 2] = True
    sim.set_wet_mask(wet_mask)

    # Add pigment
    pigment_km = {"K": np.array([0.7, 0.2, 0.1]), "S": np.array([0.1, 0.2, 0.8])}
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((size, size), dtype=float)
    mask[3 : size - 3, 3 : size - 3] = 0.8
    sim.set_pigment_water(idx, mask)

    # Time the full cycle (move_water includes update_velocities and relax_divergence)
    start_time = time.time()

    print("Running move_water...")
    water_start = time.time()
    sim.move_water()
    water_end = time.time()
    print(f"move_water completed in {water_end - water_start:.4f} seconds")

    print("Running move_pigment...")
    pigment_start = time.time()
    sim.move_pigment()
    pigment_end = time.time()
    print(f"move_pigment completed in {pigment_end - pigment_start:.4f} seconds")

    print("Running transfer_pigment...")
    transfer_start = time.time()
    sim.transfer_pigment()
    transfer_end = time.time()
    print(f"transfer_pigment completed in {transfer_end - transfer_start:.4f} seconds")

    print("Running simulate_capillary_flow...")
    capillary_start = time.time()
    sim.simulate_capillary_flow()
    capillary_end = time.time()
    print(
        f"simulate_capillary_flow completed in {capillary_end - capillary_start:.4f} seconds"
    )

    # Add flow_outward for completeness
    print("Running flow_outward...")
    flow_start = time.time()
    sim.flow_outward()
    flow_end = time.time()
    print(f"flow_outward completed in {flow_end - flow_start:.4f} seconds")

    end_time = time.time()
    print(f"Full simulation cycle completed in {end_time - start_time:.4f} seconds")
    assert end_time - start_time < 120  # Check against timeout


@pytest.mark.timeout(180)  # Longer timeout for multi-step performance
def test_multi_step_performance():
    """Test performance over multiple simulation steps."""
    print("\nTesting multi-step simulation performance:")
    size = 30  # Moderate size
    steps = 10  # Moderate number of steps

    sim = WatercolorSimulation(size, size)
    sim.generate_paper(method="perlin", seed=1)
    wet_mask = np.zeros((size, size), dtype=bool)
    wet_mask[5:-5, 5:-5] = True
    sim.set_wet_mask(wet_mask)
    pigment_km = {"K": np.array([0.7, 0.2, 0.1]), "S": np.array([0.1, 0.2, 0.8])}
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((size, size), dtype=float)
    mask[8:-8, 8:-8] = 0.8
    sim.set_pigment_water(idx, mask)

    start_time = time.time()
    sim.main_loop(num_steps=steps)
    end_time = time.time()
    duration = end_time - start_time
    print(
        f"{steps} steps on {size}x{size} grid completed in {duration:.4f} seconds ({duration/steps:.4f} sec/step)"
    )
    assert duration < 180  # Check against timeout


# Remove tests moved to other files
# @pytest.mark.timeout(60)
# def test_update_velocities_performance(): ... MOVED to test_fluid.py

# @pytest.mark.timeout(60)
# def test_move_water_performance(): ... MOVED to test_simulation.py (covered by test_move_water)

# def test_relax_divergence_performance(): ... MOVED to test_fluid.py


# Keep if __name__ == "__main__" block if needed for direct execution,
# but pytest typically handles test running.
# if __name__ == "__main__":
#     test_full_cycle_performance()
#     test_multi_step_performance()

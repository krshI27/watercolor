#!/usr/bin/env python3
"""
Performance testing for the glazes creation process
"""

import time
import numpy as np
import cv2
from pathlib import Path
import argparse

from src.simulation.watercolor_simulation import WatercolorSimulation
from scripts.watercolorize_image import (
    create_paper_structure,
    create_wetness_distribution,
    run_simulation_chunk,
)


def create_mock_args():
    """Create mock arguments for testing."""
    args = argparse.Namespace()
    args.width = 5
    args.height = 5
    args.output_dir = "test_glazes_output"
    args.paper_height = None
    args.paper_capacity = None
    args.paper_wetness = None
    args.num_glazes = 2
    args.steps_per_glaze = 2
    args.edge_darkening = 0.01
    args.viscosity = 0.05
    args.drag = 0.005
    args.verbose = True
    args.save_stages = False
    args.num_pigments = 2
    args.input_image = "test_data/simple_10x10.png"
    args.output = "test_output.png"
    return args


def test_simulation_stages():
    """Test the performance of different simulation stages."""
    print("\n=== Testing Simulation Stages Performance ===")

    # Create a small simulation for testing
    size = 5
    sim = WatercolorSimulation(size, size)
    sim.generate_paper(method="perlin", seed=1)

    # Set wet mask
    wet_mask = np.zeros((size, size), dtype=bool)
    wet_mask[2:4, 2:5] = True
    sim.set_wet_mask(wet_mask)

    # Add pigment
    pigment_km = {"K": np.array([0.7, 0.2, 0.1]), "S": np.array([0.1, 0.2, 0.8])}
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((size, size), dtype=float)
    mask[8:12, 8:12] = 0.8
    sim.set_pigment_water(idx, mask)

    # Timing each stage
    print("\nRunning move_water test (single step)...")
    start_time = time.time()
    sim.move_water()
    end_time = time.time()
    print(f"move_water: {end_time - start_time:.4f} seconds")

    print("\nRunning move_pigment test (single step)...")
    start_time = time.time()
    sim.move_pigment()
    end_time = time.time()
    print(f"move_pigment: {end_time - start_time:.4f} seconds")

    print("\nRunning transfer_pigment test (single step)...")
    start_time = time.time()
    sim.transfer_pigment()
    end_time = time.time()
    print(f"transfer_pigment: {end_time - start_time:.4f} seconds")

    print("\nRunning simulate_capillary_flow test (single step)...")
    start_time = time.time()
    sim.simulate_capillary_flow()
    end_time = time.time()
    print(f"simulate_capillary_flow: {end_time - start_time:.4f} seconds")

    print("\nRunning flow_outward test...")
    start_time = time.time()
    sim.flow_outward()
    end_time = time.time()
    print(f"flow_outward: {end_time - start_time:.4f} seconds")

    # Test multiple simulation steps
    steps = 5
    print(f"\nRunning {steps} full simulation steps...")
    start_time = time.time()
    for _ in range(steps):
        sim.move_water()
        sim.move_pigment()
        sim.transfer_pigment()
        sim.simulate_capillary_flow()
    end_time = time.time()
    print(
        f"{steps} steps: {end_time - start_time:.4f} seconds ({(end_time - start_time)/steps:.4f} seconds per step)"
    )


def optimize_glazes_function():
    """Create an optimized version of the create_glazes function."""
    from scripts.watercolorize_image import create_glazes

    # Create parameters for a small test
    args = create_mock_args()

    # Create pigment parameters and masks
    pigment_params = [
        {"K": np.array([0.7, 0.3, 0.1]), "S": np.array([0.2, 0.3, 0.8])},
        {"K": np.array([0.1, 0.7, 0.3]), "S": np.array([0.8, 0.2, 0.3])},
    ]
    pigment_masks = [
        np.ones((args.height, args.width)) * 0.5,
        np.ones((args.height, args.width)) * 0.5,
    ]

    # Test the standard function
    print("\nRunning create_glazes with minimal parameters...")
    start_time = time.time()
    result = create_glazes(args, pigment_params, pigment_masks)
    end_time = time.time()
    print(f"Standard create_glazes: {end_time - start_time:.4f} seconds")

    return result


if __name__ == "__main__":
    test_simulation_stages()
    optimize_glazes_function()

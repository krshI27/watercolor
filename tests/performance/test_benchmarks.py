# filepath: /app/tests/test_benchmarks.py
"""
Performance benchmark tests for the watercolor simulation.

This module contains tests that measure performance and ensure it meets specified benchmarks.
"""

import pytest
import time
import random
import numpy as np
from pathlib import Path
import json

from watercolor.simulation.watercolor_simulation import WatercolorSimulation
from watercolor.simulation.paper import Paper
from watercolor.simulation.fluid_simulation import FluidSimulation

# --- Constants ---
# Define benchmark thresholds based on typical performance
BENCHMARK_THRESHOLDS = {
    "fluid_update": 0.05,  # 50ms for single fluid update step
    "watercolor_update": 0.2,  # 200ms for watercolor update
    "full_render": 2.0,  # 2s for a full render
}

# Load benchmarks from file if available
BENCHMARK_FILE = Path(__file__).parent / "test_data" / "benchmarks.json"
if BENCHMARK_FILE.exists():
    with open(BENCHMARK_FILE, "r") as f:
        loaded_benchmarks = json.load(f)
        BENCHMARK_THRESHOLDS.update(loaded_benchmarks)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds before each test for reproducibility."""
    random.seed(42)
    np.random.seed(42)
    return


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.mark.performance
    def test_fluid_update_performance(self, fluid_sim, paper, wet_mask_all):
        """Benchmark fluid simulation update performance."""
        # Initialize with some variation
        fluid_sim.p[fluid_sim.height // 2, fluid_sim.width // 2] = 0.2
        fluid_sim.u[:, :] = 0.05
        fluid_sim.v[:, :] = 0.05

        # Measure performance
        start_time = time.time()

        # Run multiple iterations to get a stable measurement
        num_iterations = 10
        for _ in range(num_iterations):
            fluid_sim.update(paper, wet_mask_all, dt=0.1, num_steps=5)

        execution_time = (time.time() - start_time) / num_iterations

        # Check against benchmark threshold
        threshold = BENCHMARK_THRESHOLDS["fluid_update"]
        assert execution_time < threshold, (
            f"Performance regression detected: {execution_time:.4f}s vs {threshold:.4f}s threshold. "
            f"Fluid update is {execution_time/threshold:.1f}x slower than the benchmark."
        )

        # Optional: Print performance metrics
        print(f"\nFluid update: {execution_time:.4f}s (threshold: {threshold:.4f}s)")

    @pytest.mark.performance
    @pytest.mark.parametrize("size", [(20, 15), (50, 40), (100, 75)])
    def test_simulation_scaling(self, size):
        """Test how performance scales with simulation size."""
        width, height = size

        # Create objects for this size
        paper = Paper(width, height)
        sim = WatercolorSimulation(width, height)

        # Set up initial conditions
        sim.wet_mask = np.ones((height, width), dtype=bool)
        sim.fluid_amount = np.ones((height, width)) * 0.5
        sim.pigment[:, :, :] = 0.5

        # Measure single update
        start_time = time.time()
        sim.update(paper, dt=0.1)
        execution_time = time.time() - start_time

        # Log performance data
        pixels = width * height
        print(f"\nSize {width}x{height} ({pixels} pixels): {execution_time:.4f}s")
        print(f"Time per pixel: {execution_time / pixels * 1000:.4f} ms")

        # For really large tests, just log without failing
        if pixels <= 2000:  # Skip assertion for larger sizes
            assert execution_time < BENCHMARK_THRESHOLDS["watercolor_update"], (
                f"Performance regression for {width}x{height}: "
                f"{execution_time:.4f}s vs {BENCHMARK_THRESHOLDS['watercolor_update']:.4f}s threshold"
            )


# Helper function to save current benchmark results
def save_current_benchmarks():
    """Save current performance as benchmark reference."""
    # Run key benchmarks
    fluid_sim = FluidSimulation(20, 15)
    paper = Paper(20, 15)
    wet_mask = np.ones((15, 20), dtype=bool)

    # Measure fluid update
    start = time.time()
    for _ in range(10):
        fluid_sim.update(paper, wet_mask, dt=0.1, num_steps=5)
    fluid_time = (time.time() - start) / 10

    # Measure watercolor update
    sim = WatercolorSimulation(20, 15)
    sim.wet_mask = wet_mask
    sim.fluid_amount[wet_mask] = 0.5
    start = time.time()
    for _ in range(5):
        sim.update(paper, dt=0.1)
    watercolor_time = (time.time() - start) / 5

    # Add safety margin (20%)
    benchmarks = {
        "fluid_update": fluid_time * 1.2,
        "watercolor_update": watercolor_time * 1.2,
    }

    # Save to file
    with open(BENCHMARK_FILE, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"Benchmark thresholds saved to {BENCHMARK_FILE}")


# Uncomment to generate benchmark references
# if __name__ == "__main__":
#     save_current_benchmarks()

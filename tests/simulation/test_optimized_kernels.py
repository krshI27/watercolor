#!/usr/bin/env python3
"""
Tests for the optimized_kernels module.

This file contains tests for the optimized numerical computation kernels used
in the watercolor simulation, primarily implemented with Numba.
"""
import pytest
import numpy as np

from watercolor.simulation.optimized_kernels import advect_pigment_kernel


# --- Kernel Tests ---


def test_advect_pigment_kernel():
    """Test the performance-optimized advection kernel for pigment transport."""
    # Create test arrays
    height, width = 10, 10
    pigment = np.ones((height, width), dtype=np.float32) * 0.5
    u = np.ones((height, width + 1), dtype=np.float32) * 0.1  # rightward flow
    v = np.zeros((height + 1, width), dtype=np.float32)  # no vertical flow
    wet_mask = np.ones((height, width), dtype=np.float32)
    dt = 0.1

    # Make a copy of the original pigment array
    pigment_orig = pigment.copy()

    # Run the kernel
    advect_pigment_kernel(pigment, u, v, wet_mask, dt, height, width)

    # The pigment should have moved rightward
    # Left side should have less pigment, right side should have more
    assert np.mean(pigment[:, : width // 2]) < np.mean(pigment_orig[:, : width // 2])
    assert np.mean(pigment[:, width // 2 :]) > np.mean(pigment_orig[:, width // 2 :])

    # Total pigment should be conserved (approximately)
    assert np.isclose(np.sum(pigment), np.sum(pigment_orig), rtol=1e-4)

    # Boundaries should have zero pigment due to no-flow boundary conditions
    assert np.all(pigment[:, 0] <= pigment_orig[:, 0])

    # Test with zero velocity - should not change the pigment
    pigment = np.ones((height, width), dtype=np.float32) * 0.5
    pigment_orig = pigment.copy()
    u = np.zeros((height, width + 1), dtype=np.float32)
    v = np.zeros((height + 1, width), dtype=np.float32)

    advect_pigment_kernel(pigment, u, v, wet_mask, dt, height, width)
    assert np.allclose(pigment, pigment_orig)

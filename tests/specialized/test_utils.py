"""
Shared utility functions for tests.

This module contains common helper functions and constants used across test files
to reduce code duplication and improve test maintainability.
"""

import numpy as np
import cv2
from pathlib import Path
import os

# --- Constants for Test Parameterization ---
# Common test parameters for reuse across test files
TEST_TIME_STEPS = [0.01, 0.05, 0.1]
TEST_ITERATIONS = [1, 5, 10]
TEST_PIGMENT_VALUES = [
    {"K": np.array([0.7, 0.2, 0.1]), "S": np.array([0.1, 0.2, 0.8])},
    {"K": np.array([0.2, 0.7, 0.1]), "S": np.array([0.2, 0.1, 0.6])},
]
TEST_PAPER_PARAMS = [{"c_min": 0.1, "c_max": 0.5}, {"c_min": 0.4, "c_max": 0.9}]

# --- Image Comparison Functions ---


def compare_images(img1, img2, threshold=0.95):
    """
    Compare two images for similarity.

    Args:
        img1: First image array
        img2: Second image array
        threshold: Similarity threshold (0-1)

    Returns:
        bool: True if images are similar above threshold
        float: Actual similarity score

    Raises:
        AssertionError: If images have different dimensions
    """
    if img1.shape != img2.shape:
        raise AssertionError(
            f"Image dimensions don't match: {img1.shape} vs {img2.shape}"
        )

    # Convert to same format if needed
    if img1.dtype != img2.dtype:
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

    # Calculate similarity (normalized correlation)
    similarity = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)[0][0]
    return similarity >= threshold, similarity


def assert_numpy_arrays_almost_equal(arr1, arr2, msg=None, rtol=1e-5, atol=1e-8):
    """
    Assert that two numpy arrays are almost equal with detailed error messages.

    Args:
        arr1: First array
        arr2: Second array
        msg: Optional message prefix
        rtol: Relative tolerance parameter
        atol: Absolute tolerance parameter

    Raises:
        AssertionError: If arrays are not almost equal with details about differences
    """
    if not np.allclose(arr1, arr2, rtol=rtol, atol=atol):
        # Find indices of differing elements
        diff_mask = ~np.isclose(arr1, arr2, rtol=rtol, atol=atol)
        diff_indices = np.where(diff_mask)

        # Get sample differences
        sample_diffs = []
        for idx in zip(*diff_indices):
            sample_diffs.append(
                f"at {idx}: {arr1[idx]} vs {arr2[idx]} (diff: {arr1[idx] - arr2[idx]})"
            )

        # Limit sample size
        if len(sample_diffs) > 5:
            sample_diffs = sample_diffs[:5] + [
                f"... and {len(sample_diffs) - 5} more differences"
            ]

        diff_msg = "\n".join(sample_diffs)
        error_msg = (
            f"{msg or 'Arrays not almost equal'}\nSample differences:\n{diff_msg}"
        )

        # Include some statistics
        error_msg += f"\nMax absolute difference: {np.max(np.abs(arr1 - arr2))}"
        error_msg += f"\nMean absolute difference: {np.mean(np.abs(arr1 - arr2))}"

        raise AssertionError(error_msg)


# --- File and Path Helpers ---


def ensure_output_directory(directory):
    """
    Ensure an output directory exists.

    Args:
        directory: Path to directory

    Returns:
        Path: The directory path
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_test_data_path(filename):
    """
    Get the absolute path to a test data file.

    Args:
        filename: Name of the test data file

    Returns:
        str: Absolute path to the test data file
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_data", filename
    )

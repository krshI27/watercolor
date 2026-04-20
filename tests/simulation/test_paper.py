#!/usr/bin/env python3
"""
Tests for the Paper module of the watercolor simulation.

This file contains all tests related to the Paper class, which models the behavior
of watercolor paper including height field generation, fluid capacity, and paper sizing.
"""
import pytest
import numpy as np
import cv2
import os
from pathlib import Path

from watercolor.simulation.paper import Paper


# --- Basic Paper Tests ---


def test_paper_init(sim_size):
    """Test proper initialization of Paper with default parameters."""
    width, height = sim_size
    paper = Paper(width, height)

    assert paper.width == width
    assert paper.height == height
    assert paper.c_min == 0.3  # Default
    assert paper.c_max == 0.7  # Default
    assert paper.height_field.shape == (height, width)
    assert paper.fluid_capacity.shape == (height, width)
    assert paper.sizing.shape == (height, width)

    # Check default generation happened
    assert not np.allclose(paper.height_field, 0.0)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)


@pytest.mark.parametrize("c_min,c_max", [(0.1, 0.5), (0.4, 0.9)])
def test_paper_custom_params(sim_size, c_min, c_max):
    """Test paper initialization with custom parameters."""
    width, height = sim_size
    paper = Paper(width, height, c_min=c_min, c_max=c_max)

    assert paper.c_min == c_min
    assert paper.c_max == c_max
    assert np.min(paper.fluid_capacity) >= c_min
    assert np.max(paper.fluid_capacity) <= c_max


def test_update_capacity(paper):
    """Test capacity update based on height field."""
    original_capacity = paper.fluid_capacity.copy()

    # Change the height field
    paper.height_field = np.linspace(0, 1, paper.width * paper.height).reshape(
        paper.height, paper.width
    )

    # Update capacity
    paper.update_capacity()

    # Capacity should have changed and reflect the new height field
    assert not np.allclose(original_capacity, paper.fluid_capacity)

    # Higher height should generally correlate with lower capacity
    assert (
        np.corrcoef(paper.height_field.flatten(), paper.fluid_capacity.flatten())[0, 1]
        < 0
    )

    # Verify capacity is within c_min and c_max
    assert np.all(paper.fluid_capacity >= paper.c_min)
    assert np.all(paper.fluid_capacity <= paper.c_max)


# --- Paper Generation Tests ---


def test_paper_generation():
    """Test various paper generation methods."""
    paper = Paper(20, 20)
    initial_height = paper.height_field.copy()

    paper.generate("random", seed=1)
    assert paper.height_field.shape == (20, 20)
    assert not np.allclose(initial_height, paper.height_field)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)
    assert np.allclose(
        paper.fluid_capacity,
        paper.height_field * (paper.c_max - paper.c_min) + paper.c_min,
    )

    random_height = paper.height_field.copy()
    paper.generate("perlin", seed=1)
    assert paper.height_field.shape == (20, 20)
    assert not np.allclose(random_height, paper.height_field)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)

    perlin_height = paper.height_field.copy()
    paper.generate("fractal", seed=1)
    assert paper.height_field.shape == (20, 20)
    assert not np.allclose(perlin_height, paper.height_field)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)


# --- Paper Loading Tests ---


def test_load_heightmap(create_test_image):
    """Test loading heightmap from file."""
    # Create a grayscale test image with a gradient
    height_path = create_test_image(
        "test_heightmap.png", channels=1, pattern="gradient"
    )

    # Create a new paper with this heightmap
    paper = Paper(20, 15)
    paper.load_heightmap(height_path)

    # Verify height field reflects the gradient
    # Check values increase along the width
    for y in range(paper.height):
        assert np.all(np.diff(paper.height_field[y, :]) > 0)

    # Verify capacity updated correctly
    assert np.all(paper.fluid_capacity >= paper.c_min)
    assert np.all(paper.fluid_capacity <= paper.c_max)

    # Higher height should correlate with lower capacity
    assert (
        np.corrcoef(paper.height_field.flatten(), paper.fluid_capacity.flatten())[0, 1]
        < 0
    )


def test_load_sizing(create_test_image):
    """Test loading sizing map from file."""
    # Create a grayscale test image with regions
    sizing_path = create_test_image(
        "test_sizing.png", channels=1, pattern="color_regions"
    )

    # Create a new paper and load sizing
    paper = Paper(20, 15)
    original_sizing = paper.sizing.copy()

    paper.load_sizing(sizing_path)

    # Sizing should have changed
    assert not np.allclose(original_sizing, paper.sizing)


def test_paper_loading(tmp_path):
    """Test loading paper properties from images."""
    width, height = 20, 15
    paper = Paper(width, height)

    # Create test images
    height_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    height_path = tmp_path / "height.png"
    cv2.imwrite(str(height_path), height_img)

    sizing_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    sizing_path = tmp_path / "sizing.png"
    cv2.imwrite(str(sizing_path), sizing_img)

    # Load height field
    paper.load_from_image(str(height_path))
    expected_height = height_img.astype(np.float32) / 255.0
    assert np.allclose(paper.height_field, expected_height)

    # Check capacity was updated
    expected_capacity = expected_height * (paper.c_max - paper.c_min) + paper.c_min
    assert np.allclose(paper.fluid_capacity, expected_capacity)

    # Load sizing
    paper.load_sizing(str(sizing_path))
    expected_sizing = sizing_img.astype(np.float32) / 255.0
    assert np.allclose(paper.sizing, expected_sizing)

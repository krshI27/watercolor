# filepath: /app/test_paper.py
import pytest
import numpy as np
import cv2
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent) + "/src")

from src.simulation.paper import Paper


# Fixtures
@pytest.fixture
def paper():
    """Fixture for a basic Paper instance."""
    return Paper(20, 15)  # Use non-square dimensions


@pytest.fixture
def tmp_image_path(tmp_path):
    """Helper fixture to create temporary image files."""

    def _create_image(filename, shape=(15, 20), channels=1):
        if channels == 1:
            img = np.random.randint(0, 256, shape, dtype=np.uint8)
        else:
            img = np.random.randint(0, 256, (*shape, channels), dtype=np.uint8)
        path = tmp_path / filename
        cv2.imwrite(str(path), img)
        return str(path)

    return _create_image


# --- Paper Tests ---


def test_paper_init(paper):
    assert paper.width == 20
    assert paper.height == 15
    assert paper.c_min == 0.3  # Default
    assert paper.c_max == 0.7  # Default
    assert paper.height_field.shape == (15, 20)
    assert paper.fluid_capacity.shape == (15, 20)
    assert paper.sizing.shape == (15, 20)
    # Check default generation happened
    assert not np.allclose(paper.height_field, 0.0)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)
    assert np.all(paper.fluid_capacity >= paper.c_min) and np.all(
        paper.fluid_capacity <= paper.c_max
    )
    assert np.allclose(paper.sizing, 1.0)  # Default sizing


def test_paper_slope(paper):
    """Test calculation of paper surface slope."""
    # Create a simple ramp height field
    paper.height_field = np.linspace(0, 1, paper.width * paper.height).reshape(
        paper.height, paper.width
    )
    dy, dx = paper.slope

    assert dy.shape == (paper.height, paper.width)
    assert dx.shape == (paper.height, paper.width)
    # Slope should be non-zero
    assert np.mean(np.abs(dy)) > 1e-3
    assert np.mean(np.abs(dx)) > 1e-3
    # For this ramp shape using np.linspace with reshape, dy actually has steeper gradient
    assert np.mean(np.abs(dy)) > np.mean(np.abs(dx))

    # Test flat paper
    paper.height_field[:, :] = 0.5
    dy_flat, dx_flat = paper.slope
    assert np.allclose(dy_flat, 0.0)
    assert np.allclose(dx_flat, 0.0)


def test_paper_generation_methods():
    paper = Paper(10, 8)
    # Save initial state
    initial = paper.height_field.copy()
    # Perlin
    paper.generate("perlin", seed=42)
    assert paper.height_field.shape == (8, 10)
    assert not np.allclose(initial, paper.height_field)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)
    # Random
    paper.generate("random", seed=42)
    assert paper.height_field.shape == (8, 10)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)
    # Fractal
    paper.generate("fractal", seed=42)
    assert paper.height_field.shape == (8, 10)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)
    # Capacity always matches
    assert np.allclose(
        paper.fluid_capacity,
        paper.height_field * (paper.c_max - paper.c_min) + paper.c_min,
    )


def test_paper_update_capacity_and_normalize():
    paper = Paper(6, 6)
    paper.height_field[:] = np.arange(36).reshape(6, 6)
    paper._normalize_height()
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)
    paper.update_capacity()
    expected = paper.height_field * (paper.c_max - paper.c_min) + paper.c_min
    assert np.allclose(paper.fluid_capacity, expected)


def test_paper_load_from_image_and_sizing(tmp_path):
    width, height = 7, 5
    paper = Paper(width, height)
    # Height image
    img = (np.linspace(0, 255, width * height).reshape(height, width)).astype(np.uint8)
    img_path = tmp_path / "height.png"
    cv2.imwrite(str(img_path), img)
    paper.load_from_image(str(img_path))
    expected = img.astype(np.float32) / 255.0
    assert np.allclose(paper.height_field, expected)
    # Sizing image
    sizing = (np.ones((height, width)) * 128).astype(np.uint8)
    sizing_path = tmp_path / "sizing.png"
    cv2.imwrite(str(sizing_path), sizing)
    paper.load_sizing(str(sizing_path))
    assert np.allclose(paper.sizing, 128 / 255.0)


def test_paper_load_from_image_invalid(tmp_path):
    paper = Paper(5, 5)
    with pytest.raises(ValueError):
        paper.load_from_image(str(tmp_path / "does_not_exist.png"))
    with pytest.raises(ValueError):
        paper.load_sizing(str(tmp_path / "does_not_exist.png"))


def test_paper_slope_edges_and_corners():
    paper = Paper(4, 4)
    # Flat
    paper.height_field[:] = 1.0
    dy, dx = paper.slope
    assert np.allclose(dy, 0.0)
    assert np.allclose(dx, 0.0)
    # Single spike
    paper.height_field[:] = 0.0
    paper.height_field[0, 0] = 10.0
    dy, dx = paper.slope
    assert dy.shape == (4, 4) and dx.shape == (4, 4)
    # Slope should be largest near the spike
    assert np.max(np.abs(dy)) > 0
    assert np.max(np.abs(dx)) > 0

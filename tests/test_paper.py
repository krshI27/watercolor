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


def test_paper_generation_methods(paper):
    """Test different paper generation methods."""
    initial_height = paper.height_field.copy()

    # Test random
    paper.generate("random", seed=1)
    assert paper.height_field.shape == (15, 20)
    assert not np.allclose(initial_height, paper.height_field)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)
    assert np.allclose(
        paper.fluid_capacity,
        paper.height_field * (paper.c_max - paper.c_min) + paper.c_min,
    )
    random_height = paper.height_field.copy()

    # Test perlin
    paper.generate("perlin", seed=1)
    assert paper.height_field.shape == (15, 20)
    assert not np.allclose(random_height, paper.height_field)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)
    perlin_height = paper.height_field.copy()

    # Test fractal
    paper.generate("fractal", seed=1)
    assert paper.height_field.shape == (15, 20)
    assert not np.allclose(perlin_height, paper.height_field)
    assert np.all(paper.height_field >= 0) and np.all(paper.height_field <= 1)

    # Test invalid method defaults to random
    fractal_height = paper.height_field.copy()
    paper.generate("invalid_method", seed=1)  # Should default to 'random'
    # Re-generate random with same seed to compare
    temp_paper = Paper(20, 15)
    temp_paper.generate("random", seed=1)
    assert np.allclose(paper.height_field, temp_paper.height_field)


def test_paper_normalize_height(paper):
    """Test the internal height normalization."""
    paper.height_field = np.random.rand(15, 20) * 10 - 5  # Values outside [0, 1]
    paper._normalize_height()
    assert np.min(paper.height_field) >= 0.0
    assert np.max(paper.height_field) <= 1.0
    # Check that relative differences are preserved (roughly)
    # This is hard to test precisely, but min should be near 0, max near 1


def test_paper_update_capacity(paper):
    """Test fluid capacity update based on height."""
    paper.height_field = np.linspace(0, 1, 15 * 20).reshape(15, 20)
    paper.update_capacity()
    expected_capacity = paper.height_field * (paper.c_max - paper.c_min) + paper.c_min
    assert np.allclose(paper.fluid_capacity, expected_capacity)
    assert np.min(paper.fluid_capacity) >= paper.c_min
    assert np.max(paper.fluid_capacity) <= paper.c_max


def test_paper_load_from_image(paper, tmp_image_path):
    """Test loading paper height field from an image."""
    height_path = tmp_image_path("height.png", shape=(15, 20))
    img = cv2.imread(height_path, cv2.IMREAD_GRAYSCALE)
    expected_height = img.astype(np.float32) / 255.0

    paper.load_from_image(height_path)

    assert np.allclose(paper.height_field, expected_height)
    # Check if capacity was updated based on loaded height
    expected_capacity = expected_height * (paper.c_max - paper.c_min) + paper.c_min
    assert np.allclose(paper.fluid_capacity, expected_capacity)

    # Test loading image with different size (should resize)
    height_path_resized = tmp_image_path("height_resized.png", shape=(30, 40))
    paper.load_from_image(height_path_resized)
    assert paper.height_field.shape == (15, 20)  # Should match paper dimensions
    # Check if values are reasonable after resize (mean should be similar)
    img_resized = cv2.imread(height_path_resized, cv2.IMREAD_GRAYSCALE)
    expected_mean = np.mean(img_resized) / 255.0
    assert abs(np.mean(paper.height_field) - expected_mean) < 0.1

    # Test loading non-existent file
    with pytest.raises(ValueError, match="Could not load image"):
        paper.load_from_image("non_existent.png")


def test_paper_load_sizing(paper, tmp_image_path):
    """Test loading paper sizing field from an image."""
    sizing_path = tmp_image_path("sizing.png", shape=(15, 20))
    img = cv2.imread(sizing_path, cv2.IMREAD_GRAYSCALE)
    expected_sizing = img.astype(np.float32) / 255.0

    paper.load_sizing(sizing_path)
    assert np.allclose(paper.sizing, expected_sizing)

    # Test loading image with different size (should resize)
    sizing_path_resized = tmp_image_path("sizing_resized.png", shape=(10, 10))
    paper.load_sizing(sizing_path_resized)
    assert paper.sizing.shape == (15, 20)  # Should match paper dimensions

    # Test loading non-existent file
    with pytest.raises(ValueError, match="Could not load image"):
        paper.load_sizing("non_existent.png")


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
    # For this ramp shape (increasing faster along width), dx should be larger
    assert np.mean(np.abs(dx)) > np.mean(np.abs(dy))

    # Test flat paper
    paper.height_field[:, :] = 0.5
    dy_flat, dx_flat = paper.slope
    assert np.allclose(dy_flat, 0.0)
    assert np.allclose(dx_flat, 0.0)

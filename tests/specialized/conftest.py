# filepath: /app/tests/conftest.py
import pytest
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from unittest import mock

# Common imports for all tests
from src.simulation.watercolor_simulation import WatercolorSimulation
from src.simulation.paper import Paper
from src.simulation.fluid_simulation import FluidSimulation
from src.simulation.main import load_input_image

# --- Simulation Fixtures ---


@pytest.fixture
def sim_size():
    """Standard simulation size for tests."""
    return (20, 15)  # width, height - use non-square to catch dimension issues


@pytest.fixture
def sim(sim_size):
    """Fixture for a basic WatercolorSimulation instance."""
    return WatercolorSimulation(sim_size[0], sim_size[1])


@pytest.fixture
def fluid_sim(sim_size):
    """Fixture for a basic FluidSimulation instance."""
    return FluidSimulation(sim_size[0], sim_size[1])


@pytest.fixture
def paper(sim_size):
    """Fixture for a Paper instance."""
    p = Paper(sim_size[0], sim_size[1])
    return p


@pytest.fixture
def paper_with_slope(paper):
    """Fixture for a Paper instance with a slope."""
    width, height = paper.width, paper.height
    # Create a simple slope for testing
    paper.height_field = np.linspace(0, 0.1, width * height).reshape(height, width)
    paper.update_capacity()
    return paper


@pytest.fixture
def pigment_km():
    """Fixture for sample Kubelka-Munk parameters."""
    return {"K": np.array([0.7, 0.2, 0.1]), "S": np.array([0.1, 0.2, 0.8])}


# --- Mask Fixtures ---


@pytest.fixture
def wet_mask_all(sim_size):
    """Fixture for a fully wet mask."""
    return np.ones((sim_size[1], sim_size[0]), dtype=bool)


@pytest.fixture
def wet_mask_partial(sim_size):
    """Fixture for a partially wet mask."""
    mask = np.zeros((sim_size[1], sim_size[0]), dtype=bool)
    # Set central region to wet
    h_center, w_center = sim_size[1] // 2, sim_size[0] // 2
    h_size, w_size = sim_size[1] // 3, sim_size[0] // 3
    mask[
        h_center - h_size : h_center + h_size, w_center - w_size : w_center + w_size
    ] = True
    return mask


# --- Image Fixtures ---


@pytest.fixture
def create_test_image(tmp_path):
    """Helper fixture to create test images with customizable parameters."""

    def _create_image(filename, shape=None, channels=3, pattern=None):
        """
        Create a test image with given parameters.

        Args:
            filename: Name of the file
            shape: Tuple (height, width)
            channels: Number of color channels
            pattern: Optional pattern ('random', 'gradient', 'color_regions')
        """
        if shape is None:
            shape = (15, 20)  # Default to standard test size

        if pattern == "color_regions":
            img = np.zeros((*shape, channels), dtype=np.uint8)
            # Add some distinct color areas
            h_half, w_half = shape[0] // 2, shape[1] // 2
            if channels == 3:
                img[:h_half, :w_half, :] = [200, 50, 50]  # Reddish TL
                img[:h_half, w_half:, :] = [50, 200, 50]  # Greenish TR
                img[h_half:, :w_half, :] = [50, 50, 200]  # Blueish BL
                img[h_half:, w_half:, :] = [200, 200, 50]  # Yellow BR
            else:
                img[:h_half, :w_half] = 50
                img[:h_half, w_half:] = 100
                img[h_half:, :w_half] = 150
                img[h_half:, w_half:] = 200
        elif pattern == "gradient":
            if channels == 1:
                x = np.linspace(0, 255, shape[1])
                img = np.tile(x, (shape[0], 1)).astype(np.uint8)
            else:
                img = np.zeros((*shape, channels), dtype=np.uint8)
                for c in range(channels):
                    factor = (c + 1) / channels
                    x = np.linspace(0, 255 * factor, shape[1])
                    img[:, :, c] = np.tile(x, (shape[0], 1)).astype(np.uint8)
        else:  # random
            if channels == 1:
                img = np.random.randint(0, 256, shape, dtype=np.uint8)
            else:
                img = np.random.randint(0, 256, (*shape, channels), dtype=np.uint8)

        path = tmp_path / filename
        cv2.imwrite(str(path), img)
        return str(path)

    return _create_image


@pytest.fixture
def test_image_path(create_test_image, sim_size):
    """Fixture to create a standard test image."""
    return create_test_image("test_image.png", shape=(sim_size[1], sim_size[0]))


@pytest.fixture
def test_image_with_regions_path(create_test_image, sim_size):
    """Fixture to create a test image with distinct color regions."""
    return create_test_image(
        "test_regions.png", shape=(sim_size[1], sim_size[0]), pattern="color_regions"
    )


@pytest.fixture
def test_image(test_image_path, sim_size):
    """Fixture for a loaded test image."""
    return load_input_image(test_image_path, target_size=sim_size)


# --- Output Fixtures ---


@pytest.fixture
def output_dir(tmp_path):
    """Fixture for a temporary output directory."""
    path = tmp_path / "test_output"
    path.mkdir()
    return str(path)


@pytest.fixture
def mock_args(test_image_path, output_dir, sim_size):
    """Fixture for mock command line arguments."""
    args = argparse.Namespace()
    args.input_image = test_image_path
    args.output = os.path.join(output_dir, "output.png")
    args.width = sim_size[0]
    args.height = sim_size[1]
    args.save_stages = True
    args.output_dir = output_dir
    args.seed = 42
    args.verbose = False
    args.paper_height = None
    args.paper_capacity = None
    args.paper_wetness = None
    args.num_pigments = 2
    args.num_glazes = 1
    args.steps_per_glaze = 5
    args.edge_darkening = 0.03
    args.viscosity = 0.1
    args.drag = 0.01
    return args


# --- Mock Classes ---


class MockExecutor:
    """Mock executor for testing parallel functions sequentially."""

    def __init__(self, max_workers=None):
        pass

    def map(self, func, iterable, chunksize=1):
        return list(map(func, iterable))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_pool_executor():
    """Fixture to mock ProcessPoolExecutor for deterministic testing."""
    with mock.patch("concurrent.futures.ProcessPoolExecutor", MockExecutor):
        yield

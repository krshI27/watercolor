# filepath: /app/test_io_utils.py
import pytest
import numpy as np
import cv2
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation_main import load_input_image, save_output_image


# Fixtures (Copied from test_simulation_main.py)
@pytest.fixture
def test_image_path(tmp_path):
    """Fixture to create a dummy 10x10 PNG image."""
    img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    path = tmp_path / "test_image.png"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def grayscale_image_path(tmp_path):
    """Fixture to create a dummy 10x10 grayscale PNG image."""
    img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    path = tmp_path / "grayscale_image.png"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def output_dir(tmp_path):
    """Fixture for a temporary output directory."""
    path = tmp_path / "test_output"
    path.mkdir()
    return str(path)


# --- Function Tests (Moved from test_simulation_main.py) ---


def test_load_input_image_ok(test_image_path):
    img = load_input_image(test_image_path)
    assert isinstance(img, np.ndarray)
    assert img.shape == (10, 10, 3)
    assert img.dtype == np.float32
    assert np.all(img >= 0) and np.all(img <= 1)


def test_load_input_image_resize(test_image_path):
    img = load_input_image(test_image_path, target_size=(20, 15))
    assert img.shape == (15, 20, 3)


def test_load_input_image_grayscale(grayscale_image_path):
    # Should convert grayscale to BGR
    img = load_input_image(grayscale_image_path)
    assert img.shape == (10, 10, 3)
    assert img.dtype == np.float32


def test_load_input_image_not_found():
    with pytest.raises(ValueError, match="Could not load image"):
        load_input_image("non_existent_image.png")


def test_save_output_image(output_dir):
    img_float = np.random.rand(10, 10, 3).astype(np.float32)
    output_path = os.path.join(output_dir, "output_test.png")
    save_output_image(img_float, output_path)
    assert os.path.exists(output_path)
    # Load back and check
    img_saved = cv2.imread(output_path)
    assert img_saved is not None
    assert img_saved.shape == (10, 10, 3)
    # Check if conversion to uint8 happened correctly (approximate check)
    assert np.allclose(img_saved, (img_float * 255).astype(np.uint8), atol=1)

import pytest
import numpy as np
import cv2
import os
from pathlib import Path
import sys
import argparse
from unittest import mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation_main import load_input_image, save_output_image, parse_arguments, main
from simulation.watercolor_simulation import (
    WatercolorSimulation,
)  # Import needed for MockSim


# Fixtures
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


# --- Function Tests ---


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


def test_parse_arguments():
    # Test with minimal arguments
    test_args = ["script_name.py", "input.png"]
    with mock.patch("sys.argv", test_args):
        args = parse_arguments()
        assert args.input_image == "input.png"
        assert args.output == "output.png"  # Default
        assert args.width is None  # Default

    # Test with more arguments
    test_args_full = [
        "script_name.py",
        "in.jpg",
        "--output",
        "out.bmp",
        "--width",
        "100",
        "--height",
        "80",
        "--steps",
        "20",
        "--seed",
        "123",
        "--verbose",
    ]
    with mock.patch("sys.argv", test_args_full):
        args = parse_arguments()
        assert args.input_image == "in.jpg"
        assert args.output == "out.bmp"
        assert args.width == 100
        assert args.height == 80
        assert args.steps == 20
        assert args.seed == 123
        assert args.verbose is True


# --- Main Function Integration Test ---


@mock.patch("simulation_main.argparse.ArgumentParser")
# @mock.patch("simulation_main.main") # Don't mock main itself for integration test
def test_main_flow(mock_argparser, test_image_path, output_dir):
    """Test the overall structure and flow of the main() function."""
    # Setup mock arguments
    output_filename = os.path.join(output_dir, "main_output.png")
    mock_args = argparse.Namespace(
        input_image=test_image_path,
        output=output_filename,
        width=12,  # Use different size for resize check
        height=8,
        steps=3,  # Keep low for testing
        seed=42,
        verbose=True,
    )
    # Configure the mock parser to return these args
    mock_parser_instance = mock_argparser.return_value
    mock_parser_instance.parse_args.return_value = mock_args

    # Mock functions called by main
    with (
        mock.patch(
            "simulation_main.load_input_image", wraps=load_input_image
        ) as mock_load,
        mock.patch("simulation_main.WatercolorSimulation") as MockSim,
        mock.patch(
            "simulation_main.save_output_image", wraps=save_output_image
        ) as mock_save,
    ):

        # Configure the mock simulation
        mock_sim_instance = MockSim.return_value
        # Create a dummy result image matching the target size
        dummy_result = np.random.rand(mock_args.height, mock_args.width, 3).astype(
            np.float32
        )
        mock_sim_instance.get_result.return_value = dummy_result

        # Call the actual main function
        try:
            main()
        except Exception as e:
            pytest.fail(f"main() function execution failed: {e}")

        # Assertions
        mock_load.assert_called_once_with(
            test_image_path, target_size=(mock_args.width, mock_args.height)
        )
        MockSim.assert_called_once_with(width=mock_args.width, height=mock_args.height)
        # Check if seed was set if provided
        # (Need to inspect calls to np.random.seed or random.seed if main uses them)
        # For now, assume seed is handled internally by WatercolorSimulation if needed

        mock_sim_instance.main_loop.assert_called_once_with(steps=mock_args.steps)
        mock_sim_instance.get_result.assert_called_once()
        mock_save.assert_called_once_with(dummy_result, output_filename)

        # Check if the output file was actually created by the wrapped save_output_image
        assert os.path.exists(output_filename)
        img_saved = cv2.imread(output_filename)
        assert img_saved is not None
        assert img_saved.shape == (mock_args.height, mock_args.width, 3)

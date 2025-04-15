import pytest
import numpy as np
import cv2
import os
from pathlib import Path
import sys
import argparse
from unittest import mock
from concurrent.futures import ProcessPoolExecutor  # Keep original for mock target

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent) + "/src")

import scripts.watercolorize_image as watercolorize_image
from src.simulation.main import load_input_image, save_output_image, main
from src.simulation.watercolor_simulation import WatercolorSimulation


# Mock PoolExecutor for testing parallel functions sequentially
class MockExecutor:
    def __init__(self, max_workers=None):
        pass

    def map(self, func, iterable, chunksize=1):
        return list(map(func, iterable))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Fixtures
@pytest.fixture
def test_image_path(tmp_path):
    """Fixture to create a dummy 20x15 PNG image for integration tests."""
    img = np.random.randint(0, 256, (15, 20, 3), dtype=np.uint8)
    # Add some distinct color areas
    img[:7, :10, :] = [200, 50, 50]  # Reddish TL
    img[7:, 10:, :] = [50, 50, 200]  # Blueish BR
    path = tmp_path / "test_integration_image.png"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def test_image(test_image_path):
    """Fixture for a loaded test image."""
    # Load with target size used in tests
    return load_input_image(test_image_path, target_size=(20, 15))


@pytest.fixture
def output_dir(tmp_path):
    """Fixture for a temporary output directory."""
    path = tmp_path / "integration_output"
    path.mkdir()
    return str(path)


@pytest.fixture
def mock_args_integration(test_image_path, output_dir):
    """Fixture for mock command line arguments for integration tests."""
    args = argparse.Namespace()
    args.input_image = test_image_path
    args.output = os.path.join(output_dir, "integration_output.png")
    args.width = 20  # Match test image size
    args.height = 15  # Match test image size
    args.save_stages = True
    args.output_dir = output_dir
    args.seed = 42
    args.verbose = False
    args.paper_height = None
    args.paper_capacity = None
    args.paper_wetness = None
    args.num_pigments = 2
    args.num_glazes = 1
    args.steps_per_glaze = 3  # Keep low for testing
    args.edge_darkening = 0.03
    args.viscosity = 0.1
    args.drag = 0.01
    return args


# --- Integration Tests ---


@pytest.mark.timeout(60)  # Apply timeout to the integration test
@mock.patch(
    "watercolorize_image.ProcessPoolExecutor", MockExecutor
)  # Mock parallel execution
def test_create_glazes_integration(mock_args_integration, test_image, output_dir):
    """
    Test the full create_glazes function which orchestrates simulation for glazes.
    This is an integration test for the core simulation pipeline within watercolorize_image.
    """
    args = mock_args_integration
    img_resized = test_image  # Already loaded with correct size

    # 1. Run color separation (using mocked executor)
    pigment_params, pigment_masks = watercolorize_image.color_separation(
        img_resized, args.num_pigments
    )
    assert len(pigment_params) == args.num_pigments
    assert len(pigment_masks) == args.num_pigments

    # 2. Run create_glazes
    # This function internally creates paper, wetness, simulation, runs steps, renders
    final_image = watercolorize_image.create_glazes(args, pigment_params, pigment_masks)

    # 3. Validate results
    assert isinstance(final_image, np.ndarray)
    assert final_image.shape == (args.height, args.width, 3)
    assert final_image.dtype == np.float32  # Check dtype
    assert np.all(final_image >= 0) and np.all(final_image <= 1)

    # Check that the output image is not just flat white or black
    assert not np.allclose(final_image, 1.0)
    assert not np.allclose(final_image, 0.0)

    # Check if stages were saved (as requested by args.save_stages)
    assert os.path.exists(os.path.join(output_dir, "paper_height.png"))
    assert os.path.exists(os.path.join(output_dir, "initial_wetness.png"))
    # Check for glaze-specific stages (adjust names based on actual implementation)
    assert os.path.exists(
        os.path.join(output_dir, f"glaze_0_pigment_{args.num_pigments-1}_mask.png")
    )
    assert os.path.exists(os.path.join(output_dir, f"glaze_0_step_0_saturation.png"))
    assert os.path.exists(
        os.path.join(
            output_dir, f"glaze_0_step_{args.steps_per_glaze-1}_saturation.png"
        )
    )
    assert os.path.exists(os.path.join(output_dir, f"glaze_0_final_pigment_paper.png"))
    assert os.path.exists(
        os.path.join(output_dir, "final_watercolor_result.png")
    )  # Check final composite name


@pytest.mark.timeout(120)  # Longer timeout for full pipeline test
@mock.patch("watercolorize_image.plt.show")  # Prevent showing plot
@mock.patch("watercolorize_image.cv2.imwrite")  # Mock saving final image
@mock.patch("watercolorize_image.save_stage_output")  # Mock stage saving
@mock.patch(
    "watercolorize_image.ProcessPoolExecutor", MockExecutor
)  # Mock parallel execution
def test_watercolorize_image_main_pipeline(
    mock_save_stage,
    mock_imwrite,
    mock_plt_show,
    mock_args_integration,
    test_image_path,
    output_dir,
):
    """
    Test the main execution flow of watercolorize_image.py script.
    Mocks saving and showing images, focuses on pipeline execution.
    """
    args = mock_args_integration
    # Adjust args for this test if needed
    args.width = 15  # Smaller for speed
    args.height = 10
    args.steps_per_glaze = 2
    args.num_glazes = 1
    args.num_pigments = 2
    args.save_stages = True  # Ensure stage saving logic is hit

    # Prepare command line arguments for the main function call
    cli_args = [
        "watercolorize_image.py",  # Script name
        args.input_image,  # Positional argument
        "--output",
        args.output,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--steps-per-glaze",
        str(args.steps_per_glaze),
        "--num-glazes",
        str(args.num_glazes),
        "--num-pigments",
        str(args.num_pigments),
        "--seed",
        str(args.seed),
        "--output-dir",
        args.output_dir,
        "--save-stages",
    ]

    # Use mock.patch to simulate running the script with arguments
    with mock.patch("sys.argv", cli_args):
        # Call the main function that's guarded by __name__ == "__main__"
        # Need to execute the script context or call main directly
        try:
            watercolorize_image.main()
        except Exception as e:
            pytest.fail(f"watercolorize_image.main() raised an exception: {e}")

    # --- Assertions ---
    # Check that the pipeline ran and attempted to save the final image
    mock_imwrite.assert_called_once()
    # Check the filename passed to imwrite
    imwrite_args, _ = mock_imwrite.call_args
    assert imwrite_args[0] == args.output
    # Check the image data passed to imwrite (shape and type)
    assert isinstance(imwrite_args[1], np.ndarray)
    assert imwrite_args[1].shape == (args.height, args.width, 3)
    assert imwrite_args[1].dtype == np.uint8  # Should be converted back for saving

    # Check that stage saving was called multiple times
    assert (
        mock_save_stage.call_count > 3
    )  # Expect paper, wetness, glaze steps, final etc.

    # Check that plot showing was not called
    mock_plt_show.assert_not_called()


# --- Main Function Integration Test (Moved from test_simulation.main.py) ---


@mock.patch("simulation.main.argparse.ArgumentParser")
# @mock.patch("simulation.main.main") # Don't mock main itself for integration test
def test_main_flow(mock_argparser, test_image_path, output_dir):
    """Test the overall structure and flow of the main() function from simulation.main.py."""
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
            "simulation.main.load_input_image", wraps=load_input_image
        ) as mock_load,
        mock.patch("simulation.main.WatercolorSimulation") as MockSim,
        mock.patch(
            "simulation.main.save_output_image", wraps=save_output_image
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

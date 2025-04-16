import pytest
import numpy as np
import cv2
import os
import argparse
from src.simulation.watercolor_simulation import WatercolorSimulation


@pytest.fixture
def sim():
    """Fixture for a basic WatercolorSimulation instance."""
    return WatercolorSimulation(10, 10)


@pytest.fixture
def pigment_km():
    """Fixture for sample Kubelka-Munk parameters."""
    return {"K": np.array([0.7, 0.2, 0.1]), "S": np.array([0.1, 0.2, 0.8])}


@pytest.fixture
def test_image_path(tmp_path):
    """Fixture to create a dummy 10x10 PNG image."""
    img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    path = tmp_path / "test_image.png"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def test_image(test_image_path):
    """Fixture for a loaded 10x10 test image."""
    from src.simulation.main import load_input_image

    return load_input_image(test_image_path, target_size=(10, 10))


@pytest.fixture
def output_dir(tmp_path):
    """Fixture for a temporary output directory."""
    path = tmp_path / "test_output"
    path.mkdir()
    return str(path)


@pytest.fixture
def mock_args(test_image_path, output_dir):
    """Fixture for mock command line arguments."""
    args = argparse.Namespace()
    args.input_image = test_image_path
    args.output = os.path.join(output_dir, "output.png")
    args.width = 10
    args.height = 10
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

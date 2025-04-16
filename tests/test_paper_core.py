import numpy as np
import cv2
import pytest
from src.simulation.paper import Paper


def test_paper_generation():
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


def test_paper_loading(tmp_path):
    width, height = 20, 15
    paper = Paper(width, height)
    height_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    height_path = tmp_path / "height.png"
    cv2.imwrite(str(height_path), height_img)
    capacity_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    capacity_path = tmp_path / "capacity.png"
    cv2.imwrite(str(capacity_path), capacity_img)
    sizing_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    sizing_path = tmp_path / "sizing.png"
    cv2.imwrite(str(sizing_path), sizing_img)
    paper.load_from_image(str(height_path))
    expected_height = height_img.astype(np.float32) / 255.0
    assert np.allclose(paper.height_field, expected_height)
    expected_capacity = expected_height * (paper.c_max - paper.c_min) + paper.c_min
    assert np.allclose(paper.fluid_capacity, expected_capacity)
    paper.load_sizing(str(sizing_path))
    expected_sizing = sizing_img.astype(np.float32) / 255.0
    assert np.allclose(paper.sizing, expected_sizing)

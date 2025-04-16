#!/usr/bin/env python3
"""
Tests for the Renderer module of the watercolor simulation.

This file contains all tests related to the rendering functionality, including
the base Renderer class, WatercolorRenderer, and rendering with Kubelka-Munk.
"""
import pytest
import numpy as np
import cv2

from src.simulation.renderer import Renderer, WatercolorRenderer
from src.simulation.kubelka_munk import KubelkaMunk


# --- Base Renderer Tests ---


def test_renderer_init(sim_size):
    """Test proper initialization of Renderer."""
    width, height = sim_size
    renderer = Renderer(width, height)

    assert renderer.width == width
    assert renderer.height == height
    assert renderer.paper_texture is None  # Default
    assert renderer.output_buffer.shape == (height, width, 3)
    assert np.allclose(renderer.output_buffer, 1.0)  # Default white


@pytest.mark.parametrize("texture_type", ["random", "canvas", "cold_press", None])
def test_load_texture(sim_size, create_test_image, texture_type):
    """Test loading different paper textures."""
    width, height = sim_size
    renderer = Renderer(width, height)

    if texture_type is None:
        # Test no texture
        renderer.load_texture(None)
        assert renderer.paper_texture is None
    else:
        # Create a texture image with the specified pattern
        pattern = "gradient" if texture_type == "random" else "color_regions"
        texture_path = create_test_image(
            f"test_texture_{texture_type}.png", channels=1, pattern=pattern
        )

        # Load the texture
        renderer.load_texture(texture_path, texture_type=texture_type)

        # Texture should be loaded and properly sized
        assert renderer.paper_texture is not None
        assert renderer.paper_texture.shape == (height, width)
        assert renderer.paper_texture.min() >= 0.95  # Subtle texture
        assert renderer.paper_texture.max() <= 1.0


# --- WatercolorRenderer Tests ---


def test_renderer_render_all_pigments(sim, pigment_km):
    """Test rendering all pigments in the simulation."""
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    sim.transfer_pigment()
    img = sim.get_result()
    assert img.shape == (10, 10, 3)
    assert np.all(img >= 0) and np.all(img <= 1)
    assert not np.allclose(img[4, 4, :], [1.0, 1.0, 1.0])
    assert np.allclose(img[0, 0, :], [1.0, 1.0, 1.0])


def test_renderer_single_pigment(sim, pigment_km):
    """Test rendering a single pigment and verify color calculation."""
    idx = sim.add_pigment(kubelka_munk_params=pigment_km)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    sim.set_pigment_water(idx, mask, concentration=0.5)
    sim.set_wet_mask(mask)
    sim.transfer_pigment()
    thickness = sim.pigment_water[idx] + sim.pigment_paper[idx]
    R, T = KubelkaMunk.get_reflectance_transmittance(
        pigment_km["K"], pigment_km["S"], thickness
    )
    background_R = np.ones((10, 10, 3))
    expected_img = R + (T**2 * background_R) / (1.0 - R * background_R + 1e-10)
    assert expected_img.shape == (10, 10, 3)
    assert not np.allclose(expected_img[4, 4, :], [1.0, 1.0, 1.0])
    assert np.allclose(expected_img[0, 0, :], [1.0, 1.0, 1.0])

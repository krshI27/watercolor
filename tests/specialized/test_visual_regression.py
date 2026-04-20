# filepath: /app/tests/test_visual_regression.py
"""
Visual regression tests for the watercolor simulation.

This module provides tests that capture rendered outputs and compare
them against reference images to detect unintended visual changes.
"""

import pytest
import numpy as np
import cv2
import os
import json
from pathlib import Path

from watercolor.simulation.watercolor_simulation import WatercolorSimulation
from watercolor.simulation.paper import Paper
from watercolor.simulation.main import load_input_image
from tests.test_utils import compare_images, ensure_output_directory


# Reference directory for visual test images
VISUAL_REFS_DIR = Path(__file__).parent / "test_data" / "visual_references"

# Tolerance for visual comparison
VISUAL_SIMILARITY_THRESHOLD = 0.95


@pytest.fixture(scope="module")
def visual_refs_dir():
    """Ensure visual references directory exists."""
    ensure_output_directory(VISUAL_REFS_DIR)
    return VISUAL_REFS_DIR


class TestVisualRegression:
    """Visual regression tests to ensure rendering quality and consistency."""

    @pytest.fixture(autouse=True)
    def setup_test(self, sim_size, visual_refs_dir):
        """Set up test environment."""
        self.width, self.height = sim_size
        self.visual_refs_dir = visual_refs_dir

        # Create output directory for current test run
        self.output_dir = ensure_output_directory(
            Path(__file__).parent / "test_data" / "visual_output"
        )

    @pytest.mark.visual
    def test_basic_render(self, test_image, paper):
        """Test basic watercolor rendering against reference image."""
        # Set up simulation with test image
        sim = WatercolorSimulation(self.width, self.height)

        # Configure simulation
        sim.configure_from_image(test_image)

        # Set up controlled test conditions
        sim.fluid_amount[sim.wet_mask] = 0.7

        # Run simulation for fixed number of steps
        for _ in range(5):
            sim.update(paper, dt=0.1)

        # Render final image
        rendered = sim.render()

        # Save current output
        output_path = self.output_dir / "basic_render.png"
        cv2.imwrite(str(output_path), rendered)

        # Reference image path
        ref_path = self.visual_refs_dir / "basic_render.png"

        # Create reference if it doesn't exist
        if not ref_path.exists():
            cv2.imwrite(str(ref_path), rendered)
            pytest.skip(f"Reference image created at {ref_path}")

        # Compare with reference
        reference = cv2.imread(str(ref_path))
        is_similar, similarity = compare_images(
            rendered, reference, VISUAL_SIMILARITY_THRESHOLD
        )

        assert is_similar, (
            f"Visual regression detected. Similarity: {similarity:.2%} "
            f"(threshold: {VISUAL_SIMILARITY_THRESHOLD:.2%}). "
            f"Current output saved to {output_path}"
        )

    @pytest.mark.visual
    @pytest.mark.parametrize("edge_darkening", [0.0, 0.03, 0.06])
    def test_edge_darkening_effect(
        self, test_image_with_regions_path, paper, edge_darkening
    ):
        """Test edge darkening visual effect with different parameter values."""
        # Load test image with distinct regions
        img = load_input_image(
            test_image_with_regions_path, target_size=(self.width, self.height)
        )

        # Set up simulation
        sim = WatercolorSimulation(self.width, self.height)
        sim.configure_from_image(img)

        # Configure edge darkening
        sim.fluid_sim.edge_darkening = edge_darkening

        # Run simulation
        for _ in range(5):
            sim.update(paper, dt=0.1)

        # Render result
        rendered = sim.render()

        # Save current output
        filename = f"edge_darkening_{edge_darkening:.2f}.png"
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), rendered)

        # Reference path
        ref_path = self.visual_refs_dir / filename

        # Create reference if needed
        if not ref_path.exists():
            cv2.imwrite(str(ref_path), rendered)
            pytest.skip(f"Reference image created at {ref_path}")

        # Compare with reference
        reference = cv2.imread(str(ref_path))
        is_similar, similarity = compare_images(
            rendered, reference, VISUAL_SIMILARITY_THRESHOLD
        )

        assert is_similar, (
            f"Visual regression for edge_darkening={edge_darkening}. "
            f"Similarity: {similarity:.2%} (threshold: {VISUAL_SIMILARITY_THRESHOLD:.2%})."
        )


def generate_visual_references():
    """Generate all visual reference images."""
    import pytest

    # Run with a special flag to generate references
    os.environ["GENERATE_REFERENCES"] = "1"

    # Run the visual tests - they'll generate references when missing
    pytest.main(["-xvs", __file__])

    del os.environ["GENERATE_REFERENCES"]


if __name__ == "__main__":
    # This allows running just this file to generate reference images
    generate_visual_references()

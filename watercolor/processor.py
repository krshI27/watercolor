import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .config import WatercolorConfig
from .utils import ProgressTracker, find_edge_pixels, mix_colors, plot_debug


class WatercolorProcessor:
    def __init__(self, config: WatercolorConfig):
        self.config = config
        self.num_workers = (
            mp.cpu_count() if config.num_workers == -1 else config.num_workers
        )

    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess the input image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}")

        image = cv2.resize(image, self.config.output_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        return image

    def _decompose_colors(self, image: np.ndarray):
        """Decompose image into palette and labels"""
        from .core import decompose_colors

        return decompose_colors(image, self.config.num_colors)

    def _diffuse_pigment(
        self, pigment: np.ndarray, edges: np.ndarray, rate: float
    ) -> np.ndarray:
        """Apply diffusion to pigment layer"""
        from .core import apply_directional_diffusion

        num_points = int(self.config.num_points_factor * rate)
        max_distance = int(self.config.max_distance_factor * rate)
        falloff = self.config.falloff_base / rate

        return apply_directional_diffusion(
            pigment,
            num_points=num_points,
            max_distance=max_distance,
            falloff=falloff,
            stored_edges=edges,
            branch_probability=self.config.branch_probability,
            branch_length_factor=self.config.branch_length_factor,
            jitter_sigma=self.config.jitter_sigma,
            debug=self.config.debug,
        )

    def process_color_layer(self, args) -> tuple:
        """Process a single color layer"""
        pigment_layer, pigment_color, color_idx = args

        # Store original edges
        original_edges = find_edge_pixels(pigment_layer)

        if self.config.debug:
            plot_debug(f"Initial Layer {color_idx}", pigment_layer, cmap="gray_r")

        for iter_idx in range(self.config.iterations):
            pigment_layer = self._diffuse_pigment(
                pigment_layer, original_edges, self.config.diffusion_rate
            )
            pigment_layer = np.clip(pigment_layer, 0, 1)

            if (
                self.config.debug
                and color_idx == 0
                and iter_idx
                in [0, self.config.iterations // 2, self.config.iterations - 1]
            ):
                plot_debug(f"Diffusion Step {iter_idx}", pigment_layer, cmap="gray_r")

        return pigment_layer, pigment_color, color_idx

    def process_image(self, image_path: str) -> np.ndarray:
        """Main processing pipeline"""
        try:
            # Load and decompose image
            image = self._load_and_preprocess(image_path)
            if self.config.debug:
                plot_debug("Original Image", image)

            palette, labels = self._decompose_colors(image)
            if self.config.debug:
                plot_debug("Color-Reduced Image", palette[labels])

            # Prepare color layers
            color_layers = [
                ((labels == idx).astype(float), color, idx)
                for idx, color in enumerate(palette)
            ]

            # Initialize accumulation
            accumulation = np.ones((*labels.shape, 3))
            progress = ProgressTracker(len(palette), "Processing colors")

            # Process layers in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for pigment_layer, color, idx in executor.map(
                    self.process_color_layer, color_layers
                ):
                    accumulation = mix_colors(accumulation, pigment_layer, color)
                    if self.config.debug:
                        plot_debug(f"Accumulation after color {idx}", accumulation)
                    progress.update()

            if self.config.debug:
                plot_debug("Final Result", accumulation)

            return accumulation

        except Exception as e:
            raise RuntimeError(f"Failed to process image: {e}")

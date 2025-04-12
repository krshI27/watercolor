import cv2
import numpy as np

from .utils import find_edge_pixels, mix_colors, plot_debug


def simulate_watercolor(image_path, **kwargs):
    """Main entry point for watercolor simulation."""
    from .config import WatercolorConfig
    from .processor import WatercolorProcessor

    config = WatercolorConfig(**kwargs)
    processor = WatercolorProcessor(config)
    return processor.process_image(image_path)


def decompose_colors(image, num_colors=5):
    """Decompose image into color palette and labels."""
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    centers = np.clip(centers / np.max(centers, axis=0), 0, 1)

    # Sort palette by brightness
    brightness = np.sum(centers, axis=1)
    sorted_indices = np.argsort(brightness)[::-1]
    centers = centers[sorted_indices]

    # Remap labels
    sorted_labels = np.zeros_like(labels)
    for i, idx in enumerate(sorted_indices):
        sorted_labels[labels == idx] = i

    return centers, sorted_labels.reshape(image.shape[:2])


def apply_directional_diffusion(
    pigment,
    num_points=100,
    max_distance=10,
    falloff=0.5,
    branch_probability=0.15,
    branch_length_factor=0.3,
    jitter_sigma=0.2,
    debug=False,
    stored_edges=None,
):
    """Apply directional diffusion with stored edges."""
    height, width = pigment.shape
    result = pigment.copy()

    if stored_edges is None:
        edges = find_edge_pixels(pigment)
    else:
        edges = stored_edges

    if not edges:
        return result

    for _ in range(num_points):
        x, y, dx, dy = edges[np.random.randint(len(edges))]

        # Add jitter to direction
        if jitter_sigma > 0:
            dx += np.random.normal(0, jitter_sigma)
            dy += np.random.normal(0, jitter_sigma)
            mag = np.sqrt(dx * dx + dy * dy)
            dx, dy = dx / mag, dy / mag

        # Apply diffusion along the direction
        for dist in range(1, max_distance + 1):
            new_x = int(x + dx * dist)
            new_y = int(y + dy * dist)

            if 0 <= new_x < width and 0 <= new_y < height:
                intensity = result[int(y), int(x)] * np.exp(-falloff * dist)
                result[new_y, new_x] = max(result[new_y, new_x], intensity)

                # Branch creation
                if np.random.random() < branch_probability:
                    branch_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
                    branch_dx = dx * np.cos(branch_angle) - dy * np.sin(branch_angle)
                    branch_dy = dx * np.sin(branch_angle) + dy * np.cos(branch_angle)

                    for branch_dist in range(
                        1, int(max_distance * branch_length_factor) + 1
                    ):
                        branch_x = int(new_x + branch_dx * branch_dist)
                        branch_y = int(new_y + branch_dy * branch_dist)

                        if 0 <= branch_x < width and 0 <= branch_y < height:
                            branch_intensity = intensity * np.exp(
                                -falloff * branch_dist
                            )
                            result[branch_y, branch_x] = max(
                                result[branch_y, branch_x], branch_intensity
                            )

    return result


def process_single_edge(args):
    """Process single edge point for parallel processing"""
    # ... existing implementation from watercolor.py ...
    pass

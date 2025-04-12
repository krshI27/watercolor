from functools import lru_cache
from typing import Any, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_edge_pixels(pigment, threshold=0.01):
    """Find edge pixels and their outward directions using Sobel operators."""
    # Calculate gradients using Sobel
    gx = cv2.Sobel(pigment, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(pigment, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)

    # For each potential edge pixel
    height, width = pigment.shape
    edges = []

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if magnitude[y, x] > threshold:
                # Get direction from gradients
                dx = -gx[y, x]
                dy = -gy[y, x]
                # Normalize direction vector
                mag = np.sqrt(dx * dx + dy * dy) + 1e-6
                dx, dy = dx / mag, dy / mag
                edges.append((x, y, dx, dy))

    return edges  # Return list of (x, y, dx, dy) tuples


@lru_cache(maxsize=1000)
def cached_find_edge_pixels(pigment_bytes: bytes, threshold: float = 0.01):
    """Cached version of find_edge_pixels."""
    # Convert bytes back to numpy array
    pigment = np.frombuffer(pigment_bytes, dtype=np.float64).reshape(-1)
    return find_edge_pixels(pigment, threshold)


def plot_debug(title: str, image: np.ndarray, cmap: str = None):
    """Enhanced debug plotting with error handling"""
    try:
        plt.figure(figsize=(5, 5))
        plt.title(title)
        if cmap:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Warning: Failed to plot debug image '{title}': {e}")


def mix_colors(accumulation, pigment_layer, pigment_color):
    """Blend pigment_color into accumulation where pigment_layer > 0."""
    mask = pigment_layer[..., None]  # Expand dims for broadcasting
    blended = accumulation * (1 - mask) + pigment_color * mask
    return blended


class ProgressTracker:
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total = total_steps
        self.current = 0
        self.description = description

    def update(self, amount: int = 1) -> None:
        self.current += amount
        self._print_progress()

    def _print_progress(self) -> None:
        percentage = (self.current / self.total) * 100
        print(f"\r{self.description}: {percentage:.1f}%", end="")
        if self.current >= self.total:
            print()  # New line when done

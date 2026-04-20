"""Top-level `watercolorize()` callable.

Pragmatic v1: pigment-cluster separation + gaussian bleed + edge
darkening + procedural paper texture. Runs in seconds, not minutes.
The heavy physics simulation lives in ``src/simulation/`` for research
work; this module is the production API siblings call into.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans


def _procedural_paper(shape, scale, strength, rng):
    h, w = shape
    noise = rng.normal(0.0, 1.0, size=(h, w))
    fine = gaussian_filter(noise, sigma=scale / 8.0)
    coarse = gaussian_filter(noise, sigma=scale)
    field = 0.7 * fine + 0.3 * coarse
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return 1.0 - strength * (1.0 - field)


def _edge_darken(mask, intensity):
    blurred = gaussian_filter(mask.astype(np.float32), sigma=3.0)
    edge = np.clip(mask.astype(np.float32) - blurred, 0.0, 1.0)
    return 1.0 - intensity * edge


def watercolorize_array(
    image: np.ndarray,
    n_pigments: int = 6,
    bleed: float = 2.5,
    edge_darkness: float = 0.35,
    paper_strength: float = 0.1,
    paper_scale: float = 40.0,
    seed: int = 0,
) -> np.ndarray:
    """Input/output: uint8 RGB ``(H, W, 3)`` numpy array."""
    rgb = image.astype(np.float32)
    h, w, _ = rgb.shape
    km = KMeans(n_clusters=n_pigments, n_init=3, random_state=seed).fit(rgb.reshape(-1, 3))
    labels = km.labels_.reshape(h, w)
    palette = km.cluster_centers_

    canvas = np.ones_like(rgb) * 255.0
    for idx in range(n_pigments):
        mask = (labels == idx).astype(np.float32)
        if mask.sum() == 0:
            continue
        soft = np.clip(gaussian_filter(mask, sigma=bleed), 0.0, 1.0)
        darkened = _edge_darken(mask, edge_darkness)
        layer = palette[idx][None, None, :] * darkened[..., None]
        alpha = soft[..., None]
        canvas = canvas * (1.0 - alpha) + layer * alpha

    rng = np.random.default_rng(seed)
    paper = _procedural_paper((h, w), paper_scale, paper_strength, rng)
    canvas = canvas * paper[..., None]
    return np.clip(canvas, 0, 255).astype(np.uint8)


def watercolorize(image: Union[Image.Image, np.ndarray], **params) -> Image.Image:
    """Convenience: accepts PIL or numpy, returns a PIL ``RGB`` image."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    out = watercolorize_array(arr, **params)
    return Image.fromarray(out, mode="RGB")

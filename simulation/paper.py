#!/usr/bin/env python3
"""
Paper model for watercolor simulation.
Based on Section 4.1 of 'Computer-Generated Watercolor'.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

class Paper:
    """
    Models paper properties that affect watercolor behavior:
    - Height field (rough surface texture)
    - Fluid capacity (porous structure)
    - Sizing (absorption control)
    """
    
    def __init__(self, width: int, height: int, c_min: float = 0.3, c_max: float = 0.7):
        self.width = width
        self.height = height
        self.c_min = c_min  # Minimum fluid capacity
        self.c_max = c_max  # Maximum fluid capacity
        
        # Initialize fields
        self.height_field = np.zeros((height, width))
        self.fluid_capacity = np.zeros((height, width))
        self.sizing = np.ones((height, width))  # Default to fully sized
        
        # Generate default texture
        self.generate('perlin')
        
    def generate(self, method: str = 'perlin', seed: int = None):
        """Generate paper texture using specified method."""
        if seed is not None:
            np.random.seed(seed)
            
        if method == 'perlin':
            self._generate_perlin()
        elif method == 'fractal':
            self._generate_fractal()
        else:  # random
            self._generate_random()
            
        # Update derived properties
        self.update_capacity()
        
    def _generate_perlin(self):
        """Generate Perlin-like noise for paper texture."""
        scales = [1, 2, 4, 8]
        weights = [0.5, 0.25, 0.15, 0.1]
        
        for scale, weight in zip(scales, weights):
            freq = 8 * scale
            x = np.linspace(0, freq, self.width)
            y = np.linspace(0, freq, self.height)
            X, Y = np.meshgrid(x, y)
            
            noise = np.sin(X + 0.5*np.cos(Y)) * np.cos(Y + 0.5*np.sin(X))
            noise = gaussian_filter(noise, sigma=1/scale)
            self.height_field += weight * noise
            
        self._normalize_height()
        
    def _generate_fractal(self):
        """Generate fractal noise using fBm."""
        octaves = 6
        persistence = 0.5
        
        for i in range(octaves):
            freq = 2**i
            amp = persistence**i
            noise = np.random.randn(self.height//freq, self.width//freq)
            noise = cv2.resize(noise, (self.width, self.height))
            self.height_field += amp * noise
            
        self._normalize_height()
        
    def _generate_random(self):
        """Generate simple random noise."""
        self.height_field = np.random.randn(self.height, self.width)
        self.height_field = gaussian_filter(self.height_field, sigma=2)
        self._normalize_height()
        
    def _normalize_height(self):
        """Normalize height field to [0, 1]."""
        self.height_field -= self.height_field.min()
        self.height_field /= self.height_field.max()
        
    def update_capacity(self):
        """Update fluid capacity based on height field."""
        self.fluid_capacity = (
            self.height_field * (self.c_max - self.c_min) + self.c_min
        )
        
    def load_from_image(self, path: str):
        """Load paper height field from an image."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {path}")
            
        # Resize if needed
        if img.shape != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
            
        # Normalize to [0, 1]
        self.height_field = img.astype(np.float32) / 255.0
        self.update_capacity()
        
    def load_sizing(self, path: str):
        """Load paper sizing field from an image."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {path}")
            
        if img.shape != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
            
        self.sizing = img.astype(np.float32) / 255.0
        
    @property
    def slope(self):
        """Calculate paper surface slope (gradient of height field)."""
        return np.gradient(self.height_field)

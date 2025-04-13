#!/usr/bin/env python3
"""
Paper Model Module

This module implements the paper model for the watercolor simulation as described in
Section 4.1 of the 'Computer-Generated Watercolor' paper.

The paper is modeled as a height field and a fluid capacity field.
"""

import numpy as np
import scipy.ndimage as ndimage
from typing import Tuple, Optional


class PaperModel:
    """
    Paper model for watercolor simulation.
    
    Paper is represented as a height field that affects fluid flow, backruns, and granulation.
    The paper texture is modeled as both a height field and a fluid capacity field.
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize the paper model.
        
        Parameters:
        -----------
        width : int
            Width of the paper in pixels
        height : int
            Height of the paper in pixels
        """
        self.width = width
        self.height = height
        self.height_field = None
        self.capacity_field = None
        self.paper_min_capacity = 0.3
        self.paper_max_capacity = 0.8
    
    def generate(self, method: str = 'perlin', seed: Optional[int] = None) -> None:
        """
        Generate a paper texture as a height field.
        
        Parameters:
        -----------
        method : str
            Method to use for generating paper texture ('perlin', 'random', 'fractal')
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate paper height field
        if method == 'perlin':
            # A simple approximation of Perlin noise for this example
            scale = 8
            octaves = 4
            self.height_field = np.zeros((self.height, self.width), dtype=np.float32)
            
            for octave in range(octaves):
                s = scale * (2 ** octave)
                h_samples = max(2, self.height // s)
                w_samples = max(2, self.width // s)
                
                noise = np.random.rand(h_samples, w_samples).astype(np.float32)
                noise = ndimage.zoom(noise, (self.height / h_samples, self.width / w_samples), order=1)
                self.height_field += noise / (2 ** octave)
            
            # Normalize to [0, 1]
            self.height_field -= np.min(self.height_field)
            self.height_field /= np.max(self.height_field)
            
        elif method == 'random':
            # Simple random texture
            self.height_field = np.random.rand(self.height, self.width).astype(np.float32)
            
        elif method == 'fractal':
            # Approximate fractal noise
            scale = 4
            octaves = 6
            self.height_field = np.zeros((self.height, self.width), dtype=np.float32)
            
            for octave in range(octaves):
                s = scale * (2 ** octave)
                h_samples = max(2, self.height // s)
                w_samples = max(2, self.width // s)
                
                noise = np.random.rand(h_samples, w_samples).astype(np.float32)
                noise = ndimage.zoom(noise, (self.height / h_samples, self.width / w_samples), order=1)
                self.height_field += noise / (2 ** octave)
            
            # Normalize to [0, 1]
            self.height_field -= np.min(self.height_field)
            self.height_field /= np.max(self.height_field)
        
        # Compute paper capacity from height
        self.compute_capacity_field()

    def compute_capacity_field(self) -> None:
        """Compute fluid capacity field from the height field."""
        if self.height_field is None:
            raise ValueError("Height field must be generated before computing capacity field")
        
        self.capacity_field = (self.height_field * 
                              (self.paper_max_capacity - self.paper_min_capacity) + 
                              self.paper_min_capacity)
    
    def compute_slope(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient of the paper height field.
        
        Returns:
        --------
        tuple
            (dx, dy) gradient arrays
        """
        if self.height_field is None:
            raise ValueError("Height field must be generated before computing slope")
        
        # Use Sobel filter to compute gradient
        dx = ndimage.sobel(self.height_field, axis=1)
        dy = ndimage.sobel(self.height_field, axis=0)
        
        # Normalize to reasonable values
        dx = dx / 8.0
        dy = dy / 8.0
        
        return dx, dy

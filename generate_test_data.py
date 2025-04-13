#!/usr/bin/env python3
"""
Generate test input data for watercolor simulation.
Based on 'Computer-Generated Watercolor' by Curtis et al.

Creates test data for:
1. Paper properties
   - Height field (rough surface texture)
   - Fluid capacity field (porous structure)
   - Sizing field (absorption control)

2. Pigment properties
   - Density (ρ) - controls settling rate
   - Staining power (ω) - adherence to paper
   - Granulation (γ) - valley settling tendency

3. Test patterns
   - Edge darkening
   - Backruns
   - Granulation
   - Flow effects
   - Glazing layers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import os
from typing import Tuple

# Constants from the paper
VISCOSITY = 0.1  # μ (fluid viscosity)
VISCOUS_DRAG = 0.01  # κ (viscous drag coefficient)
EDGE_DARKENING = 0.03  # η (edge darkening factor)
ABSORPTION_RATE = 0.05  # α (water absorption rate)
DIFFUSION_THRESHOLD = 0.7  # σ (saturation threshold for wet expansion)
MIN_SATURATION = 0.1  # ε (minimum saturation for diffusion)

def create_paper_height_field(width: int, height: int, method: str = 'perlin') -> np.ndarray:
    """
    Create a height field representing paper surface roughness.
    The paper texture affects fluid flow, backruns, and granulation.
    """
    if method == 'perlin':
        # Generate base noise at different scales
        scales = [1, 2, 4, 8]
        weights = [0.5, 0.25, 0.15, 0.1]
        height_field = np.zeros((height, width))
        
        for scale, weight in zip(scales, weights):
            freq = 8 * scale
            x = np.linspace(0, freq, width)
            y = np.linspace(0, freq, height)
            X, Y = np.meshgrid(x, y)
            
            # Perlin-like noise
            noise = np.sin(X + 0.5*np.cos(Y)) * np.cos(Y + 0.5*np.sin(X))
            noise = gaussian_filter(noise, sigma=1/scale)
            height_field += weight * noise
            
    elif method == 'fractal':
        # Fractal Brownian motion
        height_field = np.zeros((height, width))
        octaves = 6
        persistence = 0.5
        
        for i in range(octaves):
            freq = 2**i
            amp = persistence**i
            noise = np.random.randn(height//freq, width//freq)
            noise = cv2.resize(noise, (width, height))
            height_field += amp * noise
            
    else:  # Random
        height_field = np.random.randn(height, width)
        height_field = gaussian_filter(height_field, sigma=2)

    # Normalize to [0, 1]
    height_field = (height_field - height_field.min()) / (height_field.max() - height_field.min())
    return height_field

def create_fluid_capacity(height_field: np.ndarray, c_min: float = 0.3, c_max: float = 0.7) -> np.ndarray:
    """
    Calculate fluid-holding capacity from height field.
    c = h * (c_max - c_min) + c_min
    """
    return height_field * (c_max - c_min) + c_min

def create_sizing_field(width: int, height: int) -> np.ndarray:
    """
    Create a sizing field that controls water absorption rate.
    Sizing is typically made of cellulose and forms a barrier that slows water absorption.
    """
    # Create base gradient
    y, x = np.ogrid[0:height, 0:width]
    gradient = x / width  # More sized (less absorbent) on right
    
    # Add local variations
    noise = np.random.randn(height//4, width//4)
    noise = cv2.resize(noise, (width, height))
    noise = gaussian_filter(noise, sigma=8)
    
    # Combine with weight towards gradient
    sizing = 0.8 * gradient + 0.2 * noise
    
    # Normalize to [0, 1]
    sizing = (sizing - sizing.min()) / (sizing.max() - sizing.min())
    return sizing

def create_test_patterns(width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create test patterns for different watercolor effects.
    Returns:
    - Edge darkening mask
    - Backrun test mask
    - Granulation test mask
    """
    # Center coordinates
    y, x = np.ogrid[-height/2:height/2, -width/2:width/2]
    
    # Edge darkening test (circular region)
    radius = min(width, height) // 4
    edge_mask = (x*x + y*y <= radius*radius).astype(float)
    edge_mask = gaussian_filter(edge_mask, sigma=2)
    
    # Backrun test (overlapping circles with gradient)
    circle1 = ((x+radius/2)**2 + (y+radius/2)**2 <= (radius/2)**2).astype(float)
    circle2 = ((x-radius/2)**2 + (y-radius/2)**2 <= (radius/2)**2).astype(float)
    backrun_mask = np.maximum(circle1, circle2)
    backrun_mask = gaussian_filter(backrun_mask, sigma=2)
    
    # Granulation test (stripes with varying density)
    freq = 8 * np.pi / width
    granulation_mask = 0.5 * (1 + np.sin(x * freq))
    granulation_mask = gaussian_filter(granulation_mask, sigma=1)
    
    return edge_mask, backrun_mask, granulation_mask

def create_pigment_properties(num_pigments: int = 3) -> list:
    """
    Create a set of test pigments with different physical properties.
    Based on Section 2.1 of the paper.
    """
    pigments = [
        {
            'name': 'Dense Pigment',  # e.g., Iron Oxide
            'density': 1.4,           # ρ: settles quickly
            'staining_power': 0.8,    # ω: strongly adheres to paper
            'granularity': 0.7,       # γ: tends to settle in valleys
            'K': np.array([0.8, 0.2, 0.1]),  # High red absorption
            'S': np.array([0.1, 0.2, 0.9])   # High blue scattering
        },
        {
            'name': 'Medium Pigment',  # e.g., Ultramarine
            'density': 1.0,            # ρ: moderate settling
            'staining_power': 0.5,     # ω: moderate adhesion
            'granularity': 0.4,        # γ: moderate granulation
            'K': np.array([0.1, 0.7, 0.2]),
            'S': np.array([0.9, 0.2, 0.1])
        },
        {
            'name': 'Light Pigment',   # e.g., Quinacridone
            'density': 0.6,            # ρ: spreads further
            'staining_power': 0.3,     # ω: less adhesion
            'granularity': 0.1,        # γ: minimal granulation
            'K': np.array([0.2, 0.8, 0.7]),
            'S': np.array([0.7, 0.1, 0.2])
        }
    ]
    return pigments[:num_pigments]

def create_glaze_pattern(width: int, height: int) -> np.ndarray:
    """
    Create a test pattern for glazing effects.
    Demonstrates optical compositing of multiple translucent layers.
    """
    y, x = np.ogrid[-height/2:height/2, -width/2:width/2]
    
    # Create overlapping regions
    radius = min(width, height) // 4
    glaze1 = np.exp(-(x**2 + y**2) / (2 * (radius/2)**2))
    glaze2 = np.exp(-((x+radius/2)**2 + (y-radius/2)**2) / (2 * (radius/2)**2))
    glaze3 = np.exp(-((x-radius/2)**2 + (y+radius/2)**2) / (2 * (radius/2)**2))
    
    # Stack into RGB
    glaze_pattern = np.stack([glaze1, glaze2, glaze3], axis=2)
    return glaze_pattern

def main():
    """Generate all test data files."""
    # Create output directory
    output_dir = "demo_input"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters
    width = 800
    height = 800
    
    print("Generating test data...")
    
    # 1. Paper properties
    height_field = create_paper_height_field(width, height, method='perlin')
    capacity_field = create_fluid_capacity(height_field)
    sizing_field = create_sizing_field(width, height)
    
    plt.imsave(os.path.join(output_dir, "paper_height.png"), height_field, cmap='gray')
    plt.imsave(os.path.join(output_dir, "paper_capacity.png"), capacity_field, cmap='gray')
    plt.imsave(os.path.join(output_dir, "paper_sizing.png"), sizing_field, cmap='gray')
    
    # 2. Test patterns
    edge_mask, backrun_mask, granulation_mask = create_test_patterns(width, height)
    plt.imsave(os.path.join(output_dir, "edge_darkening_test.png"), edge_mask, cmap='gray')
    plt.imsave(os.path.join(output_dir, "backrun_test.png"), backrun_mask, cmap='gray')
    plt.imsave(os.path.join(output_dir, "granulation_test.png"), granulation_mask, cmap='gray')
    
    # 3. Pigment separation test
    glaze_pattern = create_glaze_pattern(width, height)
    plt.imsave(os.path.join(output_dir, "pigment_separation.png"), glaze_pattern)
    
    # 4. Save parameters as JSON
    import json
    pigments = create_pigment_properties()
    with open(os.path.join(output_dir, "pigment_properties.json"), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for p in pigments:
            p['K'] = p['K'].tolist()
            p['S'] = p['S'].tolist()
        json.dump(pigments, f, indent=2)
        
    print("Test data generated in 'demo_input' directory")
    print("\nTo test individual effects, run:")
    print(
        "python simulation_main.py "
        "--input-height demo_input/paper_height.png "
        "--input-capacity demo_input/paper_capacity.png "
        "--input-sizing demo_input/paper_sizing.png "
        "--input-mask demo_input/edge_darkening_test.png "
        "--save-stages"
    )

if __name__ == "__main__":
    main()

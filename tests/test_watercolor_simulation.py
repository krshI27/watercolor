#!/usr/bin/env python3
"""
Unit tests for the watercolor simulation module.
"""

import sys
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path so we can import the simulation module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.watercolor_simulation import (
    WatercolorSimulation,
    KubelkaMunk,
    WatercolorRenderer
)

class TestWatercolorSimulation(unittest.TestCase):
    """Test cases for the WatercolorSimulation class"""
    
    def setUp(self):
        """Set up a simulation instance for testing"""
        self.width = 64
        self.height = 64
        self.sim = WatercolorSimulation(self.width, self.height)
    
    def test_initialization(self):
        """Test that the simulation initializes correctly"""
        self.assertEqual(self.sim.width, self.width)
        self.assertEqual(self.sim.height, self.height)
        self.assertEqual(len(self.sim.pigment_water), 0)
        self.assertEqual(len(self.sim.pigment_paper), 0)
        self.assertEqual(self.sim.wet_mask.shape, (self.height, self.width))
    
    def test_paper_generation(self):
        """Test paper generation methods"""
        # Test perlin noise generation
        self.sim.generate_paper(method='perlin', seed=42)
        self.assertIsNotNone(self.sim.paper_height)
        self.assertEqual(self.sim.paper_height.shape, (self.height, self.width))
        self.assertTrue(0 <= np.min(self.sim.paper_height) <= np.max(self.sim.paper_height) <= 1)
        
        # Test random generation
        self.sim.generate_paper(method='random', seed=42)
        self.assertEqual(self.sim.paper_height.shape, (self.height, self.width))
        self.assertTrue(0 <= np.min(self.sim.paper_height) <= np.max(self.sim.paper_height) <= 1)
        
        # Test fractal generation
        self.sim.generate_paper(method='fractal', seed=42)
        self.assertEqual(self.sim.paper_height.shape, (self.height, self.width))
        self.assertTrue(0 <= np.min(self.sim.paper_height) <= np.max(self.sim.paper_height) <= 1)
        
        # Test that paper capacity is properly calculated
        self.assertIsNotNone(self.sim.paper_capacity)
        self.assertEqual(self.sim.paper_capacity.shape, (self.height, self.width))
        self.assertTrue(self.sim.paper_min_capacity <= np.min(self.sim.paper_capacity) <= 
                        np.max(self.sim.paper_capacity) <= self.sim.paper_max_capacity)
    
    def test_pigment_handling(self):
        """Test pigment management functions"""
        # Test adding pigment
        km_params = {
            'K': np.array([0.8, 0.2, 0.1]),
            'S': np.array([0.1, 0.2, 0.9])
        }
        
        idx = self.sim.add_pigment(
            density=1.0,
            staining_power=0.6,
            granularity=0.4,
            kubelka_munk_params=km_params
        )
        
        self.assertEqual(idx, 0)  # First pigment should have index 0
        self.assertEqual(len(self.sim.pigment_water), 1)
        self.assertEqual(len(self.sim.pigment_paper), 1)
        self.assertEqual(len(self.sim.pigment_properties), 1)
        
        # Test pigment properties
        props = self.sim.pigment_properties[0]
        self.assertEqual(props['density'], 1.0)
        self.assertEqual(props['staining_power'], 0.6)
        self.assertEqual(props['granularity'], 0.4)
        self.assertDictEqual(props['kubelka_munk_params'], km_params)
        
        # Test setting pigment and wet mask
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        mask[20:40, 20:40] = 1.0  # Create a square mask
        
        self.sim.set_pigment_water(0, mask, concentration=0.8)
        self.sim.set_wet_mask(mask)
        
        # Check that pigment was set correctly
        self.assertTrue(np.all(self.sim.pigment_water[0][20:40, 20:40] == 0.8))
        self.assertTrue(np.all(self.sim.pigment_water[0][:20, :] == 0.0))
        
        # Check that wet mask was set correctly
        self.assertTrue(np.all(self.sim.wet_mask[20:40, 20:40] == 1.0))
        self.assertTrue(np.all(self.sim.wet_mask[:20, :] == 0.0))
    
    def test_velocity_operations(self):
        """Test velocity field operations"""
        # Generate paper
        self.sim.generate_paper(method='perlin', seed=42)
        
        # Set up a simple wet area
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        mask[20:40, 20:40] = 1.0
        self.sim.set_wet_mask(mask)
        
        # Test compute_paper_slope
        dx, dy = self.sim.compute_paper_slope()
        self.assertEqual(dx.shape, (self.height, self.width))
        self.assertEqual(dy.shape, (self.height, self.width))
        
        # Test enforce_boundary_conditions
        # First, set some velocities
        self.sim.velocity_u.fill(1.0)
        self.sim.velocity_v.fill(1.0)
        
        # Apply boundary conditions
        self.sim.enforce_boundary_conditions()
        
        # Check that velocities are zero at the boundaries of wet region
        # Just check a few sample points to make sure it's working
        self.assertEqual(self.sim.velocity_u[30, 20], 0.0)  # Left boundary
        self.assertEqual(self.sim.velocity_u[30, 40], 0.0)  # Right boundary
        self.assertEqual(self.sim.velocity_v[20, 30], 0.0)  # Top boundary
        self.assertEqual(self.sim.velocity_v[40, 30], 0.0)  # Bottom boundary

    def test_simulation_steps(self):
        """Test that simulation steps run without errors"""
        # Generate paper
        self.sim.generate_paper(method='perlin', seed=42)
        
        # Add pigment
        idx = self.sim.add_pigment(
            density=1.0,
            staining_power=0.6,
            granularity=0.4,
            kubelka_munk_params={
                'K': np.array([0.8, 0.2, 0.1]),
                'S': np.array([0.1, 0.2, 0.9])
            }
        )
        
        # Set wet mask and pigment
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        mask[20:40, 20:40] = 1.0
        self.sim.set_wet_mask(mask)
        self.sim.set_pigment_water(idx, mask, concentration=0.8)
        
        # Try running individual steps
        self.sim.update_velocities()
        self.sim.relax_divergence()
        self.sim.flow_outward()
        self.sim.move_water()
        self.sim.move_pigment()
        self.sim.transfer_pigment()
        self.sim.simulate_capillary_flow()
        
        # Test main loop runs without errors
        self.sim.main_loop(num_steps=5)
        
        # Get result and check it has expected structure
        result = self.sim.get_result()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (self.height, self.width))

class TestKubelkaMunk(unittest.TestCase):
    """Test cases for the KubelkaMunk class"""
    
    def test_get_coefficients_from_colors(self):
        """Test computation of Kubelka-Munk coefficients from colors"""
        white_color = np.array([0.8, 0.6, 0.9])
        black_color = np.array([0.2, 0.1, 0.3])
        
        K, S = KubelkaMunk.get_coefficients_from_colors(white_color, black_color)
        
        self.assertEqual(K.shape, (3,))
        self.assertEqual(S.shape, (3,))
        self.assertTrue(np.all(K >= 0))
        self.assertTrue(np.all(S >= 0))
    
    def test_get_reflectance_transmittance(self):
        """Test computation of reflectance and transmittance"""
        K = np.array([0.8, 0.2, 0.1])
        S = np.array([0.1, 0.2, 0.9])
        thickness = 0.5
        
        R, T = KubelkaMunk.get_reflectance_transmittance(K, S, thickness)
        
        self.assertEqual(R.shape, (3,))
        self.assertEqual(T.shape, (3,))
        self.assertTrue(np.all(R >= 0) and np.all(R <= 1))
        self.assertTrue(np.all(T >= 0) and np.all(T <= 1))
    
    def test_composite_layers(self):
        """Test compositing of layers"""
        R1 = np.array([0.7, 0.5, 0.3])
        T1 = np.array([0.2, 0.3, 0.4])
        R2 = np.array([0.6, 0.4, 0.2])
        T2 = np.array([0.1, 0.2, 0.3])
        
        R, T = KubelkaMunk.composite_layers(R1, T1, R2, T2)
        
        self.assertEqual(R.shape, (3,))
        self.assertEqual(T.shape, (3,))
        self.assertTrue(np.all(R >= 0) and np.all(R <= 1))
        self.assertTrue(np.all(T >= 0) and np.all(T <= 1))
    
    def test_render_glazes(self):
        """Test rendering of glazes"""
        glazes = [
            {
                'K': np.array([0.8, 0.2, 0.1]),
                'S': np.array([0.1, 0.2, 0.9]),
                'thickness': 0.5
            },
            {
                'K': np.array([0.1, 0.7, 0.3]),
                'S': np.array([0.9, 0.2, 0.1]),
                'thickness': 0.3
            }
        ]
        
        background_color = np.array([1.0, 1.0, 1.0])
        
        result = KubelkaMunk.render_glazes(glazes, background_color)
        
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))

class TestWatercolorRenderer(unittest.TestCase):
    """Test cases for the WatercolorRenderer class"""
    
    def setUp(self):
        """Set up a simulation and renderer for testing"""
        self.width = 64
        self.height = 64
        self.sim = WatercolorSimulation(self.width, self.height)
        self.sim.generate_paper(method='perlin', seed=42)
        
        # Add a pigment
        self.pigment_idx = self.sim.add_pigment(
            density=1.0,
            staining_power=0.6,
            granularity=0.4,
            kubelka_munk_params={
                'K': np.array([0.8, 0.2, 0.1]),
                'S': np.array([0.1, 0.2, 0.9])
            }
        )
        
        # Set wet mask and pigment
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        mask[20:40, 20:40] = 1.0
        self.sim.set_wet_mask(mask)
        self.sim.set_pigment_water(self.pigment_idx, mask, concentration=0.8)
        
        # Run a few simulation steps
        self.sim.main_loop(num_steps=2)
        
        # Create renderer
        self.renderer = WatercolorRenderer(self.sim)
    
    def test_render_pigment(self):
        """Test rendering a single pigment"""
        result = self.renderer.render_pigment(self.pigment_idx)
        
        self.assertEqual(result.shape, (self.height, self.width, 3))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))
    
    def test_render_all_pigments(self):
        """Test rendering all pigments"""
        # Add another pigment
        self.sim.add_pigment(
            density=0.8,
            staining_power=0.4,
            granularity=0.6,
            kubelka_munk_params={
                'K': np.array([0.1, 0.7, 0.3]),
                'S': np.array([0.9, 0.2, 0.1])
            }
        )
        
        # Set second pigment
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        mask[30:50, 30:50] = 1.0
        self.sim.set_pigment_water(1, mask, concentration=0.6)
        
        # Run a few more steps
        self.sim.main_loop(num_steps=2)
        
        # Render result
        result = self.renderer.render_all_pigments()
        
        self.assertEqual(result.shape, (self.height, self.width, 3))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))

def visual_test():
    """Run visual test to generate and display a watercolor image"""
    print("Running visual test...")
    
    # Create simulation
    sim = WatercolorSimulation(256, 256)
    sim.generate_paper(method='perlin', seed=42)
    
    # Add blue pigment
    blue_km = {
        'K': np.array([0.8, 0.2, 0.1]),  # High absorption in red, low in blue
        'S': np.array([0.1, 0.2, 0.9])   # High scattering in blue
    }
    
    blue_idx = sim.add_pigment(
        density=1.0,
        staining_power=0.6,
        granularity=0.4,
        kubelka_munk_params=blue_km
    )
    
    # Create a circular mask
    y, x = np.ogrid[-128:128, -128:128]
    mask = x*x + y*y <= 80*80
    
    # Set wet mask and pigment
    sim.set_wet_mask(mask)
    sim.set_pigment_water(blue_idx, mask, concentration=0.8)
    
    # Run simulation
    print("Running simulation steps...")
    sim.main_loop(20)
    
    # Render result
    renderer = WatercolorRenderer(sim)
    result = renderer.render_all_pigments()
    
    # Save and display result
    output_path = Path(__file__).parent / "test_watercolor_output.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(np.clip(result, 0, 1))
    plt.axis('off')
    plt.title('Watercolor Simulation Test')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visual test result saved to {output_path}")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run visual test if not running from unittest
    if sys.argv[0] == __file__:
        visual_test()

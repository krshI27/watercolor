#!/usr/bin/env python3
"""
Test suite for the watercolor simulation.
Run with: python -m unittest simulation/test_watercolor_simulation.py
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.watercolor_simulation import WatercolorSimulation, KubelkaMunk, WatercolorRenderer

class TestWatercolorSimulation(unittest.TestCase):
    """Tests for the WatercolorSimulation class."""
    
    def setUp(self):
        """Set up a simple simulation for testing."""
        self.width = 64
        self.height = 64
        self.sim = WatercolorSimulation(self.width, self.height)
        self.sim.generate_paper(method='random', seed=42)
    
    def test_initialization(self):
        """Test proper initialization of a WatercolorSimulation object."""
        self.assertEqual(self.sim.width, self.width)
        self.assertEqual(self.sim.height, self.height)
        self.assertIsNotNone(self.sim.paper_height)
        self.assertIsNotNone(self.sim.paper_capacity)
        
        # Check that arrays are correctly sized
        self.assertEqual(self.sim.wet_mask.shape, (self.height, self.width))
        self.assertEqual(self.sim.velocity_u.shape, (self.height, self.width + 1))
        self.assertEqual(self.sim.velocity_v.shape, (self.height + 1, self.width))
        self.assertEqual(self.sim.pressure.shape, (self.height, self.width))
    
    def test_add_pigment(self):
        """Test adding a pigment to the simulation."""
        # Initial count should be zero
        self.assertEqual(len(self.sim.pigment_water), 0)
        
        # Add a pigment and check the index
        pigment_idx = self.sim.add_pigment(density=0.5, staining_power=0.7, granularity=0.3)
        self.assertEqual(pigment_idx, 0)
        self.assertEqual(len(self.sim.pigment_water), 1)
        self.assertEqual(len(self.sim.pigment_paper), 1)
        self.assertEqual(len(self.sim.pigment_properties), 1)
        
        # Check pigment properties
        props = self.sim.pigment_properties[0]
        self.assertEqual(props['density'], 0.5)
        self.assertEqual(props['staining_power'], 0.7)
        self.assertEqual(props['granularity'], 0.3)
    
    def test_set_pigment_water(self):
        """Test setting pigment concentration in water."""
        pigment_idx = self.sim.add_pigment()
        
        # Create a circular mask
        y, x = np.ogrid[-self.height/2:self.height/2, -self.width/2:self.width/2]
        mask = x*x + y*y <= 20*20
        
        # Set pigment and check it was applied
        self.sim.set_pigment_water(pigment_idx, mask, concentration=0.8)
        
        # Check that pigment was added where the mask is True
        pigment = self.sim.pigment_water[pigment_idx]
        self.assertTrue(np.all(pigment[mask] == 0.8))
        self.assertTrue(np.all(pigment[~mask] == 0.0))
        
        # Also check that wet mask was updated
        self.assertTrue(np.all(self.sim.wet_mask[mask] == 1.0))
        self.assertTrue(np.all(self.sim.wet_mask[~mask] == 0.0))
    
    def test_reset_simulation(self):
        """Test resetting the simulation."""
        # Add a pigment and set values
        pigment_idx = self.sim.add_pigment()
        mask = np.ones((self.height, self.width), dtype=bool)
        self.sim.set_pigment_water(pigment_idx, mask, concentration=1.0)
        
        # Reset and check that values are cleared
        self.sim.reset()
        self.assertEqual(len(self.sim.pigment_water), 0)
        self.assertTrue(np.all(self.sim.wet_mask == 0))
    
    def test_paper_generation(self):
        """Test different paper generation methods."""
        for method in ['perlin', 'random', 'fractal']:
            with self.subTest(method=method):
                self.sim.generate_paper(method=method, seed=42)
                self.assertIsNotNone(self.sim.paper_height)
                self.assertIsNotNone(self.sim.paper_capacity)
                
                # Check value ranges
                self.assertTrue(np.all(self.sim.paper_height >= 0))
                self.assertTrue(np.all(self.sim.paper_height <= 1))
                self.assertTrue(np.all(self.sim.paper_capacity >= self.sim.paper_min_capacity))
                self.assertTrue(np.all(self.sim.paper_capacity <= self.sim.paper_max_capacity))


class TestFluidDynamics(unittest.TestCase):
    """Tests for the fluid dynamics parts of the watercolor simulation."""
    
    def setUp(self):
        """Set up a simple simulation for testing."""
        self.width = 64
        self.height = 64
        self.sim = WatercolorSimulation(self.width, self.height)
        self.sim.generate_paper(method='random', seed=42)
    
    def test_compute_paper_slope(self):
        """Test computing paper slope."""
        dx, dy = self.sim.compute_paper_slope()
        self.assertEqual(dx.shape, (self.height, self.width))
        self.assertEqual(dy.shape, (self.height, self.width))
    
    def test_enforce_boundary_conditions(self):
        """Test enforcement of boundary conditions."""
        # Set all velocities to 1
        self.sim.velocity_u.fill(1.0)
        self.sim.velocity_v.fill(1.0)
        
        # Set wet mask to a small region in the center
        self.sim.wet_mask.fill(0.0)
        self.sim.wet_mask[20:40, 20:40] = 1.0
        
        # Enforce boundary conditions
        self.sim.enforce_boundary_conditions()
        
        # Check that velocities at boundaries are zero
        # (testing a few specific points at the boundary)
        self.assertEqual(self.sim.velocity_u[20, 20], 0.0)
        self.assertEqual(self.sim.velocity_u[20, 40], 0.0)
        self.assertEqual(self.sim.velocity_v[20, 20], 0.0)
        self.assertEqual(self.sim.velocity_v[40, 20], 0.0)
    
    def test_update_velocities(self):
        """Test velocity update function."""
        # Set wet mask to a circular region
        y, x = np.ogrid[-self.height/2:self.height/2, -self.width/2:self.width/2]
        mask = x*x + y*y <= 20*20
        self.sim.set_wet_mask(mask)
        
        # Create a slope in the paper
        h = np.zeros_like(self.sim.paper_height)
        h[self.height//2:, :] = 0.5
        self.sim.paper_height = h
        
        # Set initial pressure
        self.sim.pressure.fill(0.0)
        self.sim.pressure[self.height//4, self.width//2] = 1.0
        
        # Update velocities
        before_update = np.sum(np.abs(self.sim.velocity_u)) + np.sum(np.abs(self.sim.velocity_v))
        self.sim.update_velocities()
        after_update = np.sum(np.abs(self.sim.velocity_u)) + np.sum(np.abs(self.sim.velocity_v))
        
        # Ensure velocities changed
        self.assertNotEqual(before_update, after_update)


class TestKubelkaMunk(unittest.TestCase):
    """Tests for the KubelkaMunk class."""
    
    def test_get_coefficients_from_colors(self):
        """Test calculation of K and S coefficients from colors."""
        # RGB colors on white and black backgrounds
        white_color = np.array([0.8, 0.5, 0.3])
        black_color = np.array([0.4, 0.2, 0.1])
        
        K, S = KubelkaMunk.get_coefficients_from_colors(white_color, black_color)
        
        self.assertEqual(K.shape, (3,))
        self.assertEqual(S.shape, (3,))
        self.assertTrue(np.all(K >= 0))
        self.assertTrue(np.all(S >= 0))
    
    def test_reflectance_transmittance(self):
        """Test calculation of reflectance and transmittance."""
        K = np.array([0.5, 0.3, 0.1])
        S = np.array([0.1, 0.2, 0.4])
        thickness = 0.5
        
        R, T = KubelkaMunk.get_reflectance_transmittance(K, S, thickness)
        
        self.assertEqual(R.shape, (3,))
        self.assertEqual(T.shape, (3,))
        self.assertTrue(np.all(R >= 0) and np.all(R <= 1))
        self.assertTrue(np.all(T >= 0) and np.all(T <= 1))
    
    def test_composite_layers(self):
        """Test compositing of two layers."""
        R1 = np.array([0.3, 0.4, 0.5])
        T1 = np.array([0.6, 0.5, 0.4])
        R2 = np.array([0.2, 0.3, 0.4])
        T2 = np.array([0.7, 0.6, 0.5])
        
        R, T = KubelkaMunk.composite_layers(R1, T1, R2, T2)
        
        self.assertEqual(R.shape, (3,))
        self.assertEqual(T.shape, (3,))
        self.assertTrue(np.all(R >= 0) and np.all(R <= 1))
        self.assertTrue(np.all(T >= 0) and np.all(T <= 1))
    
    def test_render_glazes(self):
        """Test rendering of glazes."""
        glazes = [
            {
                'K': np.array([0.5, 0.3, 0.1]),
                'S': np.array([0.1, 0.2, 0.4]),
                'thickness': 0.5
            },
            {
                'K': np.array([0.1, 0.5, 0.3]),
                'S': np.array([0.4, 0.1, 0.2]),
                'thickness': 0.3
            }
        ]
        
        background = np.array([1.0, 1.0, 1.0])
        result = KubelkaMunk.render_glazes(glazes, background)
        
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))


class TestRenderer(unittest.TestCase):
    """Tests for the WatercolorRenderer class."""
    
    def setUp(self):
        """Set up a simulation with pigments for rendering."""
        self.width = 64
        self.height = 64
        self.sim = WatercolorSimulation(self.width, self.height)
        self.sim.generate_paper(method='random', seed=42)
        
        # Add a blue pigment
        blue_km = {
            'K': np.array([0.8, 0.2, 0.1]),
            'S': np.array([0.1, 0.2, 0.9])
        }
        self.blue_idx = self.sim.add_pigment(
            density=1.0,
            staining_power=0.6,
            granularity=0.4,
            kubelka_munk_params=blue_km
        )
        
        # Create a circular mask for pigment
        y, x = np.ogrid[-self.height/2:self.height/2, -self.width/2:self.width/2]
        self.mask = x*x + y*y <= 20*20
        
        # Set pigment
        self.sim.set_wet_mask(self.mask)
        self.sim.set_pigment_water(self.blue_idx, self.mask, concentration=0.8)
        
        # Create renderer
        self.renderer = WatercolorRenderer(self.sim)
    
    def test_render_pigment(self):
        """Test rendering a single pigment layer."""
        result = self.renderer.render_pigment(self.blue_idx)
        
        self.assertEqual(result.shape, (self.height, self.width, 3))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))
        
        # Check that pigment appears where the mask is
        self.assertTrue(np.any(result[self.mask] != 1.0))
        
        # Check that no pigment appears where there's no mask
        # (should be background color = white)
        self.assertTrue(np.all(result[~self.mask] == 1.0))
    
    def test_render_all_pigments(self):
        """Test rendering all pigment layers."""
        # Add a second pigment (red)
        red_km = {
            'K': np.array([0.1, 0.8, 0.8]),
            'S': np.array([0.9, 0.2, 0.2])
        }
        red_idx = self.sim.add_pigment(
            density=1.0,
            staining_power=0.7,
            granularity=0.3,
            kubelka_munk_params=red_km
        )
        
        # Create a different mask for the red pigment
        y, x = np.ogrid[-self.height/2:self.height/2, -self.width/2:self.width/2]
        red_mask = ((x+10)**2 + (y-10)**2) <= 15**2
        
        # Set red pigment
        self.sim.set_pigment_water(red_idx, red_mask, concentration=0.7)
        
        # Run a few simulation steps
        self.sim.main_loop(10)
        
        # Render all pigments
        result = self.renderer.render_all_pigments()
        
        self.assertEqual(result.shape, (self.height, self.width, 3))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))
        
        # Check that pigments appear where expected
        combined_mask = np.logical_or(self.mask, red_mask)
        self.assertTrue(np.any(result[combined_mask] != 1.0))
        
        # Where pigments overlap, should see a different color from either individual pigment
        overlap = np.logical_and(self.mask, red_mask)
        if np.any(overlap):
            # Sample non-overlap areas for comparison
            blue_only = np.logical_and(self.mask, ~red_mask)
            red_only = np.logical_and(~self.mask, red_mask)
            
            blue_sample = np.mean(result[blue_only], axis=0)
            red_sample = np.mean(result[red_only], axis=0)
            mixed_sample = np.mean(result[overlap], axis=0)
            
            # Mixed color should be different from either individual color
            self.assertFalse(np.allclose(mixed_sample, blue_sample, atol=0.1))
            self.assertFalse(np.allclose(mixed_sample, red_sample, atol=0.1))


class TestSimulationIntegration(unittest.TestCase):
    """Integration tests for the watercolor simulation."""
    
    def setUp(self):
        """Set up a simple simulation for testing."""
        self.width = 128
        self.height = 128
        self.sim = WatercolorSimulation(self.width, self.height)
        self.sim.generate_paper(method='perlin', seed=42)
    
    def test_main_loop(self):
        """Test running the main simulation loop."""
        # Add a pigment
        blue_km = {
            'K': np.array([0.8, 0.2, 0.1]),
            'S': np.array([0.1, 0.2, 0.9])
        }
        blue_idx = self.sim.add_pigment(
            density=1.0,
            staining_power=0.6,
            granularity=0.4,
            kubelka_munk_params=blue_km
        )
        
        # Create a circular mask for pigment
        y, x = np.ogrid[-self.height/2:self.height/2, -self.width/2:self.width/2]
        mask = x*x + y*y <= 30**2
        
        # Set wet mask and pigment
        self.sim.set_wet_mask(mask)
        self.sim.set_pigment_water(blue_idx, mask, concentration=0.8)
        
        # Store initial state for comparison
        initial_paper_pigment = self.sim.pigment_paper[0].copy()
        initial_water_pigment = self.sim.pigment_water[0].copy()
        
        # Run simulation
        self.sim.main_loop(20)
        
        # Check that something changed
        self.assertFalse(np.array_equal(initial_paper_pigment, self.sim.pigment_paper[0]))
        self.assertFalse(np.array_equal(initial_water_pigment, self.sim.pigment_water[0]))
        
        # Check that pigment deposited on paper
        self.assertTrue(np.sum(self.sim.pigment_paper[0]) > 0)
    
    def test_edge_darkening(self):
        """Test the edge-darkening effect."""
        # Add a pigment
        blue_km = {
            'K': np.array([0.8, 0.2, 0.1]),
            'S': np.array([0.1, 0.2, 0.9])
        }
        blue_idx = self.sim.add_pigment(
            density=1.0,
            staining_power=0.6,
            granularity=0.4,
            kubelka_munk_params=blue_km
        )
        
        # Create a circular mask
        y, x = np.ogrid[-self.height/2:self.height/2, -self.width/2:self.width/2]
        mask = x*x + y*y <= 30**2
        
        # Set wet mask and pigment
        self.sim.set_wet_mask(mask)
        self.sim.set_pigment_water(blue_idx, mask, concentration=0.8)
        
        # Set a higher edge darkening factor to make the effect more obvious
        self.sim.edge_darkening_factor = 0.2
        
        # Store initial state
        initial_paper_pigment = self.sim.pigment_paper[0].copy()
        
        # Run simulation
        self.sim.main_loop(30)
        
        # Check that pigment has accumulated at the edges
        # First, find the edge of the mask
        eroded_mask = np.zeros_like(mask)
        eroded_mask[mask] = 1
        from scipy.ndimage import binary_erosion
        eroded_mask = binary_erosion(eroded_mask, iterations=2)
        edge_mask = np.logical_and(mask, ~eroded_mask)
        
        # Compare average pigment density at edge vs center
        center_pigment = np.mean(self.sim.pigment_paper[0][eroded_mask])
        edge_pigment = np.mean(self.sim.pigment_paper[0][edge_mask])
        
        # Edge should have more pigment due to edge darkening
        self.assertGreater(edge_pigment, center_pigment)


if __name__ == '__main__':
    unittest.main()

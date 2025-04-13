#!/usr/bin/env python3
"""
Unit tests for the watercolor simulation module.
"""

import pytest
import numpy as np
import os
import json
from simulation.watercolor_simulation import WatercolorSimulation
from simulation.paper import Paper
from simulation.kubelka_munk import KubelkaMunk

@pytest.fixture
def test_data_dir():
    """Load test data from demo_input directory"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_input")

@pytest.fixture
def basic_simulation():
    """Create a basic simulation setup"""
    width = height = 64
    sim = WatercolorSimulation(width, height)
    sim.paper.generate("perlin", seed=42)
    return sim

def test_paper_properties(test_data_dir):
    """Test paper height field, capacity, and sizing interactions"""
    paper = Paper(64, 64)
    
    # Load test paper properties
    height_path = os.path.join(test_data_dir, "paper_height.png")
    capacity_path = os.path.join(test_data_dir, "paper_capacity.png")
    sizing_path = os.path.join(test_data_dir, "paper_sizing.png")
    
    # Test with demo data if available
    if os.path.exists(height_path):
        paper.load_from_image(height_path)
        assert paper.height_field.shape == (64, 64)
        assert np.all(paper.height_field >= 0)
        assert np.all(paper.height_field <= 1)
        
        # Verify capacity calculation
        paper.update_capacity()
        assert paper.fluid_capacity.shape == paper.height_field.shape
        assert np.all(paper.fluid_capacity >= paper.c_min)
        assert np.all(paper.fluid_capacity <= paper.c_max)

def test_edge_darkening(basic_simulation, test_data_dir):
    """Test edge darkening effect from Section 4.3.3"""
    sim = basic_simulation
    
    # Load edge darkening test mask
    mask_path = os.path.join(test_data_dir, "edge_darkening_test.png")
    if os.path.exists(mask_path):
        from PIL import Image
        import numpy as np
        mask = np.array(Image.open(mask_path).convert('L')) / 255.0
        mask = mask > 0.5
        
        # Set up test
        sim.set_wet_mask(mask)
        sim.set_pigment_water(0, mask, concentration=0.8)
        
        # Run simulation steps
        for _ in range(20):
            sim.move_water()
            sim.move_pigment()
            sim.transfer_pigment()
        
        # Check edge darkening effect
        pigment = sim.pigment_paper[0]
        center_y, center_x = mask.shape[0]//2, mask.shape[1]//2
        center_val = pigment[center_y, center_x]
        edge_y, edge_x = np.where(mask)[0][0], np.where(mask)[1][0]
        edge_val = pigment[edge_y, edge_x]
        
        assert edge_val > center_val, "Edge should have higher pigment concentration"

def test_backrun_effect(basic_simulation, test_data_dir):
    """Test backrun effect from Section 4.6"""
    sim = basic_simulation
    
    # Load backrun test mask
    mask_path = os.path.join(test_data_dir, "backrun_test.png")
    if os.path.exists(mask_path):
        from PIL import Image
        import numpy as np
        mask = np.array(Image.open(mask_path).convert('L')) / 255.0
        mask = mask > 0.5
        
        # Set up test with varying saturation
        sim.set_wet_mask(mask)
        sim.water_saturation[mask] = 0.8  # Wet region
        sim.water_saturation[~mask] = 0.3  # Damp region
        
        # Add pigment
        sim.set_pigment_water(0, mask, concentration=0.5)
        
        # Run simulation focusing on capillary flow
        initial_mask = sim.wet_mask.copy()
        for _ in range(30):
            sim.simulate_capillary_flow()
            
        # Verify backrun formation
        assert np.sum(sim.wet_mask) > np.sum(initial_mask), "Wet area should expand"
        assert not np.array_equal(sim.wet_mask, initial_mask), "Mask should change"

def test_pigment_separation(basic_simulation, test_data_dir):
    """Test pigment separation behavior from Section 2.2"""
    sim = basic_simulation
    
    # Load pigment properties
    props_path = os.path.join(test_data_dir, "pigment_properties.json")
    if os.path.exists(props_path):
        with open(props_path) as f:
            pigments = json.load(f)
            
        # Add pigments with different properties
        indices = []
        for p in pigments:
            idx = sim.add_pigment(
                density=p['density'],
                staining_power=p['staining_power'],
                granularity=p['granularity'],
                kubelka_munk_params={
                    'K': np.array(p['K']),
                    'S': np.array(p['S'])
                }
            )
            indices.append(idx)
        
        # Create test pattern
        y, x = np.ogrid[-32:32, -32:32]
        mask = x*x + y*y <= 20*20
        
        # Add pigments to water
        for idx in indices:
            sim.set_pigment_water(idx, mask, concentration=0.5)
        
        # Run simulation
        for _ in range(20):
            sim.move_water()
            sim.move_pigment()
            sim.transfer_pigment()
        
        # Check pigment separation
        # Denser pigments should settle more quickly
        for i in range(len(indices)-1):
            curr_pigment = sim.pigment_paper[indices[i]]
            next_pigment = sim.pigment_paper[indices[i+1]]
            
            # Compare settling patterns
            assert np.mean(curr_pigment) > np.mean(next_pigment), \
                "Denser pigments should settle more"

def test_glazing_optical_model(basic_simulation):
    """Test glazing and optical compositing from Section 5"""
    sim = basic_simulation
    
    # Create two test glazes with known parameters
    glaze1 = {
        'K': np.array([0.8, 0.2, 0.1]),
        'S': np.array([0.1, 0.2, 0.9]),
        'thickness': 0.5
    }
    
    glaze2 = {
        'K': np.array([0.1, 0.7, 0.3]),
        'S': np.array([0.9, 0.2, 0.1]),
        'thickness': 0.3
    }
    
    # Test Kubelka-Munk calculations
    km = KubelkaMunk()
    R1, T1 = km.compute_layer_optics(glaze1['K'], glaze1['S'], glaze1['thickness'])
    R2, T2 = km.compute_layer_optics(glaze2['K'], glaze2['S'], glaze2['thickness'])
    
    # Composite the glazes
    R_final = km.composite_layers([R1, R2], [T1, T2])
    
    # Verify physical constraints
    assert np.all(R_final >= 0) and np.all(R_final <= 1), "Invalid reflectance values"
    assert not np.array_equal(R_final, R1), "Compositing should modify reflectance"

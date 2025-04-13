#!/usr/bin/env python3
"""
Watercolor Simulation Main Entry Point

This script provides a command-line interface for running the physics-based
watercolor simulation based on 'Computer-Generated Watercolor' by Curtis et al.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the simulation components
from simulation.watercolor_simulation import WatercolorSimulation
from simulation.renderer import WatercolorRenderer

def parse_arguments():
    """Parse command line arguments for the watercolor simulation."""
    parser = argparse.ArgumentParser(description="Generate watercolor images using physics-based simulation.")
    
    parser.add_argument("--width", type=int, default=800, help="Width of the image")
    parser.add_argument("--height", type=int, default=800, help="Height of the image")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="watercolor_simulation_output.png", help="Output file path")
    
    # Paper parameters
    parser.add_argument("--paper-method", type=str, default="perlin", choices=["perlin", "random", "fractal"], 
                      help="Method for generating paper texture")
    
    # Fluid parameters
    parser.add_argument("--viscosity", type=float, default=0.1, help="Fluid viscosity")
    parser.add_argument("--drag", type=float, default=0.01, help="Viscous drag coefficient")
    parser.add_argument("--edge-darkening", type=float, default=0.03, help="Edge darkening factor")
    
    # Pigment parameters
    parser.add_argument("--pigment-density", type=float, default=1.0, help="Density of the pigment")
    parser.add_argument("--staining-power", type=float, default=0.6, help="Staining power of the pigment")
    parser.add_argument("--granularity", type=float, default=0.4, help="Granularity of the pigment")
    
    return parser.parse_args()

def main():
    """Main entry point for the watercolor simulation."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create simulation
    print(f"Creating watercolor simulation ({args.width}x{args.height})...")
    sim = WatercolorSimulation(args.width, args.height)
    
    # Set fluid parameters
    sim.viscosity = args.viscosity
    sim.viscous_drag = args.drag
    sim.edge_darkening_factor = args.edge_darkening
    
    # Generate paper texture
    print(f"Generating paper texture using {args.paper_method} method...")
    sim.generate_paper(method=args.paper_method, seed=args.seed)
    
    # Create a sample KM parameters for a blue pigment
    blue_km = {
        'K': np.array([0.8, 0.2, 0.1]),  # High absorption in red, low in blue
        'S': np.array([0.1, 0.2, 0.9])   # High scattering in blue
    }
    
    # Add a blue pigment
    print("Adding pigment...")
    blue_idx = sim.add_pigment(
        density=args.pigment_density,
        staining_power=args.staining_power,
        granularity=args.granularity,
        kubelka_munk_params=blue_km
    )
    
    # Create a circular mask for the wet area
    y, x = np.ogrid[-args.height//2:args.height//2, -args.width//2:args.width//2]
    radius = min(args.width, args.height) // 3
    mask = x*x + y*y <= radius*radius
    
    # Set wet mask and pigment
    print("Setting up wet areas...")
    sim.set_wet_mask(mask)
    sim.set_pigment_water(blue_idx, mask, concentration=0.8)
    
    # Run simulation
    print(f"Running simulation for {args.steps} steps...")
    sim.main_loop(args.steps)
    
    # Render result
    print("Rendering result...")
    renderer = WatercolorRenderer(sim)
    result = renderer.render_all_pigments()
    
    # Save output
    plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(result, 0, 1))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Watercolor image saved to {args.output}")

if __name__ == "__main__":
    main()

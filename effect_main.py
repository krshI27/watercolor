#!/usr/bin/env python3
"""
Watercolor Effect Main Entry Point

This script provides a command-line interface for generating watercolor-style 
images using the effect-based approach (without physics simulation).
"""

import sys
import os
import argparse
from pathlib import Path

# Import the effect module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from effect.watercolor_effect import generate_watercolor

def parse_arguments():
    """Parse command line arguments for the watercolor effect generator."""
    parser = argparse.ArgumentParser(description="Generate watercolor-style images using the effect-based approach.")
    
    # Canvas parameters
    parser.add_argument("--width", type=int, default=1000, help="Width of the image")
    parser.add_argument("--height", type=int, default=1500, help="Height of the image")
    
    # Shape parameters
    parser.add_argument("--min-shape-size", type=int, default=100, help="Minimum shape size")
    parser.add_argument("--max-shape-size", type=int, default=300, help="Maximum shape size")
    parser.add_argument("--min-shapes", type=int, default=20, help="Minimum shapes per layer")
    parser.add_argument("--max-shapes", type=int, default=25, help="Maximum shapes per layer")
    
    # Deformation parameters
    parser.add_argument("-i", "--initial-deform", type=float, default=120, help="Initial deformation amount")
    parser.add_argument("-d", "--deviation", type=float, default=50, help="Random deviation")
    parser.add_argument("-bd", "--base-deformations", type=int, default=1, help="Base deformation iterations")
    parser.add_argument("-fd", "--final-deformations", type=int, default=3, help="Final deformation iterations")
    
    # Layer parameters
    parser.add_argument("-l", "--layers", type=int, default=-1, help="Number of layers (-1 for automatic)")
    parser.add_argument("--boundary-overflow", type=int, default=100, help="Boundary overflow")
    
    # Color parameters
    parser.add_argument("-sa", "--shape-opacity", type=float, default=0.08, help="Shape opacity")
    parser.add_argument("--color-variation", type=float, default=0.2, help="Color variation")
    
    # Output
    parser.add_argument("--output", type=str, default="watercolor_effect_output.png", help="Output file path")
    
    return parser.parse_args()

def main():
    """Main entry point for the watercolor effect generator."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Generate watercolor effect
    print(f"Generating watercolor effect ({args.width}x{args.height})...")
    generate_watercolor(
        width=args.width,
        height=args.height,
        initial_deform=args.initial_deform,
        deviation=args.deviation,
        base_deformations=args.base_deformations,
        final_deformations=args.final_deformations,
        min_shapes=args.min_shapes,
        max_shapes=args.max_shapes,
        shape_opacity=args.shape_opacity,
        num_layers=args.layers,
        min_shape_size=args.min_shape_size,
        max_shape_size=args.max_shape_size,
        boundary_overflow=args.boundary_overflow,
        color_variation=args.color_variation,
        output_path=args.output
    )
    
    print(f"Watercolor image saved to {args.output}")

if __name__ == "__main__":
    main()

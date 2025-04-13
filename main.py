#!/usr/bin/env python
"""
Main entry point for watercolor simulation.
This script provides a command-line interface for generating watercolor images.
"""
import sys
import os
import argparse
from pathlib import Path

# Add the app directory to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the reorganized modules
from watercolor.effect import WatercolorEffect
from watercolor.utils import parse_arguments

def main():
    """Main entry point for the watercolor simulation."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create and run the watercolor effect generator
    effect = WatercolorEffect(
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
        color_variation=args.color_variation
    )
    
    # Generate the watercolor effect
    effect.generate()
    
    # Save the output
    effect.save(args.output)
    print(f"Watercolor image saved to {args.output}")

# Example commands:
# Basic: python main.py
# Advanced: python main.py --width 1000 --height 1000 --color-variation 0.1 -i 125 -d 25 -bd 6 -fd 6 -mins 5 -maxs 15 -sa 0.1 -l 10 --min-shape-size 10 --max-shape-size 20 --boundary-overflow 1
# Single shape: python main.py -mins 1 -maxs 1 -l 1 -sa 0.9 --color-variation 0.0
# Small repeating shapes: python main.py -mins 50 -maxs 50 -sa 0.04 --min-shape-size 40 --max-shape-size 40 -bd 0 -fd 0

if __name__ == "__main__":
    main()

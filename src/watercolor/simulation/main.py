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
from PIL import Image
import cv2

# Import the simulation components
from .watercolor_simulation import WatercolorSimulation
from .renderer import WatercolorRenderer


def parse_arguments():
    """Parse command line arguments for the watercolor simulation."""
    parser = argparse.ArgumentParser(
        description="Generate watercolor images using physics-based simulation."
    )

    # Input/output arguments
    parser.add_argument("--width", type=int, default=800, help="Width of the image")
    parser.add_argument("--height", type=int, default=800, help="Height of the image")
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of simulation steps"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="watercolor_simulation_output.png",
        help="Output file path",
    )
    parser.add_argument(
        "--save-stages", action="store_true", help="Save intermediate stage outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="watercolor_stages",
        help="Directory for stage outputs",
    )

    # Input image arguments
    parser.add_argument(
        "--input-image", type=str, help="Input image for pigment color and distribution"
    )
    parser.add_argument(
        "--input-mask", type=str, help="Input image for wet mask (grayscale)"
    )
    parser.add_argument(
        "--input-height",
        type=str,
        help="Input image for paper height field (grayscale)",
    )

    # Paper parameters
    parser.add_argument(
        "--paper-method",
        type=str,
        default="perlin",
        choices=["perlin", "random", "fractal", "from_image"],
        help="Method for generating paper texture",
    )
    # Additional paper parameters
    parser.add_argument(
        "--input-capacity",
        type=str,
        help="Input image for paper fluid capacity field (grayscale)",
    )
    parser.add_argument(
        "--input-sizing",
        type=str,
        help="Input image for paper sizing field (grayscale)",
    )
    parser.add_argument(
        "--c-min",
        type=float,
        default=0.3,
        help="Minimum fluid capacity value",
    )
    parser.add_argument(
        "--c-max",
        type=float,
        default=0.7,
        help="Maximum fluid capacity value",
    )

    # Fluid parameters
    parser.add_argument("--viscosity", type=float, default=0.1, help="Fluid viscosity")
    parser.add_argument(
        "--drag", type=float, default=0.01, help="Viscous drag coefficient"
    )
    parser.add_argument(
        "--edge-darkening", type=float, default=0.03, help="Edge darkening factor"
    )
    # Additional fluid parameters
    parser.add_argument(
        "--absorption-rate",
        type=float,
        default=0.05,
        help="Water absorption rate (α)",
    )
    parser.add_argument(
        "--diffusion-threshold",
        type=float,
        default=0.7,
        help="Saturation threshold for wet expansion (σ)",
    )
    parser.add_argument(
        "--min-saturation",
        type=float,
        default=0.1,
        help="Minimum saturation for diffusion (ε)",
    )

    # Pigment parameters
    parser.add_argument(
        "--pigment-density", type=float, default=1.0, help="Density of the pigment"
    )
    parser.add_argument(
        "--staining-power",
        type=float,
        default=0.6,
        help="Staining power of the pigment",
    )
    parser.add_argument(
        "--granularity", type=float, default=0.4, help="Granularity of the pigment"
    )
    parser.add_argument(
        "--concentration", type=float, default=0.8, help="Initial pigment concentration"
    )

    return parser.parse_args()


def load_input_image(path: str, target_size: tuple = None) -> np.ndarray:
    """Load and preprocess an input image."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if target size is provided
    if target_size is not None:
        img = cv2.resize(img, target_size)

    # Normalize to [0, 1]
    return img.astype(np.float32) / 255.0


def save_output_image(image: np.ndarray, output_path: str):
    """
    Saves the final simulation result image.

    Args:
        image: The image data (numpy array, float32, 0.0-1.0).
        output_path: The path to save the image file.
    """
    try:
        # Convert float32 [0, 1] to uint8 [0, 255]
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        # Ensure BGR format for OpenCV if it's color
        if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
            # Assuming input is RGB, convert to BGR for cv2.imwrite
            # If input is already BGR, this might swap channels, but
            # consistency with loading (which produces BGR) is likely intended.
            # If loading produces RGB, remove this line.
            # image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR) # Keep commented unless sure
            pass  # Assume input is already BGR or grayscale handled by imwrite

        if not cv2.imwrite(output_path, image_uint8):
            raise IOError(f"cv2.imwrite failed to save to {output_path}")
        print(f"Output image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving output image to {output_path}: {e}", file=sys.stderr)
        # Optionally re-raise or exit depending on desired error handling
        # raise # Re-raise the exception


def save_stage_output(stage_name: str, data: np.ndarray, output_dir: str):
    """Save intermediate stage output."""
    os.makedirs(output_dir, exist_ok=True)

    # Handle different types of data
    if len(data.shape) == 2:  # Single channel (e.g., height field)
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap="gray")
        plt.colorbar()
        plt.title(stage_name)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{stage_name}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:  # RGB image
        plt.figure(figsize=(10, 10))
        plt.imshow(np.clip(data, 0, 1))
        plt.title(stage_name)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{stage_name}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()


def create_pigment_from_image(image: np.ndarray) -> dict:
    """Create Kubelka-Munk parameters from an RGB image."""
    # Convert RGB to K and S coefficients
    # This is a simple approximation - in reality, you'd need spectral data
    K = 1.0 - image.mean(axis=(0, 1))  # Higher K means more absorption
    S = image.mean(axis=(0, 1))  # Higher S means more scattering

    # Normalize and create KM parameters
    K = K / K.max()
    S = S / S.max()

    return {"K": K, "S": S}


def create_mask_from_image(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Create a mask from a grayscale image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)

    return (gray / 255.0) > threshold


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

    # Generate or load paper texture
    print(f"Generating paper texture using {args.paper_method} method...")
    if args.paper_method == "from_image" and args.input_height:
        height_field = load_input_image(args.input_height, (args.width, args.height))
        if len(height_field.shape) == 3:
            height_field = (
                cv2.cvtColor((height_field * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                / 255.0
            )
        sim.paper_height = height_field
        sim.paper_capacity = (
            height_field * (sim.paper_max_capacity - sim.paper_min_capacity)
            + sim.paper_min_capacity
        )
    else:
        sim.generate_paper(method=args.paper_method, seed=args.seed)

    if args.save_stages:
        save_stage_output("01_paper_height", sim.paper_height, args.output_dir)
        save_stage_output("02_paper_capacity", sim.paper_capacity, args.output_dir)

    # Create pigment parameters
    if args.input_image:
        print("Creating pigment parameters from input image...")
        input_image = load_input_image(args.input_image, (args.width, args.height))
        pigment_km = create_pigment_from_image(input_image)
    else:
        # Default blue pigment
        pigment_km = {
            "K": np.array([0.8, 0.2, 0.1]),  # High absorption in red, low in blue
            "S": np.array([0.1, 0.2, 0.9]),  # High scattering in blue
        }

    # Add pigment
    print("Adding pigment...")
    pigment_idx = sim.add_pigment(
        density=args.pigment_density,
        staining_power=args.staining_power,
        granularity=args.granularity,
        kubelka_munk_params=pigment_km,
    )

    # Create or load wet mask
    if args.input_mask:
        print("Loading wet mask from input image...")
        mask_image = load_input_image(args.input_mask, (args.width, args.height))
        mask = create_mask_from_image(mask_image)
    else:
        print("Creating circular wet mask...")
        y, x = np.ogrid[
            -args.height // 2 : args.height // 2, -args.width // 2 : args.width // 2
        ]
        radius = min(args.width, args.height) // 3
        mask = x * x + y * y <= radius * radius

    # Set wet mask and pigment
    print("Setting up wet areas...")
    sim.set_wet_mask(mask)
    sim.set_pigment_water(pigment_idx, mask, concentration=args.concentration)

    if args.save_stages:
        save_stage_output("03_wet_mask", sim.wet_mask, args.output_dir)
        save_stage_output(
            "04_initial_pigment", sim.pigment_water[pigment_idx], args.output_dir
        )

    # Run simulation with stage saving
    print(f"Running simulation for {args.steps} steps...")
    for step in range(args.steps):
        # Move water in shallow-water layer
        sim.move_water()

        # Move pigment within water
        sim.move_pigment()

        # Transfer pigment between water and paper
        sim.transfer_pigment()

        # Simulate capillary flow
        sim.simulate_capillary_flow()

        # Save intermediate stages at certain intervals
        if args.save_stages and step % (args.steps // 10) == 0:
            save_stage_output(
                f"step_{step:03d}_pigment_water",
                sim.pigment_water[pigment_idx],
                args.output_dir,
            )
            save_stage_output(
                f"step_{step:03d}_pigment_paper",
                sim.pigment_paper[pigment_idx],
                args.output_dir,
            )

    # Render result
    print("Rendering final result...")
    renderer = WatercolorRenderer(sim)
    result = renderer.render_all_pigments()

    if args.save_stages:
        save_stage_output("final_result", result, args.output_dir)

    # Save output
    plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(result, 0, 1))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Watercolor image saved to {args.output}")


if __name__ == "__main__":
    main()

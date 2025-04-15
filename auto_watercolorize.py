#!/usr/bin/env python3
"""
Automatic Image Watercolorization

This script implements automatic conversion of photographs into watercolor paintings
based on the technique described in "Computer-Generated Watercolor" by Curtis et al.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import simulation components
from simulation.watercolor_simulation import WatercolorSimulation
from simulation.renderer import WatercolorRenderer

try:
    from simulation.optimized_kernels import CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False
from simulation_main import save_stage_output, load_input_image


def parse_arguments():
    """Parse command line arguments for automatic watercolorization."""
    parser = argparse.ArgumentParser(
        description="Automatically convert photographs into watercolor paintings."
    )

    # Input/output arguments
    parser.add_argument(
        "input_image", type=str, help="Path to input image to convert to watercolor"
    )
    parser.add_argument(
        "--output", type=str, default="watercolor_output.png", help="Output file path"
    )
    parser.add_argument(
        "--width", type=int, default=800, help="Width of the output image"
    )
    parser.add_argument(
        "--height", type=int, default=800, help="Height of the output image"
    )
    parser.add_argument(
        "--save-stages", action="store_true", help="Save intermediate stages"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="watercolor_stages",
        help="Directory for stage outputs",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debug output"
    )

    # Paper parameters
    parser.add_argument(
        "--paper-height",
        type=str,
        default=None,
        help="Input image for paper height field (grayscale)",
    )
    parser.add_argument(
        "--paper-capacity",
        type=str,
        default=None,
        help="Input image for paper fluid capacity (grayscale)",
    )
    parser.add_argument(
        "--paper-wetness",
        type=str,
        default=None,
        help="Input image for initial paper wetness distribution (grayscale)",
    )

    # Watercolor parameters
    parser.add_argument(
        "--num-pigments",
        type=int,
        default=3,
        help="Number of pigments to use for color separation",
    )
    parser.add_argument(
        "--num-glazes",
        type=int,
        default=3,
        help="Number of glazes (painting layers) to apply",
    )
    parser.add_argument(
        "--steps-per-glaze",
        type=int,
        default=50,
        help="Number of simulation steps per glaze",
    )
    parser.add_argument(
        "--edge-darkening", type=float, default=0.03, help="Edge darkening factor"
    )
    parser.add_argument("--viscosity", type=float, default=0.1, help="Fluid viscosity")
    parser.add_argument(
        "--drag", type=float, default=0.01, help="Viscous drag coefficient"
    )

    return parser.parse_args()


def process_pigment_mask(args):
    """Process a single pigment mask in parallel"""
    labels, i, shape, centers = args
    # Create pigment mask (concentration map)
    mask = (labels == i).reshape(shape[0], shape[1]).astype(np.float32)

    # Smooth the mask - use faster Gaussian blur
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Normalize to [0, 1]
    if mask.max() > 0:
        mask = mask / mask.max()

    # Extract color from cluster centers
    color = centers[i]

    # Create Kubelka-Munk parameters based on equations from Section 5.1
    # For better optical accuracy, we use formulations that align with the paper
    K = 1.0 - color  # Higher K means more absorption (eq 3)
    S = color  # Higher S means more scattering

    return {"K": K, "S": S}, mask


def color_separation(
    image: np.ndarray, num_pigments: int = 3
) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Separate an image into distinct pigments using k-means clustering.
    Uses parallel processing for improved performance.

    Args:
        image: Input RGB image normalized to [0, 1]
        num_pigments: Number of pigments to extract

    Returns:
        pigment_params: List of Kubelka-Munk parameters for each pigment
        pigment_masks: List of concentration maps for each pigment
    """
    print(f"Performing color separation into {num_pigments} pigments...")

    # Downscale very large images for faster clustering
    original_shape = image.shape
    downscale_factor = 1
    if image.shape[0] * image.shape[1] > 1_000_000:  # 1MP threshold
        downscale_factor = int(np.sqrt((image.shape[0] * image.shape[1]) / 1_000_000))
        if downscale_factor > 1:
            small_img = cv2.resize(
                image,
                (
                    image.shape[1] // downscale_factor,
                    image.shape[0] // downscale_factor,
                ),
                interpolation=cv2.INTER_AREA,
            )
            pixels = small_img.reshape(-1, 3)
            print(
                f"Downscaled image by factor of {downscale_factor} for faster clustering"
            )
        else:
            pixels = image.reshape(-1, 3)
    else:
        pixels = image.reshape(-1, 3)

    # Use mini-batch KMeans for very large images
    if len(pixels) > 1_000_000:
        from sklearn.cluster import MiniBatchKMeans

        kmeans = MiniBatchKMeans(
            n_clusters=num_pigments, random_state=42, batch_size=10000, max_iter=100
        )
    else:
        kmeans = KMeans(n_clusters=num_pigments, random_state=42, n_init=10)

    print("Clustering pixels...")
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # If we downscaled for clustering, we need to apply the clustering to the original image
    if downscale_factor > 1:
        # Predict on full-resolution image
        print("Applying clustering to full-resolution image...")
        pixels = image.reshape(-1, 3)
        labels = kmeans.predict(pixels)

    # Process pigment masks in parallel
    print("Processing pigment masks in parallel...")
    shape = image.shape[:2]
    args_list = [(labels, i, shape, centers) for i in range(num_pigments)]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_pigment_mask, args_list))

    # Unpack results
    pigment_params = [result[0] for result in results]
    pigment_masks = [result[1] for result in results]

    return pigment_params, pigment_masks


def create_paper_structure(
    width: int, height: int, height_file: str = None, capacity_file: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create or load paper height and capacity fields.

    Args:
        width: Desired width
        height: Desired height
        height_file: Optional path to height field image
        capacity_file: Optional path to capacity field image

    Returns:
        paper_height: Paper height field
        paper_capacity: Paper fluid capacity field
    """
    # Default parameters
    c_min = 0.3
    c_max = 0.7

    if height_file:
        # Load height field from image
        height_field = load_input_image(height_file, (width, height))
        if len(height_field.shape) == 3:
            height_field = (
                cv2.cvtColor((height_field * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                / 255.0
            )
    else:
        # Generate paper texture using the Paper class
        from simulation.paper import Paper

        paper = Paper(width, height, c_min=c_min, c_max=c_max)
        paper.generate("perlin")
        height_field = paper.height_field

    if capacity_file:
        # Load capacity field from image
        capacity_field = load_input_image(capacity_file, (width, height))
        if len(capacity_field.shape) == 3:
            capacity_field = (
                cv2.cvtColor(
                    (capacity_field * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                )
                / 255.0
            )
    else:
        # Derive capacity from height field
        capacity_field = height_field * (c_max - c_min) + c_min

    return height_field, capacity_field


def create_wetness_distribution(
    width: int, height: int, wetness_file: str = None, source_image: np.ndarray = None
) -> np.ndarray:
    """
    Create a wetness distribution mask based on input image or default pattern.

    Args:
        width: Desired width
        height: Desired height
        wetness_file: Optional path to wetness distribution image
        source_image: Optional source image to derive wetness from

    Returns:
        wetness: Paper wetness distribution [0-1]
    """
    if wetness_file:
        # Load wetness from image
        wetness = load_input_image(wetness_file, (width, height))
        if len(wetness.shape) == 3:
            wetness = (
                cv2.cvtColor((wetness * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                / 255.0
            )
    elif source_image is not None:
        # Derive wetness from source image edges
        gray = cv2.cvtColor((source_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        wetness = 1.0 - cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (15, 15), 0)
    else:
        # Create default wetness distribution (mostly wet)
        wetness = np.ones((height, width), dtype=np.float32) * 0.9

        # Add some randomness to the wetness
        noise = np.random.rand(height, width) * 0.2
        wetness = np.clip(wetness - noise, 0, 1)

    return wetness


def simulate_step(sim, verbose=False):
    """Run a single simulation step with all processes"""
    if verbose:
        print("        move_water...")
    sim.move_water()

    if verbose:
        print("        move_pigment...")
    sim.move_pigment()

    if verbose:
        print("        transfer_pigment...")
    sim.transfer_pigment()

    if verbose:
        print("        simulate_capillary_flow...")
    sim.simulate_capillary_flow()

    if verbose:
        print("        step completed.")


def run_simulation_chunk(sim, steps, verbose=False):
    """Run a chunk of simulation steps"""
    if verbose:
        for step in tqdm.tqdm(range(steps), desc="Simulation steps", leave=False):
            simulate_step(sim, verbose)
    else:
        for _ in range(steps):
            simulate_step(sim, verbose)
    return steps


def create_glazes(args, pigment_params, pigment_masks):
    """
    Create multiple glazes using the watercolor simulation.
    Optimized implementation with multi-scale simulation and improved
    fluid dynamics as described in the Curtis et al. paper.

    Args:
        args: Command line arguments
        pigment_params: Kubelka-Munk parameters for each pigment
        pigment_masks: Concentration maps for each pigment

    Returns:
        Final rendered result
    """
    num_pigments = len(pigment_params)

    # Check for multi-scale optimization opportunity
    use_multiscale = False
    scale_factor = 1

    # For large images, use multi-scale simulation to speed up computation
    if args.width * args.height > 640000:  # > 800x800
        use_multiscale = True
        # Calculate scale factor (power of 2) to bring dimensions down to 400-600px range
        target_size = 500
        scale_factor = max(
            1,
            int(
                np.power(
                    2, np.floor(np.log2(min(args.width, args.height) / target_size))
                )
            ),
        )

        if scale_factor > 1:
            print(f"Using multi-scale simulation with scale factor of {scale_factor}x")

    # Create paper structure
    paper_height, paper_capacity = create_paper_structure(
        args.width, args.height, args.paper_height, args.paper_capacity
    )

    # Load or create wetness distribution
    source_image = load_input_image(args.input_image, (args.width, args.height))
    wetness = create_wetness_distribution(
        args.width, args.height, args.paper_wetness, source_image
    )

    # Create wet mask (binary)
    wet_mask = wetness > 0.5

    # Save stage outputs if requested
    if args.save_stages:
        save_stage_output("paper_height", paper_height, args.output_dir)
        save_stage_output("paper_capacity", paper_capacity, args.output_dir)
        save_stage_output("wetness", wetness, args.output_dir)
        save_stage_output("wet_mask", wet_mask.astype(np.float32), args.output_dir)

    # For multi-scale approach, we create a lower-resolution simulation first
    if use_multiscale:
        # Downscale inputs for coarse simulation
        small_width = args.width // scale_factor
        small_height = args.height // scale_factor

        # Downscale fields using area interpolation for anti-aliasing
        small_paper_height = cv2.resize(
            paper_height, (small_width, small_height), interpolation=cv2.INTER_AREA
        )
        small_paper_capacity = cv2.resize(
            paper_capacity, (small_width, small_height), interpolation=cv2.INTER_AREA
        )
        small_wet_mask = (
            cv2.resize(
                wet_mask.astype(np.float32),
                (small_width, small_height),
                interpolation=cv2.INTER_AREA,
            )
            > 0.5
        )

        # Also downscale pigment masks
        small_pigment_masks = []
        for mask in pigment_masks:
            small_mask = cv2.resize(
                mask, (small_width, small_height), interpolation=cv2.INTER_AREA
            )
            small_pigment_masks.append(small_mask)

        # Create low-resolution simulation
        print(f"Creating low-resolution simulation ({small_width}x{small_height})...")
        sim_low_res = WatercolorSimulation(small_width, small_height)
        sim_low_res.paper_height = small_paper_height
        sim_low_res.paper_capacity = small_paper_capacity

        # Use slightly higher viscosity and edge darkening for stable coarse simulation
        sim_low_res.viscosity = args.viscosity * 1.2
        sim_low_res.viscous_drag = args.drag
        sim_low_res.edge_darkening_factor = args.edge_darkening * 1.5
        sim_low_res.set_wet_mask(small_wet_mask)

        # Add pigments at low resolution
        pigment_indices_low_res = []
        for i in range(num_pigments):
            pigment_idx = sim_low_res.add_pigment(
                density=1.0,
                staining_power=0.6,
                granularity=0.4,
                kubelka_munk_params=pigment_params[i],
            )
            pigment_indices_low_res.append(pigment_idx)

        # Also create full-resolution simulation for final results
        sim = WatercolorSimulation(args.width, args.height)
        sim.paper_height = paper_height
        sim.paper_capacity = paper_capacity
        sim.viscosity = args.viscosity
        sim.viscous_drag = args.drag
        sim.edge_darkening_factor = args.edge_darkening
        sim.set_wet_mask(wet_mask)

        pigment_indices = []
        for i in range(num_pigments):
            pigment_idx = sim.add_pigment(
                density=1.0,
                staining_power=0.6,
                granularity=0.4,
                kubelka_munk_params=pigment_params[i],
            )
            pigment_indices.append(pigment_idx)
    else:
        # Standard single-resolution approach
        print(f"Creating simulation ({args.width}x{args.height})...")
        sim = WatercolorSimulation(args.width, args.height)
        sim.paper_height = paper_height
        sim.paper_capacity = paper_capacity
        sim.viscosity = args.viscosity
        sim.viscous_drag = args.drag
        sim.edge_darkening_factor = args.edge_darkening
        sim.set_wet_mask(wet_mask)

        pigment_indices = []
        for i in range(num_pigments):
            pigment_idx = sim.add_pigment(
                density=1.0,
                staining_power=0.6,
                granularity=0.4,
                kubelka_munk_params=pigment_params[i],
            )
            pigment_indices.append(pigment_idx)

        # Set reference to the active simulation
        sim_active = sim
        pigment_indices_active = pigment_indices
        active_pigment_masks = pigment_masks

    # For each glaze, add pigment and simulate
    for glaze_idx in range(args.num_glazes):
        print(f"Processing glaze {glaze_idx+1}/{args.num_glazes}")

        # Select active simulation based on current stage
        if use_multiscale:
            # Use low-res for early glazes, high-res for final glazes
            if glaze_idx < args.num_glazes - 1:
                sim_active = sim_low_res
                pigment_indices_active = pigment_indices_low_res
                active_pigment_masks = small_pigment_masks
                print("  Using low-resolution simulation for this glaze")
            else:
                # For the final glaze, use full resolution
                # First, copy state from low-res to high-res sim for continuity
                print("  Upscaling to full-resolution for final glaze")

                # Upscale pigment distributions
                for i in range(num_pigments):
                    # Upscale pigment_water
                    upscaled_water = cv2.resize(
                        sim_low_res.pigment_water[pigment_indices_low_res[i]],
                        (args.width, args.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    sim.pigment_water[pigment_indices[i]] = upscaled_water

                    # Upscale pigment_paper
                    upscaled_paper = cv2.resize(
                        sim_low_res.pigment_paper[pigment_indices_low_res[i]],
                        (args.width, args.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    sim.pigment_paper[pigment_indices[i]] = upscaled_paper

                # Upscale water saturation
                upscaled_saturation = cv2.resize(
                    sim_low_res.water_saturation,
                    (args.width, args.height),
                    interpolation=cv2.INTER_LINEAR,
                )
                sim.water_saturation = upscaled_saturation

                sim_active = sim
                pigment_indices_active = pigment_indices
                active_pigment_masks = pigment_masks
        else:
            # Always use the single simulation
            sim_active = sim
            pigment_indices_active = pigment_indices
            active_pigment_masks = pigment_masks

        # --- Add pigment for this glaze with paper-specific weights ---
        # Determine which base pigment is dominant for this glaze layer
        dominant_pigment_idx = glaze_idx % num_pigments

        # Implement pigment granularity variations based on the Curtis et al. paper
        # Dense pigments like ultramarine show stronger granulation (Section 2.1)
        for i in range(num_pigments):
            # Adjust weight based on position in glazing sequence and dominant pigment
            # As described in Section 2.2 (Glazing section) of the paper
            if i == dominant_pigment_idx:
                weight = 1.0
                # For dominant pigment, use higher density for better granulation
                sim_active.pigment_properties[pigment_indices_active[i]][
                    "density"
                ] = 1.2
                # Increase granularity for dominant pigment
                sim_active.pigment_properties[pigment_indices_active[i]][
                    "granularity"
                ] = 0.5
            else:
                # Less weight for non-dominant pigments in this glaze
                weight = 0.3
                # Use lower density for better flow
                sim_active.pigment_properties[pigment_indices_active[i]][
                    "density"
                ] = 0.8
                # Less granularity for non-dominant
                sim_active.pigment_properties[pigment_indices_active[i]][
                    "granularity"
                ] = 0.3

            # Add pigment to the water layer where the paper is wet
            sim_active.pigment_water[pigment_indices_active[i]] += (
                active_pigment_masks[i] * weight * sim_active.wet_mask
            )

        # --- Optimize simulation steps based on image size ---
        total_steps_for_glaze = args.steps_per_glaze

        # Adjust step count based on resolution for multi-scale approach
        if use_multiscale and sim_active is sim_low_res:
            # Low-res can use fewer steps but still get good results
            total_steps_for_glaze = max(30, total_steps_for_glaze // 2)

        # Use adaptive control strategy from the Curtis paper (Section 6.1)
        # Run simulation in stages with edge-darkening control

        # First stage: Initial water flow (30%)
        initial_flow_steps = int(total_steps_for_glaze * 0.3)
        print(f"  Initial flow simulation ({initial_flow_steps} steps)...")
        run_simulation_chunk(sim_active, initial_flow_steps, args.verbose)

        # Apply edge-darkening effect (Section 4.3.3)
        print("  Enhancing edge darkening...")
        sim_active.flow_outward()

        # Second stage: Main pigment transfer (50%)
        main_simulation_steps = int(total_steps_for_glaze * 0.5)
        print(f"  Main simulation ({main_simulation_steps} steps)...")
        run_simulation_chunk(sim_active, main_simulation_steps, args.verbose)

        # Pigment control phase - align pigment with target masks
        print("  Applying pigment control...")
        delta_g = 0.05  # Pigment adjustment threshold
        for i, target_mask in enumerate(active_pigment_masks):
            pigment_idx = pigment_indices_active[i]
            current = (
                sim_active.pigment_water[pigment_idx]
                + sim_active.pigment_paper[pigment_idx]
            )
            diff = cv2.GaussianBlur(
                target_mask - current, (15, 15), 0
            )  # Smoothed difference

            # Areas with too little pigment
            too_little = (diff > delta_g) & sim_active.wet_mask
            if np.any(too_little):
                # Add pigment with safety limits
                sim_active.pigment_water[pigment_idx][too_little] += delta_g

                # Decrease pressure slightly with safety bounds
                sim_active.pressure[too_little] = np.maximum(
                    0, sim_active.pressure[too_little] - delta_g
                )

            # Areas with too much pigment
            too_much = (diff < -delta_g) & sim_active.wet_mask
            if np.any(too_much):
                # Add water (increase pressure) to dilute
                max_pressure = 2.0  # Set a reasonable upper bound
                sim_active.pressure[too_much] = np.minimum(
                    max_pressure, sim_active.pressure[too_much] + delta_g * 2
                )

        # Final stage: Settling and granulation (20%)
        final_steps = total_steps_for_glaze - initial_flow_steps - main_simulation_steps
        print(f"  Final settling ({final_steps} steps)...")
        run_simulation_chunk(sim_active, final_steps, args.verbose)

        # Enhance granulation effect from Section 4.5 of the paper
        # This creates more realistic paper pattern interaction
        if sim_active.wet_mask.sum() > 0:
            for i in range(num_pigments):
                pigment_idx = pigment_indices_active[i]
                granularity = sim_active.pigment_properties[pigment_idx]["granularity"]
                if granularity > 0.2:  # Only enhance noticeably granulating pigments
                    # Enhance pigment deposition in paper valleys
                    valley_mask = (sim_active.paper_height < 0.3) & sim_active.wet_mask
                    if np.any(valley_mask):
                        # Move some pigment from water to paper in valleys
                        transfer_amount = (
                            0.1 * sim_active.pigment_water[pigment_idx][valley_mask]
                        )
                        sim_active.pigment_paper[pigment_idx][
                            valley_mask
                        ] += transfer_amount
                        sim_active.pigment_water[pigment_idx][
                            valley_mask
                        ] -= transfer_amount

        print(f"  Glaze {glaze_idx+1} finished simulation.")

        # --- Save intermediate glaze state if requested ---
        if args.save_stages:
            renderer = WatercolorRenderer(sim_active)
            glaze_result = renderer.render_all_pigments()
            save_stage_output(
                f"glaze_{glaze_idx+1}_state", glaze_result, args.output_dir
            )

    # After all glazes, render the final result using Kubelka-Munk compositing
    # Always use the full resolution simulation for final rendering
    renderer = WatercolorRenderer(sim)
    final_result = renderer.render_all_pigments()
    return final_result


def main():
    """Main entry point for automatic watercolorization."""
    args = parse_arguments()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Processing image: {args.input_image}")
    print(f"Output will be saved to: {args.output}")

    # Load input image
    input_image = load_input_image(args.input_image, (args.width, args.height))

    # Perform color separation
    pigment_params, pigment_masks = color_separation(input_image, args.num_pigments)

    # Save color separation results if requested
    if args.save_stages:
        os.makedirs(args.output_dir, exist_ok=True)
        for i, mask in enumerate(pigment_masks):
            save_stage_output(f"pigment_mask_{i+1}", mask, args.output_dir)

    # Create glazes and run simulation
    result = create_glazes(args, pigment_params, pigment_masks)

    # Save final result
    plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(result, 0, 1))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Watercolor image saved to {args.output}")


if __name__ == "__main__":
    main()

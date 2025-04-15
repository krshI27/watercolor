#!/usr/bin/env python3
"""
Debug Script for Watercolor Simulation Steps

This script allows running individual steps of the watercolor simulation
and saving/loading the state between steps for debugging purposes.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import logging
import time
import multiprocessing

# Define MAX_WORKERS for parallel processing (used in auto_watercolorize)
MAX_WORKERS = multiprocessing.cpu_count() or 4

# Import simulation components and helper functions
from simulation.watercolor_simulation import WatercolorSimulation
from simulation.renderer import WatercolorRenderer
from auto_watercolorize import (
    load_input_image,
    color_separation,
    create_paper_structure,
    create_wetness_distribution,
    save_stage_output,
    parse_arguments as auto_parse_arguments,  # Reuse args definition
)

# Define which simulation arrays constitute the 'state'
SIMULATION_STATE_KEYS = [
    "width",
    "height",
    "num_pigments",
    "water_saturation",
    "water_velocity_x",
    "water_velocity_y",
    "pigment_water",
    "pigment_paper",
    "paper_height",
    "paper_capacity",
    "wet_mask",
    "pressure",
    "divergence",
    "pigment_properties",  # List of dicts, handle separately if needed
    "viscosity",
    "viscous_drag",
    "edge_darkening_factor",
    # Add any other relevant simulation parameters/arrays here
]


def parse_debug_arguments():
    """Parse command line arguments for the debug script."""
    parser = argparse.ArgumentParser(
        description="Debug individual watercolor simulation steps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Arguments from auto_watercolorize ---
    # Input/output
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument(
        "--width", type=int, default=128, help="Simulation width"
    )  # Smaller default for faster debug
    parser.add_argument(
        "--height", type=int, default=128, help="Simulation height"
    )  # Smaller default
    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug_stages",
        help="Directory for debug outputs",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debug output"
    )

    # Paper
    parser.add_argument(
        "--paper-height",
        type=str,
        default=None,
        help="Input image for paper height field",
    )
    parser.add_argument(
        "--paper-capacity",
        type=str,
        default=None,
        help="Input image for paper fluid capacity",
    )
    parser.add_argument(
        "--paper-wetness",
        type=str,
        default=None,
        help="Input image for initial paper wetness",
    )

    # Watercolor
    parser.add_argument(
        "--num-pigments", type=int, default=3, help="Number of pigments"
    )
    parser.add_argument(
        "--edge-darkening", type=float, default=0.03, help="Edge darkening factor"
    )
    parser.add_argument("--viscosity", type=float, default=0.1, help="Fluid viscosity")
    parser.add_argument(
        "--drag", type=float, default=0.01, help="Viscous drag coefficient"
    )

    # --- Debug specific arguments ---
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "init",
            "move_water",
            "move_pigment",
            "transfer_pigment",
            "capillary_flow",
            "flow_outward",
        ],
        help="Simulation step to execute ('init' performs setup only).",
    )
    parser.add_argument(
        "--load-state",
        type=str,
        default=None,
        help="Path to .npz file containing simulation state to load before running the step.",
    )
    parser.add_argument(
        "--save-state",
        type=str,
        default=None,
        help="Path to .npz file to save simulation state after running the step.",
    )
    parser.add_argument(
        "--glaze-idx",
        type=int,
        default=0,
        help="Glaze index (for pigment application logic during 'init').",
    )
    parser.add_argument(
        "--save-pre-state",
        action="store_true",
        help="Save the state *before* running the specified step (useful for comparing).",
    )

    return parser.parse_args()


def save_simulation_state(
    sim: WatercolorSimulation, filepath: str, pigment_params: list
):
    """Saves the current state of the simulation to a .npz file."""
    state_dict = {}
    for key in SIMULATION_STATE_KEYS:
        if hasattr(sim, key):
            value = getattr(sim, key)
            # Special handling for list of dicts (pigment_properties) if needed
            if key == "pigment_properties":
                # np.savez doesn't handle list of dicts well directly.
                # We can save the pigment_params passed separately,
                # or serialize properties if they change during simulation.
                # For now, let's assume they are mostly static and derived from pigment_params.
                pass  # Skip saving this directly for now, reconstruct from pigment_params
            elif isinstance(value, (np.ndarray, int, float, bool)):
                state_dict[key] = value
            elif isinstance(value, list) and all(
                isinstance(item, np.ndarray) for item in value
            ):
                # Handle lists of numpy arrays (like pigment_water, pigment_paper)
                # Save them individually with indexed names
                for i, arr in enumerate(value):
                    state_dict[f"{key}_{i}"] = arr
            else:
                logging.warning(
                    f"Skipping unsupported state key type: {key} ({type(value)})"
                )

    # Add pigment K/S parameters separately as they are needed for setup/rendering
    for i, params in enumerate(pigment_params):
        state_dict[f"pigment_{i}_K"] = params["K"]
        state_dict[f"pigment_{i}_S"] = params["S"]

    # Add dimensions explicitly
    state_dict["width"] = sim.width
    state_dict["height"] = sim.height
    state_dict["num_pigments"] = (
        len(sim.pigment_properties)
        if hasattr(sim, "pigment_properties") and sim.pigment_properties
        else len(sim.pigment_water)
    )

    logging.info(f"Saving simulation state to {filepath}")
    np.savez(filepath, **state_dict)


def load_simulation_state(sim: WatercolorSimulation, filepath: str) -> list:
    """Loads simulation state from a .npz file into the sim object."""
    logging.info(f"Loading simulation state from {filepath}")
    with np.load(filepath) as data:
        # Basic check for compatibility
        loaded_width = int(data["width"])
        loaded_height = int(data["height"])
        loaded_num_pigments = int(data["num_pigments"])

        if loaded_width != sim.width or loaded_height != sim.height:
            raise ValueError(
                f"Loaded state dimensions ({loaded_width}x{loaded_height}) "
                f"do not match simulation ({sim.width}x{sim.height})"
            )
        if loaded_num_pigments != sim.num_pigments:
            raise ValueError(
                f"Loaded state pigment count ({loaded_num_pigments}) "
                f"does not match simulation ({sim.num_pigments})"
            )

        pigment_params = [
            {"K": data[f"pigment_{i}_K"], "S": data[f"pigment_{i}_S"]}
            for i in range(loaded_num_pigments)
        ]

        # Load arrays and properties
        for key in SIMULATION_STATE_KEYS:
            if key == "pigment_properties":
                continue  # Skip loading this directly

            if key in data:
                setattr(sim, key, data[key])
            elif f"{key}_0" in data:  # Check if it was saved as a list of arrays
                loaded_list = []
                i = 0
                while f"{key}_{i}" in data:
                    loaded_list.append(data[f"{key}_{i}"])
                    i += 1
                if loaded_list:
                    setattr(sim, key, loaded_list)
            # else:
            #     logging.warning(f"State key '{key}' not found in {filepath}")

        # Re-initialize pigment properties in the simulation based on loaded K/S
        # Assuming density, staining, granularity are default or set by glaze logic later
        sim.pigment_properties = []
        for i in range(loaded_num_pigments):
            props = sim._default_pigment_properties()  # Get defaults
            props["kubelka_munk_params"] = pigment_params[i]
            sim.pigment_properties.append(props)

        # Ensure internal lists match num_pigments if loaded individually
        if (
            not isinstance(sim.pigment_water, list)
            or len(sim.pigment_water) != sim.num_pigments
        ):
            sim.pigment_water = [
                np.zeros((sim.height, sim.width), dtype=sim.dtype)
                for _ in range(sim.num_pigments)
            ]
            for i in range(sim.num_pigments):
                if f"pigment_water_{i}" in data:
                    sim.pigment_water[i] = data[f"pigment_water_{i}"]

        if (
            not isinstance(sim.pigment_paper, list)
            or len(sim.pigment_paper) != sim.num_pigments
        ):
            sim.pigment_paper = [
                np.zeros((sim.height, sim.width), dtype=sim.dtype)
                for _ in range(sim.num_pigments)
            ]
            for i in range(sim.num_pigments):
                if f"pigment_paper_{i}" in data:
                    sim.pigment_paper[i] = data[f"pigment_paper_{i}"]

        logging.info("Simulation state loaded successfully.")
        return pigment_params


def main():
    """Main entry point for the debug script."""
    args = parse_debug_arguments()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("debug_simulation")

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    sim = None
    pigment_params = []
    pigment_masks = []

    # --- Load or Initialize Simulation ---
    if args.load_state:
        logger.info(f"Attempting to load state from: {args.load_state}")
        # Initialize sim with correct dimensions first
        sim = WatercolorSimulation(
            args.width, args.height, num_pigments=args.num_pigments
        )
        # Load state into the existing sim object
        pigment_params = load_simulation_state(sim, args.load_state)
        # We might need pigment_masks for rendering/comparison, but they aren't part of the
        # core simulation state. We could re-run color separation or save/load them too.
        # For now, let's assume they aren't strictly needed unless we add pigment.
        logger.info("Loaded state. Pigment masks are not reloaded/recalculated.")

    else:
        logger.info("No load state provided, performing initial setup...")
        # 1. Load input image
        input_image = load_input_image(args.input_image, (args.width, args.height))
        save_stage_output("debug_input_image", input_image, args.output_dir)

        # 2. Color separation
        pigment_params, pigment_masks = color_separation(input_image, args.num_pigments)
        for i, mask in enumerate(pigment_masks):
            save_stage_output(f"debug_pigment_mask_{i}", mask, args.output_dir)

        # 3. Create paper structure
        paper_height, paper_capacity = create_paper_structure(
            args.width, args.height, args.paper_height, args.paper_capacity
        )
        save_stage_output("debug_paper_height", paper_height, args.output_dir)
        save_stage_output("debug_paper_capacity", paper_capacity, args.output_dir)

        # 4. Create wetness distribution
        wetness = create_wetness_distribution(
            args.width, args.height, args.paper_wetness, input_image
        )
        wet_mask = wetness > 0.5
        save_stage_output("debug_wetness", wetness, args.output_dir)
        save_stage_output(
            "debug_wet_mask", wet_mask.astype(np.float32), args.output_dir
        )

        # 5. Initialize Simulation object
        sim = WatercolorSimulation(args.width, args.height)

        # Initialize pigment arrays for the correct number of pigments
        sim.pigment_water = [
            np.zeros((args.height, args.width), dtype=np.float32)
            for _ in range(args.num_pigments)
        ]
        sim.pigment_paper = [
            np.zeros((args.height, args.width), dtype=np.float32)
            for _ in range(args.num_pigments)
        ]
        sim.pigment_properties = []

        sim.paper_height = paper_height
        sim.paper_capacity = paper_capacity
        sim.viscosity = args.viscosity
        sim.viscous_drag = args.drag
        sim.edge_darkening_factor = args.edge_darkening
        sim.set_wet_mask(wet_mask)

        # 6. Add pigments to simulation (properties only, no amounts yet)
        for i in range(args.num_pigments):
            sim.add_pigment(
                density=1.0,  # Default, can be adjusted per glaze
                staining_power=0.6,
                granularity=0.4,
                kubelka_munk_params=pigment_params[i],
            )

        # 7. Add initial pigment amounts (mimicking start of a glaze)
        # This logic is simplified from create_glazes for debugging 'init' state
        logger.info(f"Applying initial pigment based on glaze index {args.glaze_idx}")
        dominant_pigment_idx = args.glaze_idx % args.num_pigments
        for i in range(args.num_pigments):
            weight = 1.0 if i == dominant_pigment_idx else 0.3
            # Add pigment to water layer where wet
            sim.pigment_water[i] += pigment_masks[i] * weight * sim.wet_mask
            # Adjust properties based on dominance (simplified)
            if i == dominant_pigment_idx:
                sim.pigment_properties[i]["density"] = 1.2
                sim.pigment_properties[i]["granularity"] = 0.5
            else:
                sim.pigment_properties[i]["density"] = 0.8
                sim.pigment_properties[i]["granularity"] = 0.3

        logger.info("Initial setup complete.")

    # --- Save Pre-Step State (Optional) ---
    if args.save_pre_state and args.save_state:
        pre_step_save_path = Path(args.save_state)
        pre_step_save_path = pre_step_save_path.with_name(
            f"{pre_step_save_path.stem}_pre{pre_step_save_path.suffix}"
        )
        save_simulation_state(sim, str(pre_step_save_path), pigment_params)

    # --- Execute Specified Step ---
    step_function = None
    if args.step == "move_water":
        step_function = sim.move_water
    elif args.step == "move_pigment":
        step_function = sim.move_pigment
    elif args.step == "transfer_pigment":
        step_function = sim.transfer_pigment
    elif args.step == "capillary_flow":
        step_function = sim.simulate_capillary_flow
    elif args.step == "flow_outward":
        step_function = sim.flow_outward
    elif args.step == "init":
        logger.info(
            "Step 'init' selected. No simulation step executed. Saving initial state."
        )
    else:
        logger.error(f"Unknown step: {args.step}")
        sys.exit(1)

    if step_function:
        logger.info(f"Executing step: {args.step}...")
        start_time = time.time()
        step_function()
        end_time = time.time()
        logger.info(
            f"Step '{args.step}' completed in {end_time - start_time:.4f} seconds."
        )

    # --- Save Post-Step State ---
    if args.save_state:
        save_simulation_state(sim, args.save_state, pigment_params)

    # --- Optional: Render and save current state ---
    logger.info("Rendering current state...")
    renderer = WatercolorRenderer(sim)
    current_render = renderer.render_all_pigments()
    render_filename = f"debug_render_after_{args.step}.png"
    if args.load_state:
        render_filename = (
            f"debug_render_after_{args.step}_from_{Path(args.load_state).stem}.png"
        )

    save_stage_output(render_filename, current_render, args.output_dir)
    logger.info(f"Saved current render to {Path(args.output_dir) / render_filename}")

    logger.info("Debug script finished.")


if __name__ == "__main__":
    main()

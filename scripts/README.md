# Scripts for Computer-Generated Watercolor Simulation

This directory contains user-facing scripts for the project. These scripts provide high-level automation and entry points for common workflows, such as automatic image watercolorization.

## watercolorize_image.py: Automatic Image Watercolorization

This script implements the full pipeline for converting photographs into watercolor paintings, leveraging the core simulation and rendering modules. It is designed for both usability and technical flexibility.

### Processing Steps and Technical Details

1. **Input Loading and Preprocessing**
   - Loads the input image and optional masks (wet area, paper height, sizing).
   - Supports various image formats and automatic resizing.
   - *Reasoning:* Flexible input handling allows the script to be used in both research and production settings.

2. **Pigment Separation (Color Clustering)**
   - Uses KMeans clustering to separate the image into pigment layers, each with its own mask and representative color.
   - Kubelka-Munk parameters (K, S) are derived from the cluster centers.
   - *Reasoning:* This data-driven approach enables realistic pigment modeling and supports the optical compositing model.

3. **Paper Generation**
   - Loads or generates a paper height field and capacity field using Perlin noise, fractal, or user-supplied images.
   - *Reasoning:* Realistic paper structure is essential for simulating granulation, flow, and backruns.

4. **Simulation Initialization**
   - Instantiates the WatercolorSimulation object with the specified dimensions and parameters.
   - Sets up the paper, pigment layers, and wet area masks.
   - *Reasoning:* Modular initialization allows for easy extension and parameter tuning.

5. **Multi-Stage Glaze Simulation**
   - For each glaze (wash), pigments are added according to their masks and properties.
   - The simulation is run in stages: initial flow, pigment control, and final settling, with edge darkening and pigment redistribution applied as needed.
   - *Reasoning:* This staged approach mirrors real watercolor technique and improves control over the final appearance.

6. **Parallelization and Performance**
   - Uses Python's multiprocessing and/or thread pools to accelerate pigment separation and simulation steps.
   - Optionally leverages CUDA for GPU acceleration if available.
   - *Reasoning:* Parallel processing is critical for handling large images and multiple glazes efficiently.

7. **Rendering and Output**
   - Uses the WatercolorRenderer and Kubelka-Munk model to composite pigment layers and produce the final image.
   - Intermediate results and debug images can be saved for analysis.
   - *Reasoning:* Saving intermediate stages aids in debugging and understanding the simulation process.

### Script Design Rationale
- **Modularity:** Each processing step is encapsulated in a function or class, making the script easy to maintain and extend.
- **Performance:** Parallelization and optional GPU support ensure the script can handle high-resolution images and complex simulations.
- **Reproducibility:** Command-line arguments allow all parameters to be specified explicitly, supporting reproducible experiments.
- **Extensibility:** The script is designed to be a template for new workflows, such as batch processing or integration with other tools.

## Adding New Scripts

Place any new user-facing scripts in this directory. Scripts should:
- Follow the modular and documented style of `watercolorize_image.py`
- Clearly document their processing steps, technical details, and design rationale
- Use argparse for command-line argument parsing
- Save intermediate and final outputs in a user-friendly manner

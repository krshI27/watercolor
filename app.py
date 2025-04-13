import streamlit as st
import sys
import os
from PIL import Image
import time
import subprocess
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Watercolor Effect Generator",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("🎨 Watercolor Effect Generator")
st.markdown("""
This app generates beautiful watercolor-style images using mathematical algorithms.
Adjust the parameters in the sidebar and click 'Generate Watercolor' to create your artwork!
""")

# Create a sidebar for parameters
st.sidebar.title("Watercolor Parameters")

# Organize parameters in expandable sections
with st.sidebar.expander("Canvas Settings", expanded=True):
    width = st.number_input("Width", min_value=100, max_value=3000, value=1000, step=50)
    height = st.number_input("Height", min_value=100, max_value=3000, value=1500, step=50)

with st.sidebar.expander("Shape Configuration", expanded=True):
    shape_size_range = st.slider("Shape Size Range", min_value=10, max_value=500, value=(100, 300), step=10)
    min_shape_size, max_shape_size = shape_size_range
    
    shapes_per_layer_range = st.slider("Shapes per Layer Range", min_value=1, max_value=100, value=(20, 25), step=1)
    min_shapes, max_shapes = shapes_per_layer_range

with st.sidebar.expander("Deformation Settings", expanded=True):
    initial_deform = st.slider("Initial Deformation", min_value=0, max_value=300, value=120, step=10)
    deviation = st.slider("Random Deviation", min_value=0, max_value=200, value=50, step=5)
    base_deforms = st.slider("Base Deformation Iterations", min_value=0, max_value=10, value=1, step=1)
    final_deforms = st.slider("Final Deformation Iterations", min_value=0, max_value=10, value=3, step=1)

with st.sidebar.expander("Layer Settings", expanded=True):
    num_layers = st.slider("Number of Layers (-1 for automatic)", min_value=-1, max_value=50, value=-1, step=1)
    boundary_overflow = st.slider("Boundary Overflow", min_value=0, max_value=500, value=100, step=10)

with st.sidebar.expander("Color Settings", expanded=True):
    shape_alpha = st.slider("Shape Opacity", min_value=0.01, max_value=1.0, value=0.08, step=0.01)
    color_variation = st.slider("Color Variation", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

# Output path
output_path = os.path.join(os.getcwd(), "watercolor_output.png")

# Generate button
if st.sidebar.button("Generate Watercolor", type="primary"):
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Build the command
    cmd = [
        "python", "effect_main.py",
        "--width", str(width),
        "--height", str(height),
        "--initial-deform", str(initial_deform),
        "--deviation", str(deviation),
        "--base-deformations", str(base_deforms),
        "--final-deformations", str(final_deforms),
        "--min-shapes", str(min_shapes),
        "--max-shapes", str(max_shapes),
        "--shape-opacity", str(shape_alpha),
        "--layers", str(num_layers),
        "--min-shape-size", str(min_shape_size),
        "--max-shape-size", str(max_shape_size),
        "--boundary-overflow", str(boundary_overflow),
        "--color-variation", str(color_variation),
        "--output", output_path
    ]
    
    # Update status
    status_text.text("Generating watercolor image...")
    progress_bar.progress(10)
    
    try:
        # Run the command
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        progress_bar.progress(30)
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            progress_bar.progress(90)
            status_text.text("Processing complete!")
        else:
            st.error(f"Error: {stderr.decode()}")
            progress_bar.empty()
            status_text.empty()
            st.stop()
            
        progress_bar.progress(100)
        time.sleep(0.5)  # Brief pause to show 100% completion
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        st.stop()

# Display image if it exists
if os.path.exists(output_path):
    # Create columns for image display
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        st.image(output_path, caption="Generated Watercolor Image", use_container_width=True)
    
    # Add download button
    with open(output_path, "rb") as file:
        btn = st.download_button(
            label="Download Image",
            data=file,
            file_name="watercolor_art.png",
            mime="image/png"
        )
else:
    # If no image exists yet, show a placeholder or example image
    if os.path.exists("effect/watercolor.png"):
        st.image("effect/watercolor.png", caption="Example watercolor image (generate your own using the sidebar controls)", use_container_width=True)
    else:
        st.info("Generate your first watercolor image using the controls in the sidebar!")

# Add information about preset configurations
st.sidebar.subheader("Preset Configurations")
preset = st.sidebar.selectbox("Choose a preset",
    ["Custom", "Basic", "Advanced", "Single Shape", "Small Repeating Shapes"]
)

# Apply preset when selected
if preset != "Custom" and st.sidebar.button(f"Apply {preset} Preset"):
    if preset == "Basic":
        # Default settings are already set
        st.experimental_rerun()
    elif preset == "Advanced":
        # Set advanced settings
        st.session_state.width = 1000
        st.session_state.height = 1000
        st.session_state.initial_deform = 125
        st.session_state.deviation = 25
        st.session_state.base_deforms = 6
        st.session_state.final_deforms = 6
        st.session_state.min_shapes = 5
        st.session_state.max_shapes = 15
        st.session_state.shape_alpha = 0.1
        st.session_state.num_layers = 10
        st.session_state.min_shape_size = 10
        st.session_state.max_shape_size = 20
        st.session_state.boundary_overflow = 1
        st.session_state.color_variation = 0.1
        st.experimental_rerun()
    elif preset == "Single Shape":
        # Set single shape settings
        st.session_state.min_shapes = 1
        st.session_state.max_shapes = 1
        st.session_state.num_layers = 1
        st.session_state.shape_alpha = 0.9
        st.session_state.color_variation = 0.0
        st.experimental_rerun()
    elif preset == "Small Repeating Shapes":
        # Set small repeating shapes settings
        st.session_state.min_shapes = 50
        st.session_state.max_shapes = 50
        st.session_state.shape_alpha = 0.04
        st.session_state.min_shape_size = 40
        st.session_state.max_shape_size = 40
        st.session_state.base_deforms = 0
        st.session_state.final_deforms = 0
        st.experimental_rerun()

# Add footer with instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Instructions**:
1. Adjust parameters using the sliders
2. Click "Generate Watercolor"
3. Wait for the image to be generated
4. Download your creation if desired
""")

# Show documentation tooltip
with st.sidebar.expander("Parameter Descriptions"):
    st.markdown("""
    - **Width/Height**: Canvas dimensions in pixels
    - **Shape Size**: Size range for watercolor shapes
    - **Shapes per Layer**: Number of shapes to generate in each layer
    - **Initial Deformation**: Base deformation amount for shapes
    - **Random Deviation**: Random variation applied to shape points
    - **Deformation Iterations**: Number of times to apply deformation algorithm
    - **Number of Layers**: Total layers (-1 for automatic based on height)
    - **Boundary Overflow**: How far shapes can extend outside canvas
    - **Shape Opacity**: Transparency of each shape
    - **Color Variation**: How much colors can vary within a layer
    """)

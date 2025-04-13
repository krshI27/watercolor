#!/usr/bin/env python3
"""
Watercolor Renderer Module

This module implements the rendering of watercolor simulations using the Kubelka-Munk model
as described in Section 5 of the 'Computer-Generated Watercolor' paper.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt

from .fluid_simulation import WaterSimulation
from .pigment import PigmentLayer
from .kubelka_munk import KubelkaMunk


class WatercolorRenderer:
    """
    Renderer for watercolor simulations using the Kubelka-Munk model.
    
    This class takes the output from the fluid simulation and renders it using
    the Kubelka-Munk optical compositing model to produce realistic watercolor effects.
    """
    
    def __init__(self, simulation: WaterSimulation):
        """
        Initialize renderer with a watercolor simulation.
        
        Parameters:
        -----------
        simulation : WaterSimulation
            The watercolor simulation to render
        """
        self.simulation = simulation
    
    def render_pigment_layer(
        self, 
        layer_idx: int, 
        background_color: np.ndarray = np.array([1.0, 1.0, 1.0])
    ) -> np.ndarray:
        """
        Render a single pigment layer.
        
        Parameters:
        -----------
        layer_idx : int
            Index of the pigment layer to render
        background_color : np.ndarray
            RGB color of the background
        
        Returns:
        --------
        np.ndarray
            RGB image of rendered pigment
        """
        if layer_idx >= len(self.simulation.pigment_layers):
            raise ValueError(f"Invalid pigment layer index: {layer_idx}")
        
        # Get pigment layer
        pigment_layer = self.simulation.pigment_layers[layer_idx]
        
        # Get pigment properties
        kubelka_munk_params = pigment_layer.pigment.kubelka_munk_params
        
        # Check if K and S are provided
        if 'K' not in kubelka_munk_params or 'S' not in kubelka_munk_params:
            raise ValueError(f"Pigment {pigment_layer.pigment.name} has no Kubelka-Munk parameters (K and S)")
        
        K = kubelka_munk_params['K']
        S = kubelka_munk_params['S']
        
        # Get pigment thickness (combined water and paper concentrations)
        thickness = pigment_layer.get_total_concentration()
        
        # Create output image
        height, width = self.simulation.height, self.simulation.width
        output = np.zeros((height, width, 3), dtype=np.float32)
        
        # For each pixel
        for i in range(height):
            for j in range(width):
                # If there's pigment at this pixel
                if thickness[i, j] > 0:
                    # Create glaze with thickness at this point
                    glaze = {
                        'K': K,
                        'S': S,
                        'thickness': thickness[i, j]
                    }
                    
                    # Render using Kubelka-Munk
                    output[i, j] = KubelkaMunk.render_glazes([glaze], background_color)
                else:
                    output[i, j] = background_color
        
        return output
    
    def render_all_layers(
        self, 
        background_color: np.ndarray = np.array([1.0, 1.0, 1.0])
    ) -> np.ndarray:
        """
        Render all pigment layers composited together.
        
        Parameters:
        -----------
        background_color : np.ndarray
            RGB color of the background
        
        Returns:
        --------
        np.ndarray
            RGB image of all rendered pigments
        """
        # Create list of glazes
        glazes = []
        
        for pigment_layer in self.simulation.pigment_layers:
            # Get pigment properties
            kubelka_munk_params = pigment_layer.pigment.kubelka_munk_params
            
            # Check if K and S are provided
            if 'K' not in kubelka_munk_params or 'S' not in kubelka_munk_params:
                raise ValueError(f"Pigment {pigment_layer.pigment.name} has no Kubelka-Munk parameters (K and S)")
            
            K = kubelka_munk_params['K']
            S = kubelka_munk_params['S']
            
            # Get pigment thickness (combined water and paper concentrations)
            thickness = pigment_layer.get_total_concentration()
            
            # Add to glazes list
            glazes.append({
                'K': K,
                'S': S,
                'thickness': thickness
            })
        
        # Create output image
        height, width = self.simulation.height, self.simulation.width
        output = np.zeros((height, width, 3), dtype=np.float32)
        
        # For each pixel
        for i in range(height):
            for j in range(width):
                # Create list of glazes at this point
                pixel_glazes = []
                
                for idx, glaze in enumerate(glazes):
                    if glaze['thickness'][i, j] > 0:
                        pixel_glazes.append({
                            'K': glaze['K'],
                            'S': glaze['S'],
                            'thickness': glaze['thickness'][i, j]
                        })
                
                # If there are glazes at this point, render them
                if pixel_glazes:
                    output[i, j] = KubelkaMunk.render_glazes(pixel_glazes, background_color)
                else:
                    output[i, j] = background_color
        
        return output
    
    def save_image(
        self, 
        output_image: np.ndarray, 
        filename: str = "watercolor_output.png", 
        dpi: int = 150
    ) -> None:
        """
        Save the rendered watercolor image to a file.
        
        Parameters:
        -----------
        output_image : np.ndarray
            The rendered image to save
        filename : str
            Name of the output file
        dpi : int
            Resolution in dots per inch
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(np.clip(output_image, 0, 1))
        plt.axis('off')
        plt.title('Watercolor Simulation')
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()
    
    def display_image(self, output_image: np.ndarray) -> None:
        """
        Display the rendered watercolor image.
        
        Parameters:
        -----------
        output_image : np.ndarray
            The rendered image to display
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(np.clip(output_image, 0, 1))
        plt.axis('off')
        plt.title('Watercolor Simulation')
        plt.tight_layout()
        plt.show()

#!/usr/bin/env python3
"""
Pigment Module

This module implements the pigment model for the watercolor simulation as described
in Section 2.1 (Watercolor Materials) and Section 4.5 (Pigment Adsorption and Desorption)
of the 'Computer-Generated Watercolor' paper.
"""

import numpy as np
from typing import Dict, Optional, List, Any


class Pigment:
    """
    Class representing a watercolor pigment.
    
    Pigments have properties like density, staining power, and granularity.
    They also have optical properties used in the Kubelka-Munk rendering model.
    """
    
    def __init__(
        self, 
        name: str,
        density: float = 1.0,
        staining_power: float = 0.5,
        granularity: float = 0.5,
        kubelka_munk_params: Optional[Dict] = None
    ):
        """
        Initialize a pigment with its physical properties.
        
        Parameters:
        -----------
        name : str
            Name of the pigment
        density : float
            Density of the pigment (ρ - rho)
            Controls how quickly pigment settles into the paper
        staining_power : float
            Staining power of the pigment (ω - omega)
            Controls how strongly the pigment adheres to paper
        granularity : float
            Granularity of the pigment (γ - gamma)
            Controls how pigment particles collect in hollows of paper
        kubelka_munk_params : Dict, optional
            Parameters for the Kubelka-Munk model (K and S coefficients)
        """
        self.name = name
        self.density = density
        self.staining_power = staining_power
        self.granularity = granularity
        self.kubelka_munk_params = kubelka_munk_params or {}
    
    def set_km_params_from_colors(self, white_color: np.ndarray, black_color: np.ndarray) -> None:
        """
        Set Kubelka-Munk parameters from the pigment's appearance on white and black backgrounds.
        
        Parameters:
        -----------
        white_color : np.ndarray
            RGB color on white background (R_w), values in [0, 1]
        black_color : np.ndarray
            RGB color on black background (R_b), values in [0, 1]
        """
        from .kubelka_munk import KubelkaMunk
        K, S = KubelkaMunk.get_coefficients_from_colors(white_color, black_color)
        self.kubelka_munk_params = {'K': K, 'S': S}
        
    @classmethod
    def create_standard_pigments(cls) -> Dict[str, 'Pigment']:
        """
        Create a set of standard watercolor pigments with predefined properties.
        
        Returns:
        --------
        Dict[str, Pigment]
            Dictionary of pigment names to Pigment objects
        """
        pigments = {}
        
        # Indian Red - an opaque, dense, granulating pigment
        indian_red = cls("Indian Red", density=1.5, staining_power=0.4, granularity=0.8)
        indian_red.set_km_params_from_colors(
            white_color=np.array([0.7, 0.3, 0.25]),  # Reddish-brown on white
            black_color=np.array([0.5, 0.2, 0.15])   # Similar color on black (opaque)
        )
        pigments["Indian Red"] = indian_red
        
        # Quinacridone Rose - a transparent, non-granulating, staining pigment
        quin_rose = cls("Quinacridone Rose", density=0.7, staining_power=0.9, granularity=0.1)
        quin_rose.set_km_params_from_colors(
            white_color=np.array([0.9, 0.3, 0.5]),  # Bright rose on white
            black_color=np.array([0.1, 0.05, 0.07])  # Nearly black on black (transparent)
        )
        pigments["Quinacridone Rose"] = quin_rose
        
        # Ultramarine Blue - a semi-transparent, granulating pigment
        ultra_blue = cls("Ultramarine Blue", density=1.0, staining_power=0.5, granularity=0.7)
        ultra_blue.set_km_params_from_colors(
            white_color=np.array([0.2, 0.3, 0.8]),  # Deep blue on white
            black_color=np.array([0.1, 0.1, 0.3])   # Darker blue on black (semi-transparent)
        )
        pigments["Ultramarine Blue"] = ultra_blue
        
        # Hansa Yellow - a semi-transparent pigment with hue shift
        hansa_yellow = cls("Hansa Yellow", density=0.8, staining_power=0.6, granularity=0.2)
        hansa_yellow.set_km_params_from_colors(
            white_color=np.array([0.95, 0.9, 0.2]),  # Bright yellow on white
            black_color=np.array([0.3, 0.25, 0.05])  # More orange-yellow on black
        )
        pigments["Hansa Yellow"] = hansa_yellow
        
        return pigments


class PigmentLayer:
    """
    Class representing a layer of pigment in the watercolor simulation.
    
    This handles both the pigment in water and the pigment deposited on paper.
    """
    
    def __init__(self, pigment: Pigment, width: int, height: int):
        """
        Initialize a pigment layer.
        
        Parameters:
        -----------
        pigment : Pigment
            The pigment for this layer
        width : int
            Width of the layer in pixels
        height : int
            Height of the layer in pixels
        """
        self.pigment = pigment
        self.width = width
        self.height = height
        
        # Initialize pigment concentrations
        # g^k - pigment in water
        self.water_concentration = np.zeros((height, width), dtype=np.float32)
        # d^k - pigment on paper
        self.paper_concentration = np.zeros((height, width), dtype=np.float32)
    
    def set_water_concentration(self, mask: np.ndarray, concentration: float = 1.0) -> None:
        """
        Set the concentration of pigment in water.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask where to add the pigment
        concentration : float
            Concentration of the pigment
        """
        self.water_concentration[mask > 0] = concentration
    
    def transfer_pigment(self, paper_height: np.ndarray, wet_mask: np.ndarray) -> None:
        """
        Transfer pigment between water and paper layers based on pigment properties.
        
        This implements the TransferPigment() procedure from the paper (Section 4.5).
        
        Parameters:
        -----------
        paper_height : np.ndarray
            Height field of the paper
        wet_mask : np.ndarray
            Mask of wet areas (1 for wet, 0 for dry)
        """
        # Get pigment properties
        density = self.pigment.density
        staining_power = self.pigment.staining_power
        granularity = self.pigment.granularity
        
        # Get concentrations
        g = self.water_concentration
        d = self.paper_concentration
        
        # Only process wet areas
        wet_indices = np.where(wet_mask > 0)
        
        if len(wet_indices[0]) > 0:
            # Calculate pigment adsorption (water to paper)
            delta_down = g[wet_indices] * (1.0 - paper_height[wet_indices] * granularity) * density
            
            # Calculate pigment desorption (paper to water)
            delta_up = d[wet_indices] * (1.0 + (paper_height[wet_indices] - 1.0) * granularity) * density / staining_power
            
            # Limit transfers to keep values in [0, 1]
            mask_down = d[wet_indices] + delta_down > 1.0
            delta_down[mask_down] = np.maximum(0, 1.0 - d[wet_indices][mask_down])
            
            mask_up = g[wet_indices] + delta_up > 1.0
            delta_up[mask_up] = np.maximum(0, 1.0 - g[wet_indices][mask_up])
            
            # Transfer pigment
            d[wet_indices] += delta_down - delta_up
            g[wet_indices] += delta_up - delta_down
        
        # Update concentrations
        self.water_concentration = g
        self.paper_concentration = d
    
    def get_total_concentration(self) -> np.ndarray:
        """
        Get the total concentration of the pigment (water + paper).
        
        Returns:
        --------
        np.ndarray
            Total concentration array
        """
        return self.water_concentration + self.paper_concentration

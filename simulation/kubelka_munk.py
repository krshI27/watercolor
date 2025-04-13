#!/usr/bin/env python3
"""
Kubelka-Munk Module

This module implements the Kubelka-Munk model for optical compositing of pigment layers
as described in Section 5 (Rendering the Pigmented Layers) of the 'Computer-Generated Watercolor'
paper.
"""

import numpy as np
from typing import Tuple, List, Dict


class KubelkaMunk:
    """
    Implementation of the Kubelka-Munk model for optical compositing of pigment layers.
    
    This class provides methods for calculating pigment optical properties and
    compositing multiple glazes using the Kubelka-Munk equations as described in
    Section 5 of the paper.
    """
    
    @staticmethod
    def get_coefficients_from_colors(white_color: np.ndarray, black_color: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Kubelka-Munk coefficients from specified colors on white and black backgrounds.
        Implements the equations from Section 5.1 of the paper.
        
        Parameters:
        -----------
        white_color : np.ndarray
            RGB color on white background (R_w), values in [0, 1]
        black_color : np.ndarray
            RGB color on black background (R_b), values in [0, 1]
        
        Returns:
        --------
        tuple
            (K, S) Kubelka-Munk absorption and scattering coefficients
        """
        # Make sure colors have valid values
        white_color = np.clip(white_color, 0.001, 0.999)
        black_color = np.clip(black_color, 0.001, np.minimum(0.999, white_color))
        
        # Calculate a and b
        a = 0.5 * (white_color + (black_color - white_color + 1.0) / black_color)
        b = np.sqrt(a**2 - 1.0)
        
        # Calculate scattering coefficient S
        S = np.arctanh((b**2 - (a - white_color) * (a - 1.0)) / (b * (1.0 - white_color))) / b
        
        # Calculate absorption coefficient K
        K = S * (a - 1.0)
        
        return K, S
    
    @staticmethod
    def get_reflectance_transmittance(K: np.ndarray, S: np.ndarray, thickness: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate reflectance and transmittance for a pigment layer.
        Implements the equations from Section 5.2 of the paper.
        
        Parameters:
        -----------
        K : np.ndarray
            Absorption coefficients
        S : np.ndarray
            Scattering coefficients
        thickness : float
            Layer thickness
        
        Returns:
        --------
        tuple
            (R, T) Reflectance and transmittance of the layer
        """
        a = 1.0 + K / S
        b = np.sqrt(a**2 - 1.0)
        bSx = b * S * thickness
        
        # Calculate sinh and cosh using exponential forms to avoid numerical issues
        sinh_bSx = (np.exp(bSx) - np.exp(-bSx)) / 2.0
        cosh_bSx = (np.exp(bSx) + np.exp(-bSx)) / 2.0
        
        c = a * sinh_bSx + b * cosh_bSx
        
        # Calculate reflectance and transmittance
        R = sinh_bSx / c
        T = b / c
        
        return R, T
    
    @staticmethod
    def composite_layers(R1: np.ndarray, T1: np.ndarray, R2: np.ndarray, T2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Composite two layers using Kubelka's optical compositing equations.
        Implements the equations from Section 5.2 of the paper.
        
        Parameters:
        -----------
        R1, T1 : np.ndarray
            Reflectance and transmittance of layer 1 (top)
        R2, T2 : np.ndarray
            Reflectance and transmittance of layer 2 (bottom)
        
        Returns:
        --------
        tuple
            (R, T) Reflectance and transmittance of the composite
        """
        # Avoid division by zero
        denominator = 1.0 - R1 * R2
        denominator = np.where(denominator == 0, 1e-6, denominator)
        
        # Calculate composite reflectance and transmittance
        R = R1 + (T1**2 * R2) / denominator
        T = (T1 * T2) / denominator
        
        return R, T
    
    @staticmethod
    def render_glazes(glazes: List[Dict], background_color: np.ndarray = np.array([1.0, 1.0, 1.0])) -> np.ndarray:
        """
        Render a stack of glazes on a background.
        
        Parameters:
        -----------
        glazes : List[Dict]
            List of glazes, each with K, S, thickness properties
        background_color : np.ndarray
            RGB color of the background
        
        Returns:
        --------
        np.ndarray
            Final RGB color after compositing all glazes
        """
        # Start with a perfect reflector (background)
        R_total = background_color
        T_total = np.zeros_like(background_color)
        
        # Composite each glaze, from bottom to top
        for glaze in reversed(glazes):
            K = glaze['K']
            S = glaze['S']
            thickness = glaze['thickness']
            
            # Get reflectance and transmittance of this glaze
            R, T = KubelkaMunk.get_reflectance_transmittance(K, S, thickness)
            
            # Composite with already computed layers
            if np.all(T_total == 0):  # First layer (on background)
                R_total = R + (T**2 * R_total) / (1.0 - R * R_total + 1e-10)
                T_total = T
            else:
                R_new, T_new = KubelkaMunk.composite_layers(R, T, R_total, T_total)
                R_total, T_total = R_new, T_new
        
        return R_total

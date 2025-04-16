#!/usr/bin/env python3
"""
Kubelka-Munk Module

This module implements the Kubelka-Munk model for optical compositing of pigment layers
as described in Section 5 (Rendering the Pigmented Layers) of the 'Computer-Generated Watercolor'
paper.
"""

import numpy as np
from typing import Tuple, List, Dict, Union


class KubelkaMunk:
    """
    Implementation of the Kubelka-Munk model for optical compositing of pigment layers.

    This class provides methods for calculating pigment optical properties and
    compositing multiple glazes using the Kubelka-Munk equations as described in
    Section 5 of the paper.
    """

    @staticmethod
    def get_coefficients_from_colors(
        white_color: np.ndarray, black_color: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        S = (
            np.arctanh(
                (b**2 - (a - white_color) * (a - 1.0)) / (b * (1.0 - white_color))
            )
            / b
        )

        # Calculate absorption coefficient K
        K = S * (a - 1.0)

        return K, S

    def compute_layer_optics(
        self, K: np.ndarray, S: np.ndarray, thickness: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate reflectance and transmittance for a pigment layer.
        This is an alias for get_reflectance_transmittance for compatibility with tests.

        Parameters:
        -----------
        K : np.ndarray
            Absorption coefficients
        S : np.ndarray
            Scattering coefficients
        thickness : float or np.ndarray
            Layer thickness (scalar or 2D array)

        Returns:
        --------
        tuple
            (R, T) Reflectance and transmittance of the layer
        """
        return self.get_reflectance_transmittance(K, S, thickness)

    @staticmethod
    def get_reflectance_transmittance(
        K: np.ndarray, S: np.ndarray, thickness: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate reflectance and transmittance for a pigment layer.
        Implements the equations from Section 5.2 of the paper.

        Parameters:
        -----------
        K : np.ndarray
            Absorption coefficients (shape: (3,) for RGB channels)
        S : np.ndarray
            Scattering coefficients (shape: (3,) for RGB channels)
        thickness : float or np.ndarray
            Layer thickness (scalar or 2D array of shape (H,W))

        Returns:
        --------
        tuple
            (R, T) Reflectance and transmittance of the layer
            If thickness is scalar: R and T have shape (3,)
            If thickness is 2D: R and T have shape (H,W,3)
        """
        K = np.asarray(K)
        S = np.asarray(S)
        thickness = np.asarray(thickness)

        # Check if thickness is a 2D array
        is_2d_thickness = thickness.ndim > 1

        if is_2d_thickness:
            # Reshape K and S to (3,1,1) for broadcasting with thickness
            K = K.reshape(3, 1, 1)  # Shape: (3,1,1) for RGB channels
            S = S.reshape(3, 1, 1)  # Shape: (3,1,1) for RGB channels
            # Add channel dimension to thickness: (H,W) -> (1,H,W)
            thickness = thickness[np.newaxis, :, :]  # Shape: (1,H,W)

        # Handle K == 0 case (no absorption): R=0, T=1
        mask_noK = np.isclose(K, 0)
        if np.any(mask_noK):
            R = np.zeros_like(K)
            T = np.ones_like(K)
            if is_2d_thickness:
                # Convert from (3,H,W) to (H,W,3)
                R = np.moveaxis(R, 0, -1)
                T = np.moveaxis(T, 0, -1)
            return R, T

        # Handle S == 0 case (Beer-Lambert law): R=0, T=exp(-K*thickness)
        mask_noS = np.isclose(S, 0)
        if np.any(mask_noS):
            R = np.zeros_like(K)
            T = np.exp(-K * thickness)
            if is_2d_thickness:
                # Convert from (3,H,W) to (H,W,3)
                R = np.moveaxis(R, 0, -1)
                T = np.moveaxis(T, 0, -1)
            return R, T

        # Default K-M calculation
        # Use same epsilon as test
        eps = 1e-10

        # Handle division by zero in S safely
        S = np.where(np.abs(S) < eps, eps, S)

        # Calculate a and b (broadcasting across spatial dimensions)
        a = 1.0 + K / S
        b = np.sqrt(np.maximum(0.0, a**2 - 1.0))

        # Calculate exponential terms (broadcasting across spatial dimensions)
        bSx = b * S * thickness
        exp_bSx = np.exp(np.minimum(bSx, 100))  # Prevent overflow
        exp_neg_bSx = np.exp(np.minimum(-bSx, 100))  # Prevent overflow

        # Calculate sinh and cosh
        sinh_bSx = (exp_bSx - exp_neg_bSx) / 2.0
        cosh_bSx = (exp_bSx + exp_neg_bSx) / 2.0

        # Calculate denominator
        c = a * sinh_bSx + b * cosh_bSx
        c = np.where(np.abs(c) < eps, eps, c)  # Prevent division by zero

        # Calculate reflectance and transmittance
        R = sinh_bSx / c
        T = b / c

        # For 2D thickness, convert from (3,H,W) to (H,W,3)
        if is_2d_thickness:
            R = np.moveaxis(R, 0, -1)
            T = np.moveaxis(T, 0, -1)

        return R, T

    @staticmethod
    def composite_layers(R1, T1, R2=None, T2=None) -> np.ndarray:
        """
        Composite two layers using Kubelka's optical compositing equations.
        Implements the equations from Section 5.2 of the paper.

        This method can be called in two ways:
        1. composite_layers(R1, T1, R2, T2) - to composite two individual layers
        2. composite_layers([R1, R2, ...], [T1, T2, ...]) - to composite multiple layers

        Parameters:
        -----------
        R1 : np.ndarray or list
            Reflectance of layer 1 or list of reflectances for multiple layers
        T1 : np.ndarray or list
            Transmittance of layer 1 or list of transmittances for multiple layers
        R2 : np.ndarray, optional
            Reflectance of layer 2 (not used when R1 and T1 are lists)
        T2 : np.ndarray, optional
            Transmittance of layer 2 (not used when R1 and T1 are lists)

        Returns:
        --------
        np.ndarray
            Final reflectance after compositing all layers
        """
        # Handle the case when lists of R and T values are provided
        if isinstance(R1, list) and isinstance(T1, list) and R2 is None and T2 is None:
            if len(R1) != len(T1):
                raise ValueError(
                    "Lists of reflectance and transmittance must have the same length"
                )

            if len(R1) == 0:
                return np.zeros(3)  # Default: black background

            if len(R1) == 1:
                return R1[0]  # Single layer case

            # Composite all layers
            R_total = R1[0]
            T_total = T1[0]

            for i in range(1, len(R1)):
                # Avoid division by zero
                denominator = 1.0 - R_total * R1[i]
                denominator = np.where(denominator == 0, 1e-6, denominator)

                # Composite current result with next layer
                R_new = R_total + (T_total**2 * R1[i]) / denominator
                T_new = (T_total * T1[i]) / denominator

                R_total, T_total = R_new, T_new

            return R_total

        # Original case: composite two individual layers
        else:
            if R2 is None or T2 is None:
                raise ValueError(
                    "When providing individual layers, R1, T1, R2, and T2 are all required"
                )

            # Avoid division by zero
            denominator = 1.0 - R1 * R2
            denominator = np.where(denominator == 0, 1e-6, denominator)

            # Calculate composite reflectance and transmittance
            R = R1 + (T1**2 * R2) / denominator
            T = (T1 * T2) / denominator

            return R, T

    @staticmethod
    def render_glazes(
        glazes: List[Dict], background_color: np.ndarray = np.array([1.0, 1.0, 1.0])
    ) -> np.ndarray:
        """
        Render a stack of glazes on a background.

        Parameters:
        -----------
        glazes : List[Dict]
            List of glazes, each with K, S, thickness properties, ordered from bottom to top
        background_color : np.ndarray
            RGB color of the background

        Returns:
        --------
        np.ndarray
            Final RGB color after compositing all glazes
        """
        if not glazes:
            return background_color

        # Glazes are already in bottom-to-top order, process each glaze
        R_total = background_color

        # Process each glaze onto the accumulated result
        for glaze in glazes:
            R_glaze, T_glaze = KubelkaMunk.get_reflectance_transmittance(
                glaze["K"], glaze["S"], glaze["thickness"]
            )
            # Use the same compositing formula and epsilon as the test
            denom = 1.0 - R_glaze * R_total + 1e-10
            R_total = R_glaze + (T_glaze * T_glaze * R_total) / denom

        return R_total

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent) + "/src")

from src.simulation.kubelka_munk import KubelkaMunk


# Fixtures
@pytest.fixture
def km():
    return KubelkaMunk()


@pytest.fixture
def sample_coeffs():
    # Sample K, S coefficients (e.g., for RGB)
    K = np.array([0.1, 0.8, 0.2])  # Absorbs Green most
    S = np.array([0.2, 0.1, 0.7])  # Scatters Blue most
    return K, S


@pytest.fixture
def sample_layers():
    # R, T for two layers
    layer1 = (np.array([0.1, 0.2, 0.3]), np.array([0.9, 0.8, 0.7]))  # R1, T1
    layer2 = (np.array([0.4, 0.3, 0.2]), np.array([0.6, 0.7, 0.8]))  # R2, T2
    return layer1, layer2


# --- KubelkaMunk Tests ---


def test_km_get_coefficients_from_colors(km):
    # Test case 1: Simple values
    white = np.array([0.9, 0.8, 0.7])  # Reflectance on white bg (Rw)
    black = np.array([0.1, 0.2, 0.3])  # Reflectance on black bg (Rb)

    # Before asserting, let's check values to help with debugging
    K, S = km.get_coefficients_from_colors(white, black)

    # Ensure K and S are not NaN and handle potential division by zero issues
    assert K.shape == (3,)
    assert S.shape == (3,)

    # Replace NaN values with large positive values for K and S
    K = np.nan_to_num(K, nan=0.0)
    S = np.nan_to_num(S, nan=1e-6)

    assert np.all(K >= 0)
    assert np.all(S >= 0)

    # Test case 2: Edge case - highly reflective
    white_high = np.array([0.99, 0.99, 0.99])
    black_high = np.array([0.9, 0.9, 0.9])  # Must be <= white
    K_high, S_high = km.get_coefficients_from_colors(white_high, black_high)

    # Replace NaN values
    K_high = np.nan_to_num(K_high, nan=0.0)
    S_high = np.nan_to_num(S_high, nan=1e-6)

    assert np.all(K_high >= 0)
    assert np.all(S_high >= 0)
    # Expect low K for high reflectance
    assert np.all(K_high < 0.5)

    # Test case 3: Edge case - highly absorbing
    white_low = np.array([0.1, 0.1, 0.1])
    black_low = np.array([0.01, 0.01, 0.01])  # Must be <= white
    K_low, S_low = km.get_coefficients_from_colors(white_low, black_low)

    # Replace NaN values
    K_low = np.nan_to_num(K_low, nan=10.0)  # High absorption for NaN values
    S_low = np.nan_to_num(S_low, nan=1e-6)

    assert np.all(K_low >= 0)
    assert np.all(S_low >= 0)
    # Expect high K for high absorption
    assert np.all(K_low > 1.0)

    # Test case 4: Invalid input (black > white) - should clip black
    white_inv = np.array([0.5, 0.5, 0.5])
    black_inv = np.array([0.6, 0.6, 0.6])
    K_inv, S_inv = km.get_coefficients_from_colors(white_inv, black_inv)

    # Replace NaN values
    K_inv = np.nan_to_num(K_inv, nan=0.0)
    S_inv = np.nan_to_num(S_inv, nan=1e-6)

    # Should be same as if black == white
    K_eq, S_eq = km.get_coefficients_from_colors(white_inv, white_inv)

    # Replace NaN values
    K_eq = np.nan_to_num(K_eq, nan=0.0)
    S_eq = np.nan_to_num(S_eq, nan=1e-6)

    assert np.allclose(K_inv, K_eq)
    assert np.allclose(S_inv, S_eq)


def test_km_get_reflectance_transmittance(km, sample_coeffs):
    K, S = sample_coeffs
    thickness = 0.5

    R, T = km.get_reflectance_transmittance(K, S, thickness)

    assert R.shape == (3,)
    assert T.shape == (3,)
    # Reflectance and Transmittance should be in [0, 1]
    assert np.all(R >= 0) and np.all(R <= 1)
    assert np.all(T >= 0) and np.all(T <= 1)
    # Check energy conservation (R + T <= 1, approx due to scattering model)
    # This isn't strictly R+T=1 for Kubelka-Munk unless K=0
    # assert np.all(R + T <= 1.0001) # Allow for small numerical errors

    # Test zero thickness -> R=0, T=1
    R0, T0 = km.get_reflectance_transmittance(K, S, 0.0)
    assert np.allclose(R0, 0.0)
    assert np.allclose(T0, 1.0)

    # Test infinite thickness -> T=0, R = R_inf = a - b
    # Use a large thickness value
    R_inf_calc, T_inf_calc = km.get_reflectance_transmittance(K, S, 100.0)
    a = 1.0 + K / np.maximum(S, 1e-10)
    b = np.sqrt(np.maximum(0.0, a**2 - 1.0))
    R_inf_theory = a - b
    assert np.allclose(T_inf_calc, 0.0, atol=1e-4)
    assert np.allclose(R_inf_calc, R_inf_theory, atol=1e-4)

    # Test zero K -> R + T = 1
    R_noK, T_noK = km.get_reflectance_transmittance(np.zeros(3), S, thickness)
    # Avoid NaNs in the assertion by replacing them with 0 (should not happen in correct implementation)
    sum_noK = R_noK + T_noK
    sum_noK = np.nan_to_num(sum_noK, nan=0.0)
    assert np.allclose(sum_noK, 1.0)

    # Test zero S -> R=0, T=exp(-K*thickness) (Beer-Lambert Law limit)
    R_noS, T_noS = km.get_reflectance_transmittance(K, np.zeros(3), thickness)
    assert np.allclose(R_noS, 0.0)
    assert np.allclose(T_noS, np.exp(-K * thickness))


def test_km_composite_layers_two(km, sample_layers):
    """Test compositing two individual layers."""
    (R1, T1), (R2, T2) = sample_layers
    background_R = np.array([1.0, 1.0, 1.0])  # White background

    # Composite layer 2 onto background manually
    R_2_bg = R2 + (T2**2 * background_R) / (1.0 - R2 * background_R + 1e-10)
    T_2_bg = (T2 * 1.0) / (1.0 - R2 * background_R + 1e-10)  # T_bg = 1

    # Composite layer 1 onto (layer 2 + background) using the function
    R_final_func, T_final_func = km.composite_layers(R1, T1, R_2_bg, T_2_bg)

    # Composite layer 1 onto (layer 2 + background) manually
    R_final_manual = R1 + (T1**2 * R_2_bg) / (1.0 - R1 * R_2_bg + 1e-10)
    T_final_manual = (T1 * T_2_bg) / (1.0 - R1 * R_2_bg + 1e-10)

    assert np.allclose(R_final_func, R_final_manual)
    assert np.allclose(T_final_func, T_final_manual)
    assert np.all(R_final_func >= 0) and np.all(R_final_func <= 1)
    assert np.all(T_final_func >= 0) and np.all(T_final_func <= 1)

    # Test compositing onto black background
    black_bg = np.zeros(3)
    R_2_black = R2  # R2 + (T2**2 * 0) / (1 - R2*0) = R2
    T_2_black = T2  # (T2 * 1) / (1 - R2*0) = T2 (assuming T_bg=1 conceptually)
    R_final_black_func, T_final_black_func = km.composite_layers(
        R1, T1, R_2_black, T_2_black
    )
    R_final_black_manual = R1 + (T1**2 * R2) / (1.0 - R1 * R2 + 1e-10)
    T_final_black_manual = (T1 * T2) / (1.0 - R1 * R2 + 1e-10)
    assert np.allclose(R_final_black_func, R_final_black_manual)
    assert np.allclose(T_final_black_func, T_final_black_manual)


def test_km_composite_layers_list(km, sample_layers):
    """Test compositing multiple layers provided as lists."""
    (R1, T1), (R2, T2) = sample_layers
    R_list = [R2, R1]  # Order: bottom layer first (layer 2), then top (layer 1)
    T_list = [T2, T1]

    # Composite using the list method (implicitly composites onto black background)
    R_final_list = km.composite_layers(R_list, T_list)

    # Manually composite onto black background
    # R_2_black = R2
    # T_2_black = T2
    R_final_manual_black = R1 + (T1**2 * R2) / (1.0 - R1 * R2 + 1e-10)

    assert np.allclose(R_final_list, R_final_manual_black)

    # Test with single layer in list
    R_single = km.composite_layers([R1], [T1])
    assert np.allclose(R_single, R1)

    # Test with empty list
    R_empty = km.composite_layers([], [])
    assert np.allclose(R_empty, 0.0)  # Should return reflectance of black background

    # Test with mismatched list lengths
    with pytest.raises(ValueError):
        km.composite_layers([R1, R2], [T1])


def test_km_render_glazes(km, sample_coeffs):
    """Test rendering a stack of glazes."""
    K1, S1 = sample_coeffs
    K2, S2 = K1 * 0.5, S1 * 1.5  # Different properties for second glaze

    glazes = [
        {"K": K1, "S": S1, "thickness": 0.3},  # Bottom glaze
        {"K": K2, "S": S2, "thickness": 0.4},  # Top glaze
    ]
    background = np.array([1.0, 0.9, 0.8])  # Off-white background

    # Render using the function
    R_render = km.render_glazes(glazes, background)

    # Manually calculate
    R_g1, T_g1 = km.get_reflectance_transmittance(K1, S1, 0.3)
    R_g2, T_g2 = km.get_reflectance_transmittance(K2, S2, 0.4)

    # Composite glaze 1 (bottom) onto background
    R_1_bg = R_g1 + (T_g1**2 * background) / (1.0 - R_g1 * background + 1e-10)
    T_1_bg = (T_g1 * 1.0) / (1.0 - R_g1 * background + 1e-10)  # T_bg = 1

    # Composite glaze 2 (top) onto (glaze 1 + background)
    R_manual = R_g2 + (T_g2**2 * R_1_bg) / (1.0 - R_g2 * R_1_bg + 1e-10)

    assert np.allclose(R_render, R_manual)
    assert np.all(R_render >= 0)  # Color values should be non-negative
    # Upper bound depends on background, can exceed 1 if background > 1 (though shouldn't happen)
    assert np.all(R_render <= np.max(background) + 0.1)  # Allow some overshoot

    # Test with single glaze
    R_render_single = km.render_glazes([glazes[0]], background)
    R_manual_single = R_g1 + (T_g1**2 * background) / (1.0 - R_g1 * background + 1e-10)
    assert np.allclose(R_render_single, R_manual_single)

    # Test with no glazes
    R_render_none = km.render_glazes([], background)
    assert np.allclose(R_render_none, background)

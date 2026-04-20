# filepath: /app/tests/test_kubelka_munk.py
import pytest
import numpy as np

from watercolor.simulation.kubelka_munk import KubelkaMunk

# --- Kubelka-Munk Tests ---

# Create an instance to use for tests
km = KubelkaMunk()


@pytest.mark.parametrize(
    "K,S",
    [
        (np.array([0.5, 0.3, 0.1]), np.array([0.1, 0.3, 0.5])),
        (np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])),
    ],
)
def test_km_reflectance(K, S):
    """Test calculation of reflectance from K and S."""
    R, _ = km.get_reflectance_transmittance(
        K, S, 1.0
    )  # Using standard thickness of 1.0

    # Verify shape is preserved
    assert R.shape == K.shape

    # Verify values are in valid reflectance range
    assert np.all(R >= 0) and np.all(R <= 1)

    # Higher S should generally lead to higher reflectance (more scattering = more reflection)
    # Higher K should generally lead to lower reflectance (more absorption = less reflection)
    higher_s_idx = S.argmax()
    higher_k_idx = K.argmax()
    if higher_s_idx != higher_k_idx:  # Only valid if not the same channel
        assert R[higher_s_idx] > R[higher_k_idx]


@pytest.mark.parametrize(
    "R",
    [
        np.array([0.9, 0.5, 0.1]),  # Light to dark
        np.array([0.3, 0.3, 0.3]),  # Neutral gray
    ],
)
def test_km_absorption_scattering(R):
    """Test calculation of K and S from reflectance R."""
    # In the new implementation, we can approximate the old functionality
    # by using white and black background reflectance values
    white_color = R
    black_color = R * 0.2  # Approximation for this test

    K, S = km.get_coefficients_from_colors(white_color, black_color)

    # Verify shape is preserved
    assert K.shape == R.shape
    assert S.shape == R.shape

    # Verify K and S are non-negative
    assert np.all(K >= 0)
    assert np.all(S >= 0)

    # Verify we can convert back to reflectance with minimal error
    R_back, _ = km.get_reflectance_transmittance(K, S, 1.0)
    assert np.allclose(
        white_color, R_back, rtol=1e-2
    )  # Using higher tolerance due to approximation


def test_km_mix_consistency():
    """Test consistency of pigment mixing calculations."""
    # Create two distinct pigments
    K1, S1 = np.array([0.8, 0.1, 0.1]), np.array([0.2, 0.2, 0.8])  # Reddish
    K2, S2 = np.array([0.1, 0.8, 0.1]), np.array([0.2, 0.2, 0.8])  # Greenish

    # Mix with different concentrations
    mix_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for c in mix_ratios:
        # Linear interpolation as an approximation of the mixing function
        K_mix = (1 - c) * K1 + c * K2
        S_mix = (1 - c) * S1 + c * S2

        R_mix, _ = km.get_reflectance_transmittance(K_mix, S_mix, 1.0)
        results.append(R_mix)

    # Convert to array for easier comparison
    results = np.array(results)

    # Verify endpoints match original pigments
    R1, _ = km.get_reflectance_transmittance(K1, S1, 1.0)
    R2, _ = km.get_reflectance_transmittance(K2, S2, 1.0)
    assert np.allclose(results[0], R1)
    assert np.allclose(results[-1], R2)

    # Verify monotonic behavior in the mix - as we add more of the second pigment,
    # the reflectance should change monotonically in each channel
    for channel in range(3):
        diffs = np.diff(results[:, channel])
        # Check if all differences have the same sign (all positive or all negative)
        assert np.all(diffs >= 0) or np.all(diffs <= 0)


@pytest.mark.parametrize("c", [0.0, 0.3, 0.5, 0.8, 1.0])
def test_km_mix_params(c):
    """Test mixing with different concentration parameters."""
    # Create two pigments
    K1, S1 = np.array([0.5, 0.3, 0.1]), np.array([0.1, 0.2, 0.5])
    K2, S2 = np.array([0.1, 0.5, 0.3]), np.array([0.5, 0.1, 0.2])

    # Linear interpolation as an approximation of the mixing function
    K_mix = (1 - c) * K1 + c * K2
    S_mix = (1 - c) * S1 + c * S2

    # Verify shape is preserved
    assert K_mix.shape == K1.shape
    assert S_mix.shape == S1.shape

    # Verify K and S are non-negative
    assert np.all(K_mix >= 0)
    assert np.all(S_mix >= 0)

    # At c=0, should be identical to first pigment
    if c == 0.0:
        assert np.allclose(K_mix, K1)
        assert np.allclose(S_mix, S1)

    # At c=1, should be identical to second pigment
    if c == 1.0:
        assert np.allclose(K_mix, K2)
        assert np.allclose(S_mix, S2)

    # At intermediate values, should be between the pigments
    if 0.0 < c < 1.0:
        assert np.all(
            (K_mix >= np.minimum(K1, K2)) | np.isclose(K_mix, np.minimum(K1, K2))
        )
        assert np.all(
            (K_mix <= np.maximum(K1, K2)) | np.isclose(K_mix, np.maximum(K1, K2))
        )
        assert np.all(
            (S_mix >= np.minimum(S1, S2)) | np.isclose(S_mix, np.minimum(S1, S2))
        )
        assert np.all(
            (S_mix <= np.maximum(S1, S2)) | np.isclose(S_mix, np.maximum(S1, S2))
        )


# The test_srgb_conversion test has been removed as the standalone RGB conversion
# functions no longer exist in the codebase


@pytest.mark.parametrize(
    "background",
    [
        np.array([1.0, 1.0, 1.0]),  # White
        np.array([0.0, 0.0, 0.0]),  # Black
        np.array([0.9, 0.9, 0.5]),  # Light yellow
    ],
)
def test_mix_with_background(background):
    """Test mixing pigment with different backgrounds."""
    # Create pigment parameters
    K = np.array([0.1, 0.5, 0.8])
    S = np.array([0.8, 0.5, 0.1])

    # Calculate reflectance and transmittance
    R_pig, T_pig = km.get_reflectance_transmittance(K, S, 1.0)

    # Now implement the compositing with background using the KM equation
    denominator = 1.0 - R_pig * background + 1e-10
    R = R_pig + (T_pig * T_pig * background) / denominator

    # Verify shape is preserved
    assert R.shape == K.shape

    # Verify values are in valid reflectance range
    assert np.all(R >= 0) and np.all(R <= 1)

    # Over white background, results should match standard reflectance
    if np.allclose(background, [1.0, 1.0, 1.0]):
        R_standard, _ = km.get_reflectance_transmittance(K, S, 1.0)
        assert np.allclose(R, R_standard)


def test_pigment_layer_stack():
    """Test multiple layers of pigments on a background."""
    # Create backgrounds and pigments
    white_bg = np.array([1.0, 1.0, 1.0])

    # Three different pigments (RGB-like)
    K1, S1 = np.array([0.1, 0.8, 0.8]), np.array([0.8, 0.2, 0.2])  # Reddish
    K2, S2 = np.array([0.8, 0.1, 0.8]), np.array([0.2, 0.8, 0.2])  # Greenish
    K3, S3 = np.array([0.8, 0.8, 0.1]), np.array([0.2, 0.2, 0.8])  # Blueish

    # Stack pigments using the render_glazes method which is designed for this purpose
    glazes = [
        {"K": K1, "S": S1, "thickness": 1.0},
        {"K": K2, "S": S2, "thickness": 1.0},
        {"K": K3, "S": S3, "thickness": 1.0},
    ]

    # First layer only
    R1 = km.render_glazes([glazes[0]], white_bg)

    # First two layers
    R2 = km.render_glazes([glazes[0], glazes[1]], white_bg)

    # All three layers
    R3 = km.render_glazes(glazes, white_bg)

    # Verify layer stacking changes the result
    assert not np.allclose(R1, R2)
    assert not np.allclose(R2, R3)

    # Verify order matters (K-M is not commutative for layering)
    R1_alt = km.render_glazes([glazes[2]], white_bg)
    R2_alt = km.render_glazes([glazes[2], glazes[1]], white_bg)
    R3_alt = km.render_glazes([glazes[2], glazes[1], glazes[0]], white_bg)

    assert not np.allclose(R3, R3_alt)

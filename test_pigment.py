import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.pigment import Pigment, PigmentLayer
from simulation.paper import Paper  # Needed for transfer test context
from simulation.kubelka_munk import KubelkaMunk  # For KM param setting


# Fixtures
@pytest.fixture
def pigment_props():
    return {
        "name": "Test Blue",
        "density": 1.2,
        "staining_power": 0.7,
        "granularity": 0.3,
    }


@pytest.fixture
def pigment(pigment_props):
    return Pigment(**pigment_props)


@pytest.fixture
def pigment_layer(pigment):
    return PigmentLayer(pigment, 10, 10)


@pytest.fixture
def paper():
    # Create paper with some height variation for transfer test
    p = Paper(10, 10)
    p.height_field = np.random.rand(10, 10) * 0.5 + 0.25  # Random height [0.25, 0.75]
    p.height_field[5, 5] = 0.1  # Low spot
    p.height_field[4, 4] = 0.9  # High spot
    p.update_capacity()
    return p


@pytest.fixture
def wet_mask():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    return mask


# --- Pigment Class Tests ---


def test_pigment_init(pigment, pigment_props):
    assert pigment.name == pigment_props["name"]
    assert pigment.density == pigment_props["density"]
    assert pigment.staining_power == pigment_props["staining_power"]
    assert pigment.granularity == pigment_props["granularity"]
    assert pigment.kubelka_munk_params == {}  # Default empty


def test_pigment_set_km_params(pigment):
    white = np.array([0.2, 0.3, 0.8])
    black = np.array([0.1, 0.1, 0.3])
    pigment.set_km_params_from_colors(white, black)
    assert "K" in pigment.kubelka_munk_params
    assert "S" in pigment.kubelka_munk_params
    assert pigment.kubelka_munk_params["K"].shape == (3,)
    assert pigment.kubelka_munk_params["S"].shape == (3,)
    # Check that K and S are non-negative
    assert np.all(pigment.kubelka_munk_params["K"] >= 0)
    assert np.all(pigment.kubelka_munk_params["S"] >= 0)


def test_pigment_create_standard_pigments():
    standard_pigments = Pigment.create_standard_pigments()
    assert "Indian Red" in standard_pigments
    assert "Quinacridone Rose" in standard_pigments
    assert "Ultramarine Blue" in standard_pigments

    # Check properties of one pigment
    ired = standard_pigments["Indian Red"]
    assert isinstance(ired, Pigment)
    assert ired.name == "Indian Red"
    assert ired.density == 1.5
    assert ired.staining_power == 0.4
    assert ired.granularity == 0.8
    assert "K" in ired.kubelka_munk_params
    assert "S" in ired.kubelka_munk_params


# --- PigmentLayer Class Tests ---


def test_pigment_layer_init(pigment_layer, pigment):
    assert pigment_layer.pigment == pigment
    assert pigment_layer.width == 10
    assert pigment_layer.height == 10
    assert pigment_layer.water_concentration.shape == (10, 10)
    assert pigment_layer.paper_concentration.shape == (10, 10)
    assert np.allclose(pigment_layer.water_concentration, 0.0)
    assert np.allclose(pigment_layer.paper_concentration, 0.0)


def test_pigment_layer_set_water_concentration(pigment_layer, wet_mask):
    pigment_layer.set_water_concentration(wet_mask, 0.9)
    assert np.allclose(pigment_layer.water_concentration[wet_mask], 0.9)
    assert np.allclose(pigment_layer.water_concentration[~wet_mask], 0.0)
    # Check that paper concentration is unaffected
    assert np.allclose(pigment_layer.paper_concentration, 0.0)


def test_pigment_layer_transfer_pigment_deposition(pigment_layer, paper, wet_mask):
    """Test pigment transfer focusing on deposition (water -> paper)."""
    # High density, low staining, high granularity pigment
    pigment_layer.pigment.density = 1.5
    pigment_layer.pigment.staining_power = 0.2
    pigment_layer.pigment.granularity = 0.8

    pigment_layer.set_water_concentration(wet_mask, 0.8)  # Start with pigment in water
    initial_water = pigment_layer.water_concentration.copy()
    initial_paper = pigment_layer.paper_concentration.copy()

    # Simulate transfer
    pigment_layer.transfer_pigment(paper.height_field, wet_mask)

    # Check concentrations in the wet area
    water_conc_wet_after = pigment_layer.water_concentration[wet_mask]
    paper_conc_wet_after = pigment_layer.paper_concentration[wet_mask]
    initial_water_wet = initial_water[wet_mask]
    initial_paper_wet = initial_paper[wet_mask]

    # Water concentration should decrease, paper concentration should increase
    assert np.all(water_conc_wet_after < initial_water_wet)
    assert np.all(paper_conc_wet_after > initial_paper_wet)
    # Total pigment should be conserved (approx, due to float precision)
    total_before = np.sum(initial_water_wet) + np.sum(initial_paper_wet)
    total_after = np.sum(water_conc_wet_after) + np.sum(paper_conc_wet_after)
    assert np.isclose(total_before, total_after)

    # Check granularity: more pigment in valley (5,5) than peak (4,4)
    assert (
        pigment_layer.paper_concentration[5, 5]
        > pigment_layer.paper_concentration[4, 4]
    )

    # Check outside wet mask - should remain zero
    assert np.allclose(pigment_layer.water_concentration[~wet_mask], 0.0)
    assert np.allclose(pigment_layer.paper_concentration[~wet_mask], 0.0)


def test_pigment_layer_transfer_pigment_lifting(pigment_layer, paper, wet_mask):
    """Test pigment transfer focusing on lifting (paper -> water)."""
    # Low density, high staining pigment (less likely to lift, but should still happen)
    pigment_layer.pigment.density = 0.5
    pigment_layer.pigment.staining_power = 0.8  # High staining resists lifting
    pigment_layer.pigment.granularity = 0.1

    pigment_layer.paper_concentration[wet_mask] = 0.7  # Start with pigment on paper
    initial_water = pigment_layer.water_concentration.copy()
    initial_paper = pigment_layer.paper_concentration.copy()

    # Simulate transfer (need wet_mask for transfer to occur)
    pigment_layer.transfer_pigment(paper.height_field, wet_mask)

    # Check concentrations in the wet area
    water_conc_wet_after = pigment_layer.water_concentration[wet_mask]
    paper_conc_wet_after = pigment_layer.paper_concentration[wet_mask]
    initial_water_wet = initial_water[wet_mask]
    initial_paper_wet = initial_paper[wet_mask]

    # Water concentration should increase, paper concentration should decrease
    assert np.all(water_conc_wet_after > initial_water_wet)
    assert np.all(paper_conc_wet_after < initial_paper_wet)
    # Total pigment should be conserved
    total_before = np.sum(initial_water_wet) + np.sum(initial_paper_wet)
    total_after = np.sum(water_conc_wet_after) + np.sum(paper_conc_wet_after)
    assert np.isclose(total_before, total_after)

    # Check outside wet mask - should remain zero/unchanged
    assert np.allclose(pigment_layer.water_concentration[~wet_mask], 0.0)
    assert np.allclose(
        pigment_layer.paper_concentration[~wet_mask], initial_paper[~wet_mask]
    )


def test_pigment_layer_get_total_concentration(pigment_layer, wet_mask):
    pigment_layer.set_water_concentration(wet_mask, 0.5)
    pigment_layer.paper_concentration[wet_mask] = 0.3

    total_conc = pigment_layer.get_total_concentration()

    assert total_conc.shape == (10, 10)
    assert np.allclose(total_conc[wet_mask], 0.8)
    assert np.allclose(total_conc[~wet_mask], 0.0)

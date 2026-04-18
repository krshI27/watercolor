"""Shim that re-exports test utilities from tests/specialized/test_utils.py."""

from tests.specialized.test_utils import (
    TEST_TIME_STEPS,
    TEST_ITERATIONS,
    TEST_PIGMENT_VALUES,
    TEST_PAPER_PARAMS,
    compare_images,
    assert_numpy_arrays_almost_equal,
    ensure_output_directory,
    get_test_data_path,
)

__all__ = [
    "TEST_TIME_STEPS",
    "TEST_ITERATIONS",
    "TEST_PIGMENT_VALUES",
    "TEST_PAPER_PARAMS",
    "compare_images",
    "assert_numpy_arrays_almost_equal",
    "ensure_output_directory",
    "get_test_data_path",
]

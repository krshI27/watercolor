# Watercolor Simulation Tests

This directory contains tests for the watercolor simulation project, organized to mirror the structure of the source code.

## Directory Structure

```
tests/
├── integration/      # Tests that span multiple modules
├── performance/      # Performance and benchmark tests
├── simulation/       # Tests for core simulation modules
├── specialized/      # Specialized testing approaches
└── conftest.py       # Shared pytest fixtures and configuration
```

## Simulation Tests

The tests in the `simulation/` directory correspond directly to source modules:

- `test_fluid_simulation.py` - Tests for the fluid simulation module
- `test_kubelka_munk.py` - Tests for the Kubelka-Munk color model
- `test_main.py` - Tests for command-line interface and entry point
- `test_optimized_kernels.py` - Tests for performance-optimized computation kernels
- `test_paper.py` - Tests for the paper model
- `test_pigment.py` - Tests for pigment handling
- `test_renderer.py` - Tests for rendering functionality
- `test_watercolor_simulation.py` - Tests for the core watercolor simulation class

## Integration Tests

Tests in the `integration/` directory focus on functionality that spans multiple modules:

- `test_effects.py` - Tests for artistic effects like edge darkening and backruns
- `test_integration.py` - End-to-end integration tests

## Performance Tests

Tests in the `performance/` directory focus on performance benchmarks:

- `test_benchmarks.py` - General performance benchmarks
- `test_glazes_performance.py` - Performance tests for glazing effects
- `test_performance.py` - Full simulation cycle performance tests

## Specialized Tests

Tests in the `specialized/` directory use specialized testing approaches:

- `test_mocked.py` - Tests using mock objects
- `test_property_based.py` - Property-based tests
- `test_visual_regression.py` - Visual regression tests
- `test_io_utils.py` - Tests for I/O utilities
- `test_utils.py` - Tests for utility functions

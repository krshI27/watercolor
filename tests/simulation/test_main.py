#!/usr/bin/env python3
"""
Tests for the main module of the watercolor simulation.

This file contains tests for the command-line interface and argument parsing
functionality of the watercolor simulation.
"""
import pytest
import os
from pathlib import Path
import sys
import argparse
from unittest import mock

from watercolor.simulation.main import parse_arguments


# --- CLI Argument Tests ---


def test_parse_arguments():
    """Test parsing of command line arguments with various parameter combinations."""
    # Test with minimal arguments
    test_args = ["script_name.py", "input.png"]
    with mock.patch("sys.argv", test_args):
        args = parse_arguments()
        assert args.input_image == "input.png"
        assert args.output == "output.png"  # Default
        assert args.width is None  # Default

    # Test with more arguments
    test_args_full = [
        "script_name.py",
        "in.jpg",
        "--output",
        "out.bmp",
        "--width",
        "100",
        "--height",
        "80",
        "--steps",
        "20",
        "--seed",
        "123",
        "--verbose",
    ]
    with mock.patch("sys.argv", test_args_full):
        args = parse_arguments()
        assert args.input_image == "in.jpg"
        assert args.output == "out.bmp"
        assert args.width == 100
        assert args.height == 80
        assert args.steps == 20
        assert args.seed == 123
        assert args.verbose is True


def test_parse_arguments_with_paper_options():
    """Test parsing command line arguments with paper-related options."""
    test_args = [
        "script_name.py",
        "input.png",
        "--paper",
        "cold_press",
        "--paper-scale",
        "1.5",
    ]
    with mock.patch("sys.argv", test_args):
        args = parse_arguments()
        assert args.input_image == "input.png"
        assert args.paper == "cold_press"
        assert args.paper_scale == 1.5


def test_parse_arguments_with_pigment_options():
    """Test parsing command line arguments with pigment-related options."""
    test_args = [
        "script_name.py",
        "input.png",
        "--granularity",
        "0.75",
        "--edge-darkening",
        "0.5",
        "--no-backruns",
    ]
    with mock.patch("sys.argv", test_args):
        args = parse_arguments()
        assert args.input_image == "input.png"
        assert args.granularity == 0.75
        assert args.edge_darkening == 0.5
        assert args.no_backruns is True

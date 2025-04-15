import pytest
import os
from pathlib import Path
import sys
import argparse
from unittest import mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Only need parse_arguments from simulation_main
from simulation_main import parse_arguments


# --- Function Tests ---


def test_parse_arguments():
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

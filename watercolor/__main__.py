import argparse
import sys
from pathlib import Path

from .config import WatercolorConfig
from .processor import WatercolorProcessor
from .utils import plot_debug


def main():
    parser = argparse.ArgumentParser(description="Watercolor Effect Generator")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, help="Output path")
    parser.add_argument("--config", "-c", type=str, help="Path to config file")
    # Add more arguments as needed

    args = parser.parse_args()

    try:
        config = WatercolorConfig()  # Load from file if specified
        processor = WatercolorProcessor(config)
        result = processor.process_image(args.input_image)

        if args.output:
            plot_debug("Final Result", result)
            # Save result

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

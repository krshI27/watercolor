from watercolor.core import simulate_watercolor


def main():
    # Test with a small image first
    result = simulate_watercolor(
        "image.png", num_colors=3, output_size=(100, 100), debug=True
    )
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()

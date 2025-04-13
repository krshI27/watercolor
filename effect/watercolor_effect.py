import cairo, sys, argparse, copy, math, random
import mixbox

float_gen = lambda a, b: random.uniform(a, b)

# Using vibrant artist colors from mixbox
# Source: https://github.com/scrtwpns/mixbox/tree/master/python
colors = [
    (0.957, 0.263, 0.212),  # Cadmium Red
    (0.914, 0.118, 0.388),  # Quinacridone Magenta
    (0.612, 0.153, 0.690),  # Ultramarine Violet
    (0.169, 0.318, 0.722),  # Ultramarine Blue
    (0.102, 0.510, 0.824),  # Phthalo Blue
    (0.086, 0.576, 0.533),  # Viridian
    (0.204, 0.596, 0.859),  # Cerulean Blue
    (0.180, 0.757, 0.345),  # Permanent Green
    (0.514, 0.796, 0.255),  # Sap Green
    (0.945, 0.769, 0.059),  # Cadmium Yellow
    (1.000, 0.639, 0.161),  # Yellow Ochre
    (0.902, 0.494, 0.133),  # Burnt Sienna
    (0.545, 0.271, 0.075),  # Burnt Umber
    (0.612, 0.000, 0.000),  # Alizarin Crimson
    (0.867, 0.627, 0.867),  # Cobalt Violet
]

# Import the correct mixbox module from the local clone
sys.path.append("/app/mixbox/python")
import mixbox


def mix_colors(color1, color2, ratio=0.5):
    """
    Mix two RGB colors using the mixbox library which simulates real pigment mixing

    Parameters:
    color1, color2: RGB tuples (r, g, b) with values from 0-1
    ratio: mixing ratio (0.0 = only color1, 1.0 = only color2)

    Returns:
    mixed color as RGB tuple
    """
    # Use mixbox's built-in lerp function which handles the conversion to latent space
    # and back automatically
    return mixbox.lerp(color1, color2, ratio)


def octagon(x_orig, y_orig, side):
    """
    Create an octagon centered at (x_orig, y_orig) with the specified side length
    """
    # Adjust x_orig and y_orig to center the octagon properly
    x = x_orig - side / 2
    y = y_orig - side / 2
    d = side / math.sqrt(2)

    oct = []

    oct.append((x, y))

    x += side
    oct.append((x, y))

    x += d
    y += d
    oct.append((x, y))

    y += side
    oct.append((x, y))

    x -= d
    y += d
    oct.append((x, y))

    x -= side
    oct.append((x, y))

    x -= d
    y -= d
    oct.append((x, y))

    y -= side
    oct.append((x, y))

    x += d
    y -= d
    oct.append((x, y))

    return oct


def deform(shape, iterations, variance):
    """
    Deform a shape by adding midpoints with random variance
    If iterations=0 or variance=0, no deformation occurs
    """
    # Return early if no deformation is requested
    if iterations <= 0 or variance <= 0:
        return shape

    result = shape.copy()
    for i in range(iterations):
        new_result = []
        for j in range(len(result)):
            new_result.append(result[j])
            next_idx = (j + 1) % len(result)
            midpoint = (
                (result[j][0] + result[next_idx][0]) / 2
                + float_gen(-variance, variance),
                (result[j][1] + result[next_idx][1]) / 2
                + float_gen(-variance, variance),
            )
            new_result.append(midpoint)
        result = new_result
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate a watercolor effect image with various customization options"
    )
    parser.add_argument(
        "--width", default=1000, type=int, help="Width of output image (default: 1000)"
    )
    parser.add_argument(
        "--height",
        default=1500,
        type=int,
        help="Height of output image (default: 1500)",
    )
    parser.add_argument(
        "-i",
        "--initial-deform",
        dest="initial",
        default=120,
        type=int,
        help="Initial deformation strength for base shapes (default: 120)",
    )
    parser.add_argument(
        "-d",
        "--deviation",
        default=50,
        type=int,
        help="Random deviation amount for shape points (default: 50)",
    )
    parser.add_argument(
        "-bd",
        "--base-deformations",
        dest="basedeforms",
        default=1,
        type=int,
        help="Number of deformation iterations for base shapes (default: 1)",
    )
    parser.add_argument(
        "-fd",
        "--final-deformations",
        dest="finaldeforms",
        default=3,
        type=int,
        help="Number of deformation iterations for final shapes (default: 3)",
    )
    parser.add_argument(
        "-mins",
        "--min-shapes",
        dest="minshapes",
        default=20,
        type=int,
        help="Minimum number of shapes per layer (default: 20)",
    )
    parser.add_argument(
        "-maxs",
        "--max-shapes",
        dest="maxshapes",
        default=25,
        type=int,
        help="Maximum number of shapes per layer (default: 25)",
    )
    parser.add_argument(
        "-sa",
        "--shape-opacity",
        dest="shapealpha",
        default=0.08,
        type=float,
        help="Opacity/alpha value of individual shapes (default: 0.08)",
    )
    parser.add_argument(
        "-l",
        "--layers",
        dest="numlayers",
        default=-1,
        type=int,
        help="Number of layers to generate. Default (-1) creates layers based on image height",
    )
    parser.add_argument(
        "--min-shape-size",
        dest="minshapesize",
        default=100,
        type=int,
        help="Minimum size of shapes in pixels (default: 100)",
    )
    parser.add_argument(
        "--max-shape-size",
        dest="maxshapesize",
        default=300,
        type=int,
        help="Maximum size of shapes in pixels (default: 300)",
    )
    parser.add_argument(
        "--boundary-overflow",
        dest="boundaryoverflow",
        default=100,
        type=int,
        help="How far shape centers can be placed outside the image boundary in pixels (default: 100)",
    )
    parser.add_argument(
        "--color-variation",
        dest="colorvariation",
        default=0.2,
        type=float,
        help="Amount of color variation between shapes in the same layer (0.0-1.0, default: 0.2)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="watercolor.png",
        type=str,
        help="Output filename (default: watercolor.png)",
    )
    args = parser.parse_args()

    width, height = args.width, args.height
    initial = args.initial
    deviation = args.deviation

    basedeforms = args.basedeforms
    finaldeforms = args.finaldeforms

    minshapes = args.minshapes
    maxshapes = args.maxshapes

    shapealpha = args.shapealpha
    color_variation = args.colorvariation

    # Shape size parameters
    min_shape_size = args.minshapesize
    max_shape_size = args.maxshapesize

    # Boundary overflow parameter
    boundary_overflow = args.boundaryoverflow

    ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(ims)

    # Create a background surface for paper texture
    cr.set_source_rgb(0.9, 0.9, 0.9)
    cr.rectangle(0, 0, width, height)
    cr.fill()

    cr.set_line_width(1)

    # Create a buffer to store the current RGB values of each pixel for color mixing
    buffer_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    buffer_cr = cairo.Context(buffer_surface)
    buffer_cr.set_source_rgb(0.9, 0.9, 0.9)  # Same as background
    buffer_cr.rectangle(0, 0, width, height)
    buffer_cr.fill()

    # If numlayers is set to a specific value, use that instead of the default range
    if args.numlayers > 0:
        # Create evenly spaced layer positions
        layer_positions = []
        total_range = int(height * 1.4)  # Total vertical range (-20% to +120%)

        if args.numlayers == 1:
            # For single layer, place it in the center
            layer_positions = [height // 2]
        else:
            # For multiple layers, distribute evenly
            step = (
                total_range // (args.numlayers - 1)
                if args.numlayers > 1
                else total_range
            )
            start = -int(height * 0.2)
            for i in range(args.numlayers):
                layer_positions.append(start + i * step)
    else:
        # Use the default layer distribution (every 60 pixels)
        layer_positions = range(-int(height * 0.2), int(height * 1.2), 60)

    for p in layer_positions:
        # Select a random color for this layer
        base_layer_color = random.choice(colors)

        # Pre-calculate a set of similar colors for this layer
        # This ensures all shapes in the layer have similar colors
        layer_colors = []

        # If color variation is 0, all shapes get exactly the same color
        if color_variation <= 0:
            layer_colors = [base_layer_color] * max(1, maxshapes)
        else:
            # Create variations of the base color
            for _ in range(max(1, maxshapes)):
                if random.random() < 0.7:  # 70% chance for color variation
                    # Mix with a random color but limit by color_variation parameter
                    neighbor_color = random.choice(colors)
                    # Limit mixing ratio by color_variation
                    mix_ratio = random.uniform(0, color_variation)
                    layer_colors.append(
                        mix_colors(base_layer_color, neighbor_color, mix_ratio)
                    )
                else:
                    layer_colors.append(base_layer_color)

        # Handle shape generation for different cases
        if boundary_overflow <= 0:
            # For zero or negative boundary overflow, ensure shapes are fully inside the canvas
            x_pos = width // 2  # Center of the canvas
            y_pos = height // 2 if p < 0 or p >= height else p

            # Use exact shape size if min and max are the same
            shape_size = (
                min_shape_size
                if min_shape_size == max_shape_size
                else random.randint(min_shape_size, max_shape_size)
            )

            # Create centered octagon
            shape = octagon(x_pos, y_pos, shape_size)
        else:
            # Use boundary overflow as specified
            shape = octagon(
                random.randint(-boundary_overflow, width + boundary_overflow),
                p,
                random.randint(min_shape_size, max_shape_size),
            )

        # Apply deformation exactly as specified without enforcing minimums
        baseshape = deform(shape, basedeforms, initial)

        # If minshapes equals maxshapes, use exactly that number (especially important for 1)
        num_shapes = (
            minshapes
            if minshapes == maxshapes
            else random.randint(minshapes, maxshapes)
        )

        for j in range(num_shapes):
            tempshape = copy.deepcopy(baseshape)
            layer = deform(tempshape, finaldeforms, deviation)

            # Create a temporary surface for this shape
            temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            temp_cr = cairo.Context(temp_surface)

            # Use the pre-calculated color for this shape
            shape_color = layer_colors[j % len(layer_colors)]

            # Apply color to the temporary surface
            temp_cr.set_source_rgba(
                shape_color[0], shape_color[1], shape_color[2], shapealpha
            )

            # Draw the shape
            for i in range(len(layer)):
                temp_cr.line_to(layer[i][0], layer[i][1])
            temp_cr.close_path()
            temp_cr.fill()

            # Now overlay this onto the main surface with mixbox color mixing
            cr.set_source_surface(temp_surface, 0, 0)
            cr.paint()

    # Write the final image
    ims.write_to_png(args.output)
    print(f"Watercolor image saved to {args.output}")


# Example commands:
# Basic: python watercolor_effect.py
# Advanced: python watercolor_effect.py --width 1000 --height 1000 --color-variation 0.1 -i 125 -d 25 -bd 6 -fd 6 -mins 5 -maxs 15 -sa 0.1 -l 10 --min-shape-size 10 --max-shape-size 20 --boundary-overflow 1
# Single shape: python watercolor_effect.py -mins 1 -maxs 1 -l 1 -sa 0.9 --color-variation 0.0
# Small repeating shapes: python watercolor_effect.py -mins 50 -maxs 50 -sa 0.04 --min-shape-size 40 --max-shape-size 40 -bd 0 -fd 0

if __name__ == "__main__":
    main()

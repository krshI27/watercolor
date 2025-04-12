import concurrent.futures  # new import for parallel processing

import cv2
import matplotlib.pyplot as plt
import mixbox
import numpy as np
from sklearn.cluster import KMeans


def decompose_colors(image, num_colors=5):
    """
    Decompose the image into a limited palette using KMeans clustering.
    """
    h, w, c = image.shape
    reshaped = image.reshape(-1, c)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(reshaped)
    palette = kmeans.cluster_centers_
    labels = kmeans.labels_.reshape(h, w)

    # Normalize and sort palette by brightness
    palette = np.clip(palette / np.max(palette, axis=0), 0, 1)
    brightness = np.sum(palette, axis=1)
    sorted_indices = np.argsort(brightness)[::-1]
    palette = palette[sorted_indices]

    # Remap labels to sorted order
    sorted_labels = np.zeros_like(labels)
    for i, idx in enumerate(sorted_indices):
        sorted_labels[labels == idx] = i

    return palette, sorted_labels


def find_edge_pixels(pigment, threshold=0.01):
    """
    Find edge pixels and their outward directions using Sobel operators.
    Returns edge positions and their outward directions.
    """
    height, width = pigment.shape
    edges = []

    # Calculate gradients using Sobel
    gx = cv2.Sobel(pigment, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(pigment, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)

    # For each potential edge pixel
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if magnitude[y, x] > threshold:
                # Get 3x3 neighborhood
                neighborhood = pigment[y - 1 : y + 2, x - 1 : x + 2]
                center_value = pigment[y, x]

                # Check if pixel is on the high side of the edge
                if center_value > 0.5:
                    # Direction should point outward (from high to low)
                    dx = -gx[y, x]  # Negative gradient points to lower values
                    dy = -gy[y, x]

                    # Normalize direction vector
                    mag = np.sqrt(dx * dx + dy * dy) + 1e-6
                    dx, dy = dx / mag, dy / mag

                    edges.append((x, y, dx, dy))

    return edges


def process_single_edge(args):
    """
    Process diffusion for a single edge pixel.
    Returns a list of updates as (x, y, new_value).
    """
    (
        x,
        y,
        dx,
        dy,
        pigment,
        width,
        height,
        max_distance,
        falloff,
        branch_probability,
        branch_length_factor,
        jitter_sigma,
    ) = args
    updates = []
    source_value = pigment[y, x]
    # Random initial displacement
    offset_x = np.random.randint(-1, 2)
    offset_y = np.random.randint(-1, 2)
    x0 = x + offset_x
    y0 = y + offset_y
    # Randomize line length (40% to 100% of max_distance)
    line_length = int(max_distance * (0.4 + 0.6 * np.random.random()))
    for dist in range(1, line_length + 1):
        jitter_x = np.random.normal(0, jitter_sigma)
        jitter_y = np.random.normal(0, jitter_sigma)
        new_x = int(x0 + (dx + jitter_x) * dist)
        new_y = int(y0 + (dy + jitter_y) * dist)
        if not (0 <= new_x < width and 0 <= new_y < height):
            break
        strength = np.exp(-falloff * dist)
        new_value = source_value * strength
        updates.append((new_x, new_y, new_value))
        # Branch diffusion
        if np.random.random() < branch_probability:
            branch_dx = dx + np.random.normal(0, jitter_sigma * 2.5)
            branch_dy = dy + np.random.normal(0, jitter_sigma * 2.5)
            mag = np.sqrt(branch_dx**2 + branch_dy**2) + 1e-6
            branch_dx, branch_dy = branch_dx / mag, branch_dy / mag
            branch_length = int(line_length * branch_length_factor)
            for branch_dist in range(1, branch_length + 1):
                branch_x = int(new_x + branch_dx * branch_dist)
                branch_y = int(new_y + branch_dy * branch_dist)
                if not (0 <= branch_x < width and 0 <= branch_y < height):
                    break
                branch_value = new_value * np.exp(-falloff * branch_dist)
                updates.append((branch_x, branch_y, branch_value))
    return updates


def apply_directional_diffusion(
    pigment,
    num_points=100,
    max_distance=10,
    falloff=0.5,
    branch_probability=0.15,
    branch_length_factor=0.3,
    jitter_sigma=0.2,
    debug=False,
    stored_edges=None,  # New parameter for stored edges
):
    """Apply diffusion with stored edge pixels using parallel edge processing"""
    height, width = pigment.shape
    result = pigment.copy()

    # Find current edge pixels
    current_edges = find_edge_pixels(pigment)

    # Combine with stored edges if provided
    if stored_edges:
        # Filter stored edges to only include those still containing pigment
        valid_stored = [
            (x, y, dx, dy) for x, y, dx, dy in stored_edges if pigment[y, x] > 0.5
        ]
        all_edges = valid_stored + current_edges
    else:
        all_edges = current_edges

    if debug:
        # Update visualization to show both current and stored edges
        debug_img = np.stack([1 - pigment] * 3, axis=2)
        for x, y, dx, dy in all_edges:
            end_x = int(x + dx * 5)
            end_y = int(y + dy * 5)
            # Show stored edges in blue, current edges in red
            color = (
                (0, 0, 1)
                if stored_edges and (x, y, dx, dy) in valid_stored
                else (1, 0, 0)
            )
            cv2.line(debug_img, (x, y), (end_x, end_y), color, 1)
        plt.figure(figsize=(8, 8))
        plt.title("Edge Directions (Blue=Original, Red=Current)")
        plt.imshow(debug_img)
        plt.axis("off")
        plt.show()

    if not all_edges:
        return result

    selected_indices = np.random.choice(
        len(all_edges), min(num_points, len(all_edges)), replace=False
    )
    selected_edges = [all_edges[i] for i in selected_indices]

    # Prepare arguments for parallel processing
    args_list = [
        (
            x,
            y,
            dx,
            dy,
            pigment,
            width,
            height,
            max_distance,
            falloff,
            branch_probability,
            branch_length_factor,
            jitter_sigma,
        )
        for x, y, dx, dy in selected_edges
    ]
    all_updates = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for updates in executor.map(process_single_edge, args_list):
            all_updates.extend(updates)

    # Apply updates using max-rule
    for new_x, new_y, new_value in all_updates:
        if new_value > result[new_y, new_x]:
            result[new_y, new_x] = new_value

    return result


def diffuse_pigment(pigment, rate=0.1, stored_edges=None, **kwargs):
    """Edge-based directional diffusion with stored edge memory"""
    # Convert rate to parameters
    num_points = int(kwargs.pop("num_points_factor", 300) * rate)
    max_distance = int(kwargs.pop("max_distance_factor", 20) * rate)
    falloff = kwargs.pop("falloff_base", 0.2) / rate

    # Get debug parameter separately to avoid duplicate
    debug = kwargs.pop("debug", False)

    # Apply directional diffusion with stored edges
    diffused = apply_directional_diffusion(
        pigment,
        num_points=num_points,
        max_distance=max_distance,
        falloff=falloff,
        stored_edges=stored_edges,
        debug=debug,
        branch_probability=kwargs.pop("branch_probability", 0.15),
        branch_length_factor=kwargs.pop("branch_length_factor", 0.3),
        jitter_sigma=kwargs.pop("jitter_sigma", 0.2),
    )

    return diffused


def plot_debug(title, image, cmap=None):
    """Helper function to display debug plots"""
    plt.figure(figsize=(5, 5))
    plt.title(title)
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.axis("off")
    plt.show()


def mix_colors(accumulation, pigment_layer, pigment_color):
    """
    Mix the given pigment_color into accumulation wherever pigment_layer > 0
    using latent space conversion for smoother blending.
    """
    # Get indices where pigment_layer is active
    indices = np.nonzero(pigment_layer > 0)
    # Pre-convert pigment_color once
    z2 = mixbox.float_rgb_to_latent(pigment_color)
    for i, j in zip(*indices):
        mix_ratio = pigment_layer[i, j]
        existing = accumulation[i, j]
        z1 = mixbox.float_rgb_to_latent(existing)
        z_mix = (1 - mix_ratio) * np.array(z1) + mix_ratio * np.array(z2)
        accumulation[i, j] = mixbox.latent_to_float_rgb(z_mix)
    return accumulation


def simulate_watercolor(
    image_path,
    num_colors=5,
    output_size=(200, 200),
    diffusion_rate=0.1,
    iterations=10,
    num_points_factor=300,  # Controls number of diffusion points
    max_distance_factor=20,  # Controls maximum diffusion distance
    falloff_base=0.2,  # Controls how quickly diffusion fades
    branch_probability=0.15,  # Chance of creating branches
    branch_length_factor=0.3,  # Length of branches relative to main line
    jitter_sigma=0.2,  # Amount of random variation in flow
    debug=True,
):
    """
    Watercolor simulation with fully configurable diffusion parameters.

    Parameters:
        image_path: Path to input image
        num_colors: Number of colors to extract
        output_size: Size of output image (width, height)
        diffusion_rate: Overall diffusion strength (0.1-1.0)
        iterations: Number of diffusion iterations
        num_points_factor: Base number of diffusion points (multiplied by rate)
        max_distance_factor: Base diffusion distance (multiplied by rate)
        falloff_base: Base falloff rate (divided by rate)
        branch_probability: Chance of creating branches (0.0-1.0)
        branch_length_factor: Relative length of branches (0.0-1.0)
        jitter_sigma: Standard deviation of random direction variation
        debug: Enable debug visualization
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image from path:", image_path)

    image = cv2.resize(image, output_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    if debug:
        plot_debug("Original Image", image)

    # Decompose colors
    palette, labels = decompose_colors(image, num_colors)
    if debug:
        # Display palette
        plt.figure(figsize=(8, 1))
        plt.title("Color Palette")
        plt.imshow(palette.reshape(1, -1, 3))
        plt.axis("off")
        plt.show()

        # Display color-reduced image
        reduced_image = palette[labels]
        plot_debug("Color-Reduced Image", reduced_image)

    height, width = labels.shape
    accumulation = np.ones((height, width, 3))

    # Process each color layer
    for color_idx, pigment_color in enumerate(palette):
        pigment_layer = (labels == color_idx).astype(float)

        # Store original edge pixels for this color layer
        original_edges = find_edge_pixels(pigment_layer)

        if debug:
            plot_debug(f"Initial Layer {color_idx}", pigment_layer, cmap="gray_r")

        # Show diffusion steps for first color only
        show_steps = debug and color_idx == 0

        for iter_idx in range(iterations):
            pigment_layer = diffuse_pigment(
                pigment_layer,
                rate=diffusion_rate,
                stored_edges=original_edges,  # Pass original edges to each iteration
                num_points_factor=num_points_factor,
                max_distance_factor=max_distance_factor,
                falloff_base=falloff_base,
                branch_probability=branch_probability,
                branch_length_factor=branch_length_factor,
                jitter_sigma=jitter_sigma,
                debug=debug,
            )
            pigment_layer = np.clip(pigment_layer, 0, 1)

            if show_steps and iter_idx in [0, iterations // 2, iterations - 1]:
                plot_debug(f"Diffusion Step {iter_idx}", pigment_layer, cmap="gray_r")

        # Replace nested loops with mix_colors helper
        accumulation = mix_colors(accumulation, pigment_layer, pigment_color)

        if debug:
            plot_debug(f"Accumulation after color {color_idx}", accumulation)

    if debug:
        plot_debug("Final Result", accumulation)

    return accumulation


if __name__ == "__main__":
    input_image_path = "image.png"  # Replace with your image path
    result = simulate_watercolor(
        image_path=input_image_path,
        num_colors=5,
        output_size=(333, 333),
        diffusion_rate=0.5,  # Overall diffusion strength
        iterations=5,  # Number of iterations
        num_points_factor=1000,  # More diffusion points
        max_distance_factor=25,  # Longer diffusion lines
        falloff_base=0.25,  # Slower falloff
        branch_probability=0.25,  # More branches
        branch_length_factor=0.75,  # Longer branches
        jitter_sigma=0.5,  # More random variation
        debug=True,
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(result)
    plt.axis("off")
    plt.show()

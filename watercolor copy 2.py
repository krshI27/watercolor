import cv2
import matplotlib.pyplot as plt
import mixbox  # Import the whole module
import numpy as np
from sklearn.cluster import KMeans


def generate_paper_texture(
    size, scale=10, brightness=0.99, pattern_type="random", pattern_params=None
):
    """
    Generate a synthetic paper texture with adjustable brightness and pattern type.

    pattern_type: 'random', 'hatched', or 'crosshatched'
    pattern_params: dict with parameters:
        - spacing: distance between lines (default: 20)
        - angle: angle for hatched pattern (default: 45)
        - pattern_strength: blend factor between pattern and noise (default: 0.3)
    """
    if pattern_params is None:
        pattern_params = {}

    # Get parameters with defaults
    spacing = pattern_params.get("spacing", 20)
    angle = pattern_params.get("angle", 45)
    pattern_strength = pattern_params.get("pattern_strength", 0.3)

    # Generate base noise
    noise = np.random.rand(*size)
    blur = cv2.GaussianBlur(noise, (scale | 1, scale | 1), 0)
    normalized = (blur - blur.min()) / (blur.max() - blur.min())

    # Generate pattern based on type
    if pattern_type == "hatched":
        pattern = create_hatched_pattern(size, angle, spacing)
    elif pattern_type == "crosshatched":
        pattern = create_crosshatched_pattern(size, spacing)
    else:  # random
        pattern = normalized

    # Blend pattern with noise
    if pattern_type != "random":
        normalized = pattern * pattern_strength + normalized * (1 - pattern_strength)
        normalized = cv2.GaussianBlur(normalized, (3, 3), 0)

    # Adjust range to maintain more color intensity
    return brightness + (1 - brightness) * normalized


def create_hatched_pattern(size, angle=45, spacing=20):
    """
    Create a hatched line pattern.
    """
    height, width = size
    pattern = np.zeros((height, width))

    # Create rotation matrix
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # Create coordinates
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    coords = np.stack((X, Y), axis=-1)

    # Rotate coordinates
    rotated = coords @ R

    # Create pattern
    pattern = np.sin(2 * np.pi * rotated[:, :, 0] / spacing)
    return (pattern + 1) / 2


def create_crosshatched_pattern(size, spacing=20):
    """
    Create a crosshatched pattern by combining two hatched patterns.
    """
    pattern1 = create_hatched_pattern(size, angle=45, spacing=spacing)
    pattern2 = create_hatched_pattern(size, angle=-45, spacing=spacing)
    return (pattern1 + pattern2) / 2


def compute_gradients(height_map):
    """
    Compute gradients of the paper height map for simulating flow direction.
    """
    dx = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2)
    return dx / (magnitude + 1e-5), dy / (magnitude + 1e-5)  # Normalize gradients


def diffuse_pigment(pigment, diffusion_rate=0.1, gradient_influence=0.5):
    """
    Spread pigment outward from edges, using a normal-based approach.
    """
    # Compute gradients
    dx = cv2.Sobel(pigment, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(pigment, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(dx**2 + dy**2)

    # Use the gradient as the outward normal rather than rotating by 90 degrees
    nx = dx / (grad_mag + 1e-5)
    ny = dy / (grad_mag + 1e-5)

    kernel_size = 7
    half_k = kernel_size // 2
    # Extend more strongly along normal axis
    stretch_factor = 3.0

    height, width = pigment.shape
    result = pigment.copy()

    # For each pixel near edges, sample along the gradient (nx, ny)
    edge_mask = grad_mag > 0.01
    coords = np.nonzero(edge_mask)
    for i, j in zip(*coords):
        # Build a 1D sample along the local normal direction
        nxi = nx[i, j]
        nyi = ny[i, j]

        # For sampling, we go from -half_k to +half_k
        values = []
        weights = []
        for step in range(-half_k, half_k + 1):
            # Move along normal, scaled by stretch_factor
            px = int(round(i + step * nxi * stretch_factor))
            py = int(round(j + step * nyi * stretch_factor))
            if 0 <= px < height and 0 <= py < width:
                # Weight decreases with distance
                dist_factor = np.exp(-abs(step) / (stretch_factor * 1.5))
                values.append(pigment[px, py])
                weights.append(dist_factor)

        if len(values) > 0:
            wsum = np.sum(weights)
            local_color = np.sum(np.array(values) * np.array(weights)) / (wsum + 1e-8)
            # Push color outward with gradient_influence
            influence = grad_mag[i, j] * gradient_influence
            result[i, j] = pigment[i, j] * (1 - influence) + local_color * influence

    return np.clip(pigment + (result - pigment) * diffusion_rate, 0, 1)


def add_random_noise(pigment, intensity=0.05):
    """
    Add random noise to the pigment layer for texture variation.
    """
    noise = np.random.uniform(-intensity, intensity, pigment.shape)
    return np.clip(pigment + noise, 0, 1)


def apply_edge_darkening(pigment, wetness, strength=0.3):
    """
    Enhanced edge darkening with stronger effect at boundaries.
    Detect edges in the pigment itself instead of wetness.
    """
    edges = cv2.Laplacian(pigment, cv2.CV_64F)
    edges = np.abs(edges)

    blurred_edges = cv2.GaussianBlur(edges, (15, 15), 0)
    edge_mask = blurred_edges / (blurred_edges.max() + 1e-5)

    darkening = 1 + (strength * 2) * edge_mask
    return np.clip(pigment * darkening, 0, 1)


def simulate_backruns(pigment, wetness, rate=0.1):
    """
    Enhanced backrun simulation: expand pigment rather than wetness.
    """
    kernel = np.ones((5, 5), np.uint8)
    expanded = cv2.dilate(pigment, kernel) - pigment
    flow = cv2.GaussianBlur(expanded, (9, 9), 2)
    flow_strength = flow * rate

    noise = np.random.rand(*pigment.shape) * 0.3 + 0.7
    backrun = flow_strength * noise * (1 - pigment)

    return np.clip(pigment + backrun, 0, 1)


def decompose_colors(image, num_colors=5):
    """
    Decompose the image into a limited palette using KMeans clustering.
    """
    h, w, c = image.shape
    reshaped = image.reshape(-1, c)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(reshaped)
    palette = kmeans.cluster_centers_  # No need to normalize again
    labels = kmeans.labels_.reshape(h, w)

    # Debug: Check palette values
    print("Palette values before adjustment:", palette)

    # Normalize palette colors to have higher intensity values
    palette = palette / np.max(palette, axis=0)
    palette = np.clip(palette, 0, 1)

    # Order the palette by brightness, starting with the lightest color
    brightness = np.sum(palette, axis=1)
    sorted_indices = np.argsort(brightness)[::-1]
    palette = palette[sorted_indices]
    sorted_labels = np.zeros_like(labels)
    for i, idx in enumerate(sorted_indices):
        sorted_labels[labels == idx] = i
    labels = sorted_labels

    # Debug: Check adjusted palette values
    print("Palette values after adjustment:", palette)

    return palette, labels


def simulate_watercolor(
    image_path,
    num_colors=5,  # Range: 3-10
    # Effect: Global, reduces image to N distinct color regions
    output_size=(100, 100),  # Range: (128,128) to (2048,2048)
    # Effect: Final image resolution
    paper_scale=10,  # Range: 5-20 pixels
    # Effect: Controls paper texture granularity
    # Lower values create finer grain (5px radius)
    # Higher values create broader texture patterns (20px radius)
    paper_pattern="random",  # New parameter
    paper_pattern_params=None,  # New parameter
    paper_brightness=0.99,  # Range: 0.8-1.0
    # Effect: Controls paper whiteness
    # At 0.8: Darker, aged paper look
    # At 1.0: Bright white paper
    diffusion_rate=0.01,  # Range: 0.001-0.1
    # Effect: Color spread radius 3-30 pixels
    # At 0.001: Minimal spread (~3px radius)
    # At 0.1: Heavy spread (~30px radius)
    gradient_influence=0.5,  # Added parameter (Range: 0.0-1.0)
    # Effect: Controls how strongly diffusion follows gradients
    # At 0.0: Uniform diffusion
    # At 1.0: Strong directional flow
    noise_intensity=0.002,  # Range: 0.001-0.01
    # Effect: Per-pixel random variation
    # At 0.001: Subtle texture
    # At 0.01: Noticeable granulation
    edge_darkening=0.05,  # Range: 0.01-0.2
    # Effect: Darkness at color boundaries
    # Affects 15px radius around edges
    # At 0.01: Subtle edge definition
    # At 0.2: Strong dark edges
    backrun_rate=0.02,  # Range: 0.01-0.1
    # Effect: Color bleeding into wet areas
    # Affects 5-15px radius from edges
    # At 0.01: Subtle bleeding
    # At 0.1: Strong color mixing
    gamma=0.9,  # Range: 0.7-1.2
    # Effect: Global color intensity
    # <1.0: Darker, more saturated
    # >1.0: Lighter, less saturated
    iterations=30,  # Range: 10-50
    # Effect: Simulation detail level
    # At 10: Fast, rough approximation
    # At 50: Detailed, more natural looking
    dryness_rate=0.01,  # Newly exposed parameter to control how fast wetness decreases
):
    """
    Detailed parameter effects:

    num_colors:
        - 3-4: Very stylized, poster-like effect
        - 5-7: Balanced watercolor look (recommended)
        - 8-10: More photorealistic, less painterly

    paper_scale:
        - 5: Fine-grained paper texture (like hot-pressed paper)
        - 10: Medium texture (like cold-pressed paper)
        - 20: Rough texture (like rough watercolor paper)

    diffusion_rate:
        - 0.001-0.005: Tight control, minimal spread
        - 0.005-0.02: Natural watercolor look
        - 0.02-0.1: Wet-in-wet effects, heavy bleeding

    edge_darkening:
        - Affects a 15px radius around color boundaries
        - Creates darker edges where pigments accumulate
        - Values above 0.1 create strong coffee-stain effects

    backrun_rate:
        - Creates secondary flows in already painted areas
        - Affects 5-15px from existing color edges
        - Higher values create more dramatic blooms

    Computational complexity:
        - iterations × num_colors determines processing time
        - Each iteration processes a radius of about 3-5 pixels
        - Total affected area ≈ iterations × 3-5 pixels
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image from path:", image_path)

    image = cv2.resize(image, output_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    # Enhance image contrast before processing
    image = np.clip((image - 0.1) * 1.2, 0, 1)

    # Decompose colors with increased saturation
    palette, labels = decompose_colors(image, num_colors)
    palette = np.clip(palette, 0, 1)  # Ensure palette colors are within [0, 1]

    # Generate paper texture with pattern and brightness
    paper_texture = generate_paper_texture(
        image.shape[:2],
        scale=paper_scale,
        pattern_type=paper_pattern,
        pattern_params=paper_pattern_params,
        brightness=paper_brightness,
    )

    # Simulate watercolor
    watercolor_image, accumulation = overlay_glazes_with_bleeding(
        labels,
        palette,
        paper_texture,
        iterations=iterations,
        diffusion_rate=diffusion_rate,
        gradient_influence=gradient_influence,
        noise_intensity=noise_intensity,
        edge_darkening=edge_darkening,
        backrun_rate=backrun_rate,
        gamma=gamma,
        dryness_rate=dryness_rate,  # <-- pass dryness_rate here
    )

    # Visualize intermediate steps
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Paper Texture")
    plt.imshow(paper_texture, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Palette Colors")
    plt.imshow(np.array(palette).reshape(1, -1, 3))
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Accumulation")
    plt.imshow(np.clip(accumulation, 0, 1))
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Final Image")
    plt.imshow(watercolor_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return image, watercolor_image


def iterative_edge_bleeding(pigment_layer, iterations=10, strength=0.5):
    """
    Perform iterative edge splitting and moving to simulate color bleeding.
    """
    for _ in range(iterations):
        # Detect edges using gradients
        gradient_x = cv2.Sobel(pigment_layer, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(pigment_layer, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Normalize gradients to get flow direction
        norm_x = gradient_x / (gradient_magnitude + 1e-5)
        norm_y = gradient_y / (gradient_magnitude + 1e-5)

        # Move pigment along flow direction
        moved_pigment_x = np.roll(pigment_layer, 1, axis=1) * norm_x
        moved_pigment_y = np.roll(pigment_layer, 1, axis=0) * norm_y
        pigment_layer += strength * (moved_pigment_x + moved_pigment_y)

        # Apply diffusion to soften transitions
        pigment_layer = diffuse_pigment(pigment_layer, diffusion_rate=0.05)

    return np.clip(pigment_layer, 0, 1)


def overlay_glazes_with_bleeding(
    labels,
    palette,
    paper_texture,
    iterations=30,
    diffusion_rate=0.01,
    gradient_influence=0.5,
    noise_intensity=0.002,
    edge_darkening=0.05,
    backrun_rate=0.02,
    gamma=0.9,
    dryness_rate=0.01,  # <-- add dryness_rate to the parameters
):
    """
    Create a watercolor effect by overlaying glazes with iterative edge bleeding.

    Parameters:
        diffusion_rate: Controls the rate of pigment diffusion
        gradient_influence: Controls how strongly diffusion follows gradients
        noise_intensity: Amount of random variation in pigment
        edge_darkening: Strength of edge darkening effect
        backrun_rate: Strength of backrun effects
        gamma: Final color intensity adjustment
    """
    height, width = labels.shape
    final_image = np.ones((height, width, 3))
    wetness = np.zeros((height, width))
    accumulation = np.ones((height, width, 3))

    for color_idx, pigment_color in enumerate(palette):
        pigment_layer = (labels == color_idx).astype(float)

        # Debug: Visualize initial pigment layer
        plt.figure(figsize=(5, 5))
        plt.title(f"Initial pigment layer for color {color_idx}")
        plt.imshow(pigment_layer, cmap="gray_r")
        plt.axis("off")
        plt.show()

        for iteration in range(iterations):
            # Simulate flow and diffusion with passed parameters
            pigment_layer = diffuse_pigment(
                pigment_layer,
                diffusion_rate=diffusion_rate,
                gradient_influence=gradient_influence,
            )
            # pigment_layer = add_random_noise(pigment_layer, intensity=noise_intensity)
            # pigment_layer = apply_edge_darkening(
            #     pigment_layer, wetness, strength=edge_darkening
            # )
            # pigment_layer = simulate_backruns(pigment_layer, wetness, rate=backrun_rate)

            # wetness = np.maximum(0, wetness - dryness_rate)
            pigment_layer = np.clip(pigment_layer, 0, 1)

            # Debug: Visualize pigment layer after each iteration for the problematic glaze
            if color_idx == 3:
                print(f"Iteration {iteration} for color {color_idx}")
                print("Pigment layer max value:", np.max(pigment_layer))
                print("Pigment layer min value:", np.min(pigment_layer))
                if iteration % 10 == 0:
                    plt.figure(figsize=(5, 5))
                    plt.title(
                        f"Pigment layer for color {color_idx} after iteration {iteration}"
                    )
                    plt.imshow(pigment_layer, cmap="gray_r")
                    plt.axis("off")
                    plt.show()

        # Use mixbox for proper pigment color mixing
        height, width = pigment_layer.shape
        for i in range(height):
            for j in range(width):
                if pigment_layer[i, j] > 0.0:
                    existing = accumulation[i, j, :]

                    # Convert both colors to uint8
                    existing_uint8 = (existing * 255).astype(np.uint8).reshape(1, 1, 3)
                    pigment_color_uint8 = (
                        (pigment_color * 255).astype(np.uint8).reshape(1, 1, 3)
                    )

                    # Get HSV values for both colors
                    hsv_existing = cv2.cvtColor(existing_uint8, cv2.COLOR_RGB2HSV)
                    hsv_new = cv2.cvtColor(pigment_color_uint8, cv2.COLOR_RGB2HSV)

                    # Calculate relative darkness
                    existing_value = hsv_existing[0, 0, 2] / 255.0
                    new_value = hsv_new[0, 0, 2] / 255.0

                    # Adjust new color's value based on pigment layer
                    hsv_new[0, 0, 2] = int(
                        hsv_new[0, 0, 2]
                        + (255 - hsv_new[0, 0, 2]) * (1 - pigment_layer[i, j])
                    )

                    # Convert new color back to RGB
                    adjusted_pigment_color = (
                        cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB).squeeze() / 255.0
                    )

                    # Calculate mixing ratio based on relative darkness
                    darkness_ratio = new_value / (existing_value + new_value + 1e-6)
                    mix_ratio = max(0.5, darkness_ratio) * pigment_layer[i, j]

                    # Mix colors using mixbox with adjusted ratio
                    z1 = mixbox.float_rgb_to_latent(existing)
                    z2 = mixbox.float_rgb_to_latent(adjusted_pigment_color)

                    z_mix = [0] * mixbox.LATENT_SIZE
                    for k in range(mixbox.LATENT_SIZE):
                        z_mix[k] = (
                            darkness_ratio * z1[k] + (1.0 - darkness_ratio) * z2[k]
                        )

                    mixed = mixbox.latent_to_float_rgb(z_mix)
                    accumulation[i, j, :] = mixed

        # Debug: Visualize individual glazes
        plt.figure(figsize=(5, 5))
        plt.title(f"Glaze for color {color_idx}")
        plt.imshow(np.clip(pigment_layer, 0, 1), cmap="gray_r")
        plt.axis("off")
        plt.show()

        # Debug: Visualize intermediate accumulation
        plt.figure(figsize=(5, 5))
        plt.title(f"Accumulation after color {color_idx}")
        plt.imshow(np.clip(accumulation, 0, 1))
        plt.axis("off")
        plt.show()

    # Debug: Check accumulation values
    print("Accumulation max value:", np.max(accumulation))
    print("Accumulation min value:", np.min(accumulation))

    # Apply gamma correction with passed gamma value
    final_image = np.power(
        accumulation / np.maximum(np.max(accumulation, axis=(0, 1)), 1.0), gamma
    )
    # Reduce the dark paper texture influence:
    final_image = 0.7 * final_image + 0.3 * (paper_texture[..., np.newaxis])
    final_image = np.clip(final_image, 0, 1)

    # Debug: Check final image values
    print("Final image max value:", np.max(final_image))
    print("Final image min value:", np.min(final_image))

    return np.clip(final_image, 0, 1), accumulation


# Final test run
if __name__ == "__main__":
    input_image_path = "image.png"  # Replace with your image path

    original_image, watercolor_image = simulate_watercolor(
        image_path=input_image_path,
        num_colors=3,
        output_size=(200, 200),
        paper_scale=10,
        paper_pattern="hatched",
        paper_pattern_params={"spacing": 10, "angle": 90, "pattern_strength": 0.4},
        paper_brightness=0.9,
        diffusion_rate=0.7,
        gradient_influence=1.5,
        noise_intensity=0.01,
        edge_darkening=0.001,
        backrun_rate=0.001,
        gamma=1.2,
        iterations=15,
        dryness_rate=0.25,
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Watercolor Effect")
    plt.imshow(watercolor_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

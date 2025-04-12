import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, laplace
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

@dataclass
class WatercolorParams:
    """Parameters for watercolor rendering"""
    flow_scale: float = 20.0
    turbulence: float = 0.5
    granulation: float = 0.1
    edge_darkening: float = 0.05
    wetness: float = 1.0
    gamma: float = 0.8
    color_boost: float = 1.2

class FluidSolver:
    """Enhanced fluid solver with pressure solving and proper diffusion"""
    def __init__(self, shape, dt=0.1):
        self.shape = shape
        self.dt = dt
        self.pressure = np.zeros(shape)
        self.divergence = np.zeros(shape)
        self.velocity = np.zeros((*shape, 2))
    
    def project(self, velocity):
        """Pressure projection for incompressibility"""
        div = np.zeros(self.shape)
        div[1:-1, 1:-1] = (
            (velocity[1:-1, 2:, 0] - velocity[1:-1, :-2, 0]) +
            (velocity[2:, 1:-1, 1] - velocity[:-2, 1:-1, 1])
        ) * 0.5
        
        p = np.zeros(self.shape)
        for _ in range(50):  # Jacobi iterations
            p[1:-1, 1:-1] = (
                p[1:-1, 2:] + p[1:-1, :-2] +
                p[2:, 1:-1] + p[:-2, 1:-1] - div[1:-1, 1:-1]
            ) * 0.25
        
        velocity[1:-1, 1:-1, 0] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
        velocity[1:-1, 1:-1, 1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
        return velocity

    def diffuse(self, field, rate=0.1):
        """Implicit diffusion solver"""
        result = field.copy()
        for _ in range(20):  # Jacobi iterations
            result[1:-1, 1:-1] = (
                field[1:-1, 1:-1] + rate * (
                    result[1:-1, 2:] + result[1:-1, :-2] +
                    result[2:, 1:-1] + result[:-2, 1:-1]
                )
            ) / (1 + 4 * rate)
        return result

    def curl(self, velocity):
        """Compute curl of 2D vector field"""
        if velocity.ndim < 3:
            return np.zeros(self.shape)
            
        du_dy = np.gradient(velocity[..., 0], axis=0)
        dv_dx = np.gradient(velocity[..., 1], axis=1)
        return du_dy - dv_dx
    
    def advect(self, field, velocity):
        """Semi-Lagrangian advection"""
        h, w = self.shape
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Backtrack positions
        pos_x = x - velocity[..., 0] * self.dt
        pos_y = y - velocity[..., 1] * self.dt
        
        # Ensure bounds
        pos_x = np.clip(pos_x, 0, w-1)
        pos_y = np.clip(pos_y, 0, h-1)
        
        # Interpolate
        x0 = pos_x.astype(int)
        y0 = pos_y.astype(int)
        x1 = np.minimum(x0 + 1, w-1)
        y1 = np.minimum(y0 + 1, h-1)
        
        fx = pos_x - x0
        fy = pos_y - y0
        
        # Bilinear interpolation
        c1 = field[y0, x0] * (1-fx) * (1-fy)
        c2 = field[y0, x1] * fx * (1-fy)
        c3 = field[y1, x0] * (1-fx) * fy
        c4 = field[y1, x1] * fx * fy
        
        return c1 + c2 + c3 + c4

class PigmentLayer:
    """Manages individual pigment behavior and interactions"""
    def __init__(self, mask, color, wetness=1.0):
        self.density = mask.astype(np.float32)
        self.color = np.array(color)
        self.wetness = wetness
        self.velocity = np.zeros((*mask.shape, 2))
    
    def update(self, dt=0.1):
        """Update pigment state with proper diffusion"""
        # Fixed scalar sigma for gaussian filter
        sigma = float(0.1 * self.wetness)  # Convert to scalar
        self.density = gaussian_filter(self.density, sigma=sigma)
        
        # Update wetness with fixed sigma
        edge_strength = gaussian_filter(self.density, sigma=2.0)
        self.wetness *= 0.99
        self.wetness *= 0.95 if np.mean(edge_strength) < 0.5 else 1.0

class WatercolorRenderer:
    def __init__(self, num_colors=5, output_size=(512, 512), params=None):
        self.num_colors = num_colors
        self.output_size = output_size
        self.params = params or WatercolorParams()
        self._cache = {}
        self.fluid_solver = FluidSolver(output_size)
        
    @lru_cache(maxsize=8)
    def _generate_base_noise(self, size):
        """Cached noise generation for better performance"""
        freq_x = np.fft.fftfreq(size[1])[:, None]
        freq_y = np.fft.fftfreq(size[0])[None, :]
        noise = np.exp(-2 * (freq_x**2 + freq_y**2))
        return np.fft.ifft2(noise).real

    def generate_flow_field(self, shape):
        """Enhanced flow field generation with fluid simulation"""
        if shape not in self._cache:
            noise = self._generate_base_noise(shape)
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            
            # Create initial velocity field
            velocity = np.stack([
                gaussian_filter(np.cos(2 * np.pi * noise), self.params.flow_scale),
                gaussian_filter(np.sin(2 * np.pi * noise), self.params.flow_scale)
            ], axis=-1)
            
            # Add curl-based turbulence
            curl = self.fluid_solver.curl(velocity)
            vorticity = np.stack([-curl, curl], axis=-1)
            velocity += vorticity * self.params.turbulence
            
            # Project and solve
            velocity = self.fluid_solver.project(velocity)
            fx = self.fluid_solver.advect(velocity[..., 0], velocity)
            fy = self.fluid_solver.advect(velocity[..., 1], velocity)
            
            self._cache[shape] = (fx, fy)
            
        return self._cache[shape]

    def advect_pigment(self, pigment, flow_x, flow_y):
        """Vectorized advection with improved memory efficiency"""
        h, w = pigment.shape
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        new_x = np.clip(x + flow_x, 0, w-1)
        new_y = np.clip(y + flow_y, 0, h-1)
        
        # Efficient floor operation
        x0 = new_x.astype(np.int32)
        y0 = new_y.astype(np.int32)
        x1 = np.minimum(x0 + 1, w-1)
        y1 = np.minimum(y0 + 1, h-1)
        
        # Compute weights once
        fx = new_x - x0
        fy = new_y - y0
        
        # Vectorized interpolation
        return np.clip(
            pigment[y0, x0] * (1-fx) * (1-fy) +
            pigment[y0, x1] * fx * (1-fy) +
            pigment[y1, x0] * (1-fx) * fy +
            pigment[y1, x1] * fx * fy,
            0, 1
        )

    def process_layer(self, mask, paper_texture):
        """Enhanced layer processing with improved fluid dynamics"""
        # Initialize pigment layer with higher initial wetness
        pigment = PigmentLayer(mask, color=1.0, wetness=2.0)
        
        # Multiple fluid simulation steps
        for _ in range(20):
            # Generate flow field
            flow_x, flow_y = self.generate_flow_field(mask.shape)
            velocity = np.stack([flow_x, flow_y], axis=-1)
            
            # Apply forces based on paper texture (fixed sigma)
            velocity[..., 1] += 0.1 * gaussian_filter(paper_texture, sigma=2.0)
            
            # Solve fluid dynamics
            velocity = self.fluid_solver.project(velocity)
            pigment.velocity = velocity
            
            # Update pigment state
            pigment.update(dt=0.1)
            
            # Enhanced edge effects
            edges = cv2.Canny((pigment.density * 255).astype(np.uint8), 50, 150)
            edge_dist = cv2.distanceTransform(~edges, cv2.DIST_L2, 3)
            edge_dist = 1 - edge_dist / (edge_dist.max() + 1e-6)
            
            # Stronger edge bleeding with fixed noise scale
            noise_scale = 0.2 * pigment.wetness
            noise = np.random.normal(0, noise_scale, pigment.density.shape)
            pigment.density = np.clip(
                pigment.density + noise * edge_dist * 0.1,
                0, 1
            )
            
            # Apply advection with edge-aware flow
            flow_strength = edge_dist[..., None] * 0.5
            pigment.density = self.fluid_solver.advect(
                pigment.density,
                velocity * flow_strength
            )
        
        return pigment.density

    def process_layers_parallel(self, masks, paper_texture):
        """Parallel layer processing"""
        with ThreadPoolExecutor() as executor:
            return list(executor.map(
                lambda m: self.process_layer(m, paper_texture),
                masks
            ))

    def render(self, image):
        """Enhanced rendering with improved color mixing"""
        # Improved color quantization
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(
            n_clusters=self.num_colors,
            n_init=1,
            random_state=42,
            algorithm='elkan'  # Faster algorithm
        )
        labels = kmeans.fit_predict(pixels)
        palette = kmeans.cluster_centers_
        
        # Enhanced color processing
        palette = np.clip(palette / palette.max(), 0, 1)
        palette = np.power(palette, self.params.gamma)
        palette *= self.params.color_boost
        
        # Generate paper texture
        paper = gaussian_filter(
            np.random.rand(*self.output_size),
            self.params.flow_scale/2
        )
        
        # Prepare masks for parallel processing
        masks = [
            (labels.reshape(image.shape[:2]) == i).astype(np.float32)
            for i in range(self.num_colors)
        ]
        
        # Process layers in parallel
        layers = self.process_layers_parallel(masks, paper)
        
        # Efficient compositing
        result = np.sum([
            layer[..., None] * palette[i]
            for i, layer in enumerate(layers)
        ], axis=0)
        
        return np.clip(result, 0, 1)

def apply_watercolor(image_path, num_colors=5, output_size=(512, 512), params=None):
    """Enhanced interface with parameter control"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    image = cv2.cvtColor(cv2.resize(image, output_size), cv2.COLOR_BGR2RGB) / 255.0
    
    renderer = WatercolorRenderer(
        num_colors=num_colors,
        output_size=output_size,
        params=params
    )
    result = renderer.render(image)
    
    return image, result

# ...existing visualization code...

# Load image and run simulation
input_image_path = "image.png"  # Replace with your image path
original_image, watercolor_image = apply_watercolor(input_image_path)

# Display results
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

from pathlib import Path
import subprocess
import sys
import time

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_image_path():
    print("\n" + "=" * 60)
    print("   ADVANCED MULTI-QUADRANT VECTOR & 4X RECONSTRUCTION ENGINE")
    print("=" * 60)

    # Try native Linux file picker (Zenity)
    try:
        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--title=Select an Image for Processing",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        selected_file = result.stdout.strip()
        if selected_file and Path(selected_file).is_file():
            return Path(selected_file)
    except Exception:
        pass

    # Terminal Fallback
    while True:
        path_input = input(
            "\nDrag and drop your image here (or type full path): "
        ).strip().strip("'\"")
        if not path_input:
            print("Path cannot be empty. Try again.")
            continue

        img_path = Path(path_input)
        if img_path.is_file():
            return img_path
        print(f"[Error] File not found at: {img_path}")


# Get Image Path
image_path = get_image_path()

try:
    scale_input = input(
        "Enter downsampling ratio (1 for full res, 2-4 for fast preview) [Default 1]: "
    ).strip()
    scale_factor = (
        int(scale_input)
        if scale_input.isdigit() and int(scale_input) > 0
        else 1
    )
except Exception:
    scale_factor = 1

total_start_time = time.time()

# 1. Load Image
t0 = time.time()
try:
    img = Image.open(image_path)
    sandbox_array = np.array(img.convert("RGB"))[::scale_factor, ::scale_factor]
    print(
        f"\nWorking Resolution: {sandbox_array.shape[1]}x{sandbox_array.shape[0]}"
    )
except Exception as e:
    print(f"[Error] Could not load image: {e}")
    sys.exit(1)
time_load = time.time() - t0

# 2. Vector Engine & 4-Quadrant Sub-Kernels
t1 = time.time()
print("Computing 4-Quadrant Sub-Kernels & RGB Channel Vectors...")

arr = sandbox_array.astype(np.float32)
height, width, _ = arr.shape

# Zero-pad array for 8-neighbor indexing
padded = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")

# Directional vector storage for R, G, B channels
V_r = np.zeros((height, width, 2), dtype=np.float32)
V_g = np.zeros((height, width, 2), dtype=np.float32)
V_b = np.zeros((height, width, 2), dtype=np.float32)

# Quadrant Magnitude Accumulators
quad_TL_mag = np.zeros((height, width), dtype=np.float32)
quad_TR_mag = np.zeros((height, width), dtype=np.float32)
quad_BL_mag = np.zeros((height, width), dtype=np.float32)
quad_BR_mag = np.zeros((height, width), dtype=np.float32)

# Neighbor Definitions: (dr, dc, dist, quadrant_tags)
neighbors = [
    (-1, -1, np.sqrt(2), ["TL"]),
    (-1, 0, 1.0, ["TL", "TR"]),
    (-1, 1, np.sqrt(2), ["TR"]),
    (0, -1, 1.0, ["TL", "BL"]),
    (0, 1, 1.0, ["TR", "BR"]),
    (1, -1, np.sqrt(2), ["BL"]),
    (1, 0, 1.0, ["BL", "BR"]),
    (1, 1, np.sqrt(2), ["BR"]),
]

for dr, dc, dist, quads in neighbors:
    neighbor_slice = padded[1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width]
    diff = neighbor_slice - arr

    dir_y = dr / dist  # Row direction (Y)
    dir_x = dc / dist  # Col direction (X)

    # Accumulate global RGB channel vector components
    V_r[:, :, 0] += diff[:, :, 0] * dir_y  # Vy
    V_r[:, :, 1] += diff[:, :, 0] * dir_x  # Vx

    V_g[:, :, 0] += diff[:, :, 1] * dir_y
    V_g[:, :, 1] += diff[:, :, 1] * dir_x

    V_b[:, :, 0] += diff[:, :, 2] * dir_y
    V_b[:, :, 1] += diff[:, :, 2] * dir_x

    # Accumulate local quadrant magnitudes
    diff_mag_sq = np.sum(diff**2, axis=-1)
    if "TL" in quads:
        quad_TL_mag += diff_mag_sq
    if "TR" in quads:
        quad_TR_mag += diff_mag_sq
    if "BL" in quads:
        quad_BL_mag += diff_mag_sq
    if "BR" in quads:
        quad_BR_mag += diff_mag_sq

# Channel Magnitudes
M_r = np.sqrt(V_r[:, :, 0] ** 2 + V_r[:, :, 1] ** 2)
M_g = np.sqrt(V_g[:, :, 0] ** 2 + V_g[:, :, 1] ** 2)
M_b = np.sqrt(V_b[:, :, 0] ** 2 + V_b[:, :, 1] ** 2)

total_magn = np.sqrt(M_r**2 + M_g**2 + M_b**2)

# Construct 3-Channel RGB Differential Map
p98_r = np.percentile(M_r, 98) + 1e-5
p98_g = np.percentile(M_g, 98) + 1e-5
p98_b = np.percentile(M_b, 98) + 1e-5

rgb_diff_map = np.zeros((height, width, 3), dtype=np.float32)
rgb_diff_map[:, :, 0] = np.clip(M_r / p98_r, 0, 1)  # Red differential
rgb_diff_map[:, :, 1] = np.clip(M_g / p98_g, 0, 1)  # Green differential
rgb_diff_map[:, :, 2] = np.clip(M_b / p98_b, 0, 1)  # Blue differential

# =====================================================================
# MAGNITUDE BLACK OUTLINE OVERLAY
# =====================================================================
norm_mag = np.clip(total_magn / (np.percentile(total_magn, 96) + 1e-5), 0, 1)
outline_factor = (1.0 - 0.85 * (norm_mag**1.5))[:, :, np.newaxis]
magnitude_overlay = np.clip(arr * outline_factor, 0, 255).astype(np.uint8)

# =====================================================================
# 4X RECONSTRUCTION MATRIX ASSEMBLY
# =====================================================================
print("Assembling 4x Quality Sub-Pixel Reconstruction Matrix...")

sub_TL = (4*arr +
    padded[0:height, 0:width]
    + 2*padded[0:height, 1 : 1 + width]
    + 2*padded[1 : 1 + height, 0:width]
) / 9.0

sub_TR = (4*arr +
    padded[0:height, 2 : 2 + width]
    + 2*padded[0:height, 1 : 1 + width]
    + 2*padded[1 : 1 + height, 2 : 2 + width]
) / 9.0

sub_BL = (4*arr +
    padded[2 : 2 + height, 0:width]
    + 2*padded[2 : 2 + height, 1 : 1 + width]
    + 2*padded[1 : 1 + height, 0:width]
) / 9.0

sub_BR = (4*arr +
    padded[2 : 2 + height, 2 : 2 + width]
    + 2*padded[2 : 2 + height, 1 : 1 + width]
    + 2*padded[1 : 1 + height, 2 : 2 + width]
) / 9.0

reconstructed_4x = np.zeros((2 * height, 2 * width, 3), dtype=np.float32)
reconstructed_4x[0::2, 0::2] = sub_TL
reconstructed_4x[0::2, 1::2] = sub_TR
reconstructed_4x[1::2, 0::2] = sub_BL
reconstructed_4x[1::2, 1::2] = sub_BR

img_4x_uint8 = np.clip(reconstructed_4x, 0, 255).astype(np.uint8)

# Basic Kernel Sharpening on 4x image
# Pad array by 1 pixel along rows and columns
padded_4x = np.pad(reconstructed_4x, ((1, 1), (1, 1), (0, 0)), mode="edge")

# 1. Extract Cardinals
top    = padded_4x[0:-2, 1:-1]
bottom = padded_4x[2:,   1:-1]
left   = padded_4x[1:-1, 0:-2]
right  = padded_4x[1:-1, 2:]

# 2. Extract Diagonals
top_left     = padded_4x[0:-2, 0:-2]
top_right    = padded_4x[0:-2, 2:]
bottom_left  = padded_4x[2:,   0:-2]
bottom_right = padded_4x[2:,   2:]

# 3. Apply Full 8-Neighbor Sharpening
enhanced_4x = (
    16.0 * reconstructed_4x
    - 2*(top + bottom + left + right)
    - 0.25*(top_left + top_right + bottom_left + bottom_right)
)/7

# Clip bounds to keep valid RGB range [0, 255]
enhanced_4x_uint8 = np.clip(enhanced_4x, 0, 255).astype(np.uint8)

time_vector_engine = time.time() - t1
print(f"Engine Processing Completed in {time_vector_engine:.3f}s!")

# Diagnostics
flat_magnitudes = total_magn.flatten()
mag_mean = np.mean(flat_magnitudes)
mag_max = np.max(flat_magnitudes)
p90 = np.percentile(flat_magnitudes, 90)
total_time = time.time() - total_start_time

print("\n" + "=" * 45)
print("     PERFORMANCE & MAGNITUDE PROFILE")
print("=" * 45)
print(f"Original Resolution           : {width}x{height}")
print(f"4x Reconstructed Resolution    : {2*width}x{2*height}")
print(f"Mean Vector Force              : {mag_mean:.2f}")
print(f"Peak Gradient Force            : {mag_max:.2f}")
print(f"Total Execution Time           : {total_time:.3f} seconds")
print("=" * 45 + "\n")

# =====================================================================
# DISPLAY 1: SYNCHRONIZED MULTI-VIEW PANEL (6 PANELS)
# =====================================================================
fig1, axes = plt.subplots(
    2, 3, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True
)
fig1.canvas.manager.set_window_title(
    "Synchronized Multi-Quadrant & 4x Reconstruction Analysis"
)

extent_bounds = [0, width, height, 0]

# 1. Original Photo
axes[0, 0].imshow(sandbox_array, extent=extent_bounds)
axes[0, 0].set_title("1. Original Photo", fontsize=11, fontweight="bold", pad=8)

# 2. RGB Differential Vector Field
axes[0, 1].imshow(rgb_diff_map, extent=extent_bounds)
axes[0, 1].set_title(
    "2. Edge Detection (RGB Vector)", fontsize=11, fontweight="bold", pad=8
)

# 3. Just Magnitude
axes[0, 2].imshow(total_magn, cmap="gray", extent=extent_bounds)
axes[0, 2].set_title(
    "3. Raw Gradient Magnitude Map", fontsize=11, fontweight="bold", pad=8
)

# 4. Magnitude Black Outline Overlay
axes[1, 0].imshow(magnitude_overlay, extent=extent_bounds)
axes[1, 0].set_title(
    "4. Magnitude Black Outline Overlay", fontsize=11, fontweight="bold", pad=8
)

# 5. 4x Reconstructed Original Image
axes[1, 1].imshow(img_4x_uint8, extent=extent_bounds)
axes[1, 1].set_title(
    "5. 4x Reconstructed Original (Quadrant Interleaved)",
    fontsize=11,
    fontweight="bold",
    pad=8,
)

# 6. Basic Kernel Edge-Enhanced 4x Image
axes[1, 2].imshow(enhanced_4x_uint8, extent=extent_bounds)
axes[1, 2].set_title(
    "6. Basic Kernel Edge-Enhanced 4x Image",
    fontsize=11,
    fontweight="bold",
    pad=8,
)

for ax in axes.flat:
    ax.axis("off")

fig1.suptitle(
    "Multi-Quadrant Vector Analysis & 4X Reconstruction Engine\n(All views synchronized — Zoom/Pan any panel)",
    fontsize=13,
    fontweight="bold",
)

# =====================================================================
# DISPLAY 2A: PURE RGB DIFFERENTIAL VECTOR MAP
# =====================================================================
fig_rgb1, ax_rgb1 = plt.subplots(figsize=(10, 7))
fig_rgb1.canvas.manager.set_window_title(
    "Dedicated High-Res RGB Differential Vector Map"
)

ax_rgb1.imshow(rgb_diff_map)
ax_rgb1.axis("off")
ax_rgb1.set_title(
    "Pure RGB Channel Differential Vectors (Raw Edge Dynamics)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()

# =====================================================================
# DISPLAY 2B: IMAGE-MODULATED RGB VECTOR MAP (GLOWING OUTLINE EFFECT)
# =====================================================================
fig_rgb2, ax_rgb2 = plt.subplots(figsize=(10, 7))
fig_rgb2.canvas.manager.set_window_title(
    "Dedicated Image-Modulated RGB Vector Map"
)

# Multiplies original photo colors by edge magnitude
rgb_diff_masked = np.clip(arr * rgb_diff_map, 0, 255).astype(np.uint8)

ax_rgb2.imshow(rgb_diff_masked)
ax_rgb2.axis("off")
ax_rgb2.set_title(
    "Image-Modulated RGB Vector Map (Photo Colors x Edge Dynamics)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()

# =====================================================================
# DISPLAY 3: GRID BLOCK-AVERAGED DIRECTION MAP (PURE DIRECTION)
# =====================================================================
fig_vec, ax_vector = plt.subplots(figsize=(10, 8))
fig_vec.canvas.manager.set_window_title("Grid Block-Averaged Direction Field")

# Subtle background image
ax_vector.imshow(sandbox_array, alpha=0.55, extent=[0, width, height, 0])

# Net directional vectors
U_net = (V_r[:, :, 1] + V_g[:, :, 1] + V_b[:, :, 1]) / 3.0
V_net = (V_r[:, :, 0] + V_g[:, :, 0] + V_b[:, :, 0]) / 3.0

# Define square block size for spatial averaging (e.g., ~30 blocks wide)
block_size = max(16, min(width, height) // 30)

h_blocks = height // block_size
w_blocks = width // block_size

# Crop array to exact multiples of block_size for vector reshapes
U_crop = U_net[: h_blocks * block_size, : w_blocks * block_size]
V_crop = V_net[: h_blocks * block_size, : w_blocks * block_size]
mag_crop = total_magn[: h_blocks * block_size, : w_blocks * block_size]

# Block mean calculations (average vectors across square blocks)
U_block = U_crop.reshape(h_blocks, block_size, w_blocks, block_size).mean(axis=(1, 3))
V_block = V_crop.reshape(h_blocks, block_size, w_blocks, block_size).mean(axis=(1, 3))
mag_block = mag_crop.reshape(h_blocks, block_size, w_blocks, block_size).mean(axis=(1, 3))

# Grid center coordinates
x_coords = np.arange(w_blocks) * block_size + block_size / 2.0
y_coords = np.arange(h_blocks) * block_size + block_size / 2.0
X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

# Normalize vectors to unit length -> EXACT UNIFORM ARROW LENGTH
norm = np.sqrt(U_block**2 + V_block**2) + 1e-5
U_norm = U_block / norm
V_norm = V_block / norm

# Mask out flat, low-energy regions (e.g., smooth sky) to keep plot clean
energy_thresh = np.percentile(total_magn, 45)
mask = mag_block > energy_thresh

# Fixed Quiver: 'scale_units' and 'angles' locked to 'xy' so arrows fit inside their block
q = ax_vector.quiver(
    X_grid[mask],
    Y_grid[mask],
    U_norm[mask],
    -V_norm[mask],
    mag_block[mask],
    cmap="plasma",
    pivot="middle",
    angles="xy",
    scale_units="xy",
    scale=1.0 / (block_size * 0.75),  # Fits arrow length to 75% of each grid square
    headwidth=3.5,
    headlength=3.5,
    width=0.0025,
)

fig_vec.colorbar(q, ax=ax_vector, label="Average Edge Energy")
ax_vector.set_title(
    f"Grid Block-Averaged Direction Field ({block_size}x{block_size} Pixel Squares)\n(Uniform Arrow Lengths — Thresholded Sky & Flat Zones)",
    fontsize=12,
    fontweight="bold",
)
ax_vector.axis("off")
plt.tight_layout()

# =====================================================================
# DISPLAY 4: GRADIENT FORCE DENSITY DISTRIBUTION
# =====================================================================
fig_hist, ax_mag = plt.subplots(figsize=(8, 4))
fig_hist.canvas.manager.set_window_title("Gradient Force Distribution")

ax_mag.hist(
    flat_magnitudes,
    bins=50,
    density=True,
    alpha=0.6,
    color="royalblue",
    edgecolor="black",
)

if np.std(flat_magnitudes) > 0:
    sample_size = min(20000, len(flat_magnitudes))
    sampled_mags = np.random.choice(
        flat_magnitudes, sample_size, replace=False
    )
    kde_mag = gaussian_kde(sampled_mags)
    x_axis_mag = np.linspace(0, mag_max, 500)
    ax_mag.plot(x_axis_mag, kde_mag(x_axis_mag), color="darkblue", linewidth=2)

ax_mag.axvline(
    mag_mean,
    color="orange",
    linestyle="dashed",
    linewidth=2,
    label=f"Mean Force ({mag_mean:.1f})",
)
ax_mag.axvline(
    p90,
    color="red",
    linestyle="dashdot",
    linewidth=2,
    label=f"90th %ile ({p90:.1f})",
)

ax_mag.set_title(
    "Image Gradient Force Probability Density", fontsize=11, fontweight="bold"
)
ax_mag.set_xlabel("Vector Acceleration Force Length", fontsize=10)
ax_mag.set_ylabel("Probability Density", fontsize=10)
ax_mag.grid(True, linestyle=":", alpha=0.6)
ax_mag.legend()
plt.tight_layout()

plt.show()
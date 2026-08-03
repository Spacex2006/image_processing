#fully ai code
import time
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.stats import gaussian_kde

def get_file_path():
    root = tk.Tk()
    root.withdraw() 
    
    file_types = [
        ("JPEG Images", "*.jpg"),
        ("PNG Images", "*.png"),
        ("All Files", "*.*")
    ]
    
    selected_path = filedialog.askopenfilename(
        title="Select an Image for Processing",
        initialdir=Path.home() / "Downloads",
        filetypes=file_types
    )
    
    return Path(selected_path) if selected_path else None

def save_high_res_image(image_array, default_filename="output.png"):
    """Opens a dialog allowing the user to save high-quality output images."""
    root = tk.Tk()
    root.withdraw()
    
    save_path = filedialog.asksaveasfilename(
        title="Save High Quality Output Image",
        defaultextension=".png",
        initialfile=default_filename,
        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
    )
    
    if save_path:
        out_img = Image.fromarray(image_array)
        out_img.save(save_path, quality=100)
        print(f"[SUCCESS] High-quality image saved to: {save_path}")
    else:
        print("[INFO] Save cancelled.")

image_path = get_file_path()

if not image_path:
    print("[Error] No image selected. Exiting.")
    exit()

print("\n" + "="*40)
print("   IMAGE PROCESSING & VECTOR ENGINE")
print("="*40)
scale_factor = int(input("Enter downsampling ratio (e.g., 1 for full resolution, 8 for fast preview): "))

# --- START OVERALL TIMER ---
total_start_time = time.time()

# 1. TIMING: Load & Resize
t0 = time.time()
try:
    img = Image.open(image_path)
    sandbox_array = np.array(img)[::scale_factor, ::scale_factor]
    print(f"\nWorking Resolution: {sandbox_array.shape[1]}x{sandbox_array.shape[0]}")
except FileNotFoundError:
    print("[Error] Image not found.")
    exit()
time_load = time.time() - t0

height, width, channels = sandbox_array.shape[0], sandbox_array.shape[1], sandbox_array.shape[2]

# Cast to float32 for precise mathematical operations without overflow
arr = sandbox_array.astype(np.float32)

# 2. TIMING: Fast Vectorized RGB Vector Engine
t1 = time.time()
print("Running Fast RGB Vector Engine...")

# Edge-pad the image array so 3x3 neighbor queries at borders don't throw IndexError
padded = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode='edge')

# Directional gradient vectors for Red, Green, and Blue channels
vec_r = np.zeros((height, width, 2), dtype=np.float32)
vec_g = np.zeros((height, width, 2), dtype=np.float32)
vec_b = np.zeros((height, width, 2), dtype=np.float32)

curr_R, curr_G, curr_B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

# Compute spatial differences across all 8 immediate neighbor directions simultaneously
for dr in [-1, 0, 1]:
    for dc in [-1, 0, 1]:
        if dr == 0 and dc == 0:
            continue
        
        # Euclidean distance to neighbor (1.0 for cardinal, ~1.414 for diagonal)
        mag_dist = np.sqrt(dr*dr + dc*dc)
        unit_r = dr / mag_dist
        unit_c = dc / mag_dist
        
        # Array slice representing the shifted neighbor grid
        n_R = padded[1+dr:1+dr+height, 1+dc:1+dc+width, 0]
        n_G = padded[1+dr:1+dr+height, 1+dc:1+dc+width, 1]
        n_B = padded[1+dr:1+dr+height, 1+dc:1+dc+width, 2]
        
        # Color deltas
        diff_R = n_R - curr_R
        diff_G = n_G - curr_G
        diff_B = n_B - curr_B
        
        # Accumulate directional force vectors per color channel
        vec_r[:, :, 0] += diff_R * unit_r
        vec_r[:, :, 1] += diff_R * unit_c
        
        vec_g[:, :, 0] += diff_G * unit_r
        vec_g[:, :, 1] += diff_G * unit_c
        
        vec_b[:, :, 0] += diff_B * unit_r
        vec_b[:, :, 1] += diff_B * unit_c

# Combined directional vector field (mean vector across channels)
kms = (vec_r + vec_g + vec_b) / 3.0

# Calculate total magnitude force across all color channels
mag_R = vec_r[:, :, 0]**2 + vec_r[:, :, 1]**2
mag_G = vec_g[:, :, 0]**2 + vec_g[:, :, 1]**2
mag_B = vec_b[:, :, 0]**2 + vec_b[:, :, 1]**2

magn = np.sqrt(mag_R + mag_G + mag_B)
x = np.percentile(magn, 99.5)

# Use square brackets for indexing in NumPy arrays
magn[magn > x] = x

time_vector_engine = time.time() - t1
print(f"Vector Engine Done ({time_vector_engine:.3f}s)")

# 3. TIMING: Percentile & Energy Profiling
t2 = time.time()
flat_magnitudes = magn.flatten()
mag_mean = np.mean(flat_magnitudes)
mag_std = np.std(flat_magnitudes)
mag_max = np.max(flat_magnitudes)

p50 = np.percentile(flat_magnitudes, 50)
p90 = np.percentile(flat_magnitudes, 90)
p98 = np.percentile(flat_magnitudes, 98)
p99 = np.percentile(flat_magnitudes, 99)
time_analysis = time.time() - t2

total_time = time.time() - total_start_time

# =====================================================================
# VECTOR GRADIENT MAGNITUDE ANALYSIS
# =====================================================================
print("\n" + "="*40)
print("     RAW GRADIENT MAGNITUDE PROFILE")
print("="*40)
print(f"Total Array Pixels Evaluated    : {len(flat_magnitudes)}")
print(f"Mean Vector Magnitude Force     : {mag_mean:.2f}")
print(f"Standard Deviation (σ)          : {mag_std:.2f}")
print(f"Peak Gradient Contrast Force    : {mag_max:.2f}")
print(f"Median Detail Floor (50th %ile) : {p50:.2f}")
print(f"True Boundary Filter (90th %ile): {p90:.2f}")
print(f"High-Energy Signals  (98th %ile): {p98:.2f}")
print(f"Peak Energy Signals  (99th %ile): {p99:.2f}")
print("="*40 + "\n")

print("="*40)
print("     PERFORMANCE DIAGNOSTICS")
print("="*40)
print(f"Image Load & Resize : {time_load:.3f} seconds")
print(f"RGB Vector Engine   : {time_vector_engine:.3f} seconds")
print(f"Statistical Profile : {time_analysis:.3f} seconds")
print("-" * 40)
print(f"TOTAL EXECUTION TIME: {total_time:.3f} seconds")
print("="*40)

# =====================================================================
# HIGH QUALITY SAVE EXPORT MENU
# =====================================================================
print("\n--- EXPORT OPTIONS ---")
save_choice = input("Would you like to export high quality results? [y/n]: ").strip().lower()

if save_choice == 'y':
    # Normalize magnitude array (0 to 255 uint8) for crisp visual rendering
    mag_normalized = (255 * (magn / np.max(magn))).astype(np.uint8)
    save_high_res_image(mag_normalized, default_filename="gradient_magnitude_map.png")

# --- VISUALIZATION PLOTS ---
# Plot 1: Calculated Vector Gradient Field Visualizer
fig_mag_vis, ax_mag_vis = plt.subplots(figsize=(10, 6))
mag_plot = ax_mag_vis.imshow(magn, cmap='gray')
ax_mag_vis.axis('off')
ax_mag_vis.set_title("Calculated Vector Gradient Magnitudes (Grayscale)")
fig_mag_vis.colorbar(mag_plot, ax=ax_mag_vis, label="Gradient Intensity Force")

# Plot 2: Gradient Magnitude Probability Density Graph (Histogram + KDE)
fig_mag, ax_mag = plt.subplots(figsize=(8, 5))
ax_mag.hist(flat_magnitudes, bins=50, density=True, alpha=0.6, color='royalblue', edgecolor='black', label='Pixel Energies')

if mag_std > 0:
    # Subsample max 20,000 pixels for fast KDE calculation
    sample_size = min(20000, len(flat_magnitudes))
    sampled_mags = np.random.choice(flat_magnitudes, sample_size, replace=False)
    kde_mag = gaussian_kde(sampled_mags)
    x_axis_mag = np.linspace(0, mag_max, 500)
    ax_mag.plot(x_axis_mag, kde_mag(x_axis_mag), color='darkblue', linewidth=2, label='Energy Smoothing (KDE)')

ax_mag.axvline(mag_mean, color='orange', linestyle='dashed', linewidth=2, label=f'Mean Force ({mag_mean:.1f})')
ax_mag.axvline(p98, color='red', linestyle='dashdot', linewidth=2, label=f'98th Cutoff ({p98:.1f})')

ax_mag.set_title("Image Gradient Force Probability Density Graph", fontsize=12, fontweight='bold')
ax_mag.set_xlabel("Vector Acceleration Force Length (Magnitude Scale)", fontsize=10)
ax_mag.set_ylabel("Probability Density", fontsize=10)
ax_mag.grid(True, linestyle=':', alpha=0.6)
ax_mag.legend()

plt.show()
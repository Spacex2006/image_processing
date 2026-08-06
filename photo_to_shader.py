import os
import sys
import warnings

# 1. Suppress font fallback warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt

# 2. Configure Matplotlib font settings for solid block elements
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'FreeMono', 'Noto Sans']

# Optional Scipy import for KDE smoothing
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def print_progress_bar(iteration, total, prefix='', suffix=''):
    """Prints a dynamic loading bar in the terminal."""
    percent = (iteration / float(total)) * 100
    filled_length = int(30 * iteration // total)
    bar = '█' * filled_length + '░' * (30 - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()


# ================================================================
# STEP 1: GENERATE VECTOR DATABASE FOR SOLID QUADRANT & SHADE GLYPHS
# ================================================================
print("Step 1: Generating Quadrant & Shading Block vector database...")

# The 16 Quadrant Block combinations (U+2580 to U+259F) + Shading Blocks
quadrant_glyphs = [
    ' ',  # 0: Empty
    '▘',  # 1: Top-Left
    '▝',  # 2: Top-Right
    '▀',  # 3: Top Half (TL + TR)
    '▖',  # 4: Bottom-Left
    '▌',  # 5: Left Half (TL + BL)
    '▞',  # 6: Diagonal (TR + BL)
    '▛',  # 7: TL + TR + BL
    '▗',  # 8: Bottom-Right
    '▚',  # 9: Anti-Diagonal (TL + BR)
    '▐',  # 10: Right Half (TR + BR)
    '▜',  # 11: TL + TR + BR
    '▄',  # 12: Bottom Half (BL + BR)
    '▙',  # 13: TL + BL + BR
    '▟',  # 14: TR + BL + BR
    '█',  # 15: Full Solid Block
    '░',  # 16: 25% Light Shade
    '▒',  # 17: 50% Medium Shade
    '▓'   # 18: 75% Dark Shade
]

data = []
char_vectors = []
char_mags_list = []

# Function to synthesize 20x20 pixel matrix for a Quadrant or Shade block
def render_block_matrix(glyph_idx):
    matrix = np.zeros((20, 20), dtype=float)
    
    if glyph_idx < 16:
        # Quadrant sub-regions (2x2 grid)
        # Bit 0 (1): Top-Left     | Bit 1 (2): Top-Right
        # Bit 2 (4): Bottom-Left  | Bit 3 (8): Bottom-Right
        if glyph_idx & 1: matrix[0:10, 0:10] = 255.0   # TL
        if glyph_idx & 2: matrix[0:10, 10:20] = 255.0  # TR
        if glyph_idx & 4: matrix[10:20, 0:10] = 255.0  # BL
        if glyph_idx & 8: matrix[10:20, 10:20] = 255.0 # BR
    elif glyph_idx == 16: # 25% Shade
        matrix[::2, ::2] = 255.0
    elif glyph_idx == 17: # 50% Checkerboard Shade
        matrix[::2, ::2] = 255.0
        matrix[1::2, 1::2] = 255.0
    elif glyph_idx == 18: # 75% Dark Shade
        matrix[:, :] = 255.0
        matrix[::2, ::2] = 0.0
        
    return matrix

total_quad_chars = len(quadrant_glyphs)

for idx, char in enumerate(quadrant_glyphs):
    data.append(char)
    pixel_array = render_block_matrix(idx)

    # Find bounding box of active pixels
    row_start, row_end = 0, 0
    col_start, col_end = 0, 0
    found_row, found_col = False, False

    for i in range(20):
        for j in range(20):
            if pixel_array[i, j] != 0:
                if not found_row:
                    row_start = i
                    found_row = True
                row_end = i

    for j in range(20):
        for i in range(20):
            if pixel_array[i, j] != 0:
                if not found_col:
                    col_start = j
                    found_col = True
                col_end = j

    center_y = (row_start + row_end) / 2.0
    center_x = col_start + (col_end - col_start) / 2.0

    # Sum displacement vectors for non-zero pixels
    pixel_count = 0
    sum_vert = 0.0
    sum_horiz = 0.0

    for i in range(20):
        for j in range(20):
            if pixel_array[i, j] != 0:
                pixel_count += 1
                offset_vert = center_y - i
                offset_horiz = j - center_x

                if offset_vert == 0 and offset_horiz == 0:
                    continue

                distance = np.sqrt(offset_vert**2 + offset_horiz**2)
                sum_vert += offset_vert / distance
                sum_horiz += offset_horiz / distance

    dir_length = np.sqrt(sum_vert**2 + sum_horiz**2)
    
    if dir_length > 0:
        unit_vert = sum_vert / dir_length
        unit_horiz = sum_horiz / dir_length
    else:
        unit_vert = 0.0
        unit_horiz = 0.0

    # Scale unit vector by pixel fill magnitude normalized by 400 (20x20 full block)
    final_vert = unit_vert * (pixel_count / 400.0)
    final_horiz = unit_horiz * (pixel_count / 400.0)

    char_mags_list.append(pixel_count)
    char_vectors.append((final_horiz, final_vert))

    print_progress_bar(idx + 1, total_quad_chars, prefix='Quadrant Vector DB:', suffix='Complete')

char_vec_matrix = np.array(char_vectors)
char_mags_array = np.array(char_mags_list)


# ================================================================
# STEP 2: SELECT INPUT IMAGE
# ================================================================
root = tk.Tk()
root.withdraw()

downloads_folder = os.path.expanduser("~/Downloads")
print(f"\nOpening file picker (default folder: {downloads_folder})...")

image_path = filedialog.askopenfilename(
    title="Select Image for Processing",
    initialdir=downloads_folder,
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff")]
)

if not image_path:
    print("No image selected. Exiting.")
    sys.exit()

print(f"Loaded File: {os.path.basename(image_path)}")

img = Image.open(image_path).convert("RGB")
img_w, img_h = img.size


# ================================================================
# STEP 3: CALCULATE GRID DIMENSIONS
# ================================================================
try:
    grid_x = int(input("\nEnter number of HORIZONTAL blocks (X) [Recommended: 80 to 120]: "))
except ValueError:
    grid_x = 100

font_aspect_ratio = 0.5
grid_y = max(1, int(round(grid_x * (img_h / img_w) * font_aspect_ratio)))

block_w = img_w / grid_x
block_h = img_h / grid_y

print(f"Calculated Vertical Blocks (Y): {grid_y}")
print(f"Image Resolution: {img_w}x{img_h} px | Grid: {grid_x}x{grid_y} blocks | Block Size: ~{block_w:.1f}x{block_h:.1f} px\n")

img_array = np.array(img, dtype=float)

raw_pixel_mags = np.sqrt(np.sum(img_array**2, axis=2)).ravel()
block_mag_grid = np.zeros((grid_y, grid_x))
selected_char_mag_grid = np.zeros((grid_y, grid_x))
block_vec_x_grid = np.zeros((grid_y, grid_x))
block_vec_y_grid = np.zeros((grid_y, grid_x))
selected_indices = []


# ================================================================
# STEP 4: PROCESS IMAGE BLOCKS & MATCH VIA EUCLIDEAN DISTANCE
# ================================================================
quad_rows = []

print("Processing image blocks...")
for row in range(grid_y):
    row_chars = []
    
    r_start = int(round(row * block_h))
    r_end = int(round((row + 1) * block_h))
    block_pixel_h = r_end - r_start

    for col in range(grid_x):
        c_start = int(round(col * block_w))
        c_end = int(round((col + 1) * block_w))
        block_pixel_w = c_end - c_start

        block = img_array[r_start:r_end, c_start:c_end]

        if block.size == 0:
            row_chars.append(" ")
            selected_indices.append(0)
            continue

        pixel_mags = np.sqrt(np.sum(block**2, axis=2))
        avg_block_mag = np.mean(pixel_mags)
        block_mag_grid[row, col] = avg_block_mag

        center_y = (block_pixel_h - 1) / 2.0
        center_x = (block_pixel_w - 1) / 2.0

        block_sum_vert = 0.0
        block_sum_horiz = 0.0

        for i in range(block_pixel_h):
            for j in range(block_pixel_w):
                p_mag = pixel_mags[i, j]
                
                offset_vert = center_y - i
                offset_horiz = j - center_x
                
                dist = np.sqrt(offset_vert**2 + offset_horiz**2)
                if dist == 0:
                    dist = 1e-6

                block_sum_vert += (offset_vert / dist) * p_mag
                block_sum_horiz += (offset_horiz / dist) * p_mag

        block_dir_len = np.sqrt(block_sum_vert**2 + block_sum_horiz**2)
        
        if block_dir_len > 0:
            unit_block_vert = block_sum_vert / block_dir_len
            unit_block_horiz = block_sum_horiz / block_dir_len
        else:
            unit_block_vert = 0.0
            unit_block_horiz = 0.0

        # Normalized brightness score (0.0 to 1.0)
        avg_brightness_norm = avg_block_mag / 441.67

        block_vec_horiz = unit_block_horiz * avg_brightness_norm
        block_vec_vert = unit_block_vert * avg_brightness_norm

        block_vec_x_grid[row, col] = block_vec_horiz
        block_vec_y_grid[row, col] = block_vec_vert

        block_vector = np.array([block_vec_horiz, block_vec_vert])

        # EUCLIDEAN MATCHING AGAINST QUADRANT VECTORS
        differences = char_vec_matrix - block_vector
        distances = np.sqrt(np.sum(differences**2, axis=1))
        
        best_char_idx = np.argmin(distances)

        selected_indices.append(best_char_idx)
        selected_char_mag_grid[row, col] = char_mags_array[best_char_idx]
        row_chars.append(data[best_char_idx])

    quad_rows.append("".join(row_chars))
    print_progress_bar(row + 1, grid_y, prefix='Slicing Image:', suffix='Complete')

quad_art_str = "\n".join(quad_rows)

with open("quadrant_vector_art.txt", "w", encoding="utf-8") as f:
    f.write(quad_art_str)
print(f"\nSaved raw Quadrant text to file: {os.path.abspath('quadrant_vector_art.txt')}")


# ================================================================
# DASHBOARD 1: VECTOR & SPATIAL FIELD DIAGNOSTICS
# ================================================================
print("\nOpening Dashboard 1: Vector & Spatial Field Diagnostics...")

fig_dash1 = plt.figure(figsize=(16, 12))
fig_dash1.suptitle("Dashboard 1: Vector & Spatial Field Diagnostics (Quadrant Space)", fontsize=16, fontweight='bold')

ax1 = fig_dash1.add_subplot(2, 2, 1)
x_vals = [v[0] for v in char_vectors]
y_vals = [v[1] for v in char_vectors]

ax1.scatter(x_vals, y_vals, color='red', alpha=0.7, edgecolors='black', s=40)
for char, x_p, y_p in zip(data, x_vals, y_vals):
    lbl = "' '" if char == " " else char
    ax1.annotate(lbl, (x_p, y_p), fontsize=10, ha='center', va='bottom')

ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.set_title("1. Quadrant Vector Feature Space", fontsize=11, fontweight='bold')
ax1.set_xlabel("Horizontal Displacement Sum")
ax1.set_ylabel("Vertical Displacement Sum")
ax1.grid(True, linestyle=':', alpha=0.5)

ax2 = fig_dash1.add_subplot(2, 2, 2)
ax2.imshow(img, aspect='equal')
ax2.set_title(f"2. Original Input Image ({img_w}x{img_h})", fontsize=11, fontweight='bold')
ax2.axis('off')

ax3 = fig_dash1.add_subplot(2, 2, 3)
im3 = ax3.imshow(block_mag_grid, cmap='magma', aspect='auto')
fig_dash1.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
ax3.set_title("3. Block RGB Magnitude Heatmap", fontsize=11, fontweight='bold')
ax3.set_xlabel("Grid Column")
ax3.set_ylabel("Grid Row")

ax4 = fig_dash1.add_subplot(2, 2, 4)
X_coords, Y_coords = np.meshgrid(np.arange(grid_x), np.arange(grid_y))

ax4.quiver(
    X_coords, Y_coords, 
    block_vec_x_grid, -block_vec_y_grid, 
    block_mag_grid, cmap='cool', pivot='middle'
)
ax4.invert_yaxis()
ax4.set_title("4. Image Block Displacement Vector Field", fontsize=11, fontweight='bold')
ax4.set_xlabel("Grid Column")
ax4.set_ylabel("Grid Row")

plt.tight_layout()
plt.savefig("quadrant_spatial_diagnostics.png", dpi=300)


# ================================================================
# DASHBOARD 2: MAGNITUDE & FREQUENCY ANALYTICS
# ================================================================
print("Opening Dashboard 2: Magnitude & Frequency Analytics...")

fig_dash2 = plt.figure(figsize=(16, 12))
fig_dash2.suptitle("Dashboard 2: Magnitude & Glyph Frequency Analytics", fontsize=16, fontweight='bold')

ax2_1 = fig_dash2.add_subplot(2, 2, 1)
sampled_pixels = np.random.choice(raw_pixel_mags, size=min(10000, len(raw_pixel_mags)), replace=False)
block_mags_flat = block_mag_grid.flatten()

ax2_1.hist(sampled_pixels, bins=40, density=True, alpha=0.4, color='cyan', label='Raw Pixel Magnitude')
ax2_1.hist(block_mags_flat, bins=30, density=True, alpha=0.5, color='orange', label='Block Avg Magnitude')

if HAS_SCIPY:
    kde_pixel = gaussian_kde(sampled_pixels)
    kde_block = gaussian_kde(block_mags_flat)
    x_range = np.linspace(0, max(raw_pixel_mags.max(), 441), 200)
    ax2_1.plot(x_range, kde_pixel(x_range), color='blue', linewidth=2, label='Pixel Density (KDE)')
    ax2_1.plot(x_range, kde_block(x_range), color='darkred', linewidth=2, label='Block Density (KDE)')

ax2_1.set_title("1. Image Magnitude Distribution", fontsize=11, fontweight='bold')
ax2_1.set_xlabel("Magnitude Norm sqrt(R^2 + G^2 + B^2)")
ax2_1.set_ylabel("Probability Density")
ax2_1.legend(loc='upper right')
ax2_1.grid(True, linestyle=':', alpha=0.5)

ax2_2 = fig_dash2.add_subplot(2, 2, 2)
char_counts = np.bincount(selected_indices, minlength=len(data))
top_indices = np.argsort(char_counts)[::-1][:15]

top_chars = [f"'{data[i]}'" if data[i] != " " else "'SPACE'" for i in top_indices]
top_counts = char_counts[top_indices]

bars = ax2_2.bar(top_chars, top_counts, color='purple', alpha=0.7, edgecolor='black')
ax2_2.set_title("2. Top 15 Most Used Quadrant Glyphs", fontsize=11, fontweight='bold')
ax2_2.set_xlabel("Quadrant Character Glyph")
ax2_2.set_ylabel("Occurrence Count")
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        ax2_2.text(bar.get_x() + bar.get_width()/2.0, yval + (0.01 * max(top_counts)), int(yval), ha='center', va='bottom', fontsize=8)

ax2_2.grid(True, linestyle=':', alpha=0.5, axis='y')

ax2_3 = fig_dash2.add_subplot(2, 2, 3)
flat_block_mags = block_mag_grid.flatten()
flat_char_mags = selected_char_mag_grid.flatten()

jitter = np.random.normal(0, 0.5, size=len(flat_char_mags))
ax2_3.scatter(flat_block_mags, flat_char_mags + jitter, alpha=0.4, color='green', edgecolors='none', s=15)

if len(flat_block_mags) > 1:
    m, c_val = np.polyfit(flat_block_mags, flat_char_mags, 1)
    x_line = np.linspace(flat_block_mags.min(), flat_block_mags.max(), 100)
    ax2_3.plot(x_line, m * x_line + c_val, color='red', linestyle='--', linewidth=2, label=f'Fit Slope: {m:.3f}')

ax2_3.set_title("3. Block RGB Magnitude vs. Matched Glyph Pixel Fill", fontsize=11, fontweight='bold')
ax2_3.set_xlabel("Block Average Magnitude")
ax2_3.set_ylabel("Matched Character Fill Count (Pixels)")
ax2_3.legend(loc='upper left')
ax2_3.grid(True, linestyle=':', alpha=0.5)

ax2_4 = fig_dash2.add_subplot(2, 2, 4)
padded_counts = np.pad(char_counts, (0, 20 - len(char_counts)), mode='constant')
freq_matrix = padded_counts.reshape(4, 5)

im4 = ax2_4.imshow(freq_matrix, cmap='YlOrRd')
fig_dash2.colorbar(im4, ax=ax2_4, fraction=0.046, pad=0.04)
ax2_4.set_title("4. Quadrant Spectrum Utilization Grid", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("quadrant_magnitude_analytics.png", dpi=300)


# ================================================================
# FIGURE 3: HIGH-RES RENDERED QUADRANT IMAGE
# ================================================================
print("Rendering high-res Quadrant image view...")

dynamic_fontsize = max(2, min(10, int(600 / grid_x)))

fig_ascii, ax_ascii = plt.subplots(figsize=(12, 12), facecolor='black')
ax_ascii.set_facecolor('black')

ax_ascii.text(
    0.5, 0.5, quad_art_str, 
    color='#00FF66', 
    fontsize=dynamic_fontsize, 
    ha='center', va='center', 
    transform=ax_ascii.transAxes
)
ax_ascii.axis('off')
plt.title("High-Resolution Quadrant Render", color='white', fontsize=14, pad=10)

rendered_img_path = "quadrant_art_render.png"
plt.savefig(rendered_img_path, dpi=300, bbox_inches='tight', facecolor='black')
print(f"Saved rendered Quadrant image to: {os.path.abspath(rendered_img_path)}")

plt.show()

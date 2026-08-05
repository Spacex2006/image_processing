import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Optional Scipy import for KDE smoothing on histograms
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ================================================================
# HELPER FUNCTION: TERMINAL PROGRESS BAR
# ================================================================
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
# STEP 1: GENERATE CHARACTER VECTOR DATABASE
# ================================================================
print("Step 1: Generating ASCII character vector database...")

data = []          # Stores character glyphs: [' ', '!', '"', ...]
char_vectors = []  # Stores 2D vectors: [(x_horiz, y_vert), ...]
char_mags_list = [] # Stores pixel fill counts per character

# Canvas dimensions for drawing individual characters
canvas_w = 20
canvas_h = 20
font = ImageFont.load_default()

# ASCII printable characters range from code 32 (space) to 125 (~)
total_chars = 126 - 32

for idx, ascii_code in enumerate(range(32, 126)):
    char = chr(ascii_code)
    data.append(char)

    # 1. Create a black 20x20 canvas and draw the character in white
    img = Image.new('L', (canvas_w, canvas_h), color=0)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), char, fill=255, font=font)
    pixel_array = np.array(img)

    # 2. Find the bounding box of the active white pixels
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

    # Calculate center of bounding box
    center_y = (row_start + row_end) / 2.0
    center_x = col_start + (col_end - col_start) / 2.0

    # 3. Sum displacement vectors for all non-zero pixels
    pixel_count = 0
    sum_vert = 0.0   # Vertical displacement sum
    sum_horiz = 0.0  # Horizontal displacement sum

    for i in range(20):
        for j in range(20):
            if pixel_array[i, j] != 0:
                pixel_count += 1
                
                # Distance from center
                offset_vert = center_y - i
                offset_horiz = j - center_x

                if offset_vert == 0 and offset_horiz == 0:
                    continue

                distance = np.sqrt(offset_vert**2 + offset_horiz**2)
                
                # Add unit direction
                sum_vert += offset_vert / distance
                sum_horiz += offset_horiz / distance

    # 4. Convert directional sum to a unit vector direction
    dir_length = np.sqrt(sum_vert**2 + sum_horiz**2)
    
    if dir_length > 0:
        unit_vert = sum_vert / dir_length
        unit_horiz = sum_horiz / dir_length
    else:
        unit_vert = 0.0
        unit_horiz = 0.0

    # 5. Scale unit vector by character magnitude normalized by 40
    # (40 represents max expected pixel fill density for standard font)
    final_vert = unit_vert * (pixel_count / 40.0)
    final_horiz = unit_horiz * (pixel_count / 40.0)

    # Save to database
    char_mags_list.append(pixel_count)
    char_vectors.append((final_horiz, final_vert))

    print_progress_bar(idx + 1, total_chars, prefix='Char DB:', suffix='Complete')

# Convert lists to NumPy arrays
char_vec_matrix = np.array(char_vectors)   # Shape: (94, 2) -> [Horiz, Vert]
char_mags_array = np.array(char_mags_list)


# ================================================================
# STEP 2: SELECT INPUT IMAGE (DEFAULTS TO DOWNLOADS)
# ================================================================
root = tk.Tk()
root.withdraw()  # Hide root GUI window

downloads_folder = os.path.expanduser("~/Downloads")
print(f"\nOpening file picker (default folder: {downloads_folder})...")

image_path = filedialog.askopenfilename(
    title="Select Image for ASCII Processing",
    initialdir=downloads_folder,
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff")]
)

if not image_path:
    print("No image selected. Exiting.")
    sys.exit()

print(f"Loaded File: {os.path.basename(image_path)}")

# Open image and convert to RGB
img = Image.open(image_path).convert("RGB")
img_w, img_h = img.size


# ================================================================
# STEP 3: CALCULATE GRID DIMENSIONS (X and Y)
# ================================================================
try:
    grid_x = int(input("\nEnter number of HORIZONTAL blocks (X): "))
except ValueError:
    print("Invalid input! Please enter a whole integer.")
    sys.exit()

# Font aspect ratio correction: terminal characters are twice as tall as they are wide
font_aspect_ratio = 0.5
grid_y = max(1, int(round(grid_x * (img_h / img_w) * font_aspect_ratio)))

block_w = img_w / grid_x
block_h = img_h / grid_y

print(f"Calculated Vertical Blocks (Y): {grid_y} (Aspect ratio preserved)")
print(f"Image Resolution: {img_w}x{img_h} px | Grid: {grid_x}x{grid_y} blocks | Block Size: ~{block_w:.1f}x{block_h:.1f} px\n")

img_array = np.array(img, dtype=float)

# Arrays for analytics plotting
raw_pixel_mags = np.sqrt(np.sum(img_array**2, axis=2)).ravel()
block_mag_grid = np.zeros((grid_y, grid_x))
selected_char_mag_grid = np.zeros((grid_y, grid_x))
block_vec_x_grid = np.zeros((grid_y, grid_x))
block_vec_y_grid = np.zeros((grid_y, grid_x))
selected_indices = []


# ================================================================
# STEP 4: PROCESS IMAGE BLOCKS & MATCH BY EUCLIDEAN DISTANCE
# ================================================================
ascii_rows = []

print("Processing image blocks...")
for row in range(grid_y):
    row_chars = []
    
    # Calculate pixel start/end rows for current block
    r_start = int(round(row * block_h))
    r_end = int(round((row + 1) * block_h))
    block_pixel_h = r_end - r_start

    for col in range(grid_x):
        # Calculate pixel start/end columns for current block
        c_start = int(round(col * block_w))
        c_end = int(round((col + 1) * block_w))
        block_pixel_w = c_end - c_start

        # Extract current image block slice
        block = img_array[r_start:r_end, c_start:c_end]

        if block.size == 0:
            row_chars.append(" ")
            selected_indices.append(0)
            continue

        # 1. Compute pixel RGB brightness magnitudes: sqrt(R^2 + G^2 + B^2)
        pixel_mags = np.sqrt(np.sum(block**2, axis=2))
        avg_block_mag = np.mean(pixel_mags)
        block_mag_grid[row, col] = avg_block_mag

        # 2. Block Center coordinates
        center_y = (block_pixel_h - 1) / 2.0
        center_x = (block_pixel_w - 1) / 2.0

        # 3. Sum weighted directional vectors for all pixels in block
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

        # 4. Normalize block direction to unit vector
        block_dir_len = np.sqrt(block_sum_vert**2 + block_sum_horiz**2)
        
        if block_dir_len > 0:
            unit_block_vert = block_sum_vert / block_dir_len
            unit_block_horiz = block_sum_horiz / block_dir_len
        else:
            unit_block_vert = 0.0
            unit_block_horiz = 0.0

        # 5. Normalize block brightness between 0.0 and 1.0
        # (Max possible RGB pixel norm is sqrt(255^2 + 255^2 + 255^2) = 441.67)
        avg_brightness_norm = avg_block_mag / 441.67

        # 6. Final Block Vector = (Unit Direction) * (Normalized Brightness)
        block_vec_horiz = unit_block_horiz * avg_brightness_norm
        block_vec_vert = unit_block_vert * avg_brightness_norm

        block_vec_x_grid[row, col] = block_vec_horiz
        block_vec_y_grid[row, col] = block_vec_vert

        block_vector = np.array([block_vec_horiz, block_vec_vert])

        # 7. FIND CLOSEST CHARACTER USING EUCLIDEAN DISTANCE
        # distance = sqrt((char_x - block_x)^2 + (char_y - block_y)^2)
        differences = char_vec_matrix - block_vector
        distances = np.sqrt(np.sum(differences**2, axis=1))
        
        best_char_idx = np.argmin(distances)

        selected_indices.append(best_char_idx)
        selected_char_mag_grid[row, col] = char_mags_array[best_char_idx]
        row_chars.append(data[best_char_idx])

    ascii_rows.append("".join(row_chars))
    print_progress_bar(row + 1, grid_y, prefix='Slicing Image:', suffix='Complete')

ascii_art_str = "\n".join(ascii_rows)

# Save text file (Raw ASCII format)
with open("ascii_art.txt", "w", encoding="utf-8") as f:
    f.write(ascii_art_str)


# ================================================================
# DASHBOARD 1: VECTOR & SPATIAL FIELD DIAGNOSTICS
# ================================================================
print("\nOpening Dashboard 1: Vector & Spatial Field Diagnostics...")

fig_dash1 = plt.figure(figsize=(16, 12))
fig_dash1.suptitle("Dashboard 1: Vector & Spatial Field Diagnostics", fontsize=16, fontweight='bold')

# Subplot 1: Character Feature Space Map
ax1 = fig_dash1.add_subplot(2, 2, 1)
x_vals = [v[0] for v in char_vectors]
y_vals = [v[1] for v in char_vectors]

ax1.scatter(x_vals, y_vals, color='red', alpha=0.6, edgecolors='black', s=40)
for char, x_p, y_p in zip(data, x_vals, y_vals):
    lbl = "' '" if char == " " else char.replace('$', r'\$')
    ax1.annotate(lbl, (x_p, y_p), fontsize=8, ha='center', va='bottom')

ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.set_title("1. Character Feature Space Map", fontsize=11, fontweight='bold')
ax1.set_xlabel("Horizontal Displacement Sum")
ax1.set_ylabel("Vertical Displacement Sum")
ax1.grid(True, linestyle=':', alpha=0.5)

# Subplot 2: Original Input Image (Fixed for Vertical Photos)
ax2 = fig_dash1.add_subplot(2, 2, 2)
ax2.imshow(img, aspect='equal')
ax2.set_title(f"2. Original Input Image ({img_w}x{img_h})", fontsize=11, fontweight='bold')
ax2.axis('off')

# Subplot 3: Block RGB Energy Heatmap
ax3 = fig_dash1.add_subplot(2, 2, 3)
im3 = ax3.imshow(block_mag_grid, cmap='magma', aspect='auto')
fig_dash1.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
ax3.set_title("3. Block RGB Magnitude Heatmap", fontsize=11, fontweight='bold')
ax3.set_xlabel("Grid Column")
ax3.set_ylabel("Grid Row")

# Subplot 4: Image Block Vector Field (Quiver)
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
plt.savefig("ascii_spatial_diagnostics.png", dpi=300)


# ================================================================
# DASHBOARD 2: MAGNITUDE & FREQUENCY ANALYTICS
# ================================================================
print("Opening Dashboard 2: Magnitude & Frequency Analytics...")

fig_dash2 = plt.figure(figsize=(16, 12))
fig_dash2.suptitle("Dashboard 2: Magnitude & Glyph Frequency Analytics", fontsize=16, fontweight='bold')

# Subplot 1: Raw Image vs Block Magnitude Density
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

# Subplot 2: Top 20 Most Used Glyphs
ax2_2 = fig_dash2.add_subplot(2, 2, 2)
char_counts = np.bincount(selected_indices, minlength=len(data))
top_indices = np.argsort(char_counts)[::-1][:20]

top_chars = [f"'{data[i].replace('$', r'\$')}'" if data[i] != " " else "'SPACE'" for i in top_indices]
top_counts = char_counts[top_indices]

bars = ax2_2.bar(top_chars, top_counts, color='purple', alpha=0.7, edgecolor='black')
ax2_2.set_title("2. Top 20 Most Frequently Used Glyphs", fontsize=11, fontweight='bold')
ax2_2.set_xlabel("ASCII Character Glyph")
ax2_2.set_ylabel("Occurrence Count")
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        ax2_2.text(bar.get_x() + bar.get_width()/2.0, yval + (0.01 * max(top_counts)), int(yval), ha='center', va='bottom', fontsize=8)

ax2_2.grid(True, linestyle=':', alpha=0.5, axis='y')

# Subplot 3: Block Magnitude vs Matched Character Fill Count
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

# Subplot 4: Full ASCII Spectrum Utilization Grid
ax2_4 = fig_dash2.add_subplot(2, 2, 4)
padded_counts = np.pad(char_counts, (0, 100 - len(char_counts)), mode='constant')
freq_matrix = padded_counts.reshape(10, 10)

im4 = ax2_4.imshow(freq_matrix, cmap='YlOrRd')
fig_dash2.colorbar(im4, ax=ax2_4, fraction=0.046, pad=0.04)
ax2_4.set_title("4. ASCII Spectrum Utilization Grid (32-126)", fontsize=11, fontweight='bold')
ax2_4.set_xlabel("Char Index Offset")
ax2_4.set_ylabel("Char Row Group")

plt.tight_layout()
plt.savefig("ascii_magnitude_analytics.png", dpi=300)


# ================================================================
# FIGURE 3: HIGH-RES RENDERED ASCII IMAGE
# ================================================================
dynamic_fontsize = max(2, min(10, int(600 / grid_x)))
fig_ascii, ax_ascii = plt.subplots(figsize=(12, 12), facecolor='black')
ax_ascii.set_facecolor('black')

# Escaped $ -> \$ specifically for Matplotlib text rendering
ax_ascii.text(
    0.5, 0.5, ascii_art_str.replace('$', r'\$'), 
    color='#00FF66', 
    fontfamily='monospace', 
    fontsize=dynamic_fontsize, 
    ha='center', va='center', 
    transform=ax_ascii.transAxes
)
ax_ascii.axis('off')
plt.title("High-Resolution ASCII Render", color='white', fontsize=14, pad=10)
plt.savefig("ascii_art_render.png", dpi=300, bbox_inches='tight', facecolor='black')

plt.show()

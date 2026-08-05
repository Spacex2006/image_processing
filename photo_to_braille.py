import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# ================================================================
# BRAILLE SUB-PIXEL MAPPER (WITH THEME TOGGLE)
# ================================================================
def convert_block_to_braille(sub_block_4x2, threshold=120, mode="dark"):
    """
    Takes a 4x2 array of RGB magnitude values.
    
    - Dark Mode:  Bright pixels (> threshold) turn dots ON (white dots on dark background).
    - Light Mode: Dark pixels (< threshold) turn dots ON (black ink on light background).
    """
    bit_map = [
        [0x01, 0x08],  # Row 0: Dot 1, Dot 4
        [0x02, 0x10],  # Row 1: Dot 2, Dot 5
        [0x04, 0x20],  # Row 2: Dot 3, Dot 6
        [0x40, 0x80]   # Row 3: Dot 7, Dot 8
    ]

    braille_code = 0x2800  # Base Unicode address for empty Braille cell

    for r in range(4):
        for c in range(2):
            val = sub_block_4x2[r, c]
            
            # Check condition based on theme
            if mode == "dark" and val > threshold:
                braille_code |= bit_map[r][c]
            elif mode == "light" and val < threshold:
                braille_code |= bit_map[r][c]

    return chr(braille_code)


# ================================================================
# STEP 1: FILE SELECTOR
# ================================================================
root = tk.Tk()
root.withdraw()

downloads_folder = os.path.expanduser("~/Downloads")
print(f"Opening file picker (default folder: {downloads_folder})...")

image_path = filedialog.askopenfilename(
    title="Select Image for Braille Conversion",
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
# STEP 2: USER INPUT (WIDTH AND THEME)
# ================================================================
try:
    grid_x = int(input("\nEnter HORIZONTAL width (Recommended: 80 to 120): "))
except ValueError:
    grid_x = 100

print("\nSelect Display Mode:")
print("1. Dark Mode  (For Terminal, VS Code Dark, Pop Console)")
print("2. Light Mode (For White Text Editors, PDF, Light Themes)")
print("3. Generate BOTH (Saves two separate .txt files)")

theme_choice = input("Enter choice (1, 2, or 3) [Default 1]: ").strip()

font_aspect_ratio = 0.5
grid_y = max(1, int(round(grid_x * (img_h / img_w) * font_aspect_ratio)))

sub_pixel_w = grid_x * 2
sub_pixel_h = grid_y * 4

print(f"\nGrid: {grid_x} x {grid_y} characters | Sub-Pixel Resolution: {sub_pixel_w} x {sub_pixel_h} dots\n")


# ================================================================
# STEP 3: RESIZE & SAMPLING
# ================================================================
img_resized = img.resize((sub_pixel_w, sub_pixel_h), Image.Resampling.LANCZOS)
img_np = np.array(img_resized, dtype=float)

# Compute RGB magnitude norms: sqrt(R^2 + G^2 + B^2)
mags = np.sqrt(np.sum(img_np**2, axis=2))
adaptive_threshold = np.mean(mags) * 0.85


# ================================================================
# STEP 4: GENERATE BRAILLE TEXT
# ================================================================
def build_art(mode_name):
    rows = []
    for char_row in range(grid_y):
        row_str = ""
        for char_col in range(grid_x):
            r_start = char_row * 4
            c_start = char_col * 2
            
            sub_block = mags[r_start:r_start + 4, c_start:c_start + 2]
            row_str += convert_block_to_braille(sub_block, threshold=adaptive_threshold, mode=mode_name)
        rows.append(row_str)
    return "\n".join(rows)


# Save files based on user choice
if theme_choice in ["1", ""]:
    dark_art = build_art("dark")
    with open("braille_art_dark.txt", "w", encoding="utf-8") as f:
        f.write(dark_art)
    print("Saved: braille_art_dark.txt")
    print("\n" + "=" * grid_x + "\n" + dark_art)

elif theme_choice == "2":
    light_art = build_art("light")
    with open("braille_art_light.txt", "w", encoding="utf-8") as f:
        f.write(light_art)
    print("Saved: braille_art_light.txt")
    print("\n" + "=" * grid_x + "\n" + light_art)

else:
    dark_art = build_art("dark")
    light_art = build_art("light")
    
    with open("braille_art_dark.txt", "w", encoding="utf-8") as f:
        f.write(dark_art)
    with open("braille_art_light.txt", "w", encoding="utf-8") as f:
        f.write(light_art)
        
    print("Saved BOTH files: braille_art_dark.txt & braille_art_light.txt")
    print("\n--- DARK MODE PREVIEW ---")
    print(dark_art)

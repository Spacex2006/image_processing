import time
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import queue
from scipy.stats import gaussian_kde

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='█', start_time=None):
    """
    Renders a terminal progress bar with dynamic ETA/ETC calculation.
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    eta_str = "--:--"
    if start_time and iteration > 0:
        elapsed = time.time() - start_time
        rate = iteration / elapsed
        remaining_sec = (total - iteration) / rate
        
        mins, secs = divmod(int(remaining_sec), 60)
        hrs, mins = divmod(mins, 60)
        
        if hrs > 0:
            eta_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        else:
            eta_str = f"{mins:02d}:{secs:02d}"

    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix} [ETC: {eta_str}]')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

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

image_path = get_file_path()

if not image_path:
    print("[Error] No file selected. Exiting.")
    exit()

print("\n--- Image Processing Initialization ---")
scale_factor = int(input("Enter downsampling ratio (e.g., 1 for full resolution, 8 for fast preview): "))

# --- START OVERALL TIMER ---
total_start_time = time.time()

# 1. TIMING: Load & Resize
t0 = time.time()
try:
    img = Image.open(image_path)
    sandbox_array = np.array(img)[::scale_factor, ::scale_factor]
    print(f"Working Resolution: {sandbox_array.shape[1]}x{sandbox_array.shape[0]}")
except FileNotFoundError:
    print("[Error] Image not found.")
    exit()
time_load = time.time() - t0

# Setup interactive live plotting
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(sandbox_array)
plt.axis('off')

height, width, channels = sandbox_array.shape

arr = sandbox_array[:, :, :3].astype(int)

# 2. TIMING: RGB Vector Engine
t1 = time.time()
print("\nRunning RGB Vector Engine...")

kms = np.zeros((height, width, 2))
magn = np.zeros((height, width))
magnitude = []

total_pixels = height * width
pixel_count = 0
vec_start_time = time.time()

for row in range(height):
    for col in range(width):
        vec_r = np.array([0.0, 0.0])
        vec_g = np.array([0.0, 0.0])
        vec_b = np.array([0.0, 0.0])
        
        curr_R, curr_G, curr_B = arr[row, col] 
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                n_row = row + dr
                n_col = col + dc
                
                if 0 <= n_row < height and 0 <= n_col < width:
                    nR, nG, nB = arr[n_row, n_col]
                    mag_dist = np.sqrt(dr*dr + dc*dc)
                    
                    diff_R = nR - curr_R
                    diff_G = nG - curr_G
                    diff_B = nB - curr_B
                    
                    vec_r[0] += diff_R * (dr / mag_dist)
                    vec_r[1] += diff_R * (dc / mag_dist)
                    
                    vec_g[0] += diff_G * (dr / mag_dist)
                    vec_g[1] += diff_G * (dc / mag_dist)
                    
                    vec_b[0] += diff_B * (dr / mag_dist)
                    vec_b[1] += diff_B * (dc / mag_dist)
                    
        kms[row, col] = (vec_r + vec_g + vec_b) / 3
        
        mag_R = vec_r[0]**2 + vec_r[1]**2
        mag_G = vec_g[0]**2 + vec_g[1]**2
        mag_B = vec_b[0]**2 + vec_b[1]**2
        
        total_mag = np.sqrt(mag_R + mag_G + mag_B)
        
        magn[row, col] = total_mag
        magnitude.append((total_mag, row, col))
        
        pixel_count += 1
        # Update progress bar every 1000 pixels or at completion to prevent stdout lag
        if pixel_count % 1000 == 0 or pixel_count == total_pixels:
            print_progress_bar(pixel_count, total_pixels, prefix='Vector Calculations:', suffix='Complete', length=30, start_time=vec_start_time)

time_vector_engine = time.time() - t1
print(f"Vector Engine Finished in {time_vector_engine:.2f}s")

# 3. TIMING: Sorting
t2 = time.time()
print("Sorting seed candidates by magnitude... ", end="", flush=True)
magnitude.sort(key=lambda item: item[0], reverse=True)
time_sort = time.time() - t2
print(f"Done ({time_sort:.3f}s)")

# --- TRACKING SETUP ---
flat_magnitudes = magn.flatten()
mag_mean = np.mean(flat_magnitudes)
mag_std = np.std(flat_magnitudes)
mag_max = np.max(flat_magnitudes)
global_mean_magnitude = np.mean(magn) 
a = np.percentile(flat_magnitudes, 99)
b = np.percentile(flat_magnitudes, 75)
active_threshold = a
low_threshold = b

print(f"\nGlobal Average Magnitude: {global_mean_magnitude:.2f}")
print(f"Dynamic Cutoff Threshold Set To: {active_threshold:.2f}")
maxi = magnitude[0][0]

# Create a Pitch-black canvas of the exact same size
edge_canvas = np.zeros((height, width, 3), dtype=np.uint8)

print("\n=== Launching Line Crawl Exploration ===")

idk = np.zeros((height, width), dtype=bool)
visited = np.zeros((height, width), dtype=bool)
temp = np.zeros((height, width), dtype=bool)
red = [255, 255, 255]

segment_lengths = []

# Estimate eligible seed count for the line crawl progress bar
eligible_seeds_count = sum(1 for mag, _, _ in magnitude if mag >= active_threshold)

# 4. TIMING: The Crawl
t3 = time.time()
x = 0
y = 0
crawl_start_time = time.time()

for seed_idx, (seed_mag, r_seed, c_seed) in enumerate(magnitude):
    if seed_mag < active_threshold:
        # Fill progress bar to 100% on early termination
        print_progress_bar(eligible_seeds_count, eligible_seeds_count, prefix='Line Crawling:      ', suffix='Complete', length=30, start_time=crawl_start_time)
        print(f"\n--> Halting global search: Magnitude ({seed_mag:.2f}) dropped below Target Threshold ({active_threshold:.2f})")
        break
        
    if visited[r_seed, c_seed]:
        print_progress_bar(min(seed_idx + 1, eligible_seeds_count), eligible_seeds_count, prefix='Line Crawling:      ', suffix='Complete', length=30, start_time=crawl_start_time)
        continue
        
    y += 1
    
    q = queue.Queue()
    forbidden = queue.Queue()
    q.put((r_seed, c_seed))
    pixels_traced_in_segment = 0
    
    while not q.empty():
        x += 1
        r, c = q.get()
        
        if idk[r, c] or temp[r, c]:
            continue
            
        temp[r, c] = True
        forbidden.put((r, c))
        pixels_traced_in_segment += 1
        
        curr_v = kms[r, c]
        v_y = curr_v[0]
        v_x = curr_v[1]
        
        if v_y == 0 and v_x == 0:
            continue
            
        p_y = -v_x
        p_x = v_y
        
        if abs(p_x) > 2 * abs(p_y):
            dr, dc = 0, 1 if p_x > 0 else -1
        elif abs(p_y) > 2 * abs(p_x):
            dr, dc = 1 if p_y > 0 else -1, 0
        else:
            dr = 1 if p_y > 0 else -1
            dc = 1 if p_x > 0 else -1
            
        n1_r, n1_c = r + dr, c + dc
        n2_r, n2_c = r - dr, c - dc
        
        mag1 = magn[n1_r, n1_c] if (0 <= n1_r < height and 0 <= n1_c < width) else -1
        mag2 = magn[n2_r, n2_c] if (0 <= n2_r < height and 0 <= n2_c < width) else -1
        
        if mag1 >= low_threshold and 0 <= n1_r < height and 0 <= n1_c < width and not temp[n1_r, n1_c] and not idk[n1_r, n1_c]:
            q.put((n1_r, n1_c))
        if mag2 >= low_threshold and 0 <= n2_r < height and 0 <= n2_c < width and not temp[n2_r, n2_c] and not idk[n2_r, n2_c]:
            q.put((n2_r, n2_c))
            
    if pixels_traced_in_segment > 100:
        while not forbidden.empty():
            r, c = forbidden.get()
            for repel_r in range(-5, 6):
                for repel_c in range(-5, 6):
                    n_r, n_c = r + repel_r, c + repel_c
                    if 0 <= n_r < height and 0 <= n_c < width:
                        visited[n_r, n_c] = True
                        if repel_r == 0 and repel_c == 0:
                            idk[n_r, n_c] = True
                            sandbox_array[r, c] = np.array(red) * ((magn[r, c] / maxi) ** 0.5)
                            edge_canvas[r, c] = np.array(red) * ((magn[r, c] / maxi) ** 0.5)
                            
        segment_lengths.append(pixels_traced_in_segment)

    im.set_data(sandbox_array)
    plt.draw()
    plt.pause(0.001)
    
    # Update progress bar per processed seed
    print_progress_bar(min(seed_idx + 1, eligible_seeds_count), eligible_seeds_count, prefix='Line Crawling:      ', suffix='Complete', length=30, start_time=crawl_start_time)

# =====================================================================
# INJECTED MODULE: VECTOR GRADIENT MAGNITUDE ANALYSIS
# =====================================================================
flat_magnitudes = magn.flatten()
mag_mean = np.mean(flat_magnitudes)
mag_std = np.std(flat_magnitudes)
mag_max = np.max(flat_magnitudes)

p50 = np.percentile(flat_magnitudes, 50)
p90 = np.percentile(flat_magnitudes, 90)
p99 = np.percentile(flat_magnitudes, 99)

print("\n" + "="*40)
print("     RAW GRADIENT MAGNITUDE PROFILE")
print("="*40)
print(f"Total Array Pixels Evaluated    : {len(flat_magnitudes)}")
print(f"Mean Vector Magnitude Force     : {mag_mean:.2f}")
print(f"Standard Deviation (σ)          : {mag_std:.2f}")
print(f"Peak Gradient Contrast Force    : {mag_max:.2f}")
print(f"Median Detail Floor (50th %ile) : {p50:.2f}")
print(f"True Boundary Filter (90th %ile): {p90:.2f}")
print(f"High-Energy Signals  (99th %ile): {p99:.2f}")
print("="*40 + "\n")

time_crawl = time.time() - t3
total_time = time.time() - total_start_time

print("="*40)
print("     PERFORMANCE DIAGNOSTICS")
print("="*40)
print(f"Image Load & Resize : {time_load:.3f} seconds")
print(f"RGB Vector Engine   : {time_vector_engine:.3f} seconds")
print(f"Seed Sorting        : {time_sort:.3f} seconds")
print(f"Edge Crawling       : {time_crawl:.3f} seconds")
print("-" * 40)
print(f"TOTAL EXECUTION TIME: {total_time:.3f} seconds")
print("="*40)

plt.ioff() 

# --- WINDOW 1: Overlaid ---
im.set_data(sandbox_array)
ax.set_title("Result 1: Overlaid on Original Photo")

# --- WINDOW 2: Edge Map ---
fig_edges, ax_edges = plt.subplots()
ax_edges.imshow(edge_canvas)
ax_edges.axis('off')
ax_edges.set_title("Result 2: Pure Edge Map")

# --- WINDOW 3: GRADIENT MAGNITUDE PLOT ---
fig_mag, ax_mag = plt.subplots(figsize=(8, 5))
ax_mag.hist(flat_magnitudes, bins=50, density=True, alpha=0.6, color='royalblue', edgecolor='black', label='Pixel Energies')

if mag_std > 0:
    sample_size = min(20000, len(flat_magnitudes))
    sampled_mags = np.random.choice(flat_magnitudes, sample_size, replace=False)
    kde_mag = gaussian_kde(sampled_mags)
    x_axis_mag = np.linspace(0, mag_max, 500)
    ax_mag.plot(x_axis_mag, kde_mag(x_axis_mag), color='darkblue', linewidth=2, label='Energy Smoothing (KDE)')

ax_mag.axvline(mag_mean, color='orange', linestyle='dashed', linewidth=2, label=f'Mean Force ({mag_mean:.1f})')
ax_mag.axvline(active_threshold, color='red', linestyle='dashdot', linewidth=2, label=f'Active Cutoff ({active_threshold:.1f})')

ax_mag.set_title("Image Gradient Force Probability Density Graph", fontsize=12, fontweight='bold')
ax_mag.set_xlabel("Vector Acceleration Force Length (Magnitude Scale)", fontsize=10)
ax_mag.set_ylabel("Probability Density", fontsize=10)
ax_mag.grid(True, linestyle=':', alpha=0.6)
ax_mag.legend()

# --- WINDOW 4: PATH SEGMENT LENGTH PLOT ---
if len(segment_lengths) > 0:
    lengths_arr = np.array(segment_lengths)
    avg_len = np.mean(lengths_arr)
    std_len = np.std(lengths_arr)
    max_len = np.max(lengths_arr)
    total_segments = len(lengths_arr)
    
    print("\n" + "="*40)
    print("        SEGMENT LENGTH PROFILE")
    print("="*40)
    print(f"Total New Seeds                 : {y}")
    print(f"Total Segments Evaluated        : {x}")
    print(f"Total Unique Segments Generated : {total_segments}")
    print(f"Average Segment Length          : {avg_len:.2f} pixels")
    print(f"Standard Deviation (σ)          : {std_len:.2f} pixels")
    print(f"Longest Unbroken Path Segment   : {max_len} pixels")
    print("="*40)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(lengths_arr, bins=max(10, min(50, max_len)), density=True, alpha=0.6, color='crimson', edgecolor='black', label='Segment Data')
    
    if total_segments > 1 and std_len > 0:
        kde_len = gaussian_kde(lengths_arr)
        x_axis_len = np.linspace(0, max_len, 500)
        ax2.plot(x_axis_len, kde_len(x_axis_len), color='darkred', linewidth=2, label='Density Smoothing (KDE)')
    
    ax2.axvline(avg_len, color='blue', linestyle='dashed', linewidth=2, label=f'Mean ({avg_len:.1f})')
    ax2.set_title("Segment Length Probability Density Graph", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Segment Length (Number of Pixels Plotted)", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
else:
    print("\n[Warning] No segments broke past your length criteria threshold configuration.")

plt.show()
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

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

# Standardized CLI prompt
print("\n--- Image Processing Initialization ---")
scale_factor = int(input("Enter downsampling ratio (e.g., 1 for full resolution, 8 for fast preview): "))

try:
    img = Image.open(image_path)
    sandbox_array = np.array(img)[::scale_factor, ::scale_factor]
    print(f"Working Resolution: {sandbox_array.shape[1]}x{sandbox_array.shape[0]}")
except FileNotFoundError:
    print("[Error] Image not found.")
    exit()

# Setup interactive live plotting
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(sandbox_array)
plt.axis('off')

height, width, channels = sandbox_array.shape
arr = np.zeros((height, width))

# Pass 1: Convert image to grayscale for intensity mapping
for row in range(height):
    for col in range(0, width):
        x = sandbox_array[row, col].astype(int)
        avg = (x[0] + x[1] + x[2]) // 3  # Integer division is cleaner here
        sandbox_array[row, col] = [avg, avg, avg]
        
    if row % 4 == 0:
        im.set_data(sandbox_array)
        plt.draw()
        plt.pause(0.001)

im.set_data(sandbox_array)
plt.draw()
plt.pause(0.001)

# Pass 2: Build cumulative cost matrix (Dynamic Programming)
for row in range(height):
    for col in range(width):
        x = sandbox_array[row, col].astype(int)
        if row == 0:
            arr[row, col] = x[0]
        else:
            c_start = max(0, col - 1)
            c_end = min(width, col + 2) 
            arr[row, col] = np.min(arr[row - 1, c_start:c_end]) + x[0]
            
    if row % 4 == 0:
        im.set_data(sandbox_array)
        plt.draw()
        plt.pause(0.001)

# Pass 3: Backtrack to find the minimum intensity path
mi = arr[height - 1, 0]
minid = 0
for col in range(width):
    if arr[height - 1, col] < mi:
        mi = arr[height - 1, col]
        minid = col

white = [255, 255, 255]
sandbox_array[height - 1, minid] = white

for row in reversed(range(height - 1)):
    # FIXED: Added '-1' so it checks the left diagonal, not just center/right
    c_start = max(0, minid - 1) 
    c_end = min(width, minid + 2)
    
    # FIXED: Look at the current 'row' in the cost matrix, not 'row - 1'
    row_slice = arr[row, c_start:c_end] 
    
    # FIXED: Update minid so the next iteration starts from the new column
    minid = c_start + np.argmin(row_slice) 
    
    sandbox_array[row, minid] = white

im.set_data(sandbox_array)
plt.draw()
plt.pause(0.001)

# Keep final result open
plt.ioff()
plt.show()

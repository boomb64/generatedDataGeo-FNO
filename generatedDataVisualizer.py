"""
DATA PIPELINE VISUALIZER (With Zoom)
------------------------------------
Function:
    Side-by-side comparison of Raw Input (.dat) vs. Processed Tensor (.npy).

    * LEFT IMAGE: Raw Unstructured Data (scatter plot).
    * RIGHT IMAGE: Processed Structured Tensor (heatmap).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
# 1. Path to ONE raw file you want to inspect
RAW_FILE_PATH = r"C:\Users\Daniel\PycharmProjects\generatedDataFNO\data\cases\NACA0015_0deg_section_Re2003630.dat"

# 2. Path to the folder containing your processed .npy files
PROCESSED_DIR = r"C:\Users\Daniel\PycharmProjects\generatedDataFNO\processed_full_physics"

# 3. ZOOM CONTROLS (Set to None to see the full huge domain)
#    Adjust these values to frame your wing specifically.
#    Based on your data (X starts around 2.0?), you might need to shift these.
ZOOM_X_LIM = [-0.5, 2.5]
ZOOM_Y_LIM = [-1.0, 1.0]

# ================= HELPER: PARSE RAW .DAT =================
def load_raw_dat(filepath):
    print(f"Loading RAW file: {os.path.basename(filepath)}...")
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if "ZONE" in line:
                match = re.search(r'N=(\d+)', line)
                if match:
                    n_nodes = int(match.group(1))
                    skiprows = i + 1
                    break

    # Read data
    df = pd.read_csv(filepath, skiprows=skiprows, nrows=n_nodes, sep='\s+', header=None, engine='python')
    vals = df.values
    x = vals[:, 0]
    y = vals[:, 1]
    z = vals[:, 2]

    # Auto-detect flat axis
    if (y.max() - y.min()) < 1e-6:
        print(" -> Detected X-Z plane (Y is flat)")
        x_plot, y_plot = x, z
        lbl_x, lbl_y = "X", "Z"
    else:
        print(" -> Detected X-Y plane")
        x_plot, y_plot = x, y
        lbl_x, lbl_y = "X", "Y"

    u_vel = vals[:, 5] # u velocity
    return x_plot, y_plot, u_vel, lbl_x, lbl_y

# ================= HELPER: LOAD PROCESSED .NPY =================
def load_processed_sample(npy_dir, sample_idx=0):
    print(f"Loading PROCESSED tensors from {npy_dir}...")
    try:
        inputs = np.load(os.path.join(npy_dir, "input_full.npy"))
        outputs = np.load(os.path.join(npy_dir, "output_full.npy"))

        # Grab sample 0
        # Input Channels: 0=Def_X, 1=Def_Y ...
        grid_x = inputs[sample_idx, 0, :, :]
        grid_y = inputs[sample_idx, 1, :, :]

        # Output Channels: 0=u ...
        grid_u = outputs[sample_idx, 0, :, :]

        return grid_x, grid_y, grid_u, inputs.shape
    except Exception as e:
        print(f"Error loading .npy files: {e}")
        return None, None, None, None

# ================= MAIN VISUALIZATION =================
def visualize():
    if not os.path.exists(RAW_FILE_PATH):
        print(f"ERROR: Raw file not found at {RAW_FILE_PATH}")
        return

    raw_x, raw_y, raw_u, lbl_x, lbl_y = load_raw_dat(RAW_FILE_PATH)
    proc_x, proc_y, proc_u, shape = load_processed_sample(PROCESSED_DIR, sample_idx=0)

    if proc_x is None: return

    print("\nGenerating Comparison Plot...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: BEFORE (Raw Scatter) ---
    ax1 = axes[0]
    sc1 = ax1.scatter(raw_x, raw_y, c=raw_u, cmap='jet', s=5) # Increased point size slightly
    ax1.set_title(f"BEFORE: Raw Unstructured Data\n({len(raw_x)} points)")
    ax1.set_xlabel(f"Physical {lbl_x}")
    ax1.set_ylabel(f"Physical {lbl_y}")
    ax1.axis('equal')

    # --- APPLY ZOOM TO LEFT IMAGE ---
    if ZOOM_X_LIM: ax1.set_xlim(ZOOM_X_LIM)
    if ZOOM_Y_LIM: ax1.set_ylim(ZOOM_Y_LIM)

    plt.colorbar(sc1, ax=ax1, label="U-Velocity")

    # --- PLOT 2: AFTER (Structured Grid) ---
    ax2 = axes[1]
    cm2 = ax2.pcolormesh(proc_x, proc_y, proc_u, cmap='jet', shading='auto')
    ax2.set_title(f"AFTER: Processed Structured Tensor\n(Shape: {shape[2]}x{shape[3]} Grid)")
    ax2.set_xlabel(f"Physical {lbl_x}")
    ax2.set_ylabel(f"Physical {lbl_y}")
    ax2.axis('equal')

    # --- APPLY ZOOM TO RIGHT IMAGE (For comparison) ---
    if ZOOM_X_LIM: ax2.set_xlim(ZOOM_X_LIM)
    if ZOOM_Y_LIM: ax2.set_ylim(ZOOM_Y_LIM)

    plt.colorbar(cm2, ax=ax2, label="U-Velocity")

    plt.tight_layout()
    plt.show()
    print("Done!")

if __name__ == "__main__":
    visualize()
"""
FINAL DATA PROCESSOR (C-GRID GENERATOR)
---------------------------------------
1. Generates a Body-Fitted "C-Grid".
2. SCALED 2X: Chord=2.0 to match the physical FlightStream data.
3. Rotates the C-Grid to match the specific AoA of the data.
4. Interpolates physics cleanly. No artificial masking required.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys

# ================= CONFIGURATION =================
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cases")
SAVE_DIR = os.path.join(PROJECT_ROOT, "processed_full_physics")
FILE_PATTERN = "*.dat"

TARGET_H = 221
TARGET_W = 51

NORM_FACTOR = 10.0
RE_MIN, RE_MAX = 1e5, 1e7
AOA_MIN, AOA_MAX = -20.0, 20.0

# Geometry Scaling
NACA_THICKNESS = 0.15
CHORD_LENGTH = 2.0
ROTATION_PIVOT_X = 0.0
ROTATION_PIVOT_Y = 0.0

# ---------------------------------------------------------
# Geometry Helpers
# ---------------------------------------------------------
# ---------------------------------------------------------
# Geometry Helpers & Improved Mesh Generation
# ---------------------------------------------------------
def naca4(x, t=0.15, c=1.0):
    xn = x / c
    term1 = 0.2969 * np.sqrt(np.maximum(0, xn))
    term2 = -0.1260 * xn
    term3 = -0.3516 * xn ** 2
    term4 = +0.2843 * xn ** 3
    term5 = -0.1015 * xn ** 4
    return 5 * t * c * (term1 + term2 + term3 + term4 + term5)


def calculate_wall_spacing(Re_max, chord, target_y_plus=5.0):
    """Calculates the physical first cell height required for a target y+"""
    cf = 0.058 * (Re_max ** -0.2)
    friction_velocity_ratio = np.sqrt(cf / 2.0)
    delta_y = (target_y_plus * chord) / (Re_max * friction_velocity_ratio)
    return delta_y


def geometric_stretch(n_points, initial_spacing, total_distance, max_growth_rate=1.2):
    """
    Creates a normalized 1D spacing array [0, 1] focused heavily on the wall,
    respecting a maximum growth rate between adjacent cells.
    """
    s = np.zeros(n_points)
    current_dist = 0.0
    current_step = initial_spacing

    for i in range(1, n_points):
        current_dist += current_step
        s[i] = current_dist
        # Grow the step size, but cap the growth rate for quality
        current_step *= max_growth_rate

        # If the geometric progression overshot or undershot the total distance,
    # we normalize it so it cleanly maps from 0 to 1
    s = s / s[-1]
    return s


def generate_c_grid(n_stream=221, n_normal=51, t=0.15, c=1.0, max_re=1e7):
    WAKE_LENGTH = 20.0
    OUTER_RADIUS = 20.0

    n_airfoil = n_stream // 2
    if n_airfoil % 2 == 0: n_airfoil += 1
    n_wake = (n_stream - n_airfoil) // 2

    # Airfoil Surface Points (Cosine spacing clusters points at leading/trailing edges)
    beta = np.linspace(0, np.pi, n_airfoil // 2 + 1)
    x_half = 0.5 * (1 - np.cos(beta)) * c
    y_half = naca4(x_half, t, c)

    air_x = np.concatenate([x_half[::-1][:-1], x_half])
    air_y = np.concatenate([-y_half[::-1][:-1], y_half])

    # Wake Surface Points (Quadratic spacing to slowly grow away from airfoil)
    t_vals = np.linspace(0, 1, n_wake + 1)[1:]
    wake_dist = (t_vals ** 2.0) * WAKE_LENGTH
    x_wake = c + wake_dist

    inner_x = np.concatenate([x_wake[::-1], air_x, x_wake])
    inner_y = np.concatenate([np.zeros(n_wake), air_y, np.zeros(n_wake)])

    # Outer Boundary Points
    outer_x, outer_y = np.zeros_like(inner_x), np.zeros_like(inner_y)
    outer_x[:n_wake] = x_wake[::-1]
    outer_y[:n_wake] = -OUTER_RADIUS
    outer_x[-n_wake:] = x_wake
    outer_y[-n_wake:] = OUTER_RADIUS

    n_arc = len(air_x)
    theta = np.linspace(1.5 * np.pi, 0.5 * np.pi, n_arc)
    outer_x[n_wake:-n_wake] = c + OUTER_RADIUS * np.cos(theta)
    outer_y[n_wake:-n_wake] = OUTER_RADIUS * np.sin(theta)

    # --- IMPROVED WALL-NORMAL SPACING ---
    # Target y+ of 5 based on max Reynolds to ensure boundary layer is captured
    dy_wall = calculate_wall_spacing(max_re, c, target_y_plus=5.0)

    # Generate stretching mapped from 0 to 1
    s = geometric_stretch(n_normal, initial_spacing=dy_wall, total_distance=OUTER_RADIUS, max_growth_rate=1.2)

    grid_x, grid_y = np.zeros((n_stream, n_normal)), np.zeros((n_stream, n_normal))

    for i in range(n_stream):
        grid_x[i, :] = inner_x[i] * (1 - s) + outer_x[i] * s
        grid_y[i, :] = inner_y[i] * (1 - s) + outer_y[i] * s

    return grid_x, grid_y

def get_metadata(filename):
    re_m = re.search(r'Re(\d+)', filename)
    aoa_m = re.search(r'_([m-]?\d+(?:\.\d+)?)deg', filename)
    if not re_m or not aoa_m: return None, None
    re_val = float(re_m.group(1))
    s = aoa_m.group(1)

    # FIX: Properly extract positive and negative numbers
    if s.startswith('m'):
        aoa_val = -float(s[1:])
    else:
        aoa_val = float(s) # Removed the errant '-' sign here

    return re_val, aoa_val

def parse_header(filepath):
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if "ZONE" in line:
                m = re.search(r'N=(\d+)', line)
                if m: return int(m.group(1)), i + 1
    return None, None

def check_grid_quality(gx, gy):
    v1_x, v1_y = gx[1:, :-1] - gx[:-1, :-1], gy[1:, :-1] - gy[:-1, :-1]
    v2_x, v2_y = gx[:-1, 1:] - gx[:-1, :-1], gy[:-1, 1:] - gy[:-1, :-1]
    areas = (v1_x * v2_y) - (v1_y * v2_x)
    min_area = np.min(areas)
    if min_area <= 0:
        print(f"[CRITICAL ERROR] Intersecting lines! Min Area: {min_area:.2e}")
        return False
    print(f"[SUCCESS] Grid is valid. Min Area: {min_area:.2e}")
    return True

# ---------------------------------------------------------
# Main Processor
# ---------------------------------------------------------
def process():
    files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
    if not files: return print("No files found.")

    print(f"Found {len(files)} files. Generating Scaled Master C-Grid...")
    g1_flat, g2_flat = generate_c_grid(TARGET_H, TARGET_W, t=NACA_THICKNESS, c=CHORD_LENGTH)
    if not check_grid_quality(g1_flat, g2_flat): return

    batch_x, batch_y = [], []
    count = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            re_val, aoa_val = get_metadata(fname)
            if re_val is None: continue

            # --- 1. ROTATE THE C-GRID TO MATCH DATA ---
            # Automatically inverts the angle based on your requirement: 2deg file -> -2deg rotation
            theta = np.radians(-aoa_val)
            g1_s = g1_flat - ROTATION_PIVOT_X
            g2_s = g2_flat - ROTATION_PIVOT_Y

            g1_rot = g1_s * np.cos(theta) - g2_s * np.sin(theta) + ROTATION_PIVOT_X
            g2_rot = g1_s * np.sin(theta) + g2_s * np.cos(theta) + ROTATION_PIVOT_Y

            # --- 2. LOAD & INTERPOLATE ---
            n_nodes, skiprows = parse_header(fpath)
            df = pd.read_csv(fpath, skiprows=skiprows, nrows=n_nodes, sep='\s+', header=None, engine='python')
            vals = df.values
            src_x, src_y = (vals[:, 0], vals[:, 2]) if vals[:, 1].std() < 1e-6 else (vals[:, 0], vals[:, 1])

            raw_p, raw_u, raw_v = vals[:, 3]/NORM_FACTOR, vals[:, 5]/NORM_FACTOR, vals[:, 6]/NORM_FACTOR
            raw_w = vals[:, 7]/NORM_FACTOR if vals.shape[1] > 7 else np.zeros_like(raw_u)

            pts = np.column_stack((src_x, src_y))
            gu = griddata(pts, raw_u, (g1_rot, g2_rot), method='linear', fill_value=0)
            gv = griddata(pts, raw_v, (g1_rot, g2_rot), method='linear', fill_value=0)
            gw = griddata(pts, raw_w, (g1_rot, g2_rot), method='linear', fill_value=0)
            gp = griddata(pts, raw_p, (g1_rot, g2_rot), method='linear', fill_value=0)

            # --- 3. APPLY NO-SLIP CONDITION (Wing Only!) ---
            # Calculate where the wing actually starts and ends in the array
            n_airfoil = TARGET_H // 2
            if n_airfoil % 2 == 0: n_airfoil += 1
            n_wake = (TARGET_H - n_airfoil) // 2

            # Apply 0.0 ONLY to the points physically touching the wing
            gu[n_wake:-n_wake, 0] = 0.0
            gv[n_wake:-n_wake, 0] = 0.0
            gw[n_wake:-n_wake, 0] = 0.0

            # --- VERIFICATION POPUP (RUNS ONCE) ---
            if count == 0:
                print(f"--- VERIFYING ALIGNMENT: {fname} ---")
                # Expanded to 3 subplots to show the mesh
                fig, ax = plt.subplots(1, 3, figsize=(18, 5))

                # 1. Raw Data Plot
                sc = ax[0].scatter(src_x, src_y, c=raw_u, cmap='jet', s=1)
                ax[0].set_title(f"Raw FlightStream Data (AoA: {aoa_val})")

                # 2. C-Mesh Visualization
                # We plot the 2D arrays to show the structured grid lines.
                # Low linewidth (lw) and alpha keep it from becoming a solid black blob.
                ax[1].plot(g1_rot, g2_rot, color='black', lw=0.3, alpha=0.5)  # Normal lines
                ax[1].plot(g1_rot.T, g2_rot.T, color='black', lw=0.3, alpha=0.5)  # Streamwise lines
                ax[1].plot(g1_rot[n_wake:-n_wake, 0], g2_rot[n_wake:-n_wake, 0], 'r-', lw=2, label='Airfoil Surface')
                ax[1].set_title("Generated C-Mesh Topology")
                ax[1].legend()

                # 3. Interpolated Flow Data Plot
                c_grid = ax[2].contourf(g1_rot, g2_rot, gu, levels=50, cmap='jet')
                ax[2].plot(g1_rot[n_wake:-n_wake, 0], g2_rot[n_wake:-n_wake, 0], 'w-', lw=1.5, label='Airfoil Surface')
                ax[2].set_title("Interpolated Rotated C-Grid")
                ax[2].legend()

                # Formatting for all plots
                for a in ax:
                    a.axis('equal')
                    # Zoomed in a bit tighter so you can actually see the boundary layer mesh
                    a.set_xlim(-0.5, 3.0)
                    a.set_ylim(-1.5, 1.5)

                plt.tight_layout()
                plt.show()
                if input(
                    "Does the grid perfectly align with the data and look smooth? (y/n): ").lower() != 'y': sys.exit()
            # --------------------------------------

            # Store the true parsed AoA (e.g. 2.0 or -4.0) for the network
            re_n = (re_val - RE_MIN) / (RE_MAX - RE_MIN)
            aoa_n = (aoa_val - AOA_MIN) / (AOA_MAX - AOA_MIN)

            input_stack = np.stack([g1_rot, g2_rot, g1_rot, g2_rot,
                                    np.full_like(g1_rot, re_n), np.full_like(g1_rot, aoa_n)], axis=0)
            output_stack = np.stack([np.nan_to_num(gu), np.nan_to_num(gv),
                                     np.nan_to_num(gw), np.nan_to_num(gp)], axis=0)

            batch_x.append(input_stack)
            batch_y.append(output_stack)
            count += 1
            if count % 10 == 0: print(f"Progress: {count}/{len(files)}")

        except Exception as e:
            print(f"Error {fname}: {e}")

    if batch_x:
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.save(os.path.join(SAVE_DIR, "input_full.npy"), np.array(batch_x, dtype=np.float32))
        np.save(os.path.join(SAVE_DIR, "output_full.npy"), np.array(batch_y, dtype=np.float32))
        print(f"Done! Saved {count} cases to {SAVE_DIR}")

if __name__ == "__main__":
    process()
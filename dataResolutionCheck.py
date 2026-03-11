"""
FLIGHTSTREAM DATA RESOLUTION CHECKER
------------------------------------
Checks if raw .dat files contain high enough resolution at the wall
to support the theoretical y+ requirements of the target C-Mesh.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ================= CONFIGURATION =================
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cases")
FILE_PATTERN = "*.dat"

# Parameters from your main script
NACA_THICKNESS = 0.15
CHORD_LENGTH = 2.0
RE_MAX = 1e7
TARGET_Y_PLUS = 5.0


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def naca4(x, t=0.15, c=1.0):
    xn = x / c
    term1 = 0.2969 * np.sqrt(np.maximum(0, xn))
    term2 = -0.1260 * xn
    term3 = -0.3516 * xn ** 2
    term4 = +0.2843 * xn ** 3
    term5 = -0.1015 * xn ** 4
    return 5 * t * c * (term1 + term2 + term3 + term4 + term5)


def get_metadata(filename):
    re_m = re.search(r'Re(\d+)', filename)
    aoa_m = re.search(r'_([m-]?\d+(?:\.\d+)?)deg', filename)
    if not re_m or not aoa_m: return None, None
    re_val = float(re_m.group(1))
    s = aoa_m.group(1)
    aoa_val = -float(s[1:]) if s.startswith('m') else float(s)
    return re_val, aoa_val


def parse_header(filepath):
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if "ZONE" in line:
                m = re.search(r'N=(\d+)', line)
                if m: return int(m.group(1)), i + 1
    return None, None


def calculate_wall_spacing(Re_max, chord, target_y_plus):
    """Calculates the physical first cell height required for a target y+"""
    cf = 0.058 * (Re_max ** -0.2)
    friction_velocity_ratio = np.sqrt(cf / 2.0)
    delta_y = (target_y_plus * chord) / (Re_max * friction_velocity_ratio)
    return delta_y


# ---------------------------------------------------------
# Main Checker Logic
# ---------------------------------------------------------
def run_diagnostic():
    files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
    if not files:
        print(f"No files found in {DATA_DIR}")
        return

    # Grab the first file for testing
    fpath = files[0]
    fname = os.path.basename(fpath)
    print(f"Analyzing sample file: {fname}...")

    # 1. Calculate Target Spacing
    target_dy = calculate_wall_spacing(RE_MAX, CHORD_LENGTH, TARGET_Y_PLUS)

    # 2. Load Raw Data
    re_val, aoa_val = get_metadata(fname)
    n_nodes, skiprows = parse_header(fpath)

    if n_nodes is None:
        print("Could not parse header.")
        return

    df = pd.read_csv(fpath, skiprows=skiprows, nrows=n_nodes, sep='\s+', header=None, engine='python')
    vals = df.values
    # Assuming standard 2D FlightStream output
    src_x, src_y = (vals[:, 0], vals[:, 2]) if vals[:, 1].std() < 1e-6 else (vals[:, 0], vals[:, 1])

    # 3. Generate a highly dense mathematical surface for distance checking
    # We rotate the surface to match the data's AoA for accurate measuring
    theta = np.radians(-aoa_val)
    x_dense = np.linspace(0, CHORD_LENGTH, 5000)
    y_dense = naca4(x_dense, NACA_THICKNESS, CHORD_LENGTH)

    # Combine upper and lower surfaces
    surf_x_flat = np.concatenate([x_dense, x_dense])
    surf_y_flat = np.concatenate([y_dense, -y_dense])

    # Rotate surface points
    surf_x_rot = surf_x_flat * np.cos(theta) - surf_y_flat * np.sin(theta)
    surf_y_rot = surf_x_flat * np.sin(theta) + surf_y_flat * np.cos(theta)
    surf_pts = np.column_stack((surf_x_rot, surf_y_rot))

    # 4. KD-Tree Distance Calculation
    raw_pts = np.column_stack((src_x, src_y))
    tree = cKDTree(surf_pts)
    distances, _ = tree.query(raw_pts)

    # Filter out points physically on the boundary (distance near 0)
    off_wall_dists = distances[distances > 1e-5]
    min_raw_dy = np.min(off_wall_dists) if len(off_wall_dists) > 0 else 0

    # 5. Output Report
    print("\n" + "=" * 50)
    print("RESOLUTION DIAGNOSTIC REPORT")
    print("=" * 50)
    print(f"Maximum Reynolds Number:      {RE_MAX:.1e}")
    print(f"Target y+:                    {TARGET_Y_PLUS}")
    print(f"Required C-Mesh First Cell:   {target_dy:.3e}")
    print("-" * 50)
    print(f"Closest Raw Data Point (dy):  {min_raw_dy:.3e}")

    ratio = min_raw_dy / target_dy
    print(f"Ratio (Raw dy / Target dy):   {ratio:.1f}x")

    if ratio > 3.0:
        print("\n[WARNING] Your raw data is too coarse to fully support the viscous sublayer.")
        print("Interpolation will likely smear gradients near the wall.")
    else:
        print("\n[SUCCESS] Your raw data resolution is sufficient for this C-Mesh!")
    print("=" * 50)

    # 6. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Macro View
    ax[0].scatter(src_x, src_y, color='blue', s=1, alpha=0.5, label='Raw Points')
    ax[0].plot(surf_x_rot, surf_y_rot, color='red', lw=1.5, label='Airfoil Surface')
    ax[0].set_title(f"Macro View: {fname}")
    ax[0].axis('equal')
    ax[0].legend()

    # Micro View (Leading edge zoom)
    ax[1].scatter(src_x, src_y, color='blue', s=10, alpha=0.7, label='Raw Points')
    ax[1].plot(surf_x_rot, surf_y_rot, color='red', lw=2, label='Airfoil Surface')

    # Draw a shaded region representing the target boundary layer height
    # (Approximated as a horizontal band for visual scale)
    ax[1].axhline(target_dy, color='green', linestyle='--', alpha=0.5, label='Target 1st Cell Height')
    ax[1].axhline(-target_dy, color='green', linestyle='--', alpha=0.5)

    ax[1].set_xlim(-0.02, 0.05)
    ax[1].set_ylim(-0.05, 0.05)
    ax[1].set_title("Micro View: Leading Edge Zoom")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_diagnostic()
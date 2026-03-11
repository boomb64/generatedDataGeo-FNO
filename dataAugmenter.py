"""
DATA AUGMENTER (Symmetry Mirroring)
-----------------------------------
Function:
    Doubles the dataset size by exploiting physical symmetry.
    Takes (X, Y, AoA) -> (u, v)
    Creates (X, -Y, -AoA) -> (u, -v)

    For a symmetric airfoil (NACA0015), the flow at -5 deg
    is the exact vertical mirror of the flow at +5 deg.
"""

import os
import numpy as np

# Config
DATA_PATH = "processed_full_physics"
INPUT_FILE = "input_full.npy"
OUTPUT_FILE = "output_full.npy"

# Output Filenames
AUG_INPUT_FILE = "input_augmented.npy"
AUG_OUTPUT_FILE = "output_augmented.npy"


def augment():
    print(f"Loading data from {DATA_PATH}...")
    try:
        inputs = np.load(os.path.join(DATA_PATH, INPUT_FILE))  # [N, 6, H, W]
        outputs = np.load(os.path.join(DATA_PATH, OUTPUT_FILE))  # [N, 4, H, W]
    except FileNotFoundError:
        print("Error: Original data not found.")
        return

    print(f"Original shape: {inputs.shape}")

    # --- 1. Create Copies for Mirroring ---
    inputs_mirror = inputs.copy()
    outputs_mirror = outputs.copy()

    # --- 2. Flip Spatial Grid (The 'W' dimension, axis 3) ---
    # In mgrid[X, Y], Y varies along the last axis.
    # We flip the data physically upside down.
    inputs_mirror = np.flip(inputs_mirror, axis=-1)
    outputs_mirror = np.flip(outputs_mirror, axis=-1)

    # --- 3. Invert Input Physics ---
    # Channel 1 (Grid Y) and Channel 3 (Grid Y Repeat): Negate coordinate (y -> -y)
    inputs_mirror[:, 1, :, :] *= -1
    inputs_mirror[:, 3, :, :] *= -1

    # Channel 5 (AoA): Invert Angle of Attack
    # Current norm is 0..1. Center is 0.5.
    # New = (1.0 - Old) is equivalent to flipping across the center.
    inputs_mirror[:, 5, :, :] = 1.0 - inputs_mirror[:, 5, :, :]

    # --- 4. Invert Output Physics ---
    # Channel 1 (v velocity): Vertical velocity must flip sign (updraft becomes downdraft)
    outputs_mirror[:, 1, :, :] *= -1

    # Channel 2 (w velocity): Z-velocity (spanwise). For symmetry, usually flips too.
    outputs_mirror[:, 2, :, :] *= -1

    # Note: Channel 0 (u velocity) and Channel 3 (Pressure) stay POSITIVE
    # because speed/pressure are scalar magnitudes relative to the flow direction.

    # --- 5. Combine ---
    inputs_combined = np.concatenate([inputs, inputs_mirror], axis=0)
    outputs_combined = np.concatenate([outputs, outputs_mirror], axis=0)

    print(f"Augmented shape: {inputs_combined.shape}")
    print(f"Added {len(inputs_mirror)} synthetic samples.")

    # --- 6. Save ---
    save_in = os.path.join(DATA_PATH, AUG_INPUT_FILE)
    save_out = os.path.join(DATA_PATH, AUG_OUTPUT_FILE)

    np.save(save_in, inputs_combined)
    np.save(save_out, outputs_combined)
    print(f"Saved augmented dataset to:\n  {save_in}\n  {save_out}")


if __name__ == "__main__":
    augment()
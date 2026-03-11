"""
INPUT DATA INSPECTOR
--------------------
Function:
    Checks if the Physics Channels (Re, AoA) actually contain changing data.
    If these are all 0.0 or Identical, the model is flying blind.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"processed_full_physics"


def inspect():
    try:
        # Load just the inputs
        inputs = np.load(os.path.join(DATA_PATH, "input_full.npy"))  # [N, 6, H, W]
    except FileNotFoundError:
        print("Data not found.")
        return

    print(f"Loaded Inputs: {inputs.shape}")

    # Channel Mapping (from process_data_masked.py):
    # 0: Grid X, 1: Grid Y, 2: Grid X (Repeat), 3: Grid Y (Repeat), 4: Re, 5: AoA

    # 1. Check if AoA varies across samples
    # We pick the middle pixel (110, 75) to sample the value
    aoa_values = inputs[:, 5, 110, 75]
    re_values = inputs[:, 4, 110, 75]

    print("\n--- PHYSICS CHANNEL DIAGNOSTICS ---")
    print(f"AoA Channel - Min: {aoa_values.min():.4f}, Max: {aoa_values.max():.4f}, Std Dev: {aoa_values.std():.4f}")
    print(f"Re  Channel - Min: {re_values.min():.4f}, Max: {re_values.max():.4f}, Std Dev: {re_values.std():.4f}")

    if aoa_values.std() < 1e-6:
        print("\n[CRITICAL ERROR] AoA channel is CONSTANT. The model cannot learn!")
    else:
        print("\n[OK] AoA channel varies correctly.")

    # 2. Visualize a sample input to verify normalization
    sample_idx = 0
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # Plot Grid X
    im0 = ax[0].imshow(inputs[sample_idx, 0, :, :], cmap='viridis')
    ax[0].set_title(f"Input: Grid X (Range: {inputs[sample_idx, 0].min():.1f} to {inputs[sample_idx, 0].max():.1f})")
    plt.colorbar(im0, ax=ax[0])

    # Plot AoA Channel (Should be a solid color block)
    im1 = ax[1].imshow(inputs[sample_idx, 5, :, :], cmap='plasma', vmin=0, vmax=1)
    ax[1].set_title(f"Input: AoA Channel (Val: {inputs[sample_idx, 5, 0, 0]:.3f})")
    plt.colorbar(im1, ax=ax[1])

    # Plot Re Channel
    im2 = ax[2].imshow(inputs[sample_idx, 4, :, :], cmap='plasma', vmin=0, vmax=1)
    ax[2].set_title(f"Input: Re Channel (Val: {inputs[sample_idx, 4, 0, 0]:.3f})")
    plt.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inspect()
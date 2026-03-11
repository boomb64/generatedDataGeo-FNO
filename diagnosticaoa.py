"""
AOA SENSITIVITY DIAGNOSTIC
--------------------------
Function:
    Takes ONE sample and runs it through the model 3 times with
    manually overridden Angle of Attack (AoA) values.

    Verifies if the model actually reacts to the physics controls.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.models import FNO

# --- CONFIGURATION ---
MODEL_PATH = r"model/naca_fno_final.pth"
DATA_PATH = r"processed_full_physics"
WIDTH = 64
MODES = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def diagnose():
    print(f"Loading model: {MODEL_PATH}")

    # 1. Load Data & Model
    try:
        inputs = np.load(os.path.join(DATA_PATH, "input_full.npy"))
        outputs = np.load(os.path.join(DATA_PATH, "output_full.npy"))
    except:
        print("Data not found.")
        return

    model = FNO(n_modes=(MODES * 2, MODES), hidden_channels=WIDTH, in_channels=6, out_channels=4, n_layers=4,
                non_linearity=torch.nn.GELU()).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Prepare "What If" Scenarios
    # Pick a sample (index 0)
    base_input = torch.tensor(inputs[0], dtype=torch.float).to(device)  # [6, H, W]

    # Channel 5 is AoA.
    # Norm Range: 0.0 = -20 deg, 0.5 = 0 deg, 1.0 = +20 deg

    # Scenario A: Dive (-20 deg) -> AoA = 0.0
    input_dive = base_input.clone().unsqueeze(0)
    input_dive[:, 5, :, :] = 0.0

    # Scenario B: Level (0 deg) -> AoA = 0.5
    input_level = base_input.clone().unsqueeze(0)
    input_level[:, 5, :, :] = 0.5

    # Scenario C: Climb (+20 deg) -> AoA = 1.0
    input_climb = base_input.clone().unsqueeze(0)
    input_climb[:, 5, :, :] = 1.0

    print("Running Inference for 3 Scenarios...")
    with torch.no_grad():
        out_dive = model(input_dive).cpu().numpy()[0, 0]
        out_level = model(input_level).cpu().numpy()[0, 0]
        out_climb = model(input_climb).cpu().numpy()[0, 0]

    # 3. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Common color scale for fair comparison
    vmin, vmax = out_level.min(), out_level.max()

    ax1 = axes[0]
    im1 = ax1.pcolormesh(out_dive, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
    ax1.set_title("Forced AoA: -20° (Input=0.0)")
    ax1.axis('equal')

    ax2 = axes[1]
    im2 = ax2.pcolormesh(out_level, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
    ax2.set_title("Forced AoA: 0° (Input=0.5)")
    ax2.axis('equal')

    ax3 = axes[2]
    im3 = ax3.pcolormesh(out_climb, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
    ax3.set_title("Forced AoA: +20° (Input=1.0)")
    ax3.axis('equal')

    plt.colorbar(im3, ax=axes.ravel().tolist())
    plt.suptitle("Model Sensitivity Check: Does the wake move?", fontsize=16)
    plt.tight_layout()
    plt.show()

    # 4. Numeric Proof
    diff = np.abs(out_dive - out_climb).mean()
    print(f"\nMean Difference between -20 and +20 prediction: {diff:.5f}")
    if diff < 0.01:
        print(">> FAIL: The model is ignoring the Angle of Attack.")
    else:
        print(">> PASS: The model reacts to Angle of Attack.")


if __name__ == "__main__":
    diagnose()
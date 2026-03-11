import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.models import FNO

# --- CONFIGURATION ---
MODEL_PATH = r"model/naca_fno_augmented.pth"
DATA_PATH = r"processed_full_physics"
MODES = 12
WIDTH = 64
IN_CHANNELS = 6
OUT_CHANNELS = 4

# Zoom Window Coordinates (Adjust these to focus closer/further)
X_ZOOM = (-0.5, 2.5)
Z_ZOOM = (-1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize():
    print(f"Loading model from {MODEL_PATH}...")

    # 1. Load Data
    try:
        inputs = np.load(os.path.join(DATA_PATH, "input_full.npy"))
        outputs = np.load(os.path.join(DATA_PATH, "output_full.npy"))
    except FileNotFoundError:
        print("Error: Processed data not found.")
        return

    sample_idx = -1
    x = torch.tensor(inputs[sample_idx], dtype=torch.float).unsqueeze(0).to(device)
    y = torch.tensor(outputs[sample_idx], dtype=torch.float).unsqueeze(0).to(device)

    # 2. Load Model
    model = FNO(
        n_modes=(MODES * 2, MODES),
        hidden_channels=WIDTH,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        n_layers=4,
        domain_padding=None,
        non_linearity=torch.nn.GELU()
    ).to(device)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    model.eval()

    # 3. Predict
    print("Running inference...")
    with torch.no_grad():
        pred = model(x)

    # 4. Extract
    X_grid = x[0, 0].cpu().numpy()
    Z_grid = x[0, 1].cpu().numpy()
    True_U = y[0, 0].cpu().numpy()
    Pred_U = pred[0, 0].cpu().numpy()
    Abs_Error = np.abs(True_U - Pred_U)

    # 5. Plotting (3 Rows, 2 Columns)
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Plotting Helper to avoid repetition
    def plot_row(row_idx, data, title, cmap):
        # Column 0: Global View
        ax_g = axes[row_idx, 0]
        c_g = ax_g.pcolormesh(X_grid, Z_grid, data, cmap=cmap, shading='auto')
        ax_g.set_title(f"Global: {title}")
        ax_g.axis('equal')
        fig.colorbar(c_g, ax=ax_g)

        # Column 1: Zoom View
        ax_z = axes[row_idx, 1]
        c_z = ax_z.pcolormesh(X_grid, Z_grid, data, cmap=cmap, shading='auto')
        ax_z.set_title(f"Zoom: {title}")
        ax_z.set_xlim(X_ZOOM)
        ax_z.set_ylim(Z_ZOOM)
        ax_z.set_aspect('equal', adjustable='box') # Force equal scaling in zoom
        fig.colorbar(c_z, ax=ax_z)

    # --- ROW 1: Ground Truth ---
    plot_row(0, True_U, "Ground Truth (U-Velocity)", 'jet')

    # --- ROW 2: Prediction ---
    plot_row(1, Pred_U, "FNO Prediction", 'jet')

    # --- ROW 3: Error ---
    plot_row(2, Abs_Error, "Absolute Error", 'magma')

    plt.suptitle(f"Model Inference Comparison (Sample: {sample_idx})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    visualize()
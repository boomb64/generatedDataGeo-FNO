"""
STABILIZED FNO TRAINER (MSE + Normalization)
--------------------------------------------
Fixes:
1. Switches to MSELoss (Mean Reduction) to stop gradient explosions.
2. Normalizes Grid Inputs (X, Y) to [0,1] range dynamically.
3. FIX: Removed 'verbose' arg from Scheduler (deprecated in PyTorch Nightly).
"""

import os
import torch
import numpy as np
import pandas as pd
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from neuralop.models import FNO

# ================= CONFIGURATION =================
BASE_PATH = "."
MODEL_PATH = os.path.join(BASE_PATH, "model")
DATA_PATH = os.path.join(BASE_PATH, "processed_full_physics")
FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "naca_fno_stabilized.pth")
EXCEL_FILENAME = os.path.join(BASE_PATH, "results", "training_log_stabilized.xlsx")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(EXCEL_FILENAME), exist_ok=True)

# Hyperparameters
ntrain = 270
ntest = 30
batch_size = 16
learning_rate = 0.001
epochs = 500
modes = 12
width = 32

# ================= DATASET WITH NORMALIZATION =================
class NormalizedDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx].clone() # [6, H, W]
        y_sample = self.y[idx].clone() # [4, H, W]

        # Normalize Grid X (Channel 2) to [0,1]
        x_sample[2] = (x_sample[2] - x_sample[2].min()) / (x_sample[2].max() - x_sample[2].min() + 1e-6)
        # Normalize Grid Y (Channel 3) to [0,1]
        x_sample[3] = (x_sample[3] - x_sample[3].min()) / (x_sample[3].max() - x_sample[3].min() + 1e-6)

        return {"x": x_sample, "y": y_sample}

print("Loading data...")
input_full = np.load(os.path.join(DATA_PATH, "input_full.npy"))
output_full = np.load(os.path.join(DATA_PATH, "output_full.npy"))

input_tensor = torch.tensor(input_full, dtype=torch.float)
output_tensor = torch.tensor(output_full, dtype=torch.float)

# Split
x_train = input_tensor[:ntrain]
y_train = output_tensor[:ntrain]
x_test = input_tensor[ntrain:]
y_test = output_tensor[ntrain:]

train_loader = DataLoader(NormalizedDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(NormalizedDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# ================= MODEL & OPTIMIZER =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = FNO(
    n_modes=(modes * 2, modes),
    hidden_channels=width,
    in_channels=6,
    out_channels=4,
    n_layers=4,
    non_linearity=torch.nn.GELU()
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# --- FIX: Removed 'verbose=True' (It causes crashes in PyTorch Nightly) ---
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# MSE Loss (Mean Reduction)
loss_fn = nn.MSELoss(reduction='mean')

# ================= TRAINING LOOP =================
history = []
print("Starting Stabilized Training (MSE)...")

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss_accum = 0.0

    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss_accum += loss.item()

    train_loss = train_loss_accum / len(train_loader)

    model.eval()
    test_loss_accum = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            test_loss_accum += loss_fn(out, y).item()

    test_loss = test_loss_accum / len(test_loader)

    # Step scheduler
    scheduler.step(test_loss)
    current_lr = optimizer.param_groups[0]['lr']

    t2 = default_timer()

    if (ep+1) % 10 == 0:
        print(f"Ep {ep+1}/{epochs} | Time: {t2-t1:.2f}s | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f} | LR: {current_lr:.1e}")

    history.append({'epoch': ep+1, 'train_loss': train_loss, 'test_loss': test_loss})

# Save
pd.DataFrame(history).to_excel(EXCEL_FILENAME, index=False)
torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
print("Training Complete.")
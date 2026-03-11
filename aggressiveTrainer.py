"""
FINAL FNO TRAINER (Best of Both Worlds)
---------------------------------------
1. Architecture: Width=64 (High Capacity for sharp edges).
2. Loss: MSELoss (Stable with Zero-Masked data).
3. Safety: Saves model BEFORE logging to Excel.
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
FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "naca_fno_final.pth")
EXCEL_FILENAME = os.path.join(BASE_PATH, "results", "training_log_final.xlsx")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(EXCEL_FILENAME), exist_ok=True)

# Hyperparameters
ntrain = 270
batch_size = 16
learning_rate = 0.001
epochs = 500
modes = 12
width = 64  # High Capacity

# ================= DATASET =================
class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}

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

train_loader = DataLoader(SimpleDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(SimpleDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# ================= MODEL =================
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
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
loss_fn = nn.MSELoss(reduction='mean')

# ================= TRAINING LOOP =================
history = []
print(f"Starting Final Training (Width={width}, Loss=MSE)...")

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0.0

    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            test_loss += loss_fn(out, y).item()

    test_loss /= len(test_loader)

    scheduler.step(test_loss)
    current_lr = optimizer.param_groups[0]['lr']

    t2 = default_timer()
    if (ep+1) % 10 == 0:
        print(f"Ep {ep+1}/{epochs} | Time: {t2-t1:.2f}s | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f} | LR: {current_lr:.1e}")

    history.append({'epoch': ep+1, 'train_loss': train_loss, 'test_loss': test_loss})

# --- SAFE SAVE ORDER ---
# 1. Save Model FIRST (Priority)
torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
print(f"Model saved successfully to {FINAL_MODEL_FILENAME}")

# 2. Save Log SECOND (If this fails, at least we have the model)
try:
    pd.DataFrame(history).to_excel(EXCEL_FILENAME, index=False)
    print("Log saved.")
except Exception as e:
    print(f"Warning: Could not save Excel log ({e}), but Model is safe.")
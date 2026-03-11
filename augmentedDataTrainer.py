"""
AUGMENTED FNO TRAINER
---------------------
Features:
1. Loads 'input_augmented.npy' (Double dataset size).
2. Auto-detects split sizes.
3. High Capacity (Width=64) + Low Weight Decay.
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

# New files
INPUT_FILE = "input_augmented.npy"
OUTPUT_FILE = "output_augmented.npy"

FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "naca_fno_augmented.pth")
EXCEL_FILENAME = os.path.join(BASE_PATH, "results", "training_log_augmented.xlsx")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(EXCEL_FILENAME), exist_ok=True)

# Hyperparameters
batch_size = 32         # Increased batch size for larger data
learning_rate = 0.001
epochs = 500
modes = 12
width = 64              # High Capacity

# ================= DATASET =================
class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}

print("Loading AUGMENTED data...")
try:
    input_full = np.load(os.path.join(DATA_PATH, INPUT_FILE))
    output_full = np.load(os.path.join(DATA_PATH, OUTPUT_FILE))
except FileNotFoundError:
    print(f"CRITICAL: {INPUT_FILE} not found. Run 'dataAugmenter.py' first!")
    exit()

input_tensor = torch.tensor(input_full, dtype=torch.float)
output_tensor = torch.tensor(output_full, dtype=torch.float)

total_samples = len(input_tensor)
print(f"Total Samples: {total_samples}")

# Auto-Split (90% Train / 10% Test)
ntrain = int(0.90 * total_samples)
ntest = total_samples - ntrain
print(f"Training on {ntrain}, Testing on {ntest}")

# Random Shuffle Indices before split (Crucial for augmented data)
indices = torch.randperm(total_samples)
x_shuffled = input_tensor[indices]
y_shuffled = output_tensor[indices]

x_train = x_shuffled[:ntrain]
y_train = y_shuffled[:ntrain]
x_test = x_shuffled[ntrain:]
y_test = y_shuffled[ntrain:]

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

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Lower decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
loss_fn = nn.MSELoss(reduction='mean')

# ================= TRAINING LOOP =================
history = []
print(f"Starting Augmented Training (Width={width})...")

for ep in range(epochs):
    model.train()
    t1 = default_timer()  # Start timer
    train_loss = 0.0

    # Capture LR at the start of the epoch
    current_lr = optimizer.param_groups[0]['lr']

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

    t2 = default_timer()  # End timer
    epoch_time = t2 - t1

    if (ep + 1) % 10 == 0:
        print(
            f"Ep {ep + 1}/{epochs} | Time: {epoch_time:.2f}s | Train: {train_loss:.6f} | Test: {test_loss:.6f} | LR: {current_lr:.1e}")

    # Updated to include all requested fields
    history.append({
        'epoch': ep + 1,
        'time_s': epoch_time,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'learning_rate': current_lr
    })

# Save
torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
pd.DataFrame(history).to_excel(EXCEL_FILENAME, index=False)
print(f"Model saved to {FINAL_MODEL_FILENAME}")
print(f"Training log saved to {EXCEL_FILENAME}")
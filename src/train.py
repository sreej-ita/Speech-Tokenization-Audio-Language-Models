import os
import torch
import torch.nn as nn

import sys
sys.path.append(os.path.dirname(__file__))

from dataset import SpeechFeatureDataset
from torch.utils.data import DataLoader
from model import CNNAutoencoder


# ─────────────────────────────────────────────────────
# WHAT IS HAPPENING HERE?
#
# Training loop — feeds batches through the autoencoder,
# computes MSE loss, backpropagates, updates weights.
#
# Uses speaker-split data:
#   Train : data/train/  (speakers 1089, 1188)
#   Val   : data/val/    (speakers 1320, 908)
#   Test  : data/test/   (speakers 260, 61)  ← never touched during training
#
# Usage:
#   python src/train.py
# ─────────────────────────────────────────────────────


# ── CONFIG ────────────────────────────────────────────
LATENT_DIM  = 20     # mentor's compression target: 80 → 20
BATCH_SIZE  = 16     # reduced from 32 since train set is smaller (109 clips)
EPOCHS      = 50     # more epochs to compensate for smaller dataset
LR          = 1e-3

TRAIN_PATH = "data/train/log_mel.npy"
VAL_PATH   = "data/val/log_mel.npy"


def train():

    save_dir = os.path.join("outputs", "logmel", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*52}")
    print(f"  Training CNN Autoencoder — LOG-MEL")
    print(f"  Device     : {device}")
    print(f"  Latent dim : {LATENT_DIM}  (compression: 80 → {LATENT_DIM})")
    print(f"  Epochs     : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"{'='*52}\n")

    # ── DATA ──────────────────────────────────────────
    train_ds = SpeechFeatureDataset(TRAIN_PATH)
    val_ds   = SpeechFeatureDataset(VAL_PATH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nTrain: {len(train_ds)} clips | Val: {len(val_ds)} clips\n")

    # ── MODEL ─────────────────────────────────────────
    model = CNNAutoencoder(latent_dim=LATENT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ── LOSS & OPTIMIZER ──────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── TRAINING LOOP ─────────────────────────────────
    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []

    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            x_hat, z = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validate
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                x_hat, _ = model(x)
                epoch_val_loss += criterion(x_hat, x).item()
        avg_val_loss = epoch_val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss"       : best_val_loss,
                "train_losses"   : train_losses,
                "val_losses"     : val_losses,
                "feature"        : "logmel",
                "latent_dim"     : LATENT_DIM,
                "freq_bins"      : train_ds.freq_bins,
                "time_frames"    : train_ds.time_frames,
            }, save_path)
            print(f"           → saved best model (val loss: {best_val_loss:.4f})")

    print(f"\n[✓] Training complete for LOG-MEL")
    print(f"    Best val loss : {best_val_loss:.4f}")
    print(f"    Model saved   : {save_path}")
    print(f"\nNext step → python src/encode.py")


if __name__ == "__main__":
    train()
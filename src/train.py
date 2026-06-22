import os
import torch
import torch.nn as nn

import sys
sys.path.append(os.path.dirname(__file__))

from dataset import SpeechFeatureDataset
from torch.utils.data import DataLoader
from model import CNNAutoencoder

# Train the CNN autoencoder on log-mel features and save the best checkpoint.

LATENT_DIM  = 20
BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 1e-3

TRAIN_PATH = "data/train/log_mel.npy"
VAL_PATH   = "data/val/log_mel.npy"


def train():

    save_dir = os.path.join("outputs", "logmel", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  latent={LATENT_DIM}  epochs={EPOCHS}  bs={BATCH_SIZE}  lr={LR}")

    train_ds = SpeechFeatureDataset(TRAIN_PATH)
    val_ds   = SpeechFeatureDataset(VAL_PATH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model    = CNNAutoencoder(latent_dim=LATENT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []

    for epoch in range(1, EPOCHS + 1):

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

        print(f"epoch {epoch:3d}/{EPOCHS}  train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")

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

    print(f"done  best_val={best_val_loss:.4f}  saved={save_path}")


if __name__ == "__main__":
    train()
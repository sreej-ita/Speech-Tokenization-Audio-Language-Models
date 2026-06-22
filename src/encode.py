import os
import numpy as np
import torch

import sys
sys.path.append(os.path.dirname(__file__))

from dataset import SpeechFeatureDataset
from model import CNNAutoencoder
from torch.utils.data import DataLoader


DATA_PATHS = {
    "train": "data/train/log_mel.npy",
    "val"  : "data/val/log_mel.npy",
    "test" : "data/test/log_mel.npy",
}


def encode():

    # ── LOAD TRAINED MODEL ────────────────────────────
    model_path = os.path.join("outputs", "logmel", "checkpoints", "best_model.pt")
    if not os.path.exists(model_path):
        print(f"[!] Model not found: {model_path}")
        print(f"    Run: python src/train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(model_path, map_location=device)
    model = CNNAutoencoder(latent_dim=ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()   # freeze batchnorm and dropout

    print(f"loaded epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}  latent={ckpt['latent_dim']}")

    # ── ENCODE ALL SPLITS ─────────────────────────────
    latent_dir = os.path.join("outputs", "logmel", "latents")
    os.makedirs(latent_dir, exist_ok=True)

    for split, npy_path in DATA_PATHS.items():

        if not os.path.exists(npy_path):
            print(f"[!] Not found: {npy_path} — skipping")
            continue

        ds     = SpeechFeatureDataset(npy_path)
        loader = DataLoader(ds, batch_size=16, shuffle=False)

        all_latents = []

        with torch.no_grad():   # no gradients needed — just forward pass
            for batch in loader:
                x = batch.to(device)        # (B, 1, 80, T)
                z = model.encode(x)         # (B, latent_dim, 20, T//4)
                all_latents.append(z.cpu().numpy())

        # Stack into one array: (N_clips, latent_dim, 20, T//4)
        latents = np.concatenate(all_latents, axis=0)

        save_path = os.path.join(latent_dir, f"{split}.npy")
        np.save(save_path, latents)

        print(f"[{split}] {latents.shape}  →  {save_path}")

    # ── FLATTEN TRAIN LATENTS FOR VQ ─────────────────
    # VQ needs a 2D array: (total_vectors, latent_dim)
    # Each spatial position (freq//4, time//4) gives one vector
    # So we flatten the spatial dims across all train clips
    train_latents = np.load(os.path.join(latent_dir, "train.npy"))

    # Shape: (N, latent_dim, H, W) → (N*H*W, latent_dim)
    N, D, H, W = train_latents.shape
    flat = train_latents.transpose(0, 2, 3, 1).reshape(-1, D)
    # transpose puts latent_dim last → (N, H, W, D) → flatten → (N*H*W, D)

    flat_path = os.path.join(latent_dir, "train_flat.npy")
    np.save(flat_path, flat)

    print(f"[flat] {flat.shape}  →  {flat_path}")
    print("done")


if __name__ == "__main__":
    encode()
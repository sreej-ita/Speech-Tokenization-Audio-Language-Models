import os
import numpy as np
import torch

import sys
sys.path.append(os.path.dirname(__file__))

from dataset import SpeechFeatureDataset
from model import CNNAutoencoder
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────
# WHAT IS HAPPENING HERE?
#
# The autoencoder is trained. Now we freeze the encoder
# and pass all audio through it to get latent vectors.
#
# These latent vectors are the "compressed representations"
# in the mentor's chain:
#   Raw(512) → Features(80) → Latent(20) → Token(1)
#                                  ↑
#                             we are here
#
# We do this for all 3 splits (train/val/test) so that:
#   - train latents → used to build the VQ codebook
#   - val/test latents → used to assign tokens & evaluate
#
# Usage:
#   python src/encode.py
#
# Output:
#   outputs/logmel/latents/train.npy  → (109, 20, 20, 76)
#   outputs/logmel/latents/val.npy    → (116, 20, 20, 76)
#   outputs/logmel/latents/test.npy   → (186, 20, 20, 76)
# ─────────────────────────────────────────────────────


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

    print(f"\n{'='*52}")
    print(f"  Encoding latents — LOG-MEL")
    print(f"  Loaded model from epoch {ckpt['epoch']} "
          f"(val loss: {ckpt['val_loss']:.4f})")
    print(f"  Latent dim: {ckpt['latent_dim']}")
    print(f"{'='*52}\n")

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

        print(f"  [{split:5s}] latent shape: {latents.shape} → saved to {save_path}")

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

    print(f"\n  [flat ] train latents for VQ: {flat.shape}")
    print(f"          → {N} clips × {H}×{W} spatial positions × {D} dims")
    print(f"          → saved to {flat_path}")

    print(f"\n[✓] Encoding complete for LOG-MEL")
    print(f"    Latents saved to: {latent_dir}")
    print(f"\nNext step → python src/vq.py")


if __name__ == "__main__":
    encode()
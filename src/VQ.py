import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import sys
sys.path.append(os.path.dirname(__file__))

# Single-stage VQ baseline to compare against RVQ.

CODEBOOK_SIZE = 256


def assign_tokens(latents_flat: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    z_sq  = (latents_flat ** 2).sum(axis=1, keepdims=True)
    c_sq  = (codebook ** 2).sum(axis=1)
    cross = latents_flat @ codebook.T
    dists = z_sq + c_sq - 2 * cross
    return dists.argmin(axis=1).astype(np.int32)


def quantize_single():

    latent_dir  = os.path.join("outputs", "logmel", "latents")
    out_dir     = os.path.join("outputs", "logmel", "vq_single")
    os.makedirs(out_dir, exist_ok=True)

    train_flat_path = os.path.join(latent_dir, "train_flat.npy")
    if not os.path.exists(train_flat_path):
        print(f"not found: {train_flat_path}")
        print("run src/encode.py first")
        return

    train_flat = np.load(train_flat_path)
    print(f"train={train_flat.shape}  codebook={CODEBOOK_SIZE}")

    # train single codebook
    kmeans = MiniBatchKMeans(
        n_clusters   = CODEBOOK_SIZE,
        n_init       = 5,
        max_iter     = 100,
        batch_size   = 4096,
        random_state = 42,
        verbose      = 0,
    )
    kmeans.fit(train_flat)
    codebook = kmeans.cluster_centers_.astype(np.float32)

    cb_path = os.path.join(out_dir, "codebook.npy")
    np.save(cb_path, codebook)
    print(f"codebook={codebook.shape}  →  {cb_path}")

    # assign tokens and compute MSE for all splits
    for split in ["train", "val", "test"]:
        lat_path = os.path.join(latent_dir, f"{split}.npy")
        if not os.path.exists(lat_path):
            print(f"not found: {lat_path} — skipping")
            continue

        latents     = np.load(lat_path)
        N, D, H, W  = latents.shape
        flat        = latents.transpose(0, 2, 3, 1).reshape(-1, D)

        tokens    = assign_tokens(flat, codebook)
        quantized = codebook[tokens]
        mse       = float(((flat - quantized) ** 2).mean())

        token_path = os.path.join(out_dir, f"tokens_{split}.npy")
        np.save(token_path, tokens)
        print(f"[{split}]  tokens={tokens.shape}  mse={mse:.4f}")

    print("done")


if __name__ == "__main__":
    quantize_single()
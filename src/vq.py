import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import sys
sys.path.append(os.path.dirname(__file__))


CODEBOOK_SIZE = 256   # number of codewords per stage (2^8, fits in one byte)
RVQ_STAGES    = 4     # number of residual stages
                      # effective codes = 256^4 = ~4 billion


def assign_tokens(residual: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find nearest codebook entry for each residual vector,
    then compute the new residual.

    Args:
        residual  : (N, latent_dim)
        codebook  : (codebook_size, latent_dim)

    Returns:
        token_ids : (N,) integer indices
        new_resid : (N, latent_dim) residual after quantization
    """
    z_sq  = (residual ** 2).sum(axis=1, keepdims=True)   # (N, 1)
    c_sq  = (codebook ** 2).sum(axis=1)                   # (K,)
    cross = residual @ codebook.T                          # (N, K)

    dists     = z_sq + c_sq - 2 * cross                   # (N, K)
    token_ids = dists.argmin(axis=1).astype(np.int32)     # (N,)
    quantized = codebook[token_ids]                        # (N, latent_dim)
    new_resid = residual - quantized                       # (N, latent_dim)

    return token_ids, new_resid


def train_codebook(vectors: np.ndarray) -> np.ndarray:
    """Train a single K-Means codebook on the given vectors."""
    kmeans = MiniBatchKMeans(
        n_clusters   = CODEBOOK_SIZE,
        n_init       = 5,
        max_iter     = 100,
        batch_size   = 4096,
        random_state = 42,
        verbose      = 0,
    )
    kmeans.fit(vectors)
    return kmeans.cluster_centers_.astype(np.float32)


def compute_codebook_stats(token_ids: np.ndarray, stage: int):
    """Print utilization stats for one RVQ stage."""
    unique, counts = np.unique(token_ids, return_counts=True)
    n_active       = len(unique)
    utilization    = n_active / CODEBOOK_SIZE * 100

    probs      = counts / counts.sum()
    entropy    = -np.sum(probs * np.log(probs + 1e-10))
    perplexity = np.exp(entropy)

    print(f"    stage {stage+1}: active={n_active}/{CODEBOOK_SIZE} "
          f"({utilization:.1f}%)  perplexity={perplexity:.1f}")


def quantize():

    latent_dir   = os.path.join("outputs", "logmel", "latents")
    codebook_dir = os.path.join("outputs", "logmel", "codebook")
    os.makedirs(codebook_dir, exist_ok=True)

    # ── LOAD TRAIN LATENTS ────────────────────────────
    train_flat_path = os.path.join(latent_dir, "train_flat.npy")
    if not os.path.exists(train_flat_path):
        print(f"[!] Not found: {train_flat_path}")
        print(f"    Run: python src/encode.py")
        return

    train_flat = np.load(train_flat_path)   # (N, latent_dim)
    latent_dim = train_flat.shape[1]

    print(f"train={train_flat.shape}  codebook={CODEBOOK_SIZE}x{RVQ_STAGES}  effective={CODEBOOK_SIZE**RVQ_STAGES:,}")

    # ── TRAIN RVQ CODEBOOKS ───────────────────────────
    

    codebooks = []
    residual  = train_flat.copy()

    for stage in range(RVQ_STAGES):
        cb = train_codebook(residual)
        codebooks.append(cb)

        cb_path = os.path.join(codebook_dir, f"codebook_stage{stage+1}.npy")
        np.save(cb_path, cb)

        # Advance residual
        _, residual = assign_tokens(residual, cb)
        print(f"stage {stage+1}/{RVQ_STAGES}  residual_var={residual.var():.4f}")

    print(f"final residual_var={residual.var():.4f}")

    for split in ["train", "val", "test"]:
        lat_path = os.path.join(latent_dir, f"{split}.npy")
        if not os.path.exists(lat_path):
            print(f"  [!] Not found: {lat_path} — skipping")
            continue

        latents      = np.load(lat_path)                              # (N, D, H, W)
        N, D, H, W  = latents.shape
        flat         = latents.transpose(0, 2, 3, 1).reshape(-1, D)  # (N*H*W, D)

        all_tokens = []
        residual   = flat.copy()

        for stage, cb in enumerate(codebooks):
            token_ids, residual = assign_tokens(residual, cb)
            all_tokens.append(token_ids)

        # Stack: (N*H*W, RVQ_STAGES)
        tokens = np.stack(all_tokens, axis=1)

        token_path = os.path.join(codebook_dir, f"tokens_{split}.npy")
        np.save(token_path, tokens)

        print(f"[{split}]  vectors={flat.shape}  tokens={tokens.shape}")
        for stage in range(RVQ_STAGES):
            compute_codebook_stats(tokens[:, stage], stage)

    print("done")


if __name__ == "__main__":
    quantize()
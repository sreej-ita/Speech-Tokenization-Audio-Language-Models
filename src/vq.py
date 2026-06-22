import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import sys
sys.path.append(os.path.dirname(__file__))


# ─────────────────────────────────────────────────────
# WHAT IS HAPPENING HERE?
#
# This is the final step — Residual Vector Quantization (RVQ).
# Instead of one codebook, we use a stack of Q codebooks.
# Each stage quantizes the RESIDUAL left by the previous stage.
#
# The mentor's compression chain completes here:
#   Raw(512) → Features(80) → Latent(20) → Tokens(Q integers)
#                                                  ↑
#                                             we are here
#
# HOW RVQ WORKS:
#   Stage 1: quantize z          → token_1, residual_1 = z - codebook_1[token_1]
#   Stage 2: quantize residual_1 → token_2, residual_2 = residual_1 - codebook_2[token_2]
#   ...
#   Stage Q: quantize residual_{Q-1} → token_Q
#
#   Final token sequence per vector: [token_1, token_2, ..., token_Q]
#   Reconstruction: codebook_1[t1] + codebook_2[t2] + ... + codebook_Q[tQ]
#
# WHY RVQ?
#   A single VQ with 256 codes has limited resolution.
#   RVQ with Q=4 stages and 256 codes each gives 256^4 effective codes
#   while keeping each codebook small and fast to train.
#   Used in EnCodec, SoundStream, and most modern audio codecs.
#
# Usage:
#   python src/vq.py
#
# Output:
#   outputs/logmel/codebook/codebook_stage{i}.npy  → (256, 20) per stage
#   outputs/logmel/codebook/tokens_train.npy       → (N*H*W, Q) token IDs
#   outputs/logmel/codebook/tokens_val.npy
#   outputs/logmel/codebook/tokens_test.npy
# ─────────────────────────────────────────────────────


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

    print(f"\n{'='*52}")
    print(f"  Residual Vector Quantization — LOG-MEL")
    print(f"  Train vectors : {train_flat.shape}")
    print(f"  Codebook size : {CODEBOOK_SIZE}  ×  {RVQ_STAGES} stages")
    print(f"  Effective codes: {CODEBOOK_SIZE}^{RVQ_STAGES} = {CODEBOOK_SIZE**RVQ_STAGES:,}")
    print(f"{'='*52}\n")

    # ── TRAIN RVQ CODEBOOKS ───────────────────────────
    print(f"[1/3] Training {RVQ_STAGES} codebooks with MiniBatchKMeans...")

    codebooks = []
    residual  = train_flat.copy()

    for stage in range(RVQ_STAGES):
        print(f"  Training stage {stage+1}/{RVQ_STAGES}  "
              f"(residual variance: {residual.var():.4f})")
        cb = train_codebook(residual)
        codebooks.append(cb)

        cb_path = os.path.join(codebook_dir, f"codebook_stage{stage+1}.npy")
        np.save(cb_path, cb)

        # Advance residual
        _, residual = assign_tokens(residual, cb)

    print(f"  Final residual variance: {residual.var():.4f}  "
          f"(lower = better reconstruction)")

    # ── ASSIGN TOKENS FOR ALL SPLITS ─────────────────
    print(f"\n[2/3] Assigning RVQ tokens to all splits...")

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

        print(f"\n  [{split:5s}]  vectors={flat.shape}  tokens={tokens.shape}")
        for stage in range(RVQ_STAGES):
            compute_codebook_stats(tokens[:, stage], stage)

    # ── SUMMARY ───────────────────────────────────────
    print(f"\n[3/3] Summary — LOG-MEL RVQ:")
    print(f"  Codebooks : codebook_stage1.npy … codebook_stage{RVQ_STAGES}.npy")
    print(f"  Tokens    : tokens_train.npy / tokens_val.npy / tokens_test.npy")
    print(f"              shape per split: (N*H*W, {RVQ_STAGES})")
    print(f"\n  Compression chain complete:")
    print(f"  Raw(512) → Log-Mel(80) → Latent(20) → RVQ Tokens({RVQ_STAGES} × {CODEBOOK_SIZE} codes)")

    print(f"\n[✓] RVQ complete.")
    print(f"\nNext step → python src/evaluate.py")


if __name__ == "__main__":
    quantize()
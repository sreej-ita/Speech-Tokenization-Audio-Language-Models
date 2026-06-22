import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────
# WHAT IS HAPPENING HERE?
#
# CNN Autoencoder with mentor-aligned compression chain:
#
#   Input  : (batch, 1, 80, 301)
#   Latent : (batch, 20, 20, 76)   ← 20 = LATENT_DIM
#   Output : (batch, 1, 80, 301)
#
# Compression chain per time frame:
#   Raw audio  : 512 samples  (32ms window)
#   Log-Mel    : 80 values    (mel filterbank)
#   Latent     : 20 values    (CNN encoder bottleneck)  ← mentor's target
#   Token      : 1 integer    (VQ codebook)
# ─────────────────────────────────────────────────────


class Encoder(nn.Module):
    """
    CNN Encoder: compresses (1, 80, T) → (latent_dim, 20, T//4)

    Two Conv2d layers with stride=2 each halve both spatial dimensions.
    Final Conv2d projects to latent_dim=20 channels (mentor's compression target).
    """
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 32 channels, spatial size unchanged
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2: 32 → 64, stride=2 halves height and width
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3: 64 → 128, spatial size unchanged
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 4: 128 → latent_dim=20, stride=2 halves again
            # After this: (batch, 20, 20, T//4)
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x  : (batch, 1, 80, T)
        return self.encoder(x)
        # out: (batch, latent_dim, 20, T//4)


class Decoder(nn.Module):
    """
    CNN Decoder: expands (latent_dim, 20, T//4) → (1, 80, T)

    Mirrors the encoder using ConvTranspose2d with stride=2 to upsample.
    """
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.decoder = nn.Sequential(
            # Layer 1: latent_dim → 128, stride=2 doubles spatial size
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 2: 128 → 64, spatial size unchanged
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3: 64 → 32, stride=2 doubles again → back to original size
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 4: 32 → 1, restore to original feature map
            # No activation — output should match raw feature range
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        # z  : (batch, latent_dim, 20, T//4)
        return self.decoder(z)
        # out: (batch, 1, 80, T)


class CNNAutoencoder(nn.Module):
    """
    Full Autoencoder = Encoder + Decoder.

    Usage:
        model = CNNAutoencoder(latent_dim=20)
        x_hat, z = model(x)

        x     : (batch, 1, 80, T)          ← input log-mel
        z     : (batch, 20, 20, T//4)      ← latent (compressed)
        x_hat : (batch, 1, 80, T)          ← reconstruction
    """
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.encoder    = Encoder(latent_dim)
        self.decoder    = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        # Crop to exactly match input size (ConvTranspose2d can be off by 1)
        x_hat = x_hat[:, :, :x.shape[2], :x.shape[3]]
        return x_hat, z

    def encode(self, x):
        """Use in encode.py — extract latents without decoding."""
        return self.encoder(x)


# ── SANITY CHECK ──────────────────────────────────────
if __name__ == "__main__":

    LATENT_DIM  = 20
    TIME_FRAMES = 301
    FREQ_BINS   = 80

    model    = CNNAutoencoder(latent_dim=LATENT_DIM)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {n_params:,}")

    x = torch.randn(8, 1, FREQ_BINS, TIME_FRAMES)
    print(f"  Input  shape: {tuple(x.shape)}")

    x_hat, z = model(x)
    print(f"  Latent shape: {tuple(z.shape)}  ← {z.shape[1]} latent dims (mentor target: 20)")
    print(f"  Output shape: {tuple(x_hat.shape)}")

    assert x_hat.shape == x.shape, f"Shape mismatch! {x.shape} vs {x_hat.shape}"

    loss = nn.MSELoss()(x_hat, x)
    print(f"  MSE loss (untrained): {loss.item():.4f}")
    print(f"  [✓] Checks passed")

    print("\nNext step → python src/train.py")
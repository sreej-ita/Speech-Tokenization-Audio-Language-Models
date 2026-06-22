import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Two strided Conv2d layers compress (1, 80, T) → (latent_dim, 20, T//4)."""
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.encoder = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x  : (batch, 1, 80, T)
        return self.encoder(x)
        # out: (batch, latent_dim, 20, T//4)


class Decoder(nn.Module):
    """Mirrors the encoder with ConvTranspose2d to recover (1, 80, T)."""
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
    """Full autoencoder. Returns (x_hat, z) where z is the latent."""
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
    print(f"   Checks passed")

    print(f"ok  input={tuple(x.shape)}  latent={tuple(z.shape)}  output={tuple(x_hat.shape)}")
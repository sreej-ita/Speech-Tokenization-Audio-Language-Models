import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys

sys.path.append('src')
from model import CNNAutoencoder

plt.rcParams.update({
    'figure.dpi'       : 120,
    'font.family'      : 'monospace',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})
TEAL   = '#0F6E56'
GRAY   = '#888888'

os.makedirs('outputs', exist_ok=True)

RVQ_STAGES    = 4
CODEBOOK_SIZE = 256


# ─────────────────────────────────────────────────────
# fig1: Feature Visualization
# ─────────────────────────────────────────────────────
logmel_data   = np.load('data/train/log_mel.npy')
logmel_sample = logmel_data[0]

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(logmel_sample, aspect='auto', origin='lower',
               cmap='magma', interpolation='nearest')
ax.set_title(f'Log-Mel Spectrogram  shape={logmel_sample.shape}', color=TEAL, fontsize=11)
ax.set_ylabel('Mel band')
ax.set_xlabel('Time frame  (1 frame = 10ms)')
plt.colorbar(im, ax=ax, label='Normalized amplitude')
plt.tight_layout()
plt.savefig('outputs/fig1_features.png', bbox_inches='tight')
plt.show()
print("fig1 saved")


# ─────────────────────────────────────────────────────
# fig2: Training Curves
# ─────────────────────────────────────────────────────
ckpt = torch.load('outputs/logmel/checkpoints/best_model.pt', map_location='cpu')

fig, ax = plt.subplots(figsize=(8, 4))
train_l = ckpt['train_losses']
val_l   = ckpt['val_losses']
epochs  = range(1, len(train_l) + 1)
best    = min(val_l)

ax.plot(epochs, train_l, color=TEAL, lw=2, label='Train')
ax.plot(epochs, val_l,   color=TEAL, lw=2, linestyle='--', alpha=0.7, label='Val')
ax.axhline(best, color=GRAY, lw=1, linestyle=':', label=f'Best val: {best:.4f}')
ax.set_title('Log-Mel — MSE Loss', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/fig2_training_curves.png', bbox_inches='tight')
plt.show()
print(f"fig2 saved  best_val={best:.4f}")


# ─────────────────────────────────────────────────────
# fig3: Reconstruction Quality
# ─────────────────────────────────────────────────────
model = CNNAutoencoder(latent_dim=ckpt['latent_dim'])
model.load_state_dict(ckpt['model_state'])
model.eval()

data = np.load('data/test/log_mel.npy').astype(np.float32)
x    = torch.tensor(data[0]).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    x_hat, _ = model(x)

original   = x.squeeze().numpy()
recon      = x_hat.squeeze().numpy()
mse        = float(((original - recon) ** 2).mean())
vmin, vmax = original.min(), original.max()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].imshow(original, aspect='auto', origin='lower',
               cmap='magma', vmin=vmin, vmax=vmax)
axes[0].set_title('Original', color=TEAL, fontsize=11)
axes[0].set_ylabel('Mel band')
axes[0].set_xlabel('Time frame')

axes[1].imshow(recon, aspect='auto', origin='lower',
               cmap='magma', vmin=vmin, vmax=vmax)
axes[1].set_title(f'Reconstructed  (MSE={mse:.4f})', color=TEAL, fontsize=11)
axes[1].set_ylabel('Mel band')
axes[1].set_xlabel('Time frame')

plt.suptitle('Reconstruction — test split', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/fig3_reconstruction.png', bbox_inches='tight')
plt.show()
print(f"fig3 saved  mse={mse:.4f}")


# ─────────────────────────────────────────────────────
# fig4: RVQ Per-Stage Reconstruction MSE
# ─────────────────────────────────────────────────────
codebooks = [
    np.load(f'outputs/logmel/codebook/codebook_stage{s+1}.npy')
    for s in range(RVQ_STAGES)
]

test_latents = np.load('outputs/logmel/latents/test.npy')
N, D, H, W  = test_latents.shape
test_flat    = test_latents.transpose(0, 2, 3, 1).reshape(-1, D)
tokens_test  = np.load('outputs/logmel/codebook/tokens_test.npy')

stage_mses       = []
cumulative_recon = np.zeros_like(test_flat)

for stage, cb in enumerate(codebooks):
    cumulative_recon += cb[tokens_test[:, stage]]
    mse = float(((test_flat - cumulative_recon) ** 2).mean())
    stage_mses.append(mse)
    print(f"stage {stage+1}  cumulative_mse={mse:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, RVQ_STAGES + 1), stage_mses, marker='o', color=TEAL, lw=2)
ax.set_title('RVQ — Cumulative Reconstruction MSE per Stage  (test split)', fontsize=11)
ax.set_xlabel('RVQ Stage')
ax.set_ylabel('MSE (latent space)')
ax.set_xticks(range(1, RVQ_STAGES + 1))
plt.tight_layout()
plt.savefig('outputs/fig4_rvq_stage_mse.png', bbox_inches='tight')
plt.show()
print("fig4 saved")


print("\noutputs/fig1_features.png")
print("outputs/fig2_training_curves.png")
print("outputs/fig3_reconstruction.png")
print("outputs/fig4_rvq_stage_mse.png")
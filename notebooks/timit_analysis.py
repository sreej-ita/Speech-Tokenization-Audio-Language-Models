import os
import sys
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.append('src')
from model import CNNAutoencoder

# ─────────────────────────────────────────────────────
# Visualize acoustic differences between male and female
# speakers saying the same sentence (TIMIT SA1).
# Uses the trained encoder to extract latents and RVQ
# codebooks to assign tokens.
# ─────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.dpi'       : 120,
    'font.family'      : 'monospace',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})
TEAL   = '#0F6E56'
PURPLE = '#534AB7'
GRAY   = '#888888'

# ── settings matching extract_features.py ─────────────
SR         = 16000
DURATION   = 3
N_FFT      = 512
HOP_LENGTH = 160
N_MELS     = 80
MAX_FRAMES = 301
TIMIT_DIR  = 'outputs/timit/data'
RVQ_STAGES = 4
CODEBOOK_SIZE = 256

os.makedirs('outputs/timit_analysis', exist_ok=True)


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────
def extract_logmel(filepath):
    """Extract normalized log-mel from a .WAV file."""
    waveform, _ = librosa.load(filepath, sr=SR, duration=DURATION)
    if len(waveform) < SR:
        return None
    mel     = librosa.feature.melspectrogram(y=waveform, sr=SR, n_fft=N_FFT,
                                              hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    # pad or truncate
    if log_mel.shape[1] < MAX_FRAMES:
        log_mel = np.pad(log_mel, ((0,0),(0, MAX_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :MAX_FRAMES]
    return log_mel


def collect_sa1_files(timit_dir):
    """
    Collect SA1.WAV files from TIMIT, separated by gender.
    Speaker folders starting with M = male, F = female.
    Returns dict: {'M': [...paths...], 'F': [...paths...]}
    """
    files = {'M': [], 'F': []}
    for split in ['TRAIN', 'TEST']:
        split_path = os.path.join(timit_dir, split)
        if not os.path.exists(split_path):
            continue
        for dr in os.listdir(split_path):
            dr_path = os.path.join(split_path, dr)
            for speaker in os.listdir(dr_path):
                gender   = speaker[0].upper()
                if gender not in files:
                    continue
                sa1_path = os.path.join(dr_path, speaker, 'SA1.WAV')
                if os.path.exists(sa1_path):
                    files[gender].append(sa1_path)
    print(f"found {len(files['M'])} male  speakers")
    print(f"found {len(files['F'])} female speakers")
    return files


# ─────────────────────────────────────────────────────
# LOAD MODEL AND CODEBOOKS
# ─────────────────────────────────────────────────────
ckpt      = torch.load('outputs/logmel/checkpoints/best_model.pt', map_location='cpu')
model     = CNNAutoencoder(latent_dim=ckpt['latent_dim'])
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"loaded encoder  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

codebooks = [
    np.load(f'outputs/logmel/codebook/codebook_stage{s+1}.npy')
    for s in range(RVQ_STAGES)
]


def encode_logmel(log_mel):
    """Run log-mel through encoder → latent vector (flattened)."""
    x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        z = model.encode(x)   # (1, 20, 20, 76)
    z_np    = z.squeeze(0).numpy()             # (20, 20, 76)
    N, H, W = z_np.shape
    flat    = z_np.transpose(1, 2, 0).reshape(-1, N)  # (H*W, 20)
    return flat


def assign_rvq_tokens(flat):
    """Assign RVQ tokens for a flattened latent array."""
    residual   = flat.copy()
    all_tokens = []
    for cb in codebooks:
        z_sq      = (residual**2).sum(axis=1, keepdims=True)
        c_sq      = (cb**2).sum(axis=1)
        cross     = residual @ cb.T
        dists     = z_sq + c_sq - 2*cross
        token_ids = dists.argmin(axis=1).astype(np.int32)
        residual  = residual - cb[token_ids]
        all_tokens.append(token_ids)
    return np.stack(all_tokens, axis=1)   # (H*W, RVQ_STAGES)


# ─────────────────────────────────────────────────────
# COLLECT FILES AND EXTRACT FEATURES
# ─────────────────────────────────────────────────────
sa1_files = collect_sa1_files(TIMIT_DIR)

male_logmels   = []
female_logmels = []
male_latents   = []
female_latents = []
male_tokens    = []
female_tokens  = []

for path in sa1_files['M']:
    lm = extract_logmel(path)
    if lm is not None:
        flat = encode_logmel(lm)
        male_logmels.append(lm)
        male_latents.append(flat)
        male_tokens.append(assign_rvq_tokens(flat))

for path in sa1_files['F']:
    lm = extract_logmel(path)
    if lm is not None:
        flat = encode_logmel(lm)
        female_logmels.append(lm)
        female_latents.append(flat)
        female_tokens.append(assign_rvq_tokens(flat))

print(f"extracted {len(male_logmels)} male  clips")
print(f"extracted {len(female_logmels)} female clips")


# ─────────────────────────────────────────────────────
# fig1: Log-mel comparison — one male vs one female
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].imshow(male_logmels[0], aspect='auto', origin='lower',
               cmap='magma', interpolation='nearest')
axes[0].set_title('Male Speaker — SA1', color=TEAL, fontsize=11)
axes[0].set_ylabel('Mel band')
axes[0].set_xlabel('Time frame')

axes[1].imshow(female_logmels[0], aspect='auto', origin='lower',
               cmap='magma', interpolation='nearest')
axes[1].set_title('Female Speaker — SA1', color=PURPLE, fontsize=11)
axes[1].set_ylabel('Mel band')
axes[1].set_xlabel('Time frame')

plt.suptitle('Same Sentence (SA1) — Different Acoustic Properties', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/timit_analysis/fig1_spectrogram_comparison.png', bbox_inches='tight')
plt.show()
print("fig1 saved")


# ─────────────────────────────────────────────────────
# fig2: t-SNE of latent vectors colored by gender
# ─────────────────────────────────────────────────────
# stack all latents
male_means   = np.array([lat.mean(axis=0) for lat in male_latents])
female_means = np.array([lat.mean(axis=0) for lat in female_latents])

samp = np.concatenate([male_means, female_means], axis=0)
labs = np.array(['M'] * len(male_means) + ['F'] * len(female_means))

print("running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
proj = tsne.fit_transform(samp)

fig, ax = plt.subplots(figsize=(8, 6))
for gender, color, label in [('M', TEAL, 'Male'), ('F', PURPLE, 'Female')]:
    mask = labs == gender
    ax.scatter(proj[mask, 0], proj[mask, 1],
               alpha=0.4, s=8, color=color, label=label)

ax.set_title('t-SNE of Latent Vectors — SA1\nColored by Gender', fontsize=11)
ax.set_xlabel('t-SNE dim 1')
ax.set_ylabel('t-SNE dim 2')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/timit_analysis/fig2_tsne_gender.png', bbox_inches='tight')
plt.show()
print("fig2 saved")


# ─────────────────────────────────────────────────────
# fig3: Token distribution — male vs female (stage 1)
# ─────────────────────────────────────────────────────
male_tok_all   = np.concatenate(male_tokens,   axis=0)[:, 0]  # stage 1 tokens
female_tok_all = np.concatenate(female_tokens, axis=0)[:, 0]

male_counts   = np.bincount(male_tok_all,   minlength=CODEBOOK_SIZE)
female_counts = np.bincount(female_tok_all, minlength=CODEBOOK_SIZE)

# normalize to frequency
male_freq   = male_counts   / male_counts.sum()
female_freq = female_counts / female_counts.sum()

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].bar(range(CODEBOOK_SIZE), male_freq,   color=TEAL,   alpha=0.8, width=1.0)
axes[0].set_title('Male Token Distribution — RVQ Stage 1', color=TEAL, fontsize=11)
axes[0].set_ylabel('Frequency')

axes[1].bar(range(CODEBOOK_SIZE), female_freq, color=PURPLE, alpha=0.8, width=1.0)
axes[1].set_title('Female Token Distribution — RVQ Stage 1', color=PURPLE, fontsize=11)
axes[1].set_ylabel('Frequency')
axes[1].set_xlabel('Token ID')

plt.suptitle('Token Distribution — Same Sentence (SA1), Different Genders', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/timit_analysis/fig3_token_distribution.png', bbox_inches='tight')
plt.show()
print("fig3 saved")

print("\noutputs/timit_analysis/fig1_spectrogram_comparison.png")
print("outputs/timit_analysis/fig2_tsne_gender.png")
print("outputs/timit_analysis/fig3_token_distribution.png")
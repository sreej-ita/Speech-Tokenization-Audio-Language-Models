import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
import torch
import librosa

sys.path.append('src')
from model import CNNAutoencoder
from bpe import apply_bpe, quantize_timit_sequence

plt.rcParams.update({
    'figure.dpi'       : 120,
    'font.family'      : 'monospace',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})
TEAL   = '#0F6E56'
PURPLE = '#534AB7'
GRAY   = '#888888'

os.makedirs('outputs/bpe_analysis', exist_ok=True)

BPE_DIR    = 'outputs/logmel/bpe'
VQ_DIR     = 'outputs/logmel/vq_single'
TIMIT_DIR  = 'outputs/timit/data'
SR         = 16000
DURATION   = 3
N_FFT      = 512
HOP_LENGTH = 160
N_MELS     = 80
MAX_FRAMES = 301
FREQ_DIM   = 20
TIME_DIM   = 76
INITIAL_VOCAB = 256


# ─────────────────────────────────────────────────────
# load saved BPE data
# ─────────────────────────────────────────────────────
vocab_growth = np.load(os.path.join(BPE_DIR, 'vocab_growth.npy'))
merges       = np.load(os.path.join(BPE_DIR, 'merges.npy'),
                       allow_pickle=True).tolist()
train_seqs   = np.load(os.path.join(BPE_DIR, 'sequences_train.npy'),
                       allow_pickle=True).tolist()
val_seqs     = np.load(os.path.join(BPE_DIR, 'sequences_val.npy'),
                       allow_pickle=True).tolist()
test_seqs    = np.load(os.path.join(BPE_DIR, 'sequences_test.npy'),
                       allow_pickle=True).tolist()

final_vocab = INITIAL_VOCAB + len(merges)
print(f"loaded {len(merges)} merge rules  final_vocab={final_vocab}")


# ─────────────────────────────────────────────────────
# fig1: vocabulary growth curve
# ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(vocab_growth)), vocab_growth, color=TEAL, lw=2)
ax.set_title('BPE Vocabulary Growth', fontsize=11)
ax.set_xlabel('Number of merges')
ax.set_ylabel('Vocabulary size')
ax.axhline(INITIAL_VOCAB, color=GRAY, lw=1, linestyle='--',
           label=f'Initial VQ vocab ({INITIAL_VOCAB})')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/bpe_analysis/fig1_vocab_growth.png', bbox_inches='tight')
plt.show()
print("fig1 saved")


# ─────────────────────────────────────────────────────
# fig2: sequence length reduction across splits
# ─────────────────────────────────────────────────────
splits = ['train', 'val', 'test']
before = [TIME_DIM] * 3
after  = [
    np.mean([len(s) for s in train_seqs]),
    np.mean([len(s) for s in val_seqs]),
    np.mean([len(s) for s in test_seqs]),
]

x     = np.arange(3)
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - width/2, before, width, color=GRAY, alpha=0.8, label='Before BPE')
ax.bar(x + width/2, after,  width, color=TEAL, alpha=0.8, label='After BPE')
ax.set_title('Sequence Length — Before vs After BPE', fontsize=11)
ax.set_ylabel('Avg tokens per clip')
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.legend(fontsize=9)
for i, (b, a) in enumerate(zip(before, after)):
    ax.text(i - width/2, b + 0.5, str(b),       ha='center', fontsize=9)
    ax.text(i + width/2, a + 0.5, f'{a:.1f}',   ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/bpe_analysis/fig2_sequence_length.png', bbox_inches='tight')
plt.show()
print("fig2 saved")


# ─────────────────────────────────────────────────────
# load model and VQ codebook for TIMIT processing
# ─────────────────────────────────────────────────────
ckpt        = torch.load('outputs/logmel/checkpoints/best_model.pt', map_location='cpu')
model       = CNNAutoencoder(latent_dim=ckpt['latent_dim'])
model.load_state_dict(ckpt['model_state'])
model.eval()
vq_codebook = np.load(os.path.join(VQ_DIR, 'codebook.npy'))


def extract_logmel(filepath):
    waveform, _ = librosa.load(filepath, sr=SR, duration=DURATION)
    if len(waveform) < SR:
        return None
    mel     = librosa.feature.melspectrogram(y=waveform, sr=SR, n_fft=N_FFT,
                                              hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    if log_mel.shape[1] < MAX_FRAMES:
        log_mel = np.pad(log_mel, ((0,0),(0, MAX_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :MAX_FRAMES]
    return log_mel


def get_vq_tokens(log_mel):
    x    = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        z = model.encode(x)
    z_np  = z.squeeze(0).numpy()
    flat  = z_np.transpose(1, 2, 0).reshape(-1, 20)
    z_sq  = (flat**2).sum(axis=1, keepdims=True)
    c_sq  = (vq_codebook**2).sum(axis=1)
    cross = flat @ vq_codebook.T
    dists = z_sq + c_sq - 2*cross
    return dists.argmin(axis=1).astype(np.int32)


def collect_sa1_files(timit_dir):
    files = {'M': [], 'F': []}
    for split in ['TRAIN', 'TEST']:
        split_path = os.path.join(timit_dir, split)
        if not os.path.exists(split_path):
            continue
        for dr in os.listdir(split_path):
            dr_path = os.path.join(split_path, dr)
            for speaker in os.listdir(dr_path):
                gender = speaker[0].upper()
                if gender not in files:
                    continue
                sa1_path = os.path.join(dr_path, speaker, 'SA1.WAV')
                if os.path.exists(sa1_path):
                    files[gender].append(sa1_path)
    return files


def seq_to_freq_vector(seq, vocab_size):
    """Convert token sequence to frequency vector of size vocab_size."""
    vec = np.zeros(vocab_size, dtype=np.float32)
    for t in seq:
        vec[t] += 1
    return vec / (vec.sum() + 1e-8)


# collect and process all TIMIT SA1 speakers
sa1_files = collect_sa1_files(TIMIT_DIR)

male_bpe_seqs   = []
female_bpe_seqs = []

print("extracting TIMIT BPE sequences...")
for path in sa1_files['M']:
    lm = extract_logmel(path)
    if lm is None:
        continue
    vq_tokens = get_vq_tokens(lm)
    time_seq  = quantize_timit_sequence(vq_tokens)
    bpe_seq   = apply_bpe([time_seq], merges)[0]
    male_bpe_seqs.append(bpe_seq)

for path in sa1_files['F']:
    lm = extract_logmel(path)
    if lm is None:
        continue
    vq_tokens = get_vq_tokens(lm)
    time_seq  = quantize_timit_sequence(vq_tokens)
    bpe_seq   = apply_bpe([time_seq], merges)[0]
    female_bpe_seqs.append(bpe_seq)

print(f"male   BPE sequences: {len(male_bpe_seqs)}")
print(f"female BPE sequences: {len(female_bpe_seqs)}")


# ─────────────────────────────────────────────────────
# fig3: t-SNE of BPE token frequency vectors
# ─────────────────────────────────────────────────────
male_vecs   = np.array([seq_to_freq_vector(s, final_vocab) for s in male_bpe_seqs])
female_vecs = np.array([seq_to_freq_vector(s, final_vocab) for s in female_bpe_seqs])

all_vecs = np.concatenate([male_vecs, female_vecs], axis=0)
labels   = np.array(['M'] * len(male_vecs) + ['F'] * len(female_vecs))

print("running t-SNE on BPE vectors...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
proj = tsne.fit_transform(all_vecs)

fig, ax = plt.subplots(figsize=(8, 6))
for gender, color, label in [('M', TEAL, 'Male'), ('F', PURPLE, 'Female')]:
    mask = labels == gender
    ax.scatter(proj[mask, 0], proj[mask, 1],
               alpha=0.6, s=20, color=color, label=label)
ax.set_title('t-SNE of BPE Token Vectors — SA1\nColored by Gender', fontsize=11)
ax.set_xlabel('t-SNE dim 1')
ax.set_ylabel('t-SNE dim 2')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/bpe_analysis/fig3_tsne_bpe.png', bbox_inches='tight')
plt.show()
print("fig3 saved")


# ─────────────────────────────────────────────────────
# fig4: token sequence visualization — one male vs one female
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 4))

for ax, seq, color, label in [
    (axes[0], male_bpe_seqs[0],   TEAL,   'Male'),
    (axes[1], female_bpe_seqs[0], PURPLE, 'Female'),
]:
    ax.bar(range(len(seq)), [1]*len(seq),
           color=[plt.cm.tab20(t % 20) for t in seq],
           width=1.0)
    ax.set_title(f'{label} — BPE Token Sequence  ({len(seq)} tokens)',
                 color=color, fontsize=11)
    ax.set_yticks([])

axes[1].set_xlabel('Position in sequence')
plt.suptitle('BPE Token Sequences — Same Sentence (SA1), Different Genders', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/bpe_analysis/fig4_token_sequences.png', bbox_inches='tight')
plt.show()
print("fig4 saved")
# ─────────────────────────────────────────────────────
# fig5: side by side t-SNE — latent space vs BPE space
# ─────────────────────────────────────────────────────
# latent vectors — recompute mean per speaker
male_latents   = []
female_latents = []

for path in sa1_files['M']:
    lm = extract_logmel(path)
    if lm is None:
        continue
    x = torch.tensor(lm).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        z = model.encode(x)
    z_np = z.squeeze(0).numpy()                          # (20, 20, 76)
    flat = z_np.transpose(1, 2, 0).reshape(-1, 20)       # (1520, 20)
    male_latents.append(flat.mean(axis=0))                # (20,)

for path in sa1_files['F']:
    lm = extract_logmel(path)
    if lm is None:
        continue
    x = torch.tensor(lm).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        z = model.encode(x)
    z_np = z.squeeze(0).numpy()
    flat = z_np.transpose(1, 2, 0).reshape(-1, 20)
    female_latents.append(flat.mean(axis=0))

male_lat_arr   = np.array(male_latents)
female_lat_arr = np.array(female_latents)

lat_all    = np.concatenate([male_lat_arr, female_lat_arr], axis=0)
lat_labels = np.array(['M'] * len(male_lat_arr) + ['F'] * len(female_lat_arr))

print("running t-SNE on latent vectors...")
tsne_lat  = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
proj_lat  = tsne_lat.fit_transform(lat_all)

print("running t-SNE on BPE vectors...")
tsne_bpe  = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
proj_bpe  = tsne_bpe.fit_transform(all_vecs)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, proj, labels_arr, title in [
    (axes[0], proj_lat, lat_labels, 'Latent Space'),
    (axes[1], proj_bpe, labels,     'BPE Token Space'),
]:
    for gender, color, label in [('M', TEAL, 'Male'), ('F', PURPLE, 'Female')]:
        mask = labels_arr == gender
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   alpha=0.6, s=20, color=color, label=label)
    ax.set_title(f't-SNE — {title}\nColored by Gender', fontsize=11)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(fontsize=9)

plt.suptitle('Gender Separation — Latent Space vs BPE Token Space (SA1)', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/bpe_analysis/fig5_tsne_comparison.png', bbox_inches='tight')
plt.show()
print("fig5 saved")

print("\noutputs/bpe_analysis/fig1_vocab_growth.png")
print("outputs/bpe_analysis/fig2_sequence_length.png")
print("outputs/bpe_analysis/fig3_tsne_bpe.png")
print("outputs/bpe_analysis/fig4_token_sequences.png")
print("outputs/bpe_analysis/fig5_tsne_comparison.png")
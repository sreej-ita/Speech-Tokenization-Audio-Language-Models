import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────
# WHAT IS HAPPENING HERE?
#
# PyTorch needs data delivered in a specific way:
#   - A Dataset object that knows how to load ONE sample
#   - A DataLoader that batches many samples together
#
# Our .npy files now live in speaker-split folders:
#   data/train/log_mel.npy  → (109, 80, 301)
#   data/val/log_mel.npy    → (116, 80, 301)
#   data/test/log_mel.npy   → (186, 80, 301)
#
# The autoencoder expects: (batch, 1, freq_bins, time_frames)
# The extra "1" is the channel dimension (like a grayscale image)
# ─────────────────────────────────────────────────────


class SpeechFeatureDataset(Dataset):
    """
    Loads pre-extracted log-mel features from a .npy file.

    Args:
        npy_path  : path to .npy file e.g. "data/train/log_mel.npy"
        normalize : re-normalize per sample (False since already done in extract_features.py)

    Returns per sample:
        torch.Tensor shape (1, freq_bins, time_frames)
        e.g. (1, 80, 301) for log-mel
    """

    def __init__(self, npy_path: str, normalize: bool = False):
        self.normalize = normalize

        data = np.load(npy_path)
        self.data = data.astype(np.float32)

        self.n_samples   = self.data.shape[0]
        self.freq_bins   = self.data.shape[1]
        self.time_frames = self.data.shape[2]

        print(f"[Dataset] Loaded '{npy_path}'")
        print(f"  n_samples   : {self.n_samples}")
        print(f"  freq_bins   : {self.freq_bins}")
        print(f"  time_frames : {self.time_frames}")
        print(f"  shape/sample: (1, {self.freq_bins}, {self.time_frames})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx]   # (freq_bins, time_frames)

        if self.normalize:
            mean = x.mean()
            std  = x.std() + 1e-8
            x    = (x - mean) / std

        # Add channel dim: (freq_bins, time_frames) → (1, freq_bins, time_frames)
        return torch.tensor(x).unsqueeze(0)


# ── SANITY CHECK ──────────────────────────────────────
if __name__ == "__main__":
    import os

    splits = ["train", "val", "test"]

    for split in splits:
        path = os.path.join("data", split, "log_mel.npy")
        if not os.path.exists(path):
            print(f"  [!] Not found: {path}")
            continue

        ds     = SpeechFeatureDataset(path)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        batch  = next(iter(loader))

        print(f"  [{split:5s}] batch shape : {tuple(batch.shape)} | "
              f"min: {batch.min():.2f} | max: {batch.max():.2f}")

        assert batch.shape[1] == 1,              "channel dim must be 1"
        assert batch.shape[2] == ds.freq_bins,   "freq_bins mismatch"
        assert batch.shape[3] == ds.time_frames, "time_frames mismatch"
        print(f"         [✓] checks passed")

    print("\nNext step → python src/model.py")
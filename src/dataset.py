import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SpeechFeatureDataset(Dataset):
    """Dataset wrapper for log-mel .npy files. Returns (1, freq_bins, time_frames) tensors."""

    def __init__(self, npy_path: str, normalize: bool = False):
        self.normalize = normalize

        data = np.load(npy_path)
        self.data = data.astype(np.float32)

        self.n_samples   = self.data.shape[0]
        self.freq_bins   = self.data.shape[1]
        self.time_frames = self.data.shape[2]

        print(f"loaded {npy_path}  {self.data.shape}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx]   # (freq_bins, time_frames)

        if self.normalize:
            mean = x.mean()
            std  = x.std() + 1e-8
            x    = (x - mean) / std

        return torch.tensor(x).unsqueeze(0)


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

        assert batch.shape[1] == 1             
        assert batch.shape[2] == ds.freq_bins  
        assert batch.shape[3] == ds.time_frames  
        

        print(f"[{split}] ok  batch={tuple(batch.shape)}")
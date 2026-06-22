import os
import numpy as np
import librosa

# ── SETTINGS ──────────────────────────────────────────
DATA_DIR = "test-clean/LibriSpeech/test-clean"
SAVE_DIR = "data"

# 6 male speakers from LibriSpeech test-clean
TRAIN_SPEAKERS = ["1089", "1188"]   # Peter Bobbe, Duncan Murrell
VAL_SPEAKERS   = ["1320", "908"]    # number6, Sam Stinson
TEST_SPEAKERS  = ["260",  "61"]     # Brad Bush, Paul-Gabriel Wiener

SR         = 16000   # sample rate (Hz)
DURATION   = 3       # seconds per clip
N_FFT      = 512     # FFT window — 32ms at 16kHz (power of 2)
HOP_LENGTH = 160     # 10ms stride — speech standard
N_MELS     = 80      # mel filterbank bands

# Time frames after feature extraction:
# floor((16000 * 3) / 160) + 1 = 301
MAX_FRAMES = 301


# ── HELPER: pad or truncate to MAX_FRAMES ─────────────
def fix_length(feature, max_frames):
    if feature.shape[1] < max_frames:
        pad_amount = max_frames - feature.shape[1]
        feature = np.pad(feature, ((0, 0), (0, pad_amount)), mode="constant")
    else:
        feature = feature[:, :max_frames]
    return feature


# ── COLLECT .flac FILES FOR A LIST OF SPEAKERS ────────
def collect_files(speaker_ids):
    files = []
    for speaker in speaker_ids:
        speaker_path = os.path.join(DATA_DIR, speaker)
        if not os.path.exists(speaker_path):
            print(f"  [!] Speaker {speaker} not found at {speaker_path}")
            continue
        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)
            for filename in os.listdir(chapter_path):
                if filename.endswith(".flac"):
                    files.append(os.path.join(chapter_path, filename))
    return files


# ── EXTRACT FEATURES FROM A LIST OF FILES ─────────────
def extract_features(audio_files, split_name):
    log_mel_list = []
    skipped      = 0

    print(f"\n  Processing {len(audio_files)} files for '{split_name}'...")

    for i, filepath in enumerate(audio_files):
        waveform, sr = librosa.load(filepath, sr=SR, duration=DURATION)

        if len(waveform) < SR:
            skipped += 1
            continue

        # ── LOG-MEL ───────────────────────────────────
        # Shape: (N_MELS, time_frames) = (80, ~301)
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        log_mel = fix_length(log_mel, MAX_FRAMES)   # → (80, 301)

        log_mel_list.append(log_mel)

        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{len(audio_files)} files...")

    print(f"  Done. Extracted {len(log_mel_list)} clips | Skipped: {skipped}")
    return np.stack(log_mel_list)


# ── MAIN ──────────────────────────────────────────────
if __name__ == "__main__":

    splits = {
        "train": TRAIN_SPEAKERS,
        "val"  : VAL_SPEAKERS,
        "test" : TEST_SPEAKERS,
    }

    print("Settings:")
    print(f"  SR={SR}Hz | N_FFT={N_FFT} ({N_FFT/SR*1000:.0f}ms) | "
          f"HOP={HOP_LENGTH} ({HOP_LENGTH/SR*1000:.0f}ms) | MAX_FRAMES={MAX_FRAMES}")
    print(f"  N_MELS={N_MELS}")
    print(f"\nSpeaker split:")
    for split, speakers in splits.items():
        print(f"  {split:5s}: {speakers}")

    for split_name, speaker_ids in splits.items():
        audio_files = collect_files(speaker_ids)
        print(f"\n[{split_name.upper()}] Found {len(audio_files)} audio files "
              f"from speakers {speaker_ids}")

        log_mel_array = extract_features(audio_files, split_name)

        split_dir = os.path.join(SAVE_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, "log_mel.npy"), log_mel_array)

        print(f"  Saved to '{split_dir}/'")
        print(f"    log_mel : {log_mel_array.shape}  → (n_clips, {N_MELS}, {MAX_FRAMES})")

    print("\n[✓] All splits saved.")
    print("    data/train/log_mel.npy")
    print("    data/val/log_mel.npy")
    print("    data/test/log_mel.npy")
    print("\nNext step → python src/dataset.py")
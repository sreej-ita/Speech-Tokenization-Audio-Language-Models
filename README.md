# Speech Tokenization for Audio Language Models

A pipeline for discrete speech tokenization using CNNs and Vector Quantization, built on LibriSpeech data.

## Current Status

### BPE Pipeline
- Core vocabulary tracking and frequency-counting loops implemented in pure C.
- Includes automated text normalization, leading/trailing punctuation stripping, and `<unk>` fallback logic.

### Speech Tokenization Pipeline (In Progress)
- Feature extraction implemented for Log-Mel spectrogram and MFCC using Librosa.
- CNN Autoencoder trained with MSE loss on speaker-independent splits (train/val/test).
- Latent representations being explored for discrete tokenization via Residual Vector Quantization.
- Experimenting with codebook design and token quality evaluation.

## Data & Testing

- `corpus.txt`: Adapted directly from the Wikipedia Artificial Intelligence article.
- `inference.txt`: Custom-crafted synthetic sentences designed to test normalization and `<unk>` token fallbacks.
- Audio data: LibriSpeech test-clean, 6 male speakers split by speaker identity.

## Pipeline Overview

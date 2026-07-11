# Speech Tokenization for Audio Language Models

A Python-based speech tokenization pipeline that converts raw audio into discrete acoustic tokens for language model training. This repository provides scripts for feature extraction, autoencoder training, vector quantization, BPE compression, and acoustic token validation. It also includes a from-scratch text BPE implementation in C.

## Overview

This repository includes:

- Log-mel spectrogram extraction from LibriSpeech recordings
- CNN Autoencoder for learning compact latent speech representations
- Residual Vector Quantization (RVQ) for discrete tokenization
- Single-stage VQ baseline for comparison
- Acoustic BPE compression on top of VQ token sequences
- Evaluation scripts with figures and t-SNE visualizations
- TIMIT SA1 analysis to validate that tokens capture acoustic rather than semantic information
- Text BPE tokenizer implemented from scratch in C

## Two BPE Implementations

This repository contains two independent BPE implementations:

**1. Text BPE (C implementation)**  
A from-scratch BPE tokenizer for text, implemented in pure C. Includes vocabulary tracking, frequency-counting, text normalization, punctuation stripping, and `<unk>` fallback logic. Input files: `corpus.txt` (training) and `inference.txt` (testing).

**2. Acoustic BPE (Python — `src/bpe.py`)**  
BPE applied to discrete audio token sequences, following Shen et al. (2024). Treats VQ token IDs as atomic units and merges frequent co-occurring pairs to reduce sequence length. Grows the vocabulary from 256 to 756 tokens over 500 merge operations, achieving a 27% sequence length reduction on the training split.


### Prerequisites

Python 3.10 or later.

### Installation

Clone the repository:

```bash
git clone https://github.com/sreej-ita/Speech-Tokenization-Audio-Language-Models.git
cd Speech-Tokenization-Audio-Language-Models
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download [LibriSpeech test-clean](https://www.openslr.org/12) and place at:
test-clean/LibriSpeech/test-clean/

## Running the Pipeline

Run scripts in order:

### Feature Extraction
```bash
python src/extract_features.py
```

### Train Autoencoder
```bash
python src/train.py
```

### Extract Latents
```bash
python src/encode.py
```

### Vector Quantization
```bash
python src/vq.py        # RVQ (4 stages)
python src/vq_single.py # Single-stage VQ baseline
```

### BPE Compression
```bash
python src/bpe.py
```

## Evaluation

```bash
python notebooks/analysis.py        # Core pipeline figures
python notebooks/timit_analysis.py  # TIMIT acoustic validation
python notebooks/bpe_analysis.py    # BPE analysis and t-SNE
```

## Dependencies

See `requirements.txt` for all Python packages used.

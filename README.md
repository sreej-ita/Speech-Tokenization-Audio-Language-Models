# Speech-Tokenization-Audio-Language-Models
##  Current Status (BPE Pipeline)
* Core vocabulary tracking and frequency-counting loops implemented in pure C.
* Includes automated text normalization, leading/trailing punctuation stripping, and `<unk>` fallback logic.

## Data & Testing
* **`corpus.txt`**: Adapted directly from the Wikipedia Artificial Intelligence article.
* **`inference.txt`**: Custom-crafted synthetic sentences designed to test normalization and `<unk>` token fallbacks.

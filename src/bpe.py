import os
import numpy as np
from collections import Counter

# BPE over VQ token sequences extracted from the time dimension.
# Starts from 256 VQ codes and learns 500 merge rules.

N_CLIPS_TRAIN = 109
N_CLIPS_VAL   = 116
N_CLIPS_TEST  = 186
FREQ_DIM      = 20
TIME_DIM      = 76
N_MERGES      = 500
INITIAL_VOCAB = 256

VQ_DIR      = 'outputs/logmel/vq_single'
SAVE_DIR    = 'outputs/logmel/bpe'


def load_time_sequences(token_path, n_clips):
    """
    Load flat VQ tokens and reshape to time sequences.
    (n_clips * FREQ_DIM * TIME_DIM,) → (n_clips, TIME_DIM)
    Takes mean token across freq dimension per time frame.
    """
    tokens = np.load(token_path)                              # (N*FREQ*TIME,)
    tokens = tokens.reshape(n_clips, FREQ_DIM, TIME_DIM)     # (N, FREQ, TIME)
    # take most common token across freq dim per time frame
    sequences = []
    for clip in tokens:
        time_tokens = []
        for t in range(TIME_DIM):
            freq_tokens = clip[:, t]
            most_common = Counter(freq_tokens).most_common(1)[0][0]
            time_tokens.append(most_common)
        sequences.append(tuple(time_tokens))
    return sequences


def get_pairs(sequences):
    """Count all adjacent token pairs across all sequences."""
    pair_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair_counts[(seq[i], seq[i+1])] += 1
    return pair_counts


def merge_pair(sequences, pair, new_token):
    """Replace all occurrences of pair with new_token in all sequences."""
    merged = []
    for seq in sequences:
        seq   = list(seq)
        i     = 0
        new_seq = []
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        merged.append(tuple(new_seq))
    return merged


def run_bpe(sequences, n_merges):
    """
    Run BPE for n_merges steps.
    Returns merge rules and final sequences.
    """
    merges       = []
    vocab_size   = INITIAL_VOCAB
    vocab_growth = [vocab_size]

    for merge_idx in range(n_merges):
        pair_counts = get_pairs(sequences)
        if not pair_counts:
            print(f"no more pairs at merge {merge_idx}")
            break

        best_pair = pair_counts.most_common(1)[0][0]
        new_token = vocab_size
        vocab_size += 1

        sequences = merge_pair(sequences, best_pair, new_token)
        merges.append((best_pair, new_token))
        vocab_growth.append(vocab_size)

        if (merge_idx + 1) % 50 == 0:
            avg_len = np.mean([len(s) for s in sequences])
            print(f"merge {merge_idx+1}/{n_merges}  "
                  f"vocab={vocab_size}  avg_seq_len={avg_len:.1f}")

    return merges, sequences, vocab_growth


def apply_bpe(sequences, merges):
    """Apply learned BPE merges to new sequences."""
    for pair, new_token in merges:
        sequences = merge_pair(sequences, pair, new_token)
    return sequences


def quantize_timit_sequence(flat_tokens):
    """
    Convert flat TIMIT VQ tokens to time sequence.
    flat_tokens: (FREQ_DIM * TIME_DIM,) or (N, FREQ_DIM * TIME_DIM)
    """
    if flat_tokens.ndim == 1:
        tokens = flat_tokens.reshape(FREQ_DIM, TIME_DIM)
        time_tokens = []
        for t in range(TIME_DIM):
            most_common = Counter(tokens[:, t]).most_common(1)[0][0]
            time_tokens.append(most_common)
        return tuple(time_tokens)
    else:
        return [quantize_timit_sequence(row) for row in flat_tokens]


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── load sequences ────────────────────────────────
    print("loading VQ token sequences...")
    train_seqs = load_time_sequences(
        os.path.join(VQ_DIR, 'tokens_train.npy'), N_CLIPS_TRAIN)
    val_seqs   = load_time_sequences(
        os.path.join(VQ_DIR, 'tokens_val.npy'),   N_CLIPS_VAL)
    test_seqs  = load_time_sequences(
        os.path.join(VQ_DIR, 'tokens_test.npy'),  N_CLIPS_TEST)

    print(f"train sequences : {len(train_seqs)}  len={len(train_seqs[0])}")
    print(f"val   sequences : {len(val_seqs)}")
    print(f"test  sequences : {len(test_seqs)}")

    # ── run BPE on train ──────────────────────────────
    print(f"\nrunning BPE  merges={N_MERGES}  initial_vocab={INITIAL_VOCAB}...")
    merges, train_merged, vocab_growth = run_bpe(train_seqs, N_MERGES)

    # ── apply to val and test ─────────────────────────
    val_merged  = apply_bpe(val_seqs,  merges)
    test_merged = apply_bpe(test_seqs, merges)

    # ── save ──────────────────────────────────────────
    np.save(os.path.join(SAVE_DIR, 'merges.npy'),
            np.array(merges, dtype=object), allow_pickle=True)
    np.save(os.path.join(SAVE_DIR, 'vocab_growth.npy'),
            np.array(vocab_growth))

    for name, seqs in [('train', train_merged),
                       ('val',   val_merged),
                       ('test',  test_merged)]:
        # save as ragged array since sequences have different lengths after BPE
        np.save(os.path.join(SAVE_DIR, f'sequences_{name}.npy'),
                np.array(seqs, dtype=object), allow_pickle=True)
        avg_len = np.mean([len(s) for s in seqs])
        print(f"[{name}]  sequences={len(seqs)}  avg_len={avg_len:.1f}")

    print(f"\nfinal vocab size : {INITIAL_VOCAB + len(merges)}")
    print(f"compression      : {TIME_DIM} → {np.mean([len(s) for s in train_merged]):.1f} tokens/clip")
    print("done")
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>  // Added for lowercasing characters

#define MAX_VOCAB 256
#define MAX_TOKEN_LEN 128
#define MAX_WORDS 1024
#define MAX_SEQ_LEN 62
#define MAX_PAIRS 4096
#define MAX_MERGES 256
#define EOW "</w>"
#define UNK "<unk>"

#define CORPUS_FILE "corpus.txt"
#define INFERENCE_FILE "inference.txt"

// Struct to track a word's current token sequence, its token count, and corpus frequency
typedef struct {
    char tokens[MAX_SEQ_LEN][MAX_TOKEN_LEN];
    int  len, freq;
    char word[128];
} WordSeq;

// Struct to track adjacent token pair frequencies for evaluation
typedef struct {
    char left[MAX_TOKEN_LEN], right[MAX_TOKEN_LEN];
    int  count;
} PairFreq;

// Struct to store a learned merge rule and its rank/priority order
typedef struct {
    char left[MAX_TOKEN_LEN], right[MAX_TOKEN_LEN], result[MAX_TOKEN_LEN];
    int  rank;
} MergeRule;

// Global state arrays
char vocab[MAX_VOCAB][MAX_TOKEN_LEN];
int vocab_size = 0;
MergeRule merges[MAX_MERGES];
int merge_count = 0;
PairFreq  pairs[MAX_PAIRS];
int pair_count = 0;
WordSeq seqs[MAX_WORDS];
int seq_count = 0;

// Adds a unique token to the vocabulary if it doesn't already exist
void vocab_add(const char *tok) {
    for (int i = 0; i < vocab_size; i++)
        if (strcmp(vocab[i], tok) == 0) return; // Prevent duplicates
    if (vocab_size < MAX_VOCAB) {
        strncpy(vocab[vocab_size++], tok, MAX_TOKEN_LEN - 1);
    }
}

// Dynamically tracks and updates frequencies of adjacent token pairs
void pair_update(const char *l, const char *r, int delta) {
    for (int i = 0; i < pair_count; i++) {
        if (strcmp(pairs[i].left, l) == 0 && strcmp(pairs[i].right, r) == 0) {
            pairs[i].count += delta;
            // If count drops to 0 or less, remove it by swapping with the last element
            if (pairs[i].count <= 0) pairs[i] = pairs[--pair_count];
            return;
        }
    }
    if (delta <= 0) return; // Don't insert a non-existent pair if we are reducing count
    if (pair_count < MAX_PAIRS) {
        strncpy(pairs[pair_count].left,  l, MAX_TOKEN_LEN - 1);
        strncpy(pairs[pair_count].right, r, MAX_TOKEN_LEN - 1);
        pairs[pair_count].count = delta;
        pair_count++;
    }
}

// Reads the corpus file, cleans punctuation, lowercases text, and builds base character vocabulary
void corpusF(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("Error: Could not open corpus file '%s'\n", filename);
        exit(1);
    }

    char chunk[1024];
    while (fscanf(f, "%1023s", chunk) == 1) {
        // --- NEW: Lowercase the entire chunk ---
        for (int i = 0; chunk[i]; i++) {
            chunk[i] = tolower((unsigned char)chunk[i]);
        }

        // --- NEW: Strip leading punctuation ---
        char *start = chunk;
        while (*start && strchr(".,!?;:\"'()[]{}<>_-", *start)) {
            start++;
        }

        // Strip trailing punctuation marks (Expanded punctuation set)
        int len = strlen(start);
        while (len > 0 && strchr(".,!?;:\"'()[]{}<>_-", start[len - 1])) {
            start[len - 1] = '\0';
            len--;
        }
        if (len == 0) continue;

        // Keep track of unique words and update their frequency count
        int found = 0;
        for (int i = 0; i < seq_count; i++) {
            if (strcmp(seqs[i].word, start) == 0) { 
                seqs[i].freq++; 
                found = 1; 
                break; 
            }
        }
        if (!found && seq_count < MAX_WORDS) {
            strncpy(seqs[seq_count].word, start, 127);
            seqs[seq_count].freq = 1;
            seqs[seq_count].len  = 0;
            seq_count++;
        }
    }
    fclose(f);

    // Initialize words by splitting them into individual character tokens + End-Of-Word (EOW)
    for (int i = 0; i < seq_count; i++) {
        WordSeq *s = &seqs[i];
        for (int c = 0; s->word[c]; c++) {
            s->tokens[s->len][0] = s->word[c];
            s->tokens[s->len][1] = '\0';
            vocab_add(s->tokens[s->len]);
            s->len++;
        }
        strncpy(s->tokens[s->len], EOW, MAX_TOKEN_LEN - 1);
        vocab_add(EOW);
        s->len++;
    }
}

// Populates the initial frequency of all adjacent token pairs
void count_pairs(void) {
    for (int i = 0; i < seq_count; i++) {
        WordSeq *s = &seqs[i];
        for (int j = 0; j < s->len - 1; j++) {
            if (strcmp(s->tokens[j], EOW) == 0) continue;
            pair_update(s->tokens[j], s->tokens[j+1], s->freq);
        }
    }
}

// Finds the most frequent pair; breaks ties alphabetically
int best_pair(void) {
    int best = -1;
    for (int i = 0; i < pair_count; i++) {
        if (best == -1) { best = i; continue; }
        if (pairs[i].count > pairs[best].count) {
            best = i;
        } else if (pairs[i].count == pairs[best].count) {
            char a[MAX_TOKEN_LEN*2], b[MAX_TOKEN_LEN*2];
            snprintf(a, sizeof(a), "%s%s", pairs[i].left, pairs[i].right);
            snprintf(b, sizeof(b), "%s%s", pairs[best].left, pairs[best].right);
            if (strcmp(a, b) < 0) best = i;
        }
    }
    // Halt vocabulary expansion if top pair occurs only once or less
    if (best != -1 && pairs[best].count <= 1) return -1;
    return best;
}

// Merges chosen 'left' and 'right' tokens across all corpus sequences and updates pair stats
void update(const char *left, const char *right, const char *new_tok) {
    for (int i = 0; i < seq_count; i++) {
        WordSeq *s = &seqs[i];
        int has = 0;
        for (int j = 0; j < s->len - 1; j++)
            if (strcmp(s->tokens[j], left) == 0 && strcmp(s->tokens[j+1], right) == 0)
                { has = 1; break; }
        if (!has) continue;

        // Step 1: Remove old pairs containing these tokens from tracking lists
        for (int j = 0; j < s->len - 1; j++) {
            if (strcmp(s->tokens[j], EOW) == 0) continue;
            pair_update(s->tokens[j], s->tokens[j+1], -s->freq);
        }

        // Step 2: Construct the new merged sequence representation
        char new_seq[MAX_SEQ_LEN][MAX_TOKEN_LEN];
        int  new_len = 0, j = 0;
        while (j < s->len) {
            if (j < s->len - 1 && strcmp(s->tokens[j], left) == 0 && strcmp(s->tokens[j+1], right) == 0) {
                strncpy(new_seq[new_len++], new_tok, MAX_TOKEN_LEN - 1);
                j += 2;
            } else {
                strncpy(new_seq[new_len++], s->tokens[j++], MAX_TOKEN_LEN - 1);
            }
        }
        memcpy(s->tokens, new_seq, sizeof(new_seq));
        s->len = new_len;

        // Step 3: Add new pair combinations created by the merge back into tracking list
        for (int j = 0; j < s->len - 1; j++) {
            if (strcmp(s->tokens[j], EOW) == 0) continue;
            pair_update(s->tokens[j], s->tokens[j+1], s->freq);
        }
    }
}

// Tokenizes a single string dynamically using learned merge rules, masking unknown characters as <unk>
void tokenize_inline(const char *word) {
    // Start with raw characters + EOW
    char seq[MAX_SEQ_LEN][MAX_TOKEN_LEN];
    int  len = 0;
    for (int c = 0; word[c]; c++) {
        seq[len][0] = word[c]; seq[len][1] = '\0'; len++;
    }
    strncpy(seq[len++], EOW, MAX_TOKEN_LEN - 1);

    // Iteratively apply learned merge rules in chronological training rank order
    for (int m = 0; m < merge_count; m++) {
        char new_seq[MAX_SEQ_LEN][MAX_TOKEN_LEN];
        int  new_len = 0, j = 0;
        while (j < len) {
            if (j < len - 1 &&
                strcmp(seq[j],   merges[m].left)  == 0 &&
                strcmp(seq[j+1], merges[m].right) == 0) {
                strncpy(new_seq[new_len++], merges[m].result, MAX_TOKEN_LEN - 1);
                j += 2;
            } else {
                strncpy(new_seq[new_len++], seq[j++], MAX_TOKEN_LEN - 1);
            }
        }
        memcpy(seq, new_seq, sizeof(new_seq));
        len = new_len;
    }

    // Print resulting subword layout with unknown character verification
    printf("[");
    for (int i = 0; i < len; i++) {
        // Verify if the finalized subword or character exists in the trained vocabulary
        int exists = 0;
        for (int v = 0; v < vocab_size; v++) {
            if (strcmp(vocab[v], seq[i]) == 0) {
                exists = 1;
                break;
            }
        }
        
        // Print the real token if it is known, otherwise fallback to <unk> safely
        if (exists) {
            printf("%s", seq[i]);
        } else {
            printf("<unk>");
        }
        
        if (i < len - 1) printf(", ");
    }
    printf("]\n");
}

int main(void) {
    corpusF(CORPUS_FILE);
    count_pairs();

    // Core BPE Training Loop: Find best pair, record rule, update data, repeat
    int b;
    while ((b = best_pair()) != -1 && pair_count > 0 && merge_count < MAX_MERGES) {
        char left[MAX_TOKEN_LEN], right[MAX_TOKEN_LEN], new_tok[MAX_TOKEN_LEN];
        strncpy(left,  pairs[b].left,  MAX_TOKEN_LEN - 1);
        strncpy(right, pairs[b].right, MAX_TOKEN_LEN - 1);
        
        snprintf(new_tok, sizeof(new_tok), "%s%s", left, right);

        vocab_add(new_tok);
        strncpy(merges[merge_count].left,   left,    MAX_TOKEN_LEN - 1);
        strncpy(merges[merge_count].right,  right,   MAX_TOKEN_LEN - 1);
        strncpy(merges[merge_count].result, new_tok, MAX_TOKEN_LEN - 1);
        merges[merge_count].rank = merge_count + 1;
        merge_count++;

        update(left, right, new_tok);
    }

    printf("\n=== Trained Vocabulary (Size: %d) ===\n", vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        printf("%3d: %s\n", i + 1, vocab[i]);
    }
    printf("======================================\n\n");

    // Inference Phase: Apply learned tokenizer rules to separate unseen texts
    FILE *inf_file = fopen(INFERENCE_FILE, "r");
    if (!inf_file) {
        printf("Error: Could not open inference file '%s'\n", INFERENCE_FILE);
        return 1;
    }
    
    char para[4096] = {0};
    size_t bytes_read = fread(para, 1, sizeof(para) - 1, inf_file);
    para[bytes_read] = '\0';
    fclose(inf_file);

    printf("\nInference input:\n%s\n\n", para);
    
    printf("Tokenization:\n");
    char para_buf[4096];
    strncpy(para_buf, para, sizeof(para_buf) - 1);
    para_buf[sizeof(para_buf) - 1] = '\0';
    
    // --- UPDATED: Expanded token-splitting characters to cleanly isolate raw words ---
    char *word = strtok(para_buf, " \t\n.,!?;:\"'()[]{}<>_-");
    while (word) {
        // --- NEW: Lowercase the inference word inline to match trained model expectations ---
        char clean_word[MAX_TOKEN_LEN];
        int i;
        for (i = 0; word[i] && i < MAX_TOKEN_LEN - 1; i++) {
            clean_word[i] = tolower((unsigned char)word[i]);
        }
        clean_word[i] = '\0';

        printf("%s -> ", clean_word);
        tokenize_inline(clean_word);
        word = strtok(NULL, " \t\n.,!?;:\"'()[]{}<>_-");
    }
    return 0;
}
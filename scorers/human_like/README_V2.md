# Human-Like Scorer V2

This directory contains an improved version of the human-like scorer with the following enhancements:

## Key Improvements

### 1. **New `keep-in-edit` Token**
The vocabulary now includes a special token to distinguish between:
- `keep`: Tokens outside any edit region (unchanged)
- `keep-in-edit`: Tokens that remain unchanged but are **inside** an edit region

**Example**: When changing "a very old car" → "a very new car"
- "a" → `keep-in-edit` (unchanged but inside edit)
- "very" → `keep-in-edit` (unchanged but inside edit)
- "old" → `replace` (changed token)
- "car" → `keep-in-edit` (unchanged but inside edit)

### 2. **Sentence-Level Operation**
Instead of generating one sequence per document, the new version:
- Operates on individual sentences
- Generates one sequence per sentence that contains edits
- Provides finer-grained training signal

### 3. **Edit Filtering by Sentence Boundaries**
Uses the `sents_char_pos` field from the dataset to:
- Filter out edits that span multiple sentences
- Only include edits fully contained within a single sentence
- Ensure clean training data

## Updated Vocabulary

```python
vocab = {
    '<pad>': 0,         # Padding token
    'keep': 1,          # Token outside any edit
    'del': 2,           # Token is deleted
    'add': 3,           # Token is added
    'replace': 4,       # Token is replaced
    'keep-in-edit': 5   # Token unchanged but inside edit region
}
```

## Files

### Training Pipeline

1. **`generate_edit_sequences_v2.py`** - Generate training data
   - Reads IteraTeR dataset with `sents_char_pos`
   - Partitions edits by sentence
   - Generates sentence-level sequences with `keep-in-edit` tokens
   - Processes all splits (train, test, dev) in one run

   ```bash
   # Process all splits and upload to HuggingFace
   python scorers/human_like/generate_edit_sequences_v2.py \
       --dataset-prefix data/IteraTeR/full_doc_level \
       --output-dataset-name timonziegenbein/human-like-edit-sequences-v2

   # Or save locally
   python scorers/human_like/generate_edit_sequences_v2.py \
       --dataset-prefix data/IteraTeR/full_doc_level \
       --output-dir data/human_like_sequences
   ```

2. **`train_human_like_model_v2.py`** - Train the model
   - Uses updated 6-token vocabulary
   - Same Transformer architecture as v1
   - Supports loading from CSV or HuggingFace Hub
   - Logs training metrics to W&B and Weave

   ```bash
   # Train from HuggingFace dataset (recommended)
   python scorers/human_like/train_human_like_model_v2.py \
       --dataset-name timonziegenbein/human-like-edit-sequences \
       --split train \
       --model-path scorers/human_like/human_like_language_model_v3.pth \
       --epochs 5 \
       --wandb-project human-like-scorer \
       --run-name v2-baseline

   # Train from local CSV
   python scorers/human_like/train_human_like_model_v2.py \
       --input-csv data/edit_sequences_v2.csv \
       --model-path scorers/human_like/human_like_language_model_v3.pth \
       --epochs 5
   ```

### Inference

3. **`human_like_scorer_v2.py`** - Use the scorer
   - Backward compatible with v1 model if v3 not available
   - Includes sentence-level filtering during inference
   - Can accept `sents_char_pos` parameter for filtering

   ```python
   from scorers.human_like.human_like_scorer_v2 import HumanLikeScorerV2

   scorer = HumanLikeScorerV2(device=torch.device("cuda:0"))

   reward = scorer.calculate_human_likeness(
       original_argument="Full document text...",
       original_sentence="The sentence with edit...",
       inappropriate_part="old text",
       rewritten_part="new text",
       sents_char_pos=[100, 250, 400]  # Optional: enables filtering
   )
   ```

## Architecture Details

### Model Architecture
- **Type**: Transformer Language Model
- **Embedding Dim**: 200
- **Attention Heads**: 2
- **Hidden Dim**: 200
- **Layers**: 2
- **Dropout**: 0.2
- **Max Sequence Length**: 500 tokens

### Training
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss (ignoring `<pad>`)
- **Gradient Clipping**: 0.5
- **Batch Size**: 64
- **Logging**: W&B and Weave for experiment tracking

### Logged Metrics
The training script logs the following metrics to Weights & Biases:

**Per-step metrics** (logged every training step):
- `train/loss`: Cross-entropy loss for the current batch
- `train/perplexity`: Perplexity (exp(loss)) for the current batch
- `train/epoch`: Current epoch number
- `train/step`: Global training step

**Per-epoch metrics** (logged at the end of each epoch):
- `train/epoch_avg_loss`: Average loss across all batches in the epoch
- `train/epoch_avg_perplexity`: Average perplexity for the epoch
- `epoch`: Epoch number

**Artifacts**:
- Trained model saved as W&B artifact: `human-like-model-v2`

### Evaluation
- **Metric**: Perplexity
- **Threshold**: 1.1465 (configurable)
- **Reward**: Binary (1.0 if perplexity ≤ threshold, else 0.0)

## Algorithm

### Sequence Generation

For each document with edits:

1. **Parse sentence boundaries** from `sents_char_pos`
2. **Partition edits by sentence**:
   - Check if edit's `start_char_pos` and `end_char_pos` are within sentence bounds
   - Skip edits that span multiple sentences
3. **For each sentence with edits**:
   - Tokenize the sentence with character offsets
   - Initialize all tags as `keep`
   - For each edit in the sentence:
     - Find overlapping tokens
     - Use `difflib.SequenceMatcher` to compare before/after tokens
     - Apply opcodes:
       - `equal` → `keep-in-edit`
       - `delete` → `del`
       - `replace` → `replace`
       - `insert` → `add` (at insertion position)
4. **Save sequence** as comma-separated tags

### Inference

1. Find edit location in document
2. Check if edit is within single sentence (if `sents_char_pos` provided)
3. Extract sentence containing edit
4. Generate operation sequence using same algorithm
5. Calculate perplexity of sequence
6. Return 1.0 if perplexity ≤ threshold, else 0.0

## Comparison with V1

| Feature | V1 | V2 |
|---------|----|----|
| Granularity | Document-level | Sentence-level |
| Vocabulary Size | 5 tokens | 6 tokens |
| Keep Token | Single `keep` | `keep` + `keep-in-edit` |
| Edit Filtering | None | Filters cross-sentence edits |
| Training Signal | Sparse (one seq per doc) | Dense (one seq per sentence) |
| Model Compatibility | v2.pth | v3.pth (backward compatible) |

## Expected Benefits

1. **Better Training Signal**: Sentence-level sequences provide more training examples
2. **Cleaner Data**: Filtering cross-sentence edits removes ambiguous cases
3. **Finer Distinctions**: `keep-in-edit` helps model learn edit boundaries
4. **More Robust**: Better handles documents with multiple edits

## Integration with GRPO

To use v2 in the GRPO training pipeline, update `models/grpo.py`:

```python
from scorers.human_like.human_like_scorer_v2 import HumanLikeScorerV2

# Replace initialization
human_like_scorer = HumanLikeScorerV2(device)

# In reward function, pass sents_char_pos if available
def reward_fn(prompts, completions, **kwargs):
    # Extract sents_char_pos from dataset
    sents_char_pos = kwargs.get('sents_char_pos', None)

    score = human_like_scorer.calculate_human_likeness(
        original_argument=original_arg,
        original_sentence=sent,
        inappropriate_part=before,
        rewritten_part=after,
        sents_char_pos=sents_char_pos  # Enable filtering
    )
    return score
```

## TODO

- [ ] Generate training sequences from IteraTeR dataset
- [ ] Train v3 model and evaluate perplexity
- [ ] Determine optimal threshold on validation set
- [ ] Integrate with GRPO reward functions
- [ ] Compare v2 vs v1 performance on edit quality

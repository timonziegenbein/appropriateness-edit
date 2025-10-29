# Fluency Model Training Format

This document explains the data format used for training the fluency classification model.

## Overview

The fluency model is trained to distinguish between fluent and non-fluent edits by comparing sentences before and after an edit. The model uses a **two-sequence input format** that aligns with how ModernBERT and other BERT-style models are designed.

## Input Format

### Dataset Structure

The training dataset (created by `create_fluency_eval_dataset.py`) contains:

```python
{
    'id': 'test_42_edit_0',
    'original_sentence': 'This are a test.',           # Sentence before edit
    'inappropriate_part': 'are',                        # Part to replace
    'rewritten_part': 'is',                             # Replacement text
    'expected_score': 1.0,                              # 1.0 = fluent, 0.0 = non-fluent
    'description': 'Fluent edit from fluency-data-augmented'
}
```

### Model Input Transformation

The training script (`train_fluency_model.py`) transforms this into a **two-sequence classification task**:

1. **Sequence 1 (text_before)**: The original sentence
   - Example: `"This are a test."`

2. **Sequence 2 (text_after)**: The sentence after applying the edit
   - Computed as: `original_sentence.replace(inappropriate_part, rewritten_part, 1)`
   - Example: `"This is a test."`

3. **Label**: Binary classification
   - `1` = Fluent edit (maintains or improves fluency)
   - `0` = Non-fluent edit (harms fluency)

### Tokenization

The tokenizer processes both sequences together:

```python
tokenizer(
    text_before,      # Sequence A: original sentence
    text_after,       # Sequence B: rewritten sentence
    truncation=True,
    padding="max_length",
    max_length=512
)
```

This creates input like:
```
[CLS] This are a test. [SEP] This is a test. [SEP]
```

## Why This Format?

### 1. **Alignment with Evaluation**

The evaluation script (`evaluate_fluency_scorer.py`) uses the same before/after comparison:

```python
# Line 227 in evaluate_fluency_scorer.py
rewritten_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)
```

Training and evaluation use the exact same data representation.

### 2. **Native BERT-Style Support**

ModernBERT (and BERT-style models) natively support two-sequence input:
- ✓ Single text: `tokenizer(text)`
- ✓ Text pair: `tokenizer(text_a, text_b)`
- ✗ Three texts: NOT supported

Using two sequences avoids manual concatenation and special token handling.

### 3. **Full Context Preserved**

Both sequences contain the complete sentence, giving the model:
- Full grammatical context
- Subject-verb agreement information
- Surrounding words for context-dependent judgments

### 4. **Clean Comparison**

The model learns to compare:
- Before: `"This are a test."`
- After: `"This is a test."`

And predict: ✓ Fluent (label=1)

## Dataset Creation

To create the training dataset:

```bash
python scorers/fluency/create_fluency_eval_dataset.py \
    --dataset-name timonziegenbein/fluency-augmented-pairs \
    --output-dataset-name timonziegenbein/fluency-eval-dataset \
    --num-workers 8
```

This will:
1. Load pairs of texts (one fluent, one non-fluent)
2. Extract individual edits using DirectLatexdiffParser
3. Create evaluation examples for each edit
4. Balance the dataset (50% fluent, 50% non-fluent)

## Training

To train the model:

```bash
python scorers/fluency/train_fluency_model.py \
    --dataset_name timonziegenbein/fluency-eval-dataset \
    --model_name answerdotai/ModernBERT-large \
    --output_dir ./models/fluency_modernbert \
    --results_dir ./results/fluency_modernbert
```

The script will:
1. Load the evaluation dataset
2. Transform each example into (text_before, text_after, label)
3. Tokenize using ModernBERT tokenizer
4. Train with step-based evaluation (every 500 steps)
5. Select best model based on F1 score

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: TP / (TP + FP) where positive = fluent (1)
- **Recall**: TP / (TP + FN) where positive = fluent (1)
- **F1 Score**: Harmonic mean of precision and recall

## Example Flow

```
Original Dataset Entry:
  text1: "This are a test."
  text2: "This is a test."
  label: 1 (text1→text2 is bad→good, so fluent edit)

↓ create_fluency_eval_dataset.py

Evaluation Dataset Entry:
  original_sentence: "This are a test."
  inappropriate_part: "are"
  rewritten_part: "is"
  expected_score: 1.0

↓ train_fluency_model.py

Training Example:
  text_before: "This are a test."
  text_after: "This is a test."
  label: 1

↓ ModernBERT Tokenizer

Model Input:
  [CLS] This are a test. [SEP] This is a test. [SEP]
  label: 1
```

## Notes

- The model sees complete sentences, not just the edited parts
- This allows it to judge context-dependent fluency (e.g., subject-verb agreement)
- The format is identical between training and evaluation
- ModernBERT's 8192 max sequence length easily handles most sentences

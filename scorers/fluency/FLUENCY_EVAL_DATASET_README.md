# Fluency Evaluation Dataset Creation and Usage

This document describes how to create and use the fluency evaluation dataset for evaluating fluency scoring models.

## Overview

The fluency evaluation dataset is created from `timonziegenbein/fluency-data-augmented`, which contains pairs of texts where one is more fluent than the other. The dataset creation process:

1. Extracts edit operations between the two texts using `DirectLatexdiffParser`
2. Creates evaluation examples with:
   - `original_sentence`: The source text
   - `inappropriate_part`: The part to be replaced (from diff)
   - `rewritten_part`: The replacement text (from diff)
   - `expected_score`: 1.0 if edit improves fluency (bad→good), 0.0 if it harms fluency (good→bad)

## Dataset Creation

### Step 1: Create the Evaluation Dataset

```bash
# Create dataset and upload to HuggingFace Hub
python scorers/fluency/create_fluency_eval_dataset.py \
    --dataset-name timonziegenbein/fluency-data-augmented \
    --output-dataset-name your-username/fluency-eval-dataset \
    --splits test  # Process only the test split

# For testing with limited examples
python scorers/fluency/create_fluency_eval_dataset.py \
    --max-examples 100 \
    --output-dir ./fluency_eval_dataset_sample
```

### Arguments

- `--dataset-name`: Source dataset (default: `timonziegenbein/fluency-data-augmented`)
- `--output-dataset-name`: Target dataset name on HuggingFace Hub (optional)
- `--output-dir`: Local directory to save dataset (optional)
- `--splits`: Which splits to process (e.g., `train test validation`)
- `--max-examples`: Limit number of examples per split (for testing)

### Example Output

The script will create a dataset with the following structure:

```python
{
    'id': 'test_42_edit_0',
    'original_sentence': 'This are a test.',
    'inappropriate_part': 'are',
    'rewritten_part': 'is',
    'expected_score': 1.0,  # 1.0 = fluent edit (bad→good)
    'description': 'Fluent edit from fluency-data-augmented'
}
```

## Evaluation

### Step 2: Evaluate with the New Dataset

Once you've created the dataset, you can use it with the evaluation script:

```bash
# Evaluate traditional fluency scorer
python scorers/fluency/evaluate_fluency_scorer.py \
    --eval-dataset-name your-username/fluency-eval-dataset \
    --eval-dataset-split test \
    --project fluency-eval-experiment \
    --device cuda

# Evaluate LLM-based classifier
python scorers/fluency/evaluate_fluency_scorer.py \
    --eval-dataset-name your-username/fluency-eval-dataset \
    --eval-dataset-split test \
    --use-llm \
    --llm-model-name gemini-2.0-flash-exp \
    --project fluency-eval-llm

# Limit number of examples (for quick testing)
python scorers/fluency/evaluate_fluency_scorer.py \
    --eval-dataset-name your-username/fluency-eval-dataset \
    --max-eval-examples 100
```

### Evaluation Arguments

- `--eval-dataset-name`: Name of the evaluation dataset on HuggingFace Hub
- `--eval-dataset-split`: Which split to evaluate on (train/validation/test)
- `--max-eval-examples`: Limit number of examples (None = use all)
- `--use-llm`: Use LLM-based classifier instead of traditional scorer
- `--llm-model-name`: Gemini model name (requires `GEMINI_API_KEY` env var)
- `--device`: Device for traditional scorer (cuda/cpu)
- `--project`: Weave project name for tracking results

## Metrics

The evaluation computes the following metrics:

1. **Accuracy**: Percentage of correct fluency predictions
2. **Precision**: TP / (TP + FP) where positive class = fluent (1.0)
3. **Recall**: TP / (TP + FN) where positive class = fluent (1.0)
4. **F1 Score**: Harmonic mean of precision and recall

## Dataset Statistics

After creating the dataset, the script will print statistics like:

```
================================================================================
Dataset Statistics:
================================================================================

test:
  Total examples: 15234
  Fluent edits (score=1.0): 7617
  Non-fluent edits (score=0.0): 7617
================================================================================
```

## Notes

- The dataset is balanced by design: for every edit bad→good (score=1.0), there's a corresponding edit good→bad (score=0.0)
- Edits are extracted using `DirectLatexdiffParser` and `fuzzy_post_process_edits` from `ops/latexdiff_parser.py`
- Each example may produce multiple evaluation cases (one per edit operation)
- The traditional scorer requires local model files and GPU for efficient processing
- The LLM-based scorer requires the `GEMINI_API_KEY` environment variable

## Example Workflow

```bash
# 1. Create the evaluation dataset (one-time setup)
python scorers/fluency/create_fluency_eval_dataset.py \
    --output-dataset-name username/fluency-eval \
    --splits test

# 2. Run evaluation with traditional scorer
python scorers/fluency/evaluate_fluency_scorer.py \
    --eval-dataset-name username/fluency-eval \
    --device cuda

# 3. Compare with LLM-based scorer
export GEMINI_API_KEY="your-api-key"
python scorers/fluency/evaluate_fluency_scorer.py \
    --eval-dataset-name username/fluency-eval \
    --use-llm \
    --project fluency-eval-llm-comparison
```

## Troubleshooting

### No edits extracted
If the script reports no edits for many examples, check:
- The `DirectLatexdiffParser` dependencies are installed
- The `temp_output` directory has write permissions

### Dataset too large
For testing, use `--max-examples` to limit the dataset size:
```bash
python scorers/fluency/create_fluency_eval_dataset.py --max-examples 1000
```

### Out of memory during evaluation
Limit the number of examples loaded:
```bash
python scorers/fluency/evaluate_fluency_scorer.py \
    --eval-dataset-name username/fluency-eval \
    --max-eval-examples 500
```

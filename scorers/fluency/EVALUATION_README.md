# Fluency Scorer Weave Evaluation

This directory contains a comprehensive evaluation framework for the Fluency Scorer using [Weights & Biases Weave](https://weave-docs.wandb.ai/).

## Overview

The fluency scorer evaluation tests the scorer's ability to:

1. **Identify fluent edits**: Edits that preserve grammatical correctness
2. **Reject non-fluent edits**: Edits that introduce grammar errors
3. **Handle edge cases**: Empty strings, identical replacements, special punctuation

## Files

- `evaluate_fluency_scorer.py`: Main evaluation script using Weave
- `fluency_scorer.py`: The fluency scorer implementation being evaluated
- `EVALUATION_README.md`: This documentation file

## Quick Start

### Basic Evaluation

Run the evaluation with the default test cases:

```bash
python scorers/fluency/evaluate_fluency_scorer.py --project my-fluency-eval
```

### Including Real-World Examples

Include examples from the actual fluency dataset:

```bash
python scorers/fluency/evaluate_fluency_scorer.py \
    --project my-fluency-eval \
    --include-real-examples \
    --max-real-examples 100
```

### Using CPU

If you don't have a GPU available:

```bash
python scorers/fluency/evaluate_fluency_scorer.py \
    --project my-fluency-eval \
    --device cpu
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--project` | str | `fluency-scorer-eval` | Weave project name |
| `--device` | str | `cuda` (if available) | Device to run the scorer on (`cuda` or `cpu`) |
| `--include-real-examples` | flag | `False` | Include real-world examples from the dataset |
| `--max-real-examples` | int | `50` | Maximum number of real-world examples to include |

## Test Cases

The evaluation includes 15 carefully crafted test cases covering:

### Fluent Edits (Expected score: 1.0)
1. Simple adjective replacement
2. Synonym replacement preserving grammar
3. Proper article usage
4. Complete phrase replacement
5. Simplification while maintaining grammar
6. Tone change with proper grammar
7. Transition word replacement with punctuation
8. Complex sentence structure maintenance

### Non-Fluent Edits (Expected score: 0.0)
1. Grammar error creation (e.g., "She go to the store")
2. Broken verb tense consistency
3. Article-noun mismatch
4. Incomplete sentences
5. Missing punctuation

### Edge Cases
1. Empty inappropriate part
2. Identical parts (no change)
3. Special punctuation handling

## Evaluation Metrics

The evaluation uses two scorers:

### 1. Exact Match Scorer
- Checks if the predicted fluency score exactly matches the expected score (0.0 or 1.0)
- Reports: `correct`, `expected`, `predicted`

### 2. Binary Accuracy Scorer
- Classifies results as fluent (1.0) vs. non-fluent (0.0)
- Reports: `correct`, `expected_fluent`, `predicted_fluent`

## Understanding Results

### Via Weave Dashboard

After running the evaluation, you'll see a URL to view results in the Weave dashboard:

```
View detailed results in Weave: https://wandb.ai/[username]/[project]/weave/...
```

The dashboard provides:
- Per-example predictions and scores
- Aggregate metrics across all test cases
- Confusion matrices for binary classification
- Distribution of scores

### Via Console Output

The script prints a summary to the console:

```
================================================================================
FLUENCY SCORER EVALUATION RESULTS
================================================================================

Total test cases: 15
Device used: cuda

Scorer Results:
  exact_match_scorer: 0.93
  binary_accuracy_scorer: 0.93

View detailed results in Weave: https://wandb.ai/...
================================================================================
```

## Interpreting Scores

- **1.0 (Perfect)**: All test cases passed correctly
- **0.9-0.99**: Most cases correct, investigate failures in Weave
- **0.8-0.89**: Some systematic issues, review edge cases
- **< 0.8**: Significant problems with the scorer

## Customizing Test Cases

To add your own test cases, edit the `FLUENCY_TEST_CASES` list in `evaluate_fluency_scorer.py`:

```python
{
    "original_sentence": "Your original sentence here.",
    "inappropriate_part": "part to replace",
    "rewritten_part": "replacement text",
    "expected_score": 1.0,  # or 0.0
    "description": "Description of what this tests"
}
```

## Integration with Training

This evaluation can be used to:

1. **Monitor scorer performance** during development
2. **Regression testing** after model updates
3. **A/B testing** different fluency models
4. **Threshold tuning** for the fluency classifier

## Real-World Dataset Testing

The `--include-real-examples` flag loads examples from the binary fluency dataset (`timonziegenbein/binary-fluency-data`). This provides:

- **Fluent examples** (label=1): Text that should be grammatically correct
- **Non-fluent examples** (label=0): Text with grammar issues

These examples test the scorer on real-world data beyond synthetic test cases.

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

```bash
python scorers/fluency/evaluate_fluency_scorer.py --device cpu
```

### Import Errors

Ensure you're running from the project root:

```bash
cd /mnt/home/tziegenb/appropriateness-edit
python scorers/fluency/evaluate_fluency_scorer.py
```

### Weave Authentication

If you haven't logged into W&B:

```bash
wandb login
```

## Architecture

### FluencyScorerModel (Weave Model)

The evaluation wraps `FluencyScorer` in a Weave `Model`:

```python
class FluencyScorerModel(weave.Model):
    @weave.op()
    def predict(self, original_sentence, inappropriate_part, rewritten_part):
        # Returns fluency score and metadata
        ...
```

This enables:
- Automatic logging of predictions
- Version tracking of the scorer
- Reproducible evaluations

### Evaluation Flow

1. **Initialize Weave** with project name
2. **Create dataset** from test cases
3. **Initialize model** (FluencyScorerModel)
4. **Run evaluation** with multiple scorers
5. **Log results** to Weave dashboard

## Related Files

- `scorers/fluency/fluency_scorer.py`: Main scorer implementation (lines 60-93 contain the `calculate_fluency` method)
- `scorers/reward_functions.py`: Integration with the RL training loop (lines 69-73)
- `scorers/fluency/train_fluency_model.py`: Training script for the fluency classifier
- `scorers/fluency/create_binary_fluency_dataset.py`: Dataset creation script

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{fluency_scorer_eval,
  title = {Fluency Scorer Weave Evaluation},
  author = {Ziegenbein, Timon},
  year = {2025},
  url = {https://github.com/timonziegenbein/appropriateness-edit}
}
```

## Additional Resources

- [Weave Documentation](https://weave-docs.wandb.ai/)
- [Weave Evaluations Guide](https://weave-docs.wandb.ai/guides/core-types/evaluations)
- [W&B Dashboard](https://wandb.ai/)

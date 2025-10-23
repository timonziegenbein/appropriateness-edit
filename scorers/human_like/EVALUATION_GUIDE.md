# Evaluation Guide for Human-Like Scorer

This guide explains how to evaluate and compare different versions of the human-like scorer, even though we only have positive (human-made) training examples.

## Problem Statement

The human-like scorer is trained on positive examples only (real human edits from IteraTeR). At inference time, we use perplexity with a threshold to determine if an edit is "human-like":

```python
if perplexity <= threshold:
    reward = 1.0  # Human-like
else:
    reward = 0.0  # Not human-like
```

**Challenges:**
1. How do we evaluate model quality without negative examples?
2. How do we compare different model versions (e.g., v1 vs v2)?
3. How do we choose the right threshold?

## Solution: Synthetic Negative Examples

We generate synthetic "non-human-like" edits to create a binary classification evaluation task.

### Step 1: Generate Negative Examples and Push to HuggingFace

**Option A: From HuggingFace Dataset (Recommended)**
```bash
python scorers/human_like/create_eval_dataset.py \
    --dataset-name timonziegenbein/human-like-edit-sequences \
    --output-dataset-name timonziegenbein/human-like-edit-sequences-eval \
    --n-negatives 5 \
    --splits test dev
```

**Option B: From Local CSV**
```bash
python scorers/human_like/generate_negative_examples.py \
    --positive-csv data/edit_sequences_v2.csv \
    --output-csv data/edit_sequences_eval.csv \
    --n-negatives 5
```

**Negative Generation Methods:**

1. **Random Shuffling** - Shuffle the order of edit operations
   - Mild shuffle (30% of tokens)
   - Heavy shuffle (70% of tokens)

2. **Token Substitution** - Replace random tokens with other vocab items
   - Substitutes ~30% of tokens

3. **Inefficient Edits** - Insert delete-then-add patterns
   - Mimics poor editing behavior

4. **Repetitive Patterns** - Add long sequences of repeated operations
   - E.g., `del, del, del, del, ...` (5-15 repeats)

5. **Sequence Reversal** - Reverse the entire edit sequence
   - Completely unnatural ordering

**Output:**
- CSV with both positive (`label=1`) and negative (`label=0`) examples
- `type` column indicates generation method for negatives
- Typical ratio: 1 positive : 5 negatives

### Step 2: Evaluate Model

**Option A: Using Weave (Recommended - Interactive Dashboard)**
```bash
python scorers/human_like/evaluate_human_like_scorer.py \
    --project human-like-scorer-eval \
    --eval-dataset-name timonziegenbein/human-like-edit-sequences-eval \
    --eval-dataset-split test \
    --model-path scorers/human_like/human_like_language_model_v3.pth \
    --vocab-type v2 \
    --threshold 1.1465 \
    --run-name "v2-baseline"
```

This provides an interactive Weave dashboard with real-time metrics and visualizations.

**Option B: Using Static Analysis (Local Plots)**
```bash
python scorers/human_like/evaluate_with_negatives.py \
    --model-path scorers/human_like/human_like_language_model_v3.pth \
    --test-csv data/edit_sequences_eval.csv \
    --output-dir evaluation_results_v3 \
    --vocab-type v2
```

**Weave Evaluation Features:**
- Interactive dashboard in Weights & Biases
- Compare multiple model versions side-by-side
- Track evaluation metrics over time
- Drill down into individual predictions
- Export results for further analysis

**Static Evaluation Output Files:**
1. `perplexity_distributions.png` - Histogram and boxplot comparing positive vs negative perplexities
2. `roc_curve.png` - ROC curve with AUC score
3. `precision_recall_curve.png` - PR curve with optimal F1 point
4. `threshold_metrics.csv` - Performance at different thresholds
5. `perplexity_scores.csv` - Raw perplexity scores for all examples

## Evaluation Metrics

### 1. Perplexity Distribution

**What it shows:** How well the model separates human-like from non-human-like edits.

**Interpretation:**
- **Good model**: Clear separation between positive (low perplexity) and negative (high perplexity) distributions
- **Poor model**: Overlapping distributions

**Example:**
```
Positive (Human):  mean=1.05, std=0.15
Negative (Synth):  mean=3.20, std=0.85
```

### 2. ROC Curve and AUC

**What it shows:** Model's ability to discriminate across all possible thresholds.

**Interpretation:**
- **AUC = 1.0**: Perfect discrimination
- **AUC = 0.5**: Random guessing
- **AUC > 0.9**: Excellent model
- **AUC 0.7-0.9**: Good model
- **AUC < 0.7**: Poor model

**Use for:** Comparing different model versions objectively

### 3. Precision-Recall Curve and Average Precision

**What it shows:** Trade-off between precision (avoiding false positives) and recall (catching true positives).

**Interpretation:**
- **Average Precision (AP)**: Area under PR curve
- **High AP** (> 0.9): Model maintains high precision even at high recall
- **Optimal F1 point**: Best balance between precision and recall

**Use for:** Choosing threshold based on desired precision/recall trade-off

## Choosing a Threshold

Several strategies are available:

### Strategy 1: Maximize F1 Score (Balanced)

Use the threshold that maximizes F1 = 2 × (precision × recall) / (precision + recall)

```
PR Optimal (Max F1) (threshold=1.2450):
  Precision: 0.9523
  Recall:    0.9401
  F1:        0.9462
```

**When to use:** You care equally about false positives and false negatives.

### Strategy 2: Control False Negative Rate (Conservative)

Set threshold at 95th percentile of positive distribution to ensure 95% of human edits pass.

```
95th percentile (threshold=1.3200):
  Recall:    0.9500  (by design)
  Precision: 0.9123
  FNR:       0.0500  (5% of human edits rejected)
```

**When to use:** You want to be very permissive with human edits (for GRPO training).

### Strategy 3: Maximize ROC (Discriminative)

Use threshold that maximizes TPR - FPR.

```
ROC Optimal (threshold=1.1800):
  Accuracy:  0.9234
  FPR:       0.0512
  FNR:       0.1103
```

**When to use:** You want best overall discrimination performance.

### Strategy 4: High Precision (Strict)

Use a low threshold to minimize false positives at the cost of recall.

```
50th percentile (threshold=1.0200):
  Precision: 0.9876
  Recall:    0.5012
  FPR:       0.0089
```

**When to use:** You only want to accept very confident predictions.

## Comparing Model Versions

To compare v1 vs v2:

```bash
# Evaluate v1
python scorers/human_like/evaluate_with_negatives.py \
    --model-path scorers/human_like/human_like_language_model_v2.pth \
    --test-csv data/edit_sequences_eval_v1.csv \
    --output-dir evaluation_results_v1 \
    --vocab-type v1

# Evaluate v2
python scorers/human_like/evaluate_with_negatives.py \
    --model-path scorers/human_like/human_like_language_model_v3.pth \
    --test-csv data/edit_sequences_eval_v2.csv \
    --output-dir evaluation_results_v2 \
    --vocab-type v2
```

**Comparison Criteria:**

| Metric | Better Model Has... | Why It Matters |
|--------|---------------------|----------------|
| **ROC AUC** | Higher AUC | Better overall discrimination ability |
| **Average Precision** | Higher AP | Better precision-recall trade-off |
| **Distribution Separation** | Larger gap in means | Clearer decision boundary |
| **Optimal F1** | Higher F1 | Better balanced performance |
| **Perplexity Variance** | Lower variance on positives | More consistent scoring |

**Decision Rule:**
- If Model B has **AUC > Model A by ≥0.05**, Model B is significantly better
- If **ROC AUC is similar**, prefer model with **higher AP** or **better F1**

## Baseline Comparisons with Weave

Weave makes it easy to compare your trained model against various baselines:

### Random Baseline
```bash
python scorers/human_like/evaluate_human_like_scorer.py \
    --project human-like-scorer-eval \
    --eval-dataset-name timonziegenbein/human-like-edit-sequences-eval \
    --use-random-baseline \
    --run-name "random-baseline"
```

Expected: Accuracy ~50%, F1 ~0.5

### Always Human-Like Baseline
```bash
python scorers/human_like/evaluate_human_like_scorer.py \
    --project human-like-scorer-eval \
    --eval-dataset-name timonziegenbein/human-like-edit-sequences-eval \
    --use-always-humanlike \
    --run-name "always-humanlike-baseline"
```

Expected: High recall (100%), low precision (depends on pos/neg ratio)

### Compare Multiple Models in Weave

Run multiple evaluations with different `--run-name` values for side-by-side comparison:

```bash
# Random baseline
python scorers/human_like/evaluate_human_like_scorer.py \
    --project human-like-scorer-eval \
    --eval-dataset-name timonziegenbein/human-like-edit-sequences-eval \
    --run-name "random" \
    --use-random-baseline

# Model v2
python scorers/human_like/evaluate_human_like_scorer.py \
    --project human-like-scorer-eval \
    --eval-dataset-name timonziegenbein/human-like-edit-sequences-eval \
    --run-name "v2-baseline" \
    --model-path scorers/human_like/human_like_language_model_v3.pth \
    --vocab-type v2 \
    --threshold 1.1465

# Model v2 with strict threshold
python scorers/human_like/evaluate_human_like_scorer.py \
    --project human-like-scorer-eval \
    --eval-dataset-name timonziegenbein/human-like-edit-sequences-eval \
    --run-name "v2-strict" \
    --model-path scorers/human_like/human_like_language_model_v3.pth \
    --vocab-type v2 \
    --threshold 1.0
```

Then compare all runs in the Weave UI at: `https://wandb.ai/weave/human-like-scorer-eval`

## Complete Workflow Example

### Train and Evaluate v1

```bash
# Generate training data (v1: document-level)
python scorers/human_like/generate_edit_sequences.py \
    --input-json data/IteraTeR/full_doc_level/train.json \
    --output-csv data/edit_sequences_v1_train.csv

# Train model
python scorers/human_like/train_human_like_model.py \
    --input-csv data/edit_sequences_v1_train.csv \
    --model-path scorers/human_like/human_like_language_model_v2.pth \
    --epochs 5

# Generate test data with negatives
python scorers/human_like/generate_edit_sequences.py \
    --input-json data/IteraTeR/full_doc_level/test.json \
    --output-csv data/edit_sequences_v1_test_pos.csv

python scorers/human_like/generate_negative_examples.py \
    --positive-csv data/edit_sequences_v1_test_pos.csv \
    --output-csv data/edit_sequences_v1_test.csv \
    --n-negatives 5

# Evaluate
python scorers/human_like/evaluate_with_negatives.py \
    --model-path scorers/human_like/human_like_language_model_v2.pth \
    --test-csv data/edit_sequences_v1_test.csv \
    --output-dir evaluation_results_v1 \
    --vocab-type v1
```

### Train and Evaluate v2 (Sentence-Level with keep-in-edit)

```bash
# Generate training data (v2: sentence-level)
python scorers/human_like/generate_edit_sequences_v2.py \
    --input-json data/IteraTeR/full_doc_level/train.json \
    --output-csv data/edit_sequences_v2_train.csv

# Train model
python scorers/human_like/train_human_like_model_v2.py \
    --input-csv data/edit_sequences_v2_train.csv \
    --model-path scorers/human_like/human_like_language_model_v3.pth \
    --epochs 5

# Generate test data with negatives
python scorers/human_like/generate_edit_sequences_v2.py \
    --input-json data/IteraTeR/full_doc_level/test.json \
    --output-csv data/edit_sequences_v2_test_pos.csv

python scorers/human_like/generate_negative_examples.py \
    --positive-csv data/edit_sequences_v2_test_pos.csv \
    --output-csv data/edit_sequences_v2_test.csv \
    --n-negatives 5

# Evaluate
python scorers/human_like/evaluate_with_negatives.py \
    --model-path scorers/human_like/human_like_language_model_v3.pth \
    --test-csv data/edit_sequences_v2_test.csv \
    --output-dir evaluation_results_v2 \
    --vocab-type v2
```

### Compare Results

```bash
# Compare AUC scores
echo "v1 AUC:" && grep "ROC AUC" evaluation_results_v1/threshold_metrics.csv
echo "v2 AUC:" && grep "ROC AUC" evaluation_results_v2/threshold_metrics.csv

# Visual comparison
ls -la evaluation_results_v1/*.png
ls -la evaluation_results_v2/*.png
```

## Advanced: Cross-Validation

For more robust evaluation, use k-fold cross-validation:

```bash
# Split data into k folds
python scorers/human_like/create_cv_splits.py \
    --input-csv data/edit_sequences_v2.csv \
    --n-folds 5 \
    --output-dir data/cv_splits

# Train and evaluate on each fold
for fold in {1..5}; do
    python scorers/human_like/train_human_like_model_v2.py \
        --input-csv data/cv_splits/train_fold${fold}.csv \
        --model-path models/fold${fold}_model.pth \
        --epochs 5

    python scorers/human_like/evaluate_with_negatives.py \
        --model-path models/fold${fold}_model.pth \
        --test-csv data/cv_splits/test_fold${fold}.csv \
        --output-dir evaluation_results_fold${fold} \
        --vocab-type v2
done

# Aggregate results
python scorers/human_like/aggregate_cv_results.py \
    --results-dir evaluation_results_fold* \
    --output results_summary.csv
```

## Interpretation Tips

### Good Model Signs
✅ ROC AUC > 0.9
✅ Clear separation in perplexity distributions
✅ Average Precision > 0.9
✅ Optimal F1 > 0.9
✅ Low variance in positive perplexities

### Poor Model Signs
❌ ROC AUC < 0.7
❌ Overlapping distributions
❌ Average Precision < 0.7
❌ Optimal F1 < 0.7
❌ High variance in positive perplexities
❌ Many false positives on simple negative examples

### Model is Overfitting If
- Training perplexity is very low but test perplexity is high
- High variance in test perplexities
- Poor generalization to new negative types

## References

- **ROC Curve**: Evaluates binary classifier performance across all thresholds
- **Precision-Recall Curve**: Better for imbalanced datasets
- **AUC**: Area Under Curve - single metric for discrimination ability
- **F1 Score**: Harmonic mean of precision and recall

## Troubleshooting

**Q: Model gives similar perplexities for positive and negative examples**
A: Model may be underfitting. Try training longer, increasing model capacity, or improving training data quality.

**Q: All negative examples get very high perplexity (e.g., > 100)**
A: This is actually good! But it means threshold selection is less critical. Use percentile-based threshold on positives.

**Q: ROC AUC is high but F1 is low**
A: Distributions are separated but threshold is poorly chosen. Re-examine threshold selection strategy.

**Q: Should I generate more negative examples?**
A: More negatives help with robust evaluation but don't improve the model (it only trains on positives). Use 3-5x negatives for evaluation.

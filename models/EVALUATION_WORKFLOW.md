# Evaluation Workflow

This document describes the improved evaluation workflow that separates costly edit generation from evaluation.

## Overview

The evaluation process is split into two stages:

1. **Edit Generation** (`generate_edits.py`) - Generates edits from a model (costly, run once)
2. **Edit Evaluation** (`evaluate_edits.py`) - Evaluates edits with configurable scorers (fast, run multiple times)

This separation allows you to:
- Generate edits once and evaluate with different scorer configurations
- Test different thresholds without regenerating edits
- Compare different models with the same evaluation settings
- Save computational resources by avoiding redundant model inference

## Workflow

### Step 1: Generate Edits

Generate edits from your trained model and save them to a JSONL file:

```bash
# Generate edits from a trained model
python models/generate_edits.py \
    --checkpoint_root models/checkpoints/my_model \
    --output_jsonl models/generated_edits/my_model_validation.jsonl \
    --split validation

# Or use the base model
python models/generate_edits.py \
    --use_base_model_only \
    --output_jsonl models/generated_edits/base_model_validation.jsonl \
    --split validation

# Or parse diffs from existing rewrites (e.g., human edits or other models)
python models/generate_edits.py \
    --parse_diff \
    --model_name rewrite_40a_60ss \
    --output_jsonl models/generated_edits/rewrite_40a_60ss_validation.jsonl \
    --split validation
```

**Output**: A JSONL file where each line contains:
```json
{
  "post_id": "...",
  "issue": "...",
  "argument": "...",
  "sentences": [...],
  "completion": "...",
  "edits": [...],
  "metadata": {
    "parse_success": true,
    "num_edits": 5,
    "generation_time": 2.34
  }
}
```

### Step 2: Evaluate Edits

Evaluate the generated edits with your desired scorer configuration:

```bash
# Evaluate with all scorers enabled
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model_validation.jsonl \
    --output_jsonl models/predictions/my_model_validation_scored.jsonl

# Evaluate with some scorers disabled (ablation study)
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model_validation.jsonl \
    --output_jsonl models/predictions/my_model_validation_scored_no_hl.jsonl \
    --disable_human_like

# Evaluate with only semantic similarity and fluency
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model_validation.jsonl \
    --output_jsonl models/predictions/my_model_validation_scored_ss_fl_only.jsonl \
    --disable_human_like \
    --disable_appropriateness
```

**Output**: A JSONL file compatible with the evaluation interface, containing scored edits and all metrics.

### Step 3: Visualize Results

Open the evaluation interface in a web browser:

```bash
# Open the interface
open models/evaluation_interface.html
# Or on Linux:
xdg-open models/evaluation_interface.html
```

Then load your scored JSONL file(s) in the interface to visualize results.

## Advantages

### 1. Cost Efficiency
Generate edits once, evaluate multiple times with different scorer configurations:

```bash
# Generate once (costly)
python models/generate_edits.py \
    --checkpoint_root models/checkpoints/my_model \
    --output_jsonl models/generated_edits/my_model.jsonl

# Evaluate with different scorer configurations (fast)
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/my_model_all_scorers.jsonl

python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/my_model_no_hl.jsonl \
    --disable_human_like

python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/my_model_ss_fl_only.jsonl \
    --disable_human_like \
    --disable_appropriateness
```

### 2. Ablation Studies
Test which scorers contribute most to performance:

```bash
# All scorers
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/my_model_all_scorers.jsonl

# Without human-like
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/my_model_no_hl.jsonl \
    --disable_human_like

# Without fluency
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/my_model_no_fl.jsonl \
    --disable_fluency

# Only semantic similarity
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/my_model_ss_only.jsonl \
    --disable_human_like \
    --disable_fluency
```

### 4. Model Comparison
Generate edits from multiple models once, then evaluate all with the same configuration:

```bash
# Generate edits from different models
python models/generate_edits.py \
    --checkpoint_root models/checkpoints/model_v1 \
    --output_jsonl models/generated_edits/model_v1.jsonl

python models/generate_edits.py \
    --checkpoint_root models/checkpoints/model_v2 \
    --output_jsonl models/generated_edits/model_v2.jsonl

python models/generate_edits.py \
    --use_base_model_only \
    --output_jsonl models/generated_edits/base_model.jsonl

# Evaluate all with the same configuration
for model in model_v1 model_v2 base_model; do
    python models/evaluate_edits.py \
        --input_jsonl models/generated_edits/${model}.jsonl \
        --output_jsonl models/predictions/${model}_scored.jsonl
done

# Load all three files in the evaluation interface for side-by-side comparison
```

## Configuration Options

### generate_edits.py

| Option | Description |
|--------|-------------|
| `--checkpoint_root` | Path to model checkpoint directory |
| `--output_jsonl` | Output path for generated edits |
| `--use_base_model_only` | Use base model without LoRA |
| `--parse_diff` | Parse diffs instead of generating |
| `--model_name` | Model column name (for parse_diff) |
| `--split` | Dataset split (train/validation/test) |

### evaluate_edits.py

| Option | Description |
|--------|-------------|
| `--input_jsonl` | Input path with generated edits |
| `--output_jsonl` | Output path for scored edits |
| `--disable_semantic_similarity` | Disable SS scorer (ablation) |
| `--disable_fluency` | Disable fluency scorer (ablation) |
| `--disable_human_like` | Disable HL scorer (ablation) |
| `--disable_appropriateness` | Disable app scorer (ablation) |

### Scorer Thresholds

All thresholds are defined in the scorer classes themselves and cannot be overridden via command-line arguments. If you need to change thresholds, modify the scorer class defaults:

**Local Scorers:**
- `SemanticSimilarityScorer`: 0.6144
- `HumanLikeScorer`: 2.4527
- `FluencyScorer`: Binary classifier (no threshold)

**Global Scorers:**
- `GlobalSemanticSimilarityScorer`: 0.3842
- `GlobalHumanLikeScorer`: 2.3567
- `GlobalFluencyScorer`: 0.5

## Legacy Script

The original `predict_and_evaluate.py` still exists for backward compatibility but is deprecated. It combines both generation and evaluation in one step:

```bash
# Old workflow (deprecated, but still works)
python models/predict_and_evaluate.py \
    --checkpoint_root <checkpoint_path> \
    --output_jsonl <output_file.jsonl>
```

For new experiments, use the two-step workflow described above.

## File Formats

### Generated Edits Format (output of generate_edits.py)
```json
{
  "post_id": "post_123",
  "issue": "Climate change",
  "argument": "The original argument text...",
  "sentences": ["Sentence 1 text.", "Sentence 2 text."],
  "completion": "Model's JSON output...",
  "edits": [
    {
      "sentence_id": 1,
      "reason": "Toxic Emotions",
      "inappropriate_part": "totally insane",
      "rewritten_part": "questionable"
    }
  ],
  "metadata": {
    "parse_success": true,
    "num_edits": 1,
    "generation_time": 2.5
  }
}
```

### Scored Edits Format (output of evaluate_edits.py)
```json
{
  "post_id": "post_123",
  "issue": "Climate change",
  "argument": "The original argument text...",
  "argument_after_edits": "Text with perfect edits applied...",
  "argument_after_all_edits": "Text with all valid edits applied...",
  "edits": [
    {
      "sentence_id": 1,
      "reason": "Toxic Emotions",
      "inappropriate_part": "totally insane",
      "rewritten_part": "questionable",
      "original_sentence": "Sentence 1 text.",
      "valid": true,
      "reason_correct": true,
      "rewards": {
        "semantic_similarity": 1.0,
        "fluency": 1.0,
        "human_like": 1.0,
        "app": 1.0,
        "perfect": 1.0
      }
    }
  ],
  "metrics": {
    "App": 1.0,
    "Sim": 0.95,
    "NES": 0.92,
    "PPL": 15.3,
    "GM": 0.67
  },
  "global_scores": {
    "semantic_similarity_binary": 1.0,
    "semantic_similarity_score": 0.85,
    "human_like_binary": 1.0,
    "human_like_perplexity": 2.1,
    "fluency_binary": 1.0,
    "fluency_confidence": 0.95
  },
  "predicted_scores_before": {...},
  "predicted_scores_after": {...}
}
```

## Tips

1. **Organize your output directories**:
   ```
   models/
     generated_edits/     # Store raw generated edits here
     predictions/         # Store scored evaluations here
   ```

2. **Use descriptive filenames** that include configuration details:
   ```
   model_v1_hl2.5_ss0.7_validation.jsonl
   ```

3. **Keep generated edits** for future re-evaluation:
   - Generated edits are reusable
   - Disk space is cheaper than GPU time

4. **Batch evaluations** with a shell script:
   ```bash
   #!/bin/bash
   # evaluate_all.sh

   INPUT="models/generated_edits/my_model.jsonl"

   # Test different configurations
   python models/evaluate_edits.py --input_jsonl $INPUT \
       --output_jsonl models/predictions/config_1.jsonl \
       --semantic_similarity_threshold 0.6

   python models/evaluate_edits.py --input_jsonl $INPUT \
       --output_jsonl models/predictions/config_2.jsonl \
       --semantic_similarity_threshold 0.7

   # Add more configurations...
   ```

5. **Use the evaluation interface** to compare results side-by-side:
   - Load multiple JSONL files
   - Compare metrics across configurations
   - Use radar charts for visual comparison

# Quick Start: New Evaluation Workflow

## TL;DR

**Old way (slow, inflexible):**
```bash
# Generates edits AND evaluates in one go
python models/predict_and_evaluate.py \
    --checkpoint_root models/checkpoints/my_model \
    --output_jsonl models/predictions/output.jsonl
```

**New way (fast, flexible):**
```bash
# 1. Generate edits ONCE (costly, ~2-3 hours for validation set)
python models/generate_edits.py \
    --checkpoint_root models/checkpoints/my_model \
    --output_jsonl models/generated_edits/my_model.jsonl

# 2. Evaluate MULTIPLE TIMES with different scorer configs (fast, ~10-20 minutes each)
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/all_scorers.jsonl

python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/no_human_like.jsonl \
    --disable_human_like
```

## Why This Is Better

1. **Saves Time**: Generate edits once, evaluate many times
2. **Saves Money**: Avoid redundant GPU inference
3. **Easy Experimentation**: Test different scorer configurations instantly
4. **Ablation Studies**: Disable scorers to see their impact
5. **Model Comparison**: Evaluate multiple models with same configuration

## Common Use Cases

### Use Case 1: Ablation Study

```bash
# Generate once
python models/generate_edits.py \
    --checkpoint_root models/checkpoints/my_model \
    --output_jsonl models/generated_edits/my_model.jsonl

# Test with all scorers
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/all_scorers.jsonl

# Test without human-like
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/no_hl.jsonl \
    --disable_human_like

# Test without fluency
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/no_fl.jsonl \
    --disable_fluency

# Compare to see which scorer matters most
```

### Use Case 2: Compare Multiple Models

```bash
# Generate edits from each model once
python models/generate_edits.py \
    --checkpoint_root models/checkpoints/model_v1 \
    --output_jsonl models/generated_edits/model_v1.jsonl

python models/generate_edits.py \
    --checkpoint_root models/checkpoints/model_v2 \
    --output_jsonl models/generated_edits/model_v2.jsonl

python models/generate_edits.py \
    --use_base_model_only \
    --output_jsonl models/generated_edits/base_model.jsonl

# Evaluate all with same config
for model in model_v1 model_v2 base_model; do
    python models/evaluate_edits.py \
        --input_jsonl models/generated_edits/${model}.jsonl \
        --output_jsonl models/predictions/${model}_scored.jsonl
done

# Load all 3 in interface for side-by-side comparison
```

### Use Case 3: Evaluate Human Edits

```bash
# Parse human edits (or edits from another system)
python models/generate_edits.py \
    --parse_diff \
    --model_name rewrite_40a_60ss \
    --output_jsonl models/generated_edits/human_edits.jsonl

# Evaluate them
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/human_edits.jsonl \
    --output_jsonl models/predictions/human_edits_scored.jsonl
```

## Directory Structure

Organize your files like this:

```
models/
├── generate_edits.py           # Step 1: Generate edits
├── evaluate_edits.py            # Step 2: Evaluate edits
├── predict_and_evaluate.py      # Legacy (deprecated)
├── evaluation_interface.html    # Visualize results
├── EVALUATION_WORKFLOW.md       # Full documentation
├── QUICK_START.md               # This file
├── example_evaluation_workflow.sh  # Example script
├── generated_edits/             # Store raw edits here
│   ├── model_v1.jsonl
│   ├── model_v2.jsonl
│   └── base_model.jsonl
└── predictions/                 # Store scored results here
    ├── model_v1_default.jsonl
    ├── model_v1_ss0.7.jsonl
    ├── model_v1_no_hl.jsonl
    └── ...
```

## Quick Commands Reference

### Generate Edits

```bash
# From trained model
python models/generate_edits.py \
    --checkpoint_root <path> \
    --output_jsonl <output.jsonl>

# From base model
python models/generate_edits.py \
    --use_base_model_only \
    --output_jsonl <output.jsonl>

# From diffs
python models/generate_edits.py \
    --parse_diff \
    --model_name <column_name> \
    --output_jsonl <output.jsonl>
```

### Evaluate Edits

```bash
# Default: all scorers enabled
python models/evaluate_edits.py \
    --input_jsonl <input.jsonl> \
    --output_jsonl <output.jsonl>

# Ablation: disable specific scorers
python models/evaluate_edits.py \
    --input_jsonl <input.jsonl> \
    --output_jsonl <output.jsonl> \
    --disable_human_like \
    --disable_fluency
```

### Scorer Thresholds

All thresholds are predefined in the scorer classes and cannot be overridden via CLI. To change thresholds, modify the scorer class directly:

- **Semantic Similarity**: 0.6144 (local), 0.3842 (global)
- **Human-Like**: 2.4527 (local), 2.3567 (global)
- **Fluency**: Binary classifier (local), 0.5 (global)

## Example Workflow Script

Run the example workflow:

```bash
# Edit the configuration in the script first
vim models/example_evaluation_workflow.sh

# Then run it
./models/example_evaluation_workflow.sh
```

## Next Steps

1. Read the full documentation: `models/EVALUATION_WORKFLOW.md`
2. Try the example script: `./models/example_evaluation_workflow.sh`
3. Generate edits for your model
4. Experiment with different evaluation configurations
5. Visualize results in the evaluation interface

## Tips

- Keep generated edits - they're reusable
- Use descriptive filenames with config info
- Batch multiple evaluations in a shell script
- Load multiple results in the interface to compare
- Generated edits are ~10x smaller than scored results

## Getting Help

- Full workflow: `models/EVALUATION_WORKFLOW.md`
- Example script: `models/example_evaluation_workflow.sh`
- Evaluation interface: `models/evaluation_interface.html`

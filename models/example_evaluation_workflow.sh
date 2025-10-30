#!/bin/bash

# Example workflow for evaluating models with the new two-step process
# This script demonstrates how to:
# 1. Generate edits once (costly)
# 2. Evaluate with multiple configurations (fast)

set -e  # Exit on error

# Configuration
MODEL_CHECKPOINT="models/checkpoints/my_model"
SPLIT="validation"
EDITS_DIR="models/generated_edits"
PREDICTIONS_DIR="models/predictions"

# Create output directories
mkdir -p "$EDITS_DIR"
mkdir -p "$PREDICTIONS_DIR"

echo "==================================="
echo "Step 1: Generate Edits (One Time)"
echo "==================================="

# Generate edits from trained model
echo "Generating edits from trained model..."
python models/generate_edits.py \
    --checkpoint_root "$MODEL_CHECKPOINT" \
    --output_jsonl "$EDITS_DIR/my_model_${SPLIT}.jsonl" \
    --split "$SPLIT"

# Optional: Generate edits from base model for comparison
# echo "Generating edits from base model..."
# python models/generate_edits.py \
#     --use_base_model_only \
#     --output_jsonl "$EDITS_DIR/base_model_${SPLIT}.jsonl" \
#     --split "$SPLIT"

echo ""
echo "==================================="
echo "Step 2: Evaluate with Different Configurations"
echo "==================================="

# Configuration 1: All scorers with default thresholds
echo "Evaluating with all scorers (default thresholds)..."
python models/evaluate_edits.py \
    --input_jsonl "$EDITS_DIR/my_model_${SPLIT}.jsonl" \
    --output_jsonl "$PREDICTIONS_DIR/my_model_${SPLIT}_default.jsonl"

# Configuration 2: Without human-like scorer (ablation)
echo "Evaluating without human-like scorer..."
python models/evaluate_edits.py \
    --input_jsonl "$EDITS_DIR/my_model_${SPLIT}.jsonl" \
    --output_jsonl "$PREDICTIONS_DIR/my_model_${SPLIT}_no_hl.jsonl" \
    --disable_human_like

# Configuration 3: Only semantic similarity and fluency
echo "Evaluating with only SS and fluency..."
python models/evaluate_edits.py \
    --input_jsonl "$EDITS_DIR/my_model_${SPLIT}.jsonl" \
    --output_jsonl "$PREDICTIONS_DIR/my_model_${SPLIT}_ss_fl_only.jsonl" \
    --disable_human_like \
    --disable_appropriateness

echo ""
echo "==================================="
echo "Results Summary"
echo "==================================="

echo "Generated edits saved to:"
echo "  - $EDITS_DIR/my_model_${SPLIT}.jsonl"
echo ""
echo "Evaluation results saved to:"
echo "  - $PREDICTIONS_DIR/my_model_${SPLIT}_default.jsonl (all scorers)"
echo "  - $PREDICTIONS_DIR/my_model_${SPLIT}_no_hl.jsonl (ablation: no human-like)"
echo "  - $PREDICTIONS_DIR/my_model_${SPLIT}_ss_fl_only.jsonl (only SS and fluency)"
echo ""
echo "To visualize results:"
echo "  1. Open models/evaluation_interface.html in a web browser"
echo "  2. Load the JSONL files from $PREDICTIONS_DIR"
echo "  3. Compare configurations side-by-side"
echo ""
echo "Done!"

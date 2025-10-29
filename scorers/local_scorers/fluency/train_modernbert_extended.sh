#!/bin/bash
# Train ModernBERT fluency classifier with extended dataset using accelerate

# Set environment variables
export WANDB_PROJECT="fluency-scorer-eval"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
DATASET_NAME="timonziegenbein/fluency-edits-gec-only-extended"
MODEL_NAME="answerdotai/ModernBERT-large"
OUTPUT_DIR="scorers/fluency/modernbert_gec_extended_v2"
RESULTS_DIR="scorers/fluency/modernbert_gec_extended_v2"
WANDB_PROJECT_NAME="fluency-scorer-eval"

# Training hyperparameters (optimized from previous runs)
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1

# Run training with accelerate
accelerate launch \
  --config_file scorers/fluency/accelerate_config.yaml \
  scorers/fluency/train_fluency_model.py \
  --dataset_name "${DATASET_NAME}" \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --wandb_project "${WANDB_PROJECT_NAME}" \
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}

echo ""
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "Results saved to: ${RESULTS_DIR}"

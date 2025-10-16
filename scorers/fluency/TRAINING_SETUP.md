# Fluency Model Training Setup

This document describes the complete training setup for the ModernBERT fluency classifier with multi-GPU support.

## Training Configuration

### Hyperparameters (from best run: jycpxsf5)

```yaml
Model: answerdotai/ModernBERT-large
Learning Rate: 3e-5
Batch Size: 8 per GPU
Gradient Accumulation: 1
Epochs: 3
Weight Decay: 0.001
Optimizer: AdamW (adamw_torch)
Scheduler: Cosine
Warmup Ratio: 0.1
Precision: BF16
Seed: 42
```

### Evaluation Strategy

- **Strategy**: By epoch
- **Metric for best model**: Precision (treating fluent=1 as positive class)
- **Metrics tracked**:
  - Accuracy: Overall correctness
  - Precision: TP / (TP + FP) where positive = fluent (1)
  - Recall: TP / (TP + FN) where positive = fluent (1)
  - F1: Harmonic mean of precision and recall

### Multi-GPU Setup

The training uses **4 GPUs** with Accelerate for distributed training:

```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
mixed_precision: bf16
gpu_ids: all
```

## Running Training

### Basic Usage

```bash
# Train with 4 GPUs using accelerate
accelerate launch \
    --config_file scorers/fluency/accelerate_config.yaml \
    scorers/fluency/train_fluency_model.py \
    --output_dir ./models/fluency_modernbert \
    --results_dir ./results/fluency_modernbert
```

### With Custom Hyperparameters

```bash
accelerate launch \
    --config_file scorers/fluency/accelerate_config.yaml \
    scorers/fluency/train_fluency_model.py \
    --dataset_name timonziegenbein/fluency-agumented-edits \
    --model_name answerdotai/ModernBERT-large \
    --output_dir ./models/fluency_modernbert \
    --results_dir ./results/fluency_modernbert \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --weight_decay 0.001 \
    --wandb_project fluency-scorer-eval
```

### Quick Test (Small Dataset)

```bash
accelerate launch \
    --config_file scorers/fluency/accelerate_config.yaml \
    scorers/fluency/train_fluency_model.py \
    --max_train_samples 1000 \
    --max_eval_samples 200 \
    --output_dir ./models/fluency_modernbert_test
```

## Dataset Format

The training uses the dataset created by `create_fluency_eval_dataset.py`:

```python
{
    'id': 'test_42_edit_0',
    'original_sentence': 'This are a test.',      # Before edit
    'inappropriate_part': 'are',                   # What to replace
    'rewritten_part': 'is',                        # Replacement
    'expected_score': 1.0                          # 1.0 = fluent, 0.0 = non-fluent
}
```

**Important**: Only examples with exactly **one edit** are used for training (filtered in dataset creation).

## Model Input

The model receives two sequences:

1. **Sequence 1**: Original sentence (before edit)
2. **Sequence 2**: Rewritten sentence (after edit)

Tokenized as:
```
[CLS] This are a test. [SEP] This is a test. [SEP]
```

Label:
- `1` = Fluent edit (maintains or improves fluency)
- `0` = Non-fluent edit (harms fluency)

## Optimization Goal

**Primary Metric: Precision**

The model is optimized to **maximize precision** for fluent edits (label=1). This means:
- The model should be conservative in predicting "fluent"
- When it predicts an edit is fluent, it should be highly confident
- This minimizes false positives (incorrectly accepting non-fluent edits)

## Monitoring Training

Training progress is logged to W&B project: `fluency-scorer-eval`

Key metrics to watch:
- `eval_precision`: Main optimization target
- `eval_recall`: Ensure not too low
- `eval_f1`: Overall balance
- `eval_accuracy`: General correctness

## Output

After training, the following are saved:

```
./models/fluency_modernbert/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── tokenizer.json
└── ...

./results/fluency_modernbert/
├── predictions.txt
├── labels.txt
└── results.json
```

## Performance Expectations

Based on the reference run (jycpxsf5) with ModernBERT-base:
- Model: ModernBERT-base (not large)
- Evaluation: By epoch
- Scheduler: Cosine
- Precision target: Maximize

With ModernBERT-large and the optimized setup, we expect improved performance.

## Troubleshooting

### GPU Memory Issues

If you encounter OOM errors:
1. Reduce `per_device_train_batch_size` (e.g., to 4)
2. Increase `gradient_accumulation_steps` (e.g., to 2)
3. Use `gradient_checkpointing` (add to TrainingArguments)

### Slow Training

If training is slow:
1. Increase `per_device_train_batch_size` if memory allows
2. Check `dataloader_num_workers` (currently set to 4)
3. Verify all 4 GPUs are being used: `nvidia-smi`

### Imbalanced Metrics

If precision is very high but recall is very low:
- The model is being too conservative
- Consider adjusting the classification threshold during inference
- Or use `metric_for_best_model="f1"` for better balance

## Next Steps

After training:

1. **Evaluate on test set**:
   ```bash
   python scorers/fluency/evaluate_fluency_scorer.py \
       --eval-dataset-name timonziegenbein/fluency-agumented-edits \
       --eval-dataset-split test
   ```

2. **Compare with LLM baseline**:
   ```bash
   python scorers/fluency/evaluate_fluency_scorer.py \
       --eval-dataset-name timonziegenbein/fluency-agumented-edits \
       --use-llm \
       --llm-model-name gemini-2.0-flash-exp
   ```

3. **Integrate into fluency scorer**:
   Update `scorers/fluency/fluency_scorer.py` to use the trained model

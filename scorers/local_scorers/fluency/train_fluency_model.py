
import argparse
import json
import numpy as np
import os
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
import weave

def main(args):
    # Set seed for reproducibility
    set_seed(42)

    # Initialize wandb with project name (only on main process to avoid multiple runs)
    import wandb
    from accelerate import Accelerator

    accelerator = Accelerator()

    # Only initialize wandb and weave on the main process
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project)
        # Initialize Weave for tracing (use same project as wandb)
        weave.init(project_name=args.wandb_project)

    # Load the dataset from the Hugging Face Hub
    dataset = load_dataset(args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_and_tokenize(examples):
        """
        Reconstruct before/after sentences from edit information and tokenize.

        Expected input format (from create_fluency_eval_dataset.py):
        - original_sentence: The sentence before the edit
        - inappropriate_part: The part to be replaced
        - rewritten_part: The replacement text
        - expected_score: 1.0 (fluent) or 0.0 (non-fluent)

        We create two sequences:
        - text_before: The original sentence (before edit)
        - text_after: The rewritten sentence (after edit)
        """
        text_before = []
        text_after = []
        labels = []

        for i in range(len(examples["original_sentence"])):
            original = examples["original_sentence"][i]
            inappropriate = examples["inappropriate_part"][i]
            rewritten = examples["rewritten_part"][i]

            # Text before the edit is just the original sentence
            text_before.append(original)

            # Text after the edit: replace inappropriate_part with rewritten_part
            text_after.append(original.replace(inappropriate, rewritten, 1))

            # Label: expected_score is already 0.0 or 1.0, convert to int
            labels.append(int(examples["expected_score"][i]))

        # Tokenize the before/after pairs
        tokenized = tokenizer(
            text_before,
            text_after,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        # Add labels
        tokenized["labels"] = labels

        return tokenized

    tokenized_datasets = dataset.map(preprocess_and_tokenize, batched=True)

    if args.max_train_samples is not None:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(args.max_train_samples))
    if args.max_eval_samples is not None:
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(args.max_eval_samples))
    if args.max_predict_samples is not None:
        tokenized_datasets["test"] = tokenized_datasets["test"].select(range(args.max_predict_samples))

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    @weave.op(tracing_sample_rate=0.01)
    def compute_metrics(eval_pred):
        """
        Compute metrics matching the scorers in evaluate_fluency_scorer.py.
        Treats fluent (label=1) as the positive class.
        Traces 1% of calls to Weave for monitoring.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Calculate metrics treating fluent (1) as positive class
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, pos_label=1, zero_division=0),
            'recall': recall_score(labels, predictions, pos_label=1, zero_division=0),
            'f1': f1_score(labels, predictions, pos_label=1, zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=50,
        eval_on_start=True,  # Evaluate before training starts
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="precision",  # Optimize for precision
        greater_is_better=True,
        report_to="wandb",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        bf16=True,
        seed=42,
        group_by_length=True,  # Group samples by length for efficiency
        # Accelerate/DDP settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
    )

    # Initialize model directly (not using model_init for multi-GPU)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on the test set
    results = trainer.predict(tokenized_datasets["test"])

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    np.savetxt(os.path.join(args.results_dir, "predictions.txt"), results.predictions)
    np.savetxt(os.path.join(args.results_dir, "labels.txt"), results.label_ids)

    with open(os.path.join(args.results_dir, "results.json"), "w") as f:
        json.dump(results.metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fluency classification model.")
    parser.add_argument("--dataset_name", type=str, default="timonziegenbein/fluency-agumented-edits", help="Name of the evaluation dataset on the Hugging Face Hub (created by create_fluency_eval_dataset.py).")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-large", help="Name of the pretrained model to use.")
    parser.add_argument("--output_dir", type=str, default="../models/fluency_classifier", help="Directory to save the trained model.")
    parser.add_argument("--results_dir", type=str, default="../results/fluency_classifier", help="Directory to save the evaluation results.")

    # Wandb/Weave configuration
    parser.add_argument("--wandb_project", type=str, default="fluency-scorer-eval", help="Weights & Biases project name (also used for Weave tracing).")

    # Training hyperparameters (from wandb run jycpxsf5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per GPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")

    # Data limiting (for debugging)
    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging, truncate the number of training examples to this value.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="For debugging, truncate the number of evaluation examples to this value.")
    parser.add_argument("--max_predict_samples", type=int, default=None, help="For debugging, truncate the number of prediction examples to this value.")

    args = parser.parse_args()
    main(args)

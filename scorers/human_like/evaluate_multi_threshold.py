#!/usr/bin/env python3
"""
Evaluate multiple threshold candidates using Weave and Wandb.

This script:
1. Loads threshold candidates from JSON file
2. Creates a separate model for each threshold
3. Evaluates all models using Weave
4. Logs results to Wandb for comparison

Usage:
    python scorers/human_like/evaluate_multi_threshold.py \
        --threshold-file scorers/human_like/threshold_candidates.json \
        --model-path scorers/human_like/models/human_like_v3.pth \
        --eval-dataset-name timonziegenbein/human-like-edit-sequences-eval \
        --split test \
        --project human-like-threshold-selection \
        --device cuda
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import wandb
import weave
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scorers.human_like.model_defs import LanguageModel


# Global cache for models (outside the Weave Model class to avoid serialization issues)
_GLOBAL_MODEL_CACHE: Dict[str, Any] = {}


class HumanLikeScorerModel(weave.Model):
    """Weave Model wrapper for HumanLikeScorer with configurable threshold."""

    model_path: str
    device: str
    threshold: float
    vocab_type: str = "v2"  # v2 has 6 tokens including keep-in-edit

    def _get_or_create_model(self):
        """Load model once and cache to avoid serialization issues."""
        cache_key = f"{self.model_path}_{self.device}"

        if cache_key not in _GLOBAL_MODEL_CACHE:
            device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=device)

            # Extract model state
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Model hyperparameters (must match training)
            vocab_size = 6  # V2 vocab: <pad>, keep, del, add, replace, keep-in-edit
            embedding_dim = 200
            nhead = 2
            nhid = 200
            nlayers = 2
            dropout = 0.2

            # Initialize model
            model = LanguageModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                nhead=nhead,
                nhid=nhid,
                nlayers=nlayers,
                dropout=dropout
            )

            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # Define vocab (must match training!)
            vocab = {
                '<pad>': 0,
                'keep': 1,
                'del': 2,
                'add': 3,
                'replace': 4,
                'keep-in-edit': 5
            }

            _GLOBAL_MODEL_CACHE[cache_key] = {
                'model': model,
                'vocab': vocab,
                'device': device
            }

        return _GLOBAL_MODEL_CACHE[cache_key]

    def _calculate_perplexity(self, sequence: str) -> float:
        """Calculate perplexity for a single sequence (used by Weave predict)."""
        model_data = self._get_or_create_model()
        model = model_data['model']
        vocab = model_data['vocab']
        device = model_data['device']

        # Convert sequence to integers
        tokens = sequence.split(',')
        token_ids = [vocab.get(token.strip(), vocab['<pad>']) for token in tokens]

        # Pad to at least length 2
        if len(token_ids) < 2:
            token_ids.append(vocab['<pad>'])

        # Convert to tensor
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)

        # Compute loss
        with torch.no_grad():
            logits = model(input_ids)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Compute cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Perplexity is exp(loss)
            perplexity = torch.exp(loss).item()

        return perplexity

    @weave.op()
    def predict(self, sequence: str, original_sentence: str = "", inappropriate_part: str = "", rewritten_part: str = "", **kwargs) -> Dict[str, Any]:
        """
        Predict if an edit sequence is human-like.

        Args:
            sequence: Comma-separated edit operation sequence
            original_sentence: The full original sentence (for context)
            inappropriate_part: The part that was edited
            rewritten_part: The rewritten version

        Returns:
            Dictionary with:
                - human_like_score: 1.0 if human-like, 0.0 otherwise
                - perplexity: Calculated perplexity
                - threshold: Threshold used
                - original_sentence: Echo back for tracing
                - inappropriate_part: Echo back for tracing
                - rewritten_part: Echo back for tracing
        """
        perplexity = self._calculate_perplexity(sequence)
        human_like_score = 1.0 if perplexity <= self.threshold else 0.0

        return {
            "human_like_score": human_like_score,
            "perplexity": perplexity,
            "threshold": self.threshold,
            "original_sentence": original_sentence,
            "inappropriate_part": inappropriate_part,
            "rewritten_part": rewritten_part
        }


class AccuracyScorer(weave.Scorer):
    """Calculate accuracy of human-like predictions."""

    @weave.op()
    def score(self, label: int, model_output: dict) -> dict:
        """Calculate if prediction matches label."""
        prediction = int(model_output.get('human_like_score', 0))
        correct = 1.0 if prediction == label else 0.0
        return {"correct": correct}

    @weave.op()
    def summarize(self, score_rows: list) -> dict:
        """Calculate overall accuracy."""
        if not score_rows:
            return {"accuracy": 0.0}

        correct_count = sum(row['correct'] for row in score_rows)
        accuracy = correct_count / len(score_rows)
        return {"accuracy": accuracy}


class PrecisionScorer(weave.Scorer):
    """Calculate precision of human-like predictions."""

    @weave.op()
    def score(self, label: int, model_output: dict) -> dict:
        """Track true positives and false positives."""
        prediction = int(model_output.get('human_like_score', 0))
        return {
            "true_positive": 1 if prediction == 1 and label == 1 else 0,
            "false_positive": 1 if prediction == 1 and label == 0 else 0
        }

    @weave.op()
    def summarize(self, score_rows: list) -> dict:
        """Calculate overall precision."""
        if not score_rows:
            return {"precision": 0.0}

        tp = sum(row['true_positive'] for row in score_rows)
        fp = sum(row['false_positive'] for row in score_rows)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return {"precision": precision}


class RecallScorer(weave.Scorer):
    """Calculate recall of human-like predictions."""

    @weave.op()
    def score(self, label: int, model_output: dict) -> dict:
        """Track true positives and false negatives."""
        prediction = int(model_output.get('human_like_score', 0))
        return {
            "true_positive": 1 if prediction == 1 and label == 1 else 0,
            "false_negative": 1 if prediction == 0 and label == 1 else 0
        }

    @weave.op()
    def summarize(self, score_rows: list) -> dict:
        """Calculate overall recall."""
        if not score_rows:
            return {"recall": 0.0}

        tp = sum(row['true_positive'] for row in score_rows)
        fn = sum(row['false_negative'] for row in score_rows)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return {"recall": recall}


class F1Scorer(weave.Scorer):
    """Calculate F1 score of human-like predictions."""

    @weave.op()
    def score(self, label: int, model_output: dict) -> dict:
        """Track all confusion matrix values."""
        prediction = int(model_output.get('human_like_score', 0))
        return {
            "true_positive": 1 if prediction == 1 and label == 1 else 0,
            "false_positive": 1 if prediction == 1 and label == 0 else 0,
            "false_negative": 1 if prediction == 0 and label == 1 else 0
        }

    @weave.op()
    def summarize(self, score_rows: list) -> dict:
        """Calculate F1 score."""
        if not score_rows:
            return {"f1": 0.0}

        tp = sum(row['true_positive'] for row in score_rows)
        fp = sum(row['false_positive'] for row in score_rows)
        fn = sum(row['false_negative'] for row in score_rows)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }


async def evaluate_threshold(
    threshold_name: str,
    threshold_value: float,
    model_path: str,
    device: str,
    dataset: List[Dict],
    project: str
) -> Dict[str, Any]:
    """Evaluate a single threshold value."""
    print(f"\n{'='*60}")
    print(f"Evaluating {threshold_name}: {threshold_value:.4f}")
    print(f"{'='*60}")

    # Create model with this threshold
    model = HumanLikeScorerModel(
        model_path=model_path,
        device=device,
        threshold=threshold_value
    )

    # Create evaluation
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[
            AccuracyScorer(),
            PrecisionScorer(),
            RecallScorer(),
            F1Scorer()
        ],
        name=f"human_like_threshold_{threshold_name}"
    )

    # Run evaluation
    results = await evaluation.evaluate(
        model,
        __weave={"display_name": f"HumanLike_{threshold_name}"}
    )

    # Extract summary metrics
    summary = {}
    for scorer_name, scorer_results in results.items():
        if isinstance(scorer_results, dict):
            summary.update(scorer_results)

    print(f"\nResults for {threshold_name}:")
    print(f"  Accuracy:  {summary.get('accuracy', 0.0):.4f}")
    print(f"  Precision: {summary.get('precision', 0.0):.4f}")
    print(f"  Recall:    {summary.get('recall', 0.0):.4f}")
    print(f"  F1:        {summary.get('f1', 0.0):.4f}")

    return {
        'threshold_name': threshold_name,
        'threshold_value': threshold_value,
        **summary
    }


async def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple threshold candidates')
    parser.add_argument('--threshold-file', type=str, required=True,
                        help='JSON file with threshold candidates')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--eval-dataset-name', type=str, required=True,
                        help='HuggingFace evaluation dataset name')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (default: test)')
    parser.add_argument('--project', type=str, default='human-like-scorer-eval',
                        help='Wandb/Weave project name')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom run name for Wandb')

    args = parser.parse_args()

    # Initialize Wandb
    run_name = args.run_name or f"threshold_selection_{args.split}"
    wandb.init(
        project=args.project,
        name=run_name,
        config={
            'model_path': args.model_path,
            'eval_dataset': args.eval_dataset_name,
            'split': args.split
        }
    )

    # Initialize Weave
    weave.init(project_name=args.project)

    # Load threshold candidates
    print(f"Loading threshold candidates from {args.threshold_file}...")
    with open(args.threshold_file, 'r') as f:
        threshold_data = json.load(f)

    thresholds = threshold_data['thresholds']
    print(f"Found {len(thresholds)} threshold candidates")

    # Load evaluation dataset
    print(f"\nLoading evaluation dataset {args.eval_dataset_name} (split: {args.split})...")
    hf_dataset = load_dataset(args.eval_dataset_name, split=args.split)
    print(f"Loaded {len(hf_dataset)} examples")

    # Convert to list of dicts for Weave
    dataset = [
        {
            'sequence': ex['sequence'],
            'label': ex['label'],
            'type': ex.get('type', 'positive' if ex['label'] == 1 else 'negative'),
            'original_sentence': ex.get('original_sentence', ''),
            'inappropriate_part': ex.get('inappropriate_part', ''),
            'rewritten_part': ex.get('rewritten_part', '')
        }
        for ex in hf_dataset
    ]

    # Evaluate each threshold
    all_results = []
    for threshold_name in sorted(thresholds.keys()):
        threshold_value = thresholds[threshold_name]

        result = await evaluate_threshold(
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            model_path=args.model_path,
            device=args.device,
            dataset=dataset,
            project=args.project
        )

        all_results.append(result)

        # Log to Wandb
        wandb.log({
            f"{threshold_name}/accuracy": result.get('accuracy', 0.0),
            f"{threshold_name}/precision": result.get('precision', 0.0),
            f"{threshold_name}/recall": result.get('recall', 0.0),
            f"{threshold_name}/f1": result.get('f1', 0.0),
            f"{threshold_name}/threshold_value": threshold_value
        })

    # Create summary table
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL THRESHOLDS")
    print(f"{'='*80}")
    print(f"{'Threshold':<15} {'Value':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*80}")

    for result in all_results:
        print(f"{result['threshold_name']:<15} "
              f"{result['threshold_value']:>10.4f} "
              f"{result.get('accuracy', 0.0):>10.4f} "
              f"{result.get('precision', 0.0):>10.4f} "
              f"{result.get('recall', 0.0):>10.4f} "
              f"{result.get('f1', 0.0):>10.4f}")

    print(f"{'='*80}")

    # Find best threshold by F1
    best_by_f1 = max(all_results, key=lambda x: x.get('f1', 0.0))
    print(f"\nBest threshold by F1: {best_by_f1['threshold_name']} "
          f"(value={best_by_f1['threshold_value']:.4f}, F1={best_by_f1.get('f1', 0.0):.4f})")

    # Log best threshold to Wandb
    wandb.log({
        'best_threshold_name': best_by_f1['threshold_name'],
        'best_threshold_value': best_by_f1['threshold_value'],
        'best_f1': best_by_f1.get('f1', 0.0)
    })

    # Save results to JSON
    results_file = Path(args.threshold_file).parent / 'threshold_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': all_results,
            'best_by_f1': best_by_f1,
            'config': vars(args)
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"View interactive results in Weave: https://wandb.ai/{wandb.run.entity}/{args.project}")

    wandb.finish()


if __name__ == '__main__':
    asyncio.run(main())

"""
Weave Evaluation for Human-Like Scorer

This script creates a comprehensive evaluation of the human-like scorer using Weights & Biases Weave.
It tests the scorer's ability to:
1. Identify human-like edit sequences (real human edits)
2. Reject non-human-like sequences (synthetic/corrupted edits)
3. Compare different model versions and baselines
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import weave
import torch
import argparse
from typing import List, Dict, Any
from datasets import load_dataset
import logging
import numpy as np

from scorers.human_like.model_defs import LanguageModel
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated vocabulary with keep-in-edit token (v2)
HL_VOCAB_V2 = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}
# Original vocabulary (v1)
HL_VOCAB_V1 = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4}

# Global cache for models
_MODEL_CACHE = {}


class HumanLikeScorerModel(weave.Model):
    """Weave Model wrapper for Human-Like Scorer"""

    model_path: str
    device: str
    vocab_type: str
    threshold: float
    max_len: int = 500

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 vocab_type: str = "v2", threshold: float = 1.1465, max_len: int = 500):
        super().__init__(
            model_path=model_path,
            device=device,
            vocab_type=vocab_type,
            threshold=threshold,
            max_len=max_len
        )
        self._cache_key = f"hl_model_{model_path}_{vocab_type}_{device}"

    def _get_or_create_model(self):
        """Get model from global cache or load it"""
        global _MODEL_CACHE
        if self._cache_key not in _MODEL_CACHE:
            logger.error(f"Model not found in cache: {self._cache_key}")
            raise RuntimeError("Model was not pre-loaded. This is a programming error.")
        return _MODEL_CACHE[self._cache_key]

    def _calculate_perplexity(self, sequence: str) -> float:
        """Calculate perplexity for a sequence"""
        model_data = self._get_or_create_model()
        model = model_data['model']
        vocab = model_data['vocab']

        # Convert sequence to integers
        sequence_as_int = [vocab.get(token, 0) for token in sequence.split(',')]

        if len(sequence_as_int) <= 1:
            return float('inf')

        input_seq = sequence_as_int[:-1]
        target_seq = sequence_as_int[1:]

        # Pad sequences
        padded_input = np.array(
            input_seq[:self.max_len] + [vocab['<pad>']] * (self.max_len - len(input_seq))
            if len(input_seq) < self.max_len else input_seq[:self.max_len]
        )
        padded_target = np.array(
            target_seq[:self.max_len] + [vocab['<pad>']] * (self.max_len - len(target_seq))
            if len(target_seq) < self.max_len else target_seq[:self.max_len]
        )

        inputs = torch.from_numpy(padded_input).long().unsqueeze(0).to(self.device)
        targets = torch.from_numpy(padded_target).long().unsqueeze(0).to(self.device)

        criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

        with torch.no_grad():
            output = model(inputs)
            loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
            perplexity = torch.exp(loss)
            return perplexity.item()

    @weave.op()
    def predict(self, sequence: str, original_sentence: str, inappropriate_part: str,
                rewritten_part: str) -> Dict[str, Any]:
        """
        Score whether an edit sequence is human-like.

        Args:
            sequence: Comma-separated edit operation sequence
            original_sentence: The original sentence (for display)
            inappropriate_part: The text being edited (for display)
            rewritten_part: The replacement text (for display)

        Returns:
            Dictionary containing the human-like score and metadata
        """
        perplexity = self._calculate_perplexity(sequence)

        # Binary score based on threshold
        human_like_score = 1.0 if perplexity <= self.threshold else 0.0

        return {
            "human_like_score": human_like_score,
            "perplexity": perplexity,
            "threshold": self.threshold,
            "sequence_length": len(sequence.split(','))
        }


class RandomBaselineModel(weave.Model):
    """Random baseline that predicts 0.0 or 1.0 with equal probability"""

    seed: int = 42

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        import random
        random.seed(seed)

    @weave.op()
    def predict(self, sequence: str, original_sentence: str, inappropriate_part: str,
                rewritten_part: str) -> Dict[str, Any]:
        import random
        human_like_score = float(random.choice([0, 1]))

        return {
            "human_like_score": human_like_score,
            "perplexity": 0.0,
            "threshold": 0.0,
            "sequence_length": len(sequence.split(','))
        }


class AlwaysHumanLikeModel(weave.Model):
    """Baseline that always predicts 1.0 (human-like)"""

    @weave.op()
    def predict(self, sequence: str, original_sentence: str, inappropriate_part: str,
                rewritten_part: str) -> Dict[str, Any]:
        return {
            "human_like_score": 1.0,
            "perplexity": 0.0,
            "threshold": 0.0,
            "sequence_length": len(sequence.split(','))
        }


class AlwaysNonHumanLikeModel(weave.Model):
    """Baseline that always predicts 0.0 (non-human-like)"""

    @weave.op()
    def predict(self, sequence: str, original_sentence: str, inappropriate_part: str,
                rewritten_part: str) -> Dict[str, Any]:
        return {
            "human_like_score": 0.0,
            "perplexity": 0.0,
            "threshold": 0.0,
            "sequence_length": len(sequence.split(','))
        }


class AccuracyScorer(weave.Scorer):
    """Scorer that evaluates whether the prediction matches the expected label"""

    @weave.op()
    def score(self, label: int, model_output: Dict[str, Any]) -> Dict[str, Any]:
        predicted_score = model_output["human_like_score"]
        is_correct = predicted_score == float(label)

        return {"correct": is_correct}


class PrecisionScorer(weave.Scorer):
    @weave.op()
    def score(self, label: int, model_output: Dict[str, Any]) -> Dict[str, Any]:
        predicted_score = model_output["human_like_score"]
        true_positive = bool(label and predicted_score)
        false_positive = bool(predicted_score and not label)
        return {
            "true_positive": true_positive,
            "false_positive": false_positive
        }

    @weave.op()
    def summarize(self, score_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_true_positive = sum(score["true_positive"] for score in score_rows)
        total_false_positive = sum(score["false_positive"] for score in score_rows)
        denominator = total_true_positive + total_false_positive
        precision = total_true_positive / denominator if denominator > 0 else 0
        return {"precision": precision}


class RecallScorer(weave.Scorer):
    @weave.op()
    def score(self, label: int, model_output: Dict[str, Any]) -> Dict[str, Any]:
        predicted_score = model_output["human_like_score"]
        true_positive = bool(label and predicted_score)
        false_negative = bool(label and not predicted_score)
        return {
            "true_positive": true_positive,
            "false_negative": false_negative
        }

    @weave.op()
    def summarize(self, score_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_true_positive = sum(score["true_positive"] for score in score_rows)
        total_false_negative = sum(score["false_negative"] for score in score_rows)
        denominator = total_true_positive + total_false_negative
        recall = total_true_positive / denominator if denominator > 0 else 0
        return {"recall": recall}


class F1Scorer(weave.Scorer):
    @weave.op()
    def score(self, label: int, model_output: Dict[str, Any]) -> Dict[str, Any]:
        predicted_score = model_output["human_like_score"]
        true_positive = bool(label and predicted_score)
        false_positive = bool(predicted_score and not label)
        false_negative = bool(label and not predicted_score)
        return {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative
        }

    @weave.op()
    def summarize(self, score_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_true_positive = sum(score["true_positive"] for score in score_rows)
        total_false_positive = sum(score["false_positive"] for score in score_rows)
        total_false_negative = sum(score["false_negative"] for score in score_rows)

        precision_denominator = total_true_positive + total_false_positive
        precision = total_true_positive / precision_denominator if precision_denominator > 0 else 0

        recall_denominator = total_true_positive + total_false_negative
        recall = total_true_positive / recall_denominator if recall_denominator > 0 else 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


def create_evaluation_dataset(examples: List[Dict], dataset_name: str = "human_like_test_cases"):
    """Create a Weave dataset from examples"""
    # Format examples for Weave
    formatted_examples = []
    for i, ex in enumerate(examples):
        formatted_examples.append({
            "id": f"test_case_{i+1}",
            "sequence": ex["sequence"],
            "label": ex["label"],
            "original_sentence": ex["original_sentence"],
            "inappropriate_part": ex["inappropriate_part"],
            "rewritten_part": ex["rewritten_part"],
            "type": ex.get("type", "human" if ex["label"] == 1 else "synthetic")
        })

    dataset = weave.Dataset(name=dataset_name, rows=formatted_examples)
    weave.publish(dataset)

    logger.info(f"Created Weave dataset '{dataset_name}' with {len(formatted_examples)} examples")
    return dataset


async def run_evaluation_async(args):
    """Run the evaluation asynchronously"""
    # Initialize Weave
    weave.init(args.project)
    logger.info(f"Initialized Weave project: {args.project}")

    # Load evaluation dataset
    logger.info(f"Loading evaluation dataset: {args.eval_dataset_name}, split: {args.eval_dataset_split}")
    hf_dataset = load_dataset(args.eval_dataset_name, split=args.eval_dataset_split)

    # Limit examples if specified
    if args.max_eval_examples:
        hf_dataset = hf_dataset.shuffle(seed=42).select(range(min(args.max_eval_examples, len(hf_dataset))))

    # Convert to list
    all_examples = list(hf_dataset)
    logger.info(f"Loaded {len(all_examples)} examples")

    # Statistics
    positive_count = sum(1 for ex in all_examples if ex['label'] == 1)
    negative_count = sum(1 for ex in all_examples if ex['label'] == 0)
    logger.info(f"  Positive (human): {positive_count}")
    logger.info(f"  Negative (synthetic): {negative_count}")

    # Create evaluation dataset
    dataset = create_evaluation_dataset(all_examples, "human_like_test_cases")

    # Initialize the model based on flags
    if args.use_random_baseline:
        logger.info("Initializing RandomBaselineModel...")
        model = RandomBaselineModel(seed=args.random_seed)
    elif args.use_always_humanlike:
        logger.info("Initializing AlwaysHumanLikeModel...")
        model = AlwaysHumanLikeModel()
    elif args.use_always_nonhumanlike:
        logger.info("Initializing AlwaysNonHumanLikeModel...")
        model = AlwaysNonHumanLikeModel()
    else:
        # Load the trained model
        logger.info(f"Pre-loading Human-Like Scorer model from: {args.model_path}")

        # Determine vocabulary
        vocab = HL_VOCAB_V2 if args.vocab_type == "v2" else HL_VOCAB_V1

        # Model architecture parameters
        embedding_dim = 200
        nhead = 2
        nhid = 200
        nlayers = 2
        dropout = 0.2

        # Load model
        hl_model = LanguageModel(len(vocab), embedding_dim, nhead, nhid, nlayers, dropout).to(args.device)
        hl_model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        hl_model.eval()

        # Cache the model
        cache_key = f"hl_model_{args.model_path}_{args.vocab_type}_{args.device}"
        _MODEL_CACHE[cache_key] = {
            'model': hl_model,
            'vocab': vocab
        }
        logger.info(f"Model pre-loaded successfully (vocab_type: {args.vocab_type})")

        # Initialize the model wrapper
        model = HumanLikeScorerModel(
            model_path=args.model_path,
            device=args.device,
            vocab_type=args.vocab_type,
            threshold=args.threshold
        )
        logger.info(f"Initialized HumanLikeScorerModel with threshold: {args.threshold}")

    # Create evaluation
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[
            AccuracyScorer(),
            F1Scorer(),
        ],
        name="human_like_scorer_evaluation"
    )

    logger.info("Starting evaluation...")

    # Run evaluation with optional display name
    if args.run_name:
        logger.info(f"Evaluation run name: {args.run_name}")
        results = await evaluation.evaluate(model, __weave={"display_name": args.run_name})
    else:
        results = await evaluation.evaluate(model)

    # Print summary
    print("\n" + "="*80)
    print("HUMAN-LIKE SCORER EVALUATION RESULTS")
    print("="*80)
    if args.run_name:
        print(f"\nRun name: {args.run_name}")
    print(f"Total test cases: {len(all_examples)}")
    print(f"  Positive (human): {positive_count}")
    print(f"  Negative (synthetic): {negative_count}")
    print(f"Device used: {args.device}")
    if not (args.use_random_baseline or args.use_always_humanlike or args.use_always_nonhumanlike):
        print(f"Model: {args.model_path}")
        print(f"Vocab type: {args.vocab_type}")
        print(f"Threshold: {args.threshold}")

    # Print results
    print("\nScorer Results:")
    if hasattr(results, 'scores'):
        for scorer_name, score_value in results.scores.items():
            print(f"  {scorer_name}: {score_value}")
    elif isinstance(results, dict):
        for key, value in results.items():
            if 'scorer' in key.lower() or 'score' in key.lower():
                print(f"  {key}: {value}")

    # Try to get URL
    url = None
    if hasattr(results, 'url'):
        url = results.url
    elif hasattr(results, 'run_url'):
        url = results.run_url

    if url:
        print(f"\nView detailed results in Weave: {url}")
    else:
        print(f"\nView detailed results in Weave: https://wandb.ai/weave/{args.project}")

    print("="*80 + "\n")

    logger.info("Evaluation complete!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Human-Like Scorer using Weave")
    parser.add_argument("--project", type=str, default="human-like-scorer-eval",
                       help="Weave project name")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Display name for this evaluation run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run the scorer on")

    # Dataset options
    parser.add_argument("--eval-dataset-name", type=str,
                       default="timonziegenbein/human-like-edit-sequences-eval",
                       help="Name of the evaluation dataset on HuggingFace Hub")
    parser.add_argument("--eval-dataset-split", type=str, default="test",
                       help="Which split to use (default: test)")
    parser.add_argument("--max-eval-examples", type=int, default=None,
                       help="Maximum number of examples to evaluate")

    # Model options
    parser.add_argument("--model-path", type=str,
                       default="scorers/human_like/human_like_language_model_v3.pth",
                       help="Path to trained model (.pth file)")
    parser.add_argument("--vocab-type", type=str, default="v2", choices=["v1", "v2"],
                       help="Vocabulary type (v1: 5 tokens, v2: 6 tokens with keep-in-edit)")
    parser.add_argument("--threshold", type=float, default=1.1465,
                       help="Perplexity threshold for human-likeness")

    # Baseline options
    parser.add_argument("--use-random-baseline", action="store_true",
                       help="Use random baseline")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for random baseline")
    parser.add_argument("--use-always-humanlike", action="store_true",
                       help="Use always-human-like baseline (always predicts 1.0)")
    parser.add_argument("--use-always-nonhumanlike", action="store_true",
                       help="Use always-non-human-like baseline (always predicts 0.0)")

    args = parser.parse_args()

    # Run async evaluation
    import asyncio
    results = asyncio.run(run_evaluation_async(args))

    return results


if __name__ == "__main__":
    main()

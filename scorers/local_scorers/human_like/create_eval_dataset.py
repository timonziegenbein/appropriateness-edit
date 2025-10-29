"""
Create Human-Like Scorer Evaluation Dataset

This script:
1. Loads the positive examples from HuggingFace
2. Generates synthetic negative examples
3. Combines them into an evaluation dataset
4. Pushes to HuggingFace Hub
"""

import argparse
import pandas as pd
import numpy as np
import random
from typing import List
from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def shuffle_sequence(sequence: str, shuffle_ratio: float = 0.5) -> str:
    """Randomly shuffle a portion of the edit sequence."""
    tokens = sequence.split(',')
    n_shuffle = max(2, int(len(tokens) * shuffle_ratio))

    indices = list(range(len(tokens)))
    shuffle_indices = random.sample(indices, min(n_shuffle, len(indices)))

    shuffle_values = [tokens[i] for i in shuffle_indices]
    random.shuffle(shuffle_values)

    for i, idx in enumerate(shuffle_indices):
        tokens[idx] = shuffle_values[i]

    return ','.join(tokens)


def random_substitution(sequence: str, vocab: List[str], sub_ratio: float = 0.3) -> str:
    """Randomly substitute tokens with other vocabulary items."""
    tokens = sequence.split(',')
    n_sub = max(1, int(len(tokens) * sub_ratio))

    sub_indices = random.sample(range(len(tokens)), min(n_sub, len(tokens)))

    for idx in sub_indices:
        tokens[idx] = random.choice(vocab)

    return ','.join(tokens)


def create_inefficient_edits(sequence: str) -> str:
    """Create inefficient edit patterns (delete then add)."""
    tokens = sequence.split(',')

    if len(tokens) < 3:
        return sequence

    insert_pos = random.randint(1, len(tokens) - 2)
    tokens.insert(insert_pos, 'del')
    tokens.insert(insert_pos + 1, 'add')

    return ','.join(tokens)


def create_repetitive_pattern(sequence: str) -> str:
    """Create sequences with unnatural repetitions."""
    tokens = sequence.split(',')

    operations = ['del', 'add', 'replace', 'keep-in-edit']
    repeat_op = random.choice(operations)
    n_repeats = random.randint(5, 15)

    insert_pos = random.randint(1, len(tokens))
    for _ in range(n_repeats):
        tokens.insert(insert_pos, repeat_op)

    return ','.join(tokens)


def reverse_sequence(sequence: str) -> str:
    """Reverse the entire edit sequence."""
    tokens = sequence.split(',')
    tokens.reverse()
    return ','.join(tokens)


def generate_negatives_for_split(dataset, n_negatives_per_positive: int = 5, seed: int = 42):
    """Generate negative examples for a dataset split."""
    random.seed(seed)
    np.random.seed(seed)

    vocab = ['keep', 'del', 'add', 'replace', 'keep-in-edit']

    all_examples = []

    # Keep all positive examples
    for example in dataset:
        all_examples.append({
            'sequence': example['sequence'],
            'label': 1,
            'original_sentence': example['original_sentence'],
            'inappropriate_part': example['inappropriate_part'],
            'rewritten_part': example['rewritten_part'],
            'doc_id': example.get('doc_id', 'unknown'),
            'sent_idx': example.get('sent_idx', -1),
            'edit_idx': example.get('edit_idx', -1),
            'type': 'human'
        })

    # Generate negative examples
    negative_generators = [
        ('shuffle_mild', lambda s: shuffle_sequence(s, shuffle_ratio=0.3)),
        ('shuffle_heavy', lambda s: shuffle_sequence(s, shuffle_ratio=0.7)),
        ('substitution', lambda s: random_substitution(s, vocab, sub_ratio=0.3)),
        ('inefficient', create_inefficient_edits),
        ('repetitive', create_repetitive_pattern),
        ('reverse', reverse_sequence),
    ]

    logger.info(f"Generating {n_negatives_per_positive} negative examples per positive...")

    for idx, example in enumerate(dataset):
        original_seq = example['sequence']

        for i in range(n_negatives_per_positive):
            method_name, method_func = negative_generators[i % len(negative_generators)]

            try:
                negative_seq = method_func(original_seq)

                all_examples.append({
                    'sequence': negative_seq,
                    'label': 0,
                    'original_sentence': example['original_sentence'],
                    'inappropriate_part': example['inappropriate_part'],
                    'rewritten_part': example['rewritten_part'],
                    'doc_id': example.get('doc_id', 'unknown'),
                    'sent_idx': example.get('sent_idx', -1),
                    'edit_idx': example.get('edit_idx', -1),
                    'type': method_name
                })
            except Exception as e:
                logger.warning(f"Failed to generate negative for example {idx} using {method_name}: {e}")

        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(dataset)} positive examples...")

    # Shuffle the dataset
    random.shuffle(all_examples)

    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description='Create evaluation dataset with synthetic negative examples for human-like scorer.'
    )
    parser.add_argument('--dataset-name', type=str,
                       default='timonziegenbein/human-like-edit-sequences',
                       help='HuggingFace dataset with positive examples')
    parser.add_argument('--output-dataset-name', type=str, required=True,
                       help='Name for output evaluation dataset on HuggingFace Hub')
    parser.add_argument('--n-negatives', type=int, default=5,
                       help='Number of negative examples per positive')
    parser.add_argument('--splits', type=str, nargs='+', default=['test', 'dev'],
                       help='Which splits to process (default: test dev)')

    args = parser.parse_args()

    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    eval_splits = {}

    for split_name in args.splits:
        if split_name not in dataset:
            logger.warning(f"Split '{split_name}' not found in dataset. Available: {list(dataset.keys())}")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing split: {split_name}")
        logger.info(f"{'='*80}")

        split_data = dataset[split_name]
        logger.info(f"Loaded {len(split_data)} positive examples")

        # Generate negatives
        all_examples = generate_negatives_for_split(
            split_data,
            args.n_negatives
        )

        # Create dataset
        eval_dataset = Dataset.from_dict({
            'sequence': [ex['sequence'] for ex in all_examples],
            'label': [ex['label'] for ex in all_examples],
            'original_sentence': [ex['original_sentence'] for ex in all_examples],
            'inappropriate_part': [ex['inappropriate_part'] for ex in all_examples],
            'rewritten_part': [ex['rewritten_part'] for ex in all_examples],
            'doc_id': [ex['doc_id'] for ex in all_examples],
            'sent_idx': [ex['sent_idx'] for ex in all_examples],
            'edit_idx': [ex['edit_idx'] for ex in all_examples],
            'type': [ex['type'] for ex in all_examples]
        })

        eval_splits[split_name] = eval_dataset

        # Summary
        positive_count = sum(1 for ex in all_examples if ex['label'] == 1)
        negative_count = sum(1 for ex in all_examples if ex['label'] == 0)

        logger.info(f"\n--- {split_name} Split Summary ---")
        logger.info(f"Positive examples: {positive_count}")
        logger.info(f"Negative examples: {negative_count}")
        logger.info(f"Total examples: {len(all_examples)}")

        logger.info(f"\nNegative breakdown by type:")
        type_counts = pd.Series([ex['type'] for ex in all_examples if ex['label'] == 0]).value_counts()
        for neg_type, count in type_counts.items():
            logger.info(f"  {neg_type}: {count}")

    if not eval_splits:
        logger.error("No splits were processed!")
        return

    # Create DatasetDict
    eval_dataset_dict = DatasetDict(eval_splits)

    # Print overall statistics
    logger.info(f"\n{'='*80}")
    logger.info("Combined Dataset Statistics")
    logger.info(f"{'='*80}")
    for split_name, split_data in eval_dataset_dict.items():
        pos_count = sum(1 for ex in split_data if ex['label'] == 1)
        neg_count = sum(1 for ex in split_data if ex['label'] == 0)
        logger.info(f"\nSplit '{split_name}':")
        logger.info(f"  Total examples: {len(split_data)}")
        logger.info(f"  Positive: {pos_count} ({100*pos_count/len(split_data):.1f}%)")
        logger.info(f"  Negative: {neg_count} ({100*neg_count/len(split_data):.1f}%)")

    # Push to Hub
    logger.info(f"\nPushing dataset to HuggingFace Hub: {args.output_dataset_name}")
    eval_dataset_dict.push_to_hub(args.output_dataset_name)
    logger.info(f"✓ Dataset successfully uploaded to {args.output_dataset_name}")

    logger.info("\n=== Complete ===")


if __name__ == "__main__":
    main()

"""
Generate synthetic negative examples for evaluating the human-like scorer.

These negative examples represent non-human-like or poor edit patterns that
should score poorly (high perplexity) compared to real human edits.
"""

import argparse
import pandas as pd
import numpy as np
import random
from typing import List


def shuffle_sequence(sequence: str, shuffle_ratio: float = 0.5) -> str:
    """
    Randomly shuffle a portion of the edit sequence.

    Args:
        sequence: Comma-separated edit operation sequence
        shuffle_ratio: Proportion of tokens to shuffle (0.0 to 1.0)

    Returns:
        Shuffled sequence
    """
    tokens = sequence.split(',')
    n_shuffle = max(2, int(len(tokens) * shuffle_ratio))

    # Select random positions to shuffle
    indices = list(range(len(tokens)))
    shuffle_indices = random.sample(indices, n_shuffle)

    # Shuffle the selected tokens
    shuffle_values = [tokens[i] for i in shuffle_indices]
    random.shuffle(shuffle_values)

    # Put them back
    for i, idx in enumerate(shuffle_indices):
        tokens[idx] = shuffle_values[i]

    return ','.join(tokens)


def random_substitution(sequence: str, vocab: List[str], sub_ratio: float = 0.3) -> str:
    """
    Randomly substitute tokens with other vocabulary items.

    Args:
        sequence: Comma-separated edit operation sequence
        vocab: List of valid tokens (excluding <pad>)
        sub_ratio: Proportion of tokens to substitute

    Returns:
        Modified sequence
    """
    tokens = sequence.split(',')
    n_sub = max(1, int(len(tokens) * sub_ratio))

    # Select random positions for substitution
    sub_indices = random.sample(range(len(tokens)), n_sub)

    # Substitute with random vocab items
    for idx in sub_indices:
        tokens[idx] = random.choice(vocab)

    return ','.join(tokens)


def create_inefficient_edits(sequence: str) -> str:
    """
    Create inefficient edit patterns (delete then add, or redundant operations).

    Args:
        sequence: Comma-separated edit operation sequence

    Returns:
        Modified sequence with inefficient patterns
    """
    tokens = sequence.split(',')

    # Find positions where we can insert inefficient patterns
    if len(tokens) < 3:
        return sequence

    # Insert "del" followed by "add" at random position (inefficient)
    insert_pos = random.randint(1, len(tokens) - 2)
    tokens.insert(insert_pos, 'del')
    tokens.insert(insert_pos + 1, 'add')

    return ','.join(tokens)


def create_repetitive_pattern(sequence: str) -> str:
    """
    Create sequences with unnatural repetitions.

    Args:
        sequence: Comma-separated edit operation sequence

    Returns:
        Modified sequence with repetitive patterns
    """
    tokens = sequence.split(',')

    # Pick a random operation and repeat it excessively
    operations = ['del', 'add', 'replace', 'keep-in-edit']
    repeat_op = random.choice(operations)
    n_repeats = random.randint(5, 15)

    # Insert repetition at random position
    insert_pos = random.randint(1, len(tokens))
    for _ in range(n_repeats):
        tokens.insert(insert_pos, repeat_op)

    return ','.join(tokens)


def reverse_sequence(sequence: str) -> str:
    """
    Reverse the entire edit sequence (very unnatural).

    Args:
        sequence: Comma-separated edit operation sequence

    Returns:
        Reversed sequence
    """
    tokens = sequence.split(',')
    tokens.reverse()
    return ','.join(tokens)


def generate_negatives(positive_csv: str, output_csv: str,
                      n_negatives_per_positive: int = 5,
                      seed: int = 42):
    """
    Generate negative examples from positive human edit sequences.

    Args:
        positive_csv: Path to CSV with positive examples
        output_csv: Path to save combined positive and negative examples
        n_negatives_per_positive: How many negative examples per positive
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load positive examples
    df_positive = pd.read_csv(positive_csv)
    print(f"Loaded {len(df_positive)} positive examples")

    # Vocabulary (excluding <pad>)
    vocab = ['keep', 'del', 'add', 'replace', 'keep-in-edit']

    # Store all examples (positive + negative)
    all_examples = []

    # Keep all positive examples
    for _, row in df_positive.iterrows():
        all_examples.append({
            'sequence': row['sequence'],
            'label': 1,  # Positive
            'doc_id': row.get('doc_id', 'unknown'),
            'sent_idx': row.get('sent_idx', -1),
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

    print(f"Generating {n_negatives_per_positive} negative examples per positive...")

    for idx, row in df_positive.iterrows():
        original_seq = row['sequence']

        # Generate multiple negatives using different methods
        for i in range(n_negatives_per_positive):
            # Cycle through different generation methods
            method_name, method_func = negative_generators[i % len(negative_generators)]

            try:
                negative_seq = method_func(original_seq)

                all_examples.append({
                    'sequence': negative_seq,
                    'label': 0,  # Negative
                    'doc_id': row.get('doc_id', 'unknown'),
                    'sent_idx': row.get('sent_idx', -1),
                    'type': method_name
                })
            except Exception as e:
                print(f"Warning: Failed to generate negative for row {idx} using {method_name}: {e}")

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df_positive)} positive examples...")

    # Create DataFrame and save
    df_all = pd.DataFrame(all_examples)

    # Shuffle the dataset
    df_all = df_all.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_all.to_csv(output_csv, index=False)

    print(f"\n=== Summary ===")
    print(f"Positive examples: {len(df_all[df_all['label'] == 1])}")
    print(f"Negative examples: {len(df_all[df_all['label'] == 0])}")
    print(f"Total examples: {len(df_all)}")
    print(f"\nNegative breakdown by type:")
    for neg_type in df_all[df_all['label'] == 0]['type'].value_counts().items():
        print(f"  {neg_type[0]}: {neg_type[1]}")
    print(f"\nSaved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic negative examples for human-like scorer evaluation.'
    )
    parser.add_argument('--positive-csv', type=str, required=True,
                       help='Path to CSV with positive (human) edit sequences')
    parser.add_argument('--output-csv', type=str, required=True,
                       help='Path to save combined positive and negative examples')
    parser.add_argument('--n-negatives', type=int, default=5,
                       help='Number of negative examples to generate per positive')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    generate_negatives(
        args.positive_csv,
        args.output_csv,
        args.n_negatives,
        args.seed
    )


if __name__ == "__main__":
    main()

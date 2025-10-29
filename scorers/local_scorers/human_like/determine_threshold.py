#!/usr/bin/env python3
"""
Determine optimal thresholds for the human-like scorer based on perplexity percentiles.

This script:
1. Loads the dev set from HuggingFace
2. Computes perplexity for all sequences
3. Calculates thresholds at different percentiles
4. Saves threshold candidates for evaluation

Usage:
    python scorers/human_like/determine_threshold.py \
        --model-path scorers/human_like/models/human_like_v3.pth \
        --dataset-name timonziegenbein/human-like-edit-sequences \
        --split dev \
        --percentiles 50 60 70 75 80 85 90 95 \
        --output-file scorers/human_like/threshold_candidates.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scorers.local_scorers.human_like.model_defs import LanguageModel


def load_model(model_path, device):
    """Load the trained human-like language model."""
    # Load the saved checkpoint
    checkpoint = torch.load(model_path, map_location=device)

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

    return model


def compute_perplexities_batch(model, sequences, device, batch_size=32):
    """Compute perplexities for a batch of sequences efficiently."""
    # Define vocab mapping (must match training!)
    vocab = {
        '<pad>': 0,
        'keep': 1,
        'del': 2,
        'add': 3,
        'replace': 4,
        'keep-in-edit': 5
    }

    all_perplexities = []

    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]

        # Convert all sequences to token IDs
        batch_token_ids = []
        for sequence in batch_sequences:
            tokens = sequence.split(',')
            token_ids = [vocab.get(token.strip(), vocab['<pad>']) for token in tokens]

            # Pad to at least length 2
            if len(token_ids) < 2:
                token_ids.append(vocab['<pad>'])

            batch_token_ids.append(token_ids)

        # Find max length in batch
        max_len = max(len(ids) for ids in batch_token_ids)

        # Pad all sequences to max length
        padded_batch = []
        for token_ids in batch_token_ids:
            padded = token_ids + [vocab['<pad>']] * (max_len - len(token_ids))
            padded_batch.append(padded)

        # Convert to tensor
        input_ids = torch.tensor(padded_batch, dtype=torch.long).to(device)

        # Compute losses for entire batch
        with torch.no_grad():
            logits = model(input_ids)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Compute per-example loss, ignoring PAD tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=vocab['<pad>'])
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Reshape to (batch_size, seq_len-1)
            losses = losses.view(input_ids.size(0), -1)

            # Create mask for non-PAD tokens (shifted labels)
            pad_mask = (shift_labels != vocab['<pad>']).float()

            # Compute mean loss per sequence, only over non-PAD positions
            # Sum of losses divided by number of non-PAD tokens
            per_seq_loss = (losses * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)

            # Perplexity is exp(loss)
            perplexities = torch.exp(per_seq_loss).cpu().tolist()
            all_perplexities.extend(perplexities)

    return all_perplexities


def main():
    parser = argparse.ArgumentParser(description='Determine optimal thresholds for human-like scorer')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='HuggingFace dataset name')
    parser.add_argument('--split', type=str, default='dev',
                        help='Dataset split to use (default: dev)')
    parser.add_argument('--percentiles', type=int, nargs='+',
                        default=[50, 60, 70, 75, 80, 85, 90, 95],
                        help='Percentiles to compute thresholds for')
    parser.add_argument('--output-file', type=str,
                        default='scorers/human_like/threshold_candidates.json',
                        help='Output file for threshold candidates')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing sequences (default: 32)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device)
    print("Model loaded successfully")

    # Load dataset
    print(f"Loading dataset {args.dataset_name} (split: {args.split})...")
    dataset = load_dataset(args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    # Filter only positive (human-like) examples
    positive_examples = [ex for ex in dataset if ex['label'] == 1]
    print(f"Found {len(positive_examples)} positive (human-like) examples")

    # Extract sequences
    sequences = [ex['sequence'] for ex in positive_examples]

    # Compute perplexities for all positive examples using batching
    print(f"Computing perplexities (batch_size={args.batch_size})...")
    perplexities = compute_perplexities_batch(model, sequences, device, batch_size=args.batch_size)
    perplexities = np.array(perplexities)

    # Compute statistics
    print("\n" + "="*60)
    print("Perplexity Statistics on Positive Examples:")
    print("="*60)
    print(f"Mean:   {np.mean(perplexities):.4f}")
    print(f"Median: {np.median(perplexities):.4f}")
    print(f"Std:    {np.std(perplexities):.4f}")
    print(f"Min:    {np.min(perplexities):.4f}")
    print(f"Max:    {np.max(perplexities):.4f}")
    print()

    # Compute percentile-based thresholds
    thresholds = {}
    print("Percentile-based Thresholds:")
    print("-"*60)
    for percentile in sorted(args.percentiles):
        threshold = np.percentile(perplexities, percentile)
        thresholds[f"p{percentile}"] = float(threshold)
        print(f"P{percentile:2d}: {threshold:.4f}")

    print("="*60)

    # Save threshold candidates
    output_data = {
        'model_path': args.model_path,
        'dataset_name': args.dataset_name,
        'split': args.split,
        'num_positive_examples': len(positive_examples),
        'statistics': {
            'mean': float(np.mean(perplexities)),
            'median': float(np.median(perplexities)),
            'std': float(np.std(perplexities)),
            'min': float(np.min(perplexities)),
            'max': float(np.max(perplexities))
        },
        'thresholds': thresholds,
        'percentiles': args.percentiles
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nThreshold candidates saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Review the threshold candidates above")
    print(f"2. Run multi-threshold evaluation:")
    print(f"   python scorers/human_like/evaluate_multi_threshold.py \\")
    print(f"       --threshold-file {args.output_file} \\")
    print(f"       --model-path {args.model_path} \\")
    print(f"       --eval-dataset-name <eval_dataset_name> \\")
    print(f"       --project human-like-threshold-selection")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Determine optimal thresholds for the global human-like scorer based on perplexity percentiles.

This script:
1. Loads the dev set from IteraTeR
2. Computes document-level perplexity for all sequences
3. Calculates thresholds at different percentiles
4. Saves threshold candidates for evaluation

Usage:
    python scorers/global_scorers/human_like/determine_threshold.py \
        --model-path scorers/global_scorers/human_like/models/global_human_like_v1.pth \
        --input-json datasets/scorers/IteraTeR/full_doc_level/dev.json \
        --percentiles 50 60 70 75 80 85 90 95 99 \
        --output-file scorers/global_scorers/human_like/threshold_candidates.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import difflib
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scorers.local_scorers.human_like.model_defs import LanguageModel

# V2 vocab with 'keep-in-edit' token
hl_vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}


def load_model(model_path, device):
    """Load the trained global human-like language model."""
    # Load the saved checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        max_len = checkpoint.get('max_len', 1500)
    else:
        state_dict = checkpoint
        max_len = 1500

    # Model hyperparameters (must match training)
    vocab_size = 6  # V2 vocab: <pad>, keep, del, add, replace, keep-in-edit
    embedding_dim = 200
    nhead = 2
    nhid = 200
    nlayers = 2

    # Initialize model with dropout=0 for inference
    model = LanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        dropout=0.0
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, max_len


def generate_document_edit_sequence(before_revision, edit_actions, tokenizer):
    """
    Generate a document-level edit sequence from the actual edit actions.

    Args:
        before_revision: Original document text
        edit_actions: List of edit actions from the JSON file
        tokenizer: Tokenizer for converting text to tokens

    Returns:
        List of edit operation tokens representing the document-level pattern
    """
    # Tokenize the original document
    encoding = tokenizer(before_revision, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']

    if len(tokens) == 0:
        return []

    # Initialize all tokens as 'keep'
    tags = ['keep'] * len(tokens)

    # Process each edit action
    for edit in edit_actions:
        edit_type = edit.get('type')
        start_char = edit.get('start_char_pos')
        end_char = edit.get('end_char_pos')

        if start_char is None:
            continue

        # Skip non-edit actions
        if edit_type not in ['R', 'A', 'D']:
            continue

        # Find token indices that overlap with this edit
        token_start_index = -1
        token_end_index = -1
        for i, offset in enumerate(offsets):
            token_start, token_end = offset
            if start_char < token_end and (end_char is None or end_char > token_start):
                if token_start_index == -1:
                    token_start_index = i
                token_end_index = i

        if token_start_index == -1:
            continue

        # Apply the edit operation to the tag sequence
        if edit_type == 'R':  # Replace
            after_text = edit.get('after', '')
            before_tokens = tokens[token_start_index:token_end_index+1]
            after_tokens = tokenizer.tokenize(after_text) if after_text else []

            # Use difflib to get fine-grained operations for this edit
            matcher = difflib.SequenceMatcher(None, before_tokens, after_tokens)
            current_idx = token_start_index
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    # Tokens that match within an edit region are 'keep-in-edit'
                    for idx in range(current_idx, current_idx + (i2 - i1)):
                        if idx < len(tags):
                            tags[idx] = 'keep-in-edit'
                    current_idx += (i2 - i1)
                elif tag == 'delete':
                    # Mark as deletions
                    for idx in range(current_idx, current_idx + (i2 - i1)):
                        if idx < len(tags):
                            tags[idx] = 'del'
                    current_idx += (i2 - i1)
                elif tag == 'replace':
                    # Mark as replacements
                    for idx in range(current_idx, current_idx + (i2 - i1)):
                        if idx < len(tags):
                            tags[idx] = 'replace'
                    current_idx += (i2 - i1)
                elif tag == 'insert':
                    # Insert 'add' tokens
                    for _ in range(j2 - j1):
                        tags.insert(current_idx, 'add')
                        current_idx += 1

        elif edit_type == 'D':  # Delete
            # Mark tokens as deleted
            for idx in range(token_start_index, token_end_index + 1):
                if idx < len(tags):
                    tags[idx] = 'del'

        elif edit_type == 'A':  # Add
            after_text = edit.get('after', '')
            after_tokens = tokenizer.tokenize(after_text) if after_text else []
            # Insert 'add' tokens at the position
            for _ in range(len(after_tokens)):
                tags.insert(token_start_index, 'add')

    return tags


def calculate_perplexity(model, sequence, max_len, device):
    """Calculate perplexity for an edit operation sequence."""
    sequence_as_int = [hl_vocab.get(token, 0) for token in sequence]

    if len(sequence_as_int) <= 1:
        return float('inf')

    input_seq = sequence_as_int[:-1]
    target_seq = sequence_as_int[1:]

    padded_input = np.array(input_seq[:max_len] + [hl_vocab['<pad>']]*(max_len - len(input_seq)) if len(input_seq) < max_len else input_seq[:max_len])
    padded_target = np.array(target_seq[:max_len] + [hl_vocab['<pad>']]*(max_len - len(target_seq)) if len(target_seq) < max_len else target_seq[:max_len])

    inputs = torch.from_numpy(padded_input).long().unsqueeze(0).to(device)
    targets = torch.from_numpy(padded_target).long().unsqueeze(0).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=hl_vocab['<pad>'])

    with torch.no_grad():
        output = model(inputs)
        loss = criterion(output.view(-1, len(hl_vocab)), targets.view(-1))
        perplexity = torch.exp(loss)
        return perplexity.item()


def main():
    parser = argparse.ArgumentParser(description='Determine optimal thresholds for global human-like scorer')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input-json', type=str, required=True,
                        help='Path to IteraTeR JSON file (dev split)')
    parser.add_argument('--percentiles', type=int, nargs='+',
                        default=[50, 60, 70, 75, 80, 85, 90, 95, 99],
                        help='Percentiles to compute thresholds for')
    parser.add_argument('--output-file', type=str,
                        default='scorers/global_scorers/human_like/threshold_candidates.json',
                        help='Output file for threshold candidates')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, max_len = load_model(args.model_path, device)
    print(f"Model loaded successfully (max_len={max_len})")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Load documents and compute perplexities
    print(f"Loading documents from {args.input_json}...")
    perplexities = []
    num_documents = 0

    with open(args.input_json, 'r') as f:
        for line in tqdm(f, desc="Processing documents"):
            try:
                data = json.loads(line)
                before_revision = data.get('before_revision', '')
                edit_actions = data.get('edit_actions', [])

                if not before_revision or not edit_actions:
                    continue

                # Generate document-level edit sequence
                edit_sequence = generate_document_edit_sequence(before_revision, edit_actions, tokenizer)

                if not edit_sequence or len(edit_sequence) <= 1:
                    continue

                # Calculate perplexity
                perplexity = calculate_perplexity(model, edit_sequence, max_len, device)

                if np.isfinite(perplexity):
                    perplexities.append(perplexity)
                    num_documents += 1

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing document: {e}")
                continue

    perplexities = np.array(perplexities)
    print(f"Processed {num_documents} documents with valid perplexities")

    # Compute statistics
    print("\n" + "="*60)
    print("Perplexity Statistics on Human-Edited Documents:")
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
        'input_json': args.input_json,
        'num_documents': num_documents,
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
    print(f"2. Run evaluation:")
    print(f"   python scorers/global_scorers/human_like/evaluate_global_human_like_scorer.py \\")
    print(f"       --threshold-file {args.output_file} \\")
    print(f"       --model-path {args.model_path}")


if __name__ == '__main__':
    main()

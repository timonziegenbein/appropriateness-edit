#!/usr/bin/env python3
"""
Evaluate the global human-like scorer on the IteraTeR test set.

Since all IteraTeR documents are human-edited (positive examples), this evaluation:
1. Validates that the model generalizes from dev to test
2. Shows the distribution of perplexities on the test set
3. Reports what percentage of test documents pass various thresholds

Usage:
    python scorers/global_scorers/human_like/evaluate_global_human_like_scorer.py \
        --model-path scorers/global_scorers/human_like/models/global_human_like_v1.pth \
        --test-json datasets/scorers/IteraTeR/full_doc_level/test.json \
        --threshold-file scorers/global_scorers/human_like/threshold_candidates.json \
        --project global-human-like-eval
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
import weave

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scorers.local_scorers.human_like.model_defs import LanguageModel

# V2 vocab with 'keep-in-edit' token
hl_vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}


def load_model(model_path, device):
    """Load the trained global human-like language model."""
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        max_len = checkpoint.get('max_len', 1500)
    else:
        state_dict = checkpoint
        max_len = 1500

    vocab_size = 6
    embedding_dim = 200
    nhead = 2
    nhid = 200
    nlayers = 2

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
    """Generate a document-level edit sequence from the actual edit actions."""
    encoding = tokenizer(before_revision, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']

    if len(tokens) == 0:
        return []

    tags = ['keep'] * len(tokens)

    for edit in edit_actions:
        edit_type = edit.get('type')
        start_char = edit.get('start_char_pos')
        end_char = edit.get('end_char_pos')

        if start_char is None or edit_type not in ['R', 'A', 'D']:
            continue

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

        if edit_type == 'R':
            after_text = edit.get('after', '')
            before_tokens = tokens[token_start_index:token_end_index+1]
            after_tokens = tokenizer.tokenize(after_text) if after_text else []

            matcher = difflib.SequenceMatcher(None, before_tokens, after_tokens)
            current_idx = token_start_index
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    for idx in range(current_idx, current_idx + (i2 - i1)):
                        if idx < len(tags):
                            tags[idx] = 'keep-in-edit'
                    current_idx += (i2 - i1)
                elif tag == 'delete':
                    for idx in range(current_idx, current_idx + (i2 - i1)):
                        if idx < len(tags):
                            tags[idx] = 'del'
                    current_idx += (i2 - i1)
                elif tag == 'replace':
                    for idx in range(current_idx, current_idx + (i2 - i1)):
                        if idx < len(tags):
                            tags[idx] = 'replace'
                    current_idx += (i2 - i1)
                elif tag == 'insert':
                    for _ in range(j2 - j1):
                        tags.insert(current_idx, 'add')
                        current_idx += 1

        elif edit_type == 'D':
            for idx in range(token_start_index, token_end_index + 1):
                if idx < len(tags):
                    tags[idx] = 'del'

        elif edit_type == 'A':
            after_text = edit.get('after', '')
            after_tokens = tokenizer.tokenize(after_text) if after_text else []
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
    parser = argparse.ArgumentParser(description='Evaluate global human-like scorer')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test-json', type=str, required=True,
                        help='Path to IteraTeR test JSON file')
    parser.add_argument('--threshold-file', type=str, required=True,
                        help='Path to threshold candidates JSON file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--project', type=str, default='global-human-like-eval',
                        help='Weave project name')

    args = parser.parse_args()

    # Initialize Weave
    weave.init(args.project)
    print(f"Initialized Weave project: {args.project}")

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

    # Load threshold candidates
    print(f"Loading threshold candidates from {args.threshold_file}...")
    with open(args.threshold_file, 'r') as f:
        threshold_data = json.load(f)

    thresholds = threshold_data.get('thresholds', {})
    dev_stats = threshold_data.get('statistics', {})
    print(f"Loaded {len(thresholds)} threshold candidates")

    # Load test documents and compute perplexities
    print(f"Loading test documents from {args.test_json}...")
    test_perplexities = []
    test_metadata = []
    num_documents = 0

    with open(args.test_json, 'r') as f:
        for line in tqdm(f, desc="Processing test documents"):
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
                    test_perplexities.append(perplexity)
                    test_metadata.append({
                        'num_edits': len([e for e in edit_actions if e.get('type') in ['R', 'A', 'D']]),
                        'sequence_length': len(edit_sequence),
                        'document_length': len(before_revision)
                    })
                    num_documents += 1

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing document: {e}")
                continue

    test_perplexities = np.array(test_perplexities)
    print(f"Processed {num_documents} test documents with valid perplexities")

    # Compute test set statistics
    test_stats = {
        'mean': float(np.mean(test_perplexities)),
        'median': float(np.median(test_perplexities)),
        'std': float(np.std(test_perplexities)),
        'min': float(np.min(test_perplexities)),
        'max': float(np.max(test_perplexities))
    }

    # Print comparison
    print("\n" + "="*80)
    print("GLOBAL HUMAN-LIKE SCORER EVALUATION RESULTS")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Test set: {args.test_json}")
    print(f"Test documents: {num_documents}")

    print("\n" + "-"*80)
    print("Perplexity Distribution Comparison:")
    print("-"*80)
    print(f"{'Statistic':<15} {'Dev Set':<15} {'Test Set':<15} {'Difference':<15}")
    print("-"*80)
    print(f"{'Mean':<15} {dev_stats.get('mean', 0):<15.4f} {test_stats['mean']:<15.4f} {test_stats['mean'] - dev_stats.get('mean', 0):<15.4f}")
    print(f"{'Median':<15} {dev_stats.get('median', 0):<15.4f} {test_stats['median']:<15.4f} {test_stats['median'] - dev_stats.get('median', 0):<15.4f}")
    print(f"{'Std':<15} {dev_stats.get('std', 0):<15.4f} {test_stats['std']:<15.4f} {test_stats['std'] - dev_stats.get('std', 0):<15.4f}")
    print(f"{'Min':<15} {dev_stats.get('min', 0):<15.4f} {test_stats['min']:<15.4f} {test_stats['min'] - dev_stats.get('min', 0):<15.4f}")
    print(f"{'Max':<15} {dev_stats.get('max', 0):<15.4f} {test_stats['max']:<15.4f} {test_stats['max'] - dev_stats.get('max', 0):<15.4f}")

    # Evaluate each threshold
    print("\n" + "-"*80)
    print("Threshold Performance on Test Set:")
    print("-"*80)
    print(f"{'Threshold':<15} {'Percentile':<15} {'Pass Rate':<15} {'Description':<40}")
    print("-"*80)

    threshold_results = []
    for key, threshold_value in sorted(thresholds.items(), key=lambda x: x[1]):
        pass_rate = np.mean(test_perplexities <= threshold_value)
        percentile = key.replace('p', 'P')

        # Determine description
        if pass_rate >= 0.95:
            description = "Very lenient (most documents pass)"
        elif pass_rate >= 0.85:
            description = "Lenient (many documents pass)"
        elif pass_rate >= 0.70:
            description = "Moderate (balanced)"
        elif pass_rate >= 0.50:
            description = "Strict (fewer documents pass)"
        else:
            description = "Very strict (most documents fail)"

        print(f"{threshold_value:<15.4f} {percentile:<15} {pass_rate:<15.2%} {description:<40}")

        threshold_results.append({
            'threshold': threshold_value,
            'percentile': percentile,
            'pass_rate': pass_rate,
            'description': description
        })

    # Recommended threshold
    print("\n" + "-"*80)
    print("Recommended Thresholds:")
    print("-"*80)

    # Find P95 threshold (95% of dev examples pass)
    p95_threshold = thresholds.get('p95')
    if p95_threshold:
        p95_test_pass_rate = np.mean(test_perplexities <= p95_threshold)
        print(f"P95 Threshold: {p95_threshold:.4f}")
        print(f"  - Designed so 95% of dev documents pass")
        print(f"  - Test set pass rate: {p95_test_pass_rate:.2%}")
        print(f"  - Use for: Standard validation (recommended)")

    # Find P99 threshold (99% of dev examples pass)
    p99_threshold = thresholds.get('p99')
    if p99_threshold:
        p99_test_pass_rate = np.mean(test_perplexities <= p99_threshold)
        print(f"\nP99 Threshold: {p99_threshold:.4f}")
        print(f"  - Designed so 99% of dev documents pass")
        print(f"  - Test set pass rate: {p99_test_pass_rate:.2%}")
        print(f"  - Use for: More lenient validation")

    print("="*80)

    # Log to Weave
    weave.publish({
        'evaluation_type': 'global_human_like_test_set',
        'model_path': args.model_path,
        'test_json': args.test_json,
        'num_test_documents': num_documents,
        'dev_statistics': dev_stats,
        'test_statistics': test_stats,
        'threshold_results': threshold_results,
        'recommended_thresholds': {
            'p95': {
                'value': p95_threshold,
                'test_pass_rate': float(p95_test_pass_rate)
            } if p95_threshold else None,
            'p99': {
                'value': p99_threshold,
                'test_pass_rate': float(p99_test_pass_rate)
            } if p99_threshold else None
        }
    })

    print(f"\nResults logged to Weave project: {args.project}")


if __name__ == '__main__':
    main()

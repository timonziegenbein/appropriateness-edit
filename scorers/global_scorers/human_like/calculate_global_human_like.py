import pandas as pd
import numpy as np
import torch
import sys
import os
import json
import random
import torch.nn as nn
import difflib
from transformers import AutoTokenizer

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scorers.local_scorers.human_like.model_defs import LanguageModel

# V2 vocab with 'keep-in-edit' token
hl_vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}

def load_human_like_model(device, model_path="scorers/global_scorers/human_like/models/global_human_like_v1.pth", max_len=1500):
    """Load the human-like model (same as local scorer but with longer max_len for documents)."""
    hl_embedding_dim = 200
    hl_nhead = 2
    hl_nhid = 200
    hl_nlayers = 2
    vocab_size = len(hl_vocab)

    # Create model with dropout=0 for inference
    human_like_model = LanguageModel(
        vocab_size, hl_embedding_dim, hl_nhead, hl_nhid, hl_nlayers, dropout=0.0
    ).to(device)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        human_like_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        human_like_model.load_state_dict(checkpoint)

    human_like_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    return human_like_model, tokenizer, max_len


def calculate_perplexity_for_sequence(model, sequence, max_len, device):
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


def generate_document_edit_sequence(before_revision, edit_actions, tokenizer):
    """
    Generate a document-level edit sequence from the actual edit actions in the JSON.

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

        # Skip non-edit actions (like keeping text)
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


def calculate_global_human_like_scores():
    """
    Calculates human-likeness perplexity for entire documents (all edits combined).
    This is the document-level version of the local human-like calculation.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model (same as local scorer but with longer max_len)
    model, tokenizer, max_len = load_human_like_model(device)
    print(f"Loaded human-like model with max_len={max_len}")

    results = []

    print("Starting to process examples from datasets/scorers/IteraTeR/full_doc_level/train.json...")
    with open("datasets/scorers/IteraTeR/full_doc_level/train.json", 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                before_revision = data.get('before_revision', '')
                edit_actions = data.get('edit_actions', [])

                # Skip if missing required fields
                if not before_revision or not edit_actions:
                    continue

                # Generate document-level edit sequence from actual edit actions
                edit_sequence = generate_document_edit_sequence(before_revision, edit_actions, tokenizer)

                if not edit_sequence or len(edit_sequence) <= 1:
                    continue

                # Calculate perplexity for the document-level edit sequence
                perplexity = calculate_perplexity_for_sequence(model, edit_sequence, max_len, device)

                if not np.isfinite(perplexity):
                    continue

                # Count edits (filtering for actual edit operations R, A, D)
                num_edits = len([e for e in edit_actions if e.get('type') in ['R', 'A', 'D']])

                results.append({
                    'global_human_like_perplexity': perplexity,
                    'num_edits': num_edits,
                    'sequence_length': len(edit_sequence),
                    'original_length': len(before_revision)
                })

            except json.JSONDecodeError:
                print(f"Skipping line {i+1} due to JSON decoding error.")
                continue
            except Exception as e:
                print(f"Error processing document {i+1}: {e}")
                continue

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} documents...")

    if results:
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_path = "results/global_human_like_classification.csv"
        os.makedirs("results", exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nGlobal human-like results saved to {output_path}")

        # Print statistics
        perplexity_scores = results_df['global_human_like_perplexity'].values
        print(f"\n{'='*80}")
        print("Global Human-Like Perplexity Distribution:")
        print(f"{'='*80}")
        print(f"Total documents: {len(perplexity_scores)}")
        print(f"Mean: {np.mean(perplexity_scores):.6f}")
        print(f"Std: {np.std(perplexity_scores):.6f}")
        print(f"Min: {np.min(perplexity_scores):.6f}")
        print(f"Max: {np.max(perplexity_scores):.6f}")
        print(f"\nPercentiles:")
        print(f"  P01: {np.percentile(perplexity_scores, 1):.6f}")
        print(f"  P05: {np.percentile(perplexity_scores, 5):.6f}")
        print(f"  P10: {np.percentile(perplexity_scores, 10):.6f}")
        print(f"  P25: {np.percentile(perplexity_scores, 25):.6f}")
        print(f"  P50 (median): {np.percentile(perplexity_scores, 50):.6f}")
        print(f"  P75: {np.percentile(perplexity_scores, 75):.6f}")
        print(f"  P90: {np.percentile(perplexity_scores, 90):.6f}")
        print(f"  P95: {np.percentile(perplexity_scores, 95):.6f}")
        print(f"  P99: {np.percentile(perplexity_scores, 99):.6f}")

        # Recommend threshold (e.g., P95 means 95% of human edits pass)
        recommended_threshold = np.percentile(perplexity_scores, 95)
        print(f"\nRECOMMENDED THRESHOLD (P95): {recommended_threshold:.6f}")
        print(f"  (95% of human-edited documents would pass this threshold)")

        # Also show P99 for comparison (like local scorer)
        recommended_threshold_p99 = np.percentile(perplexity_scores, 99)
        print(f"\nALTERNATIVE THRESHOLD (P99): {recommended_threshold_p99:.6f}")
        print(f"  (99% of human-edited documents would pass this threshold)")
    else:
        print("No valid global human-like scores were calculated.")

if __name__ == '__main__':
    calculate_global_human_like_scores()

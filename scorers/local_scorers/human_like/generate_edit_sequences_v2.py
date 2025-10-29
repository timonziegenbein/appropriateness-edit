import argparse
import json
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import difflib
from datasets import Dataset, DatasetDict
from pathlib import Path

# Updated vocabulary with keep-in-edit token
vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}

def get_sentence_boundaries(sents_char_pos):
    """Convert sents_char_pos array to list of (start, end) tuples for each sentence."""
    boundaries = []
    prev_pos = 0
    for pos in sents_char_pos:
        boundaries.append((prev_pos, pos))
        prev_pos = pos + 1  # Next sentence starts after the end position
    return boundaries

def filter_edits_by_sentence(edit_actions, sentence_boundaries):
    """
    Partition edits by sentence, filtering out edits that span multiple sentences.

    Returns: dict mapping sentence_idx -> list of edits
    """
    sents_to_edits = {}

    for edit in edit_actions:
        start_char = edit["start_char_pos"]
        end_char = edit["end_char_pos"]

        # Find which sentence this edit belongs to
        for sent_idx, (sent_start, sent_end) in enumerate(sentence_boundaries):
            # Check if edit is fully contained within this sentence
            if start_char >= sent_start and end_char <= sent_end:
                if sent_idx not in sents_to_edits:
                    sents_to_edits[sent_idx] = []
                sents_to_edits[sent_idx].append(edit)
                break

    return sents_to_edits

def generate_sequence_for_single_edit(sentence_text, edit, tokenizer, sent_start_char):
    """
    Generate edit operation sequence for a SINGLE edit within a sentence.

    Each sequence shows: [keep tokens before edit] + [edit operations] + [keep tokens after edit]

    Args:
        sentence_text: The text of the sentence
        edit: A single edit dict with start_char_pos, end_char_pos, and after fields
        tokenizer: The tokenizer to use
        sent_start_char: Character position where this sentence starts in the document

    Returns:
        List of tag strings representing the edit sequence, or None if invalid
    """
    # Tokenize the sentence with character offsets
    encoding = tokenizer(sentence_text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']

    if len(tokens) == 0:
        return None

    # Initialize all tags as 'keep'
    tags = ['keep'] * len(tokens)

    # Adjust edit positions to be relative to sentence start
    edit_start = edit["start_char_pos"] - sent_start_char
    edit_end = edit["end_char_pos"] - sent_start_char

    # Find token range for this edit
    tok_start = -1
    tok_end = -1
    for i, (token_start, token_end) in enumerate(offsets):
        # Check if token overlaps with edit region
        if edit_start < token_end and edit_end > token_start:
            if tok_start == -1:
                tok_start = i
            tok_end = i

    if tok_start == -1:
        # Edit doesn't align with any tokens - this shouldn't produce a sequence
        return None

    # Get before and after tokens
    before_tokens = tokens[tok_start:tok_end+1]
    after_text = edit.get("after", "")
    if not isinstance(after_text, str):
        after_text = ""
    after_tokens = tokenizer.tokenize(after_text)

    # Use difflib to compare before and after
    matcher = difflib.SequenceMatcher(None, before_tokens, after_tokens)

    # Track if we actually found any edit operations
    has_edit_operations = False

    # Apply opcodes to determine tags
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Tokens are kept but inside an edit region
            for i in range(i1, i2):
                tags[tok_start + i] = 'keep-in-edit'
            has_edit_operations = True  # keep-in-edit counts as being inside an edit
        elif tag == 'delete':
            for i in range(i1, i2):
                tags[tok_start + i] = 'del'
            has_edit_operations = True
        elif tag == 'replace':
            for i in range(i1, i2):
                tags[tok_start + i] = 'replace'
            has_edit_operations = True
        elif tag == 'insert':
            # Mark the position where insertion happens
            # Insert happens after position i1-1, or at start if i1==0
            if i1 > 0:
                insert_pos = tok_start + i1 - 1
            else:
                insert_pos = tok_start
            if insert_pos < len(tags):
                tags[insert_pos] = 'add'
            has_edit_operations = True

    # Only return sequences that actually contain edit operations
    if not has_edit_operations:
        return None

    # Verify the sequence isn't all 'keep' tokens
    if all(tag == 'keep' for tag in tags):
        return None

    return tags

def main():
    parser = argparse.ArgumentParser(description='Generate sentence-level edit sequences from human edit data (v2) and upload to HuggingFace Hub.')
    parser.add_argument('--dataset-prefix', type=str, required=True,
                       help='Base path to the IteraTeR data, e.g., data/IteraTeR/full_doc_level. Must contain train.json, test.json, and dev.json.')
    parser.add_argument('--output-dataset-name', type=str, default=None,
                       help='Name for the dataset on HuggingFace Hub (optional).')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Local directory to save the dataset (optional).')
    args = parser.parse_args()

    if not args.output_dataset_name and not args.output_dir:
        print("Error: Must specify either --output-dataset-name or --output-dir")
        return

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    dataset_splits = DatasetDict()
    splits_to_process = ["train", "test", "dev"]

    for split_name in splits_to_process:
        input_file = Path(args.dataset_prefix) / f"{split_name}.json"

        if not input_file.exists():
            print(f"Warning: Input file not found for split '{split_name}': {input_file}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing split: {split_name}")
        print(f"{'='*80}")

        sequences = []
        skipped_no_sents = 0
        skipped_no_edits = 0
        processed_edits = 0

        with open(input_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    if "before_revision" not in data or "edit_actions" not in data or "sents_char_pos" not in data:
                        skipped_no_sents += 1
                        continue

                    before_revision = data["before_revision"]
                    edit_actions = data["edit_actions"]
                    sents_char_pos = data["sents_char_pos"]

                    if len(sents_char_pos) == 0:
                        skipped_no_sents += 1
                        continue

                    sentence_boundaries = get_sentence_boundaries(sents_char_pos)
                    sents_to_edits = filter_edits_by_sentence(edit_actions, sentence_boundaries)

                    if len(sents_to_edits) == 0:
                        skipped_no_edits += 1
                        continue

                    for sent_idx, edits in sents_to_edits.items():
                        sent_start, sent_end = sentence_boundaries[sent_idx]
                        sentence_text = before_revision[sent_start:sent_end+1]

                        for edit_idx, edit in enumerate(edits):
                            # Skip if sentence contains %DIF markers (corrupted data)
                            if '%DIF' in sentence_text:
                                continue

                            tags = generate_sequence_for_single_edit(
                                sentence_text,
                                edit,
                                tokenizer,
                                sent_start
                            )

                            if tags:
                                before_text = edit.get("before", "")
                                after_text = edit.get("after", "")
                                if before_text is None: before_text = ""
                                if after_text is None: after_text = ""

                                # Skip if inappropriate_part or rewritten_part contain %DIF
                                if '%DIF' in before_text or '%DIF' in after_text:
                                    continue

                                # Skip edits where inappropriate_part is empty (pure insertions)
                                # These don't represent edits of inappropriate content
                                if not before_text.strip():
                                    continue

                                sequences.append({
                                    "sequence": ",".join(tags),
                                    "label": 1,
                                    "doc_id": data.get("doc_id", "unknown"),
                                    "sent_idx": sent_idx,
                                    "edit_idx": edit_idx,
                                    "original_sentence": sentence_text,
                                    "inappropriate_part": before_text,
                                    "rewritten_part": after_text
                                })
                                processed_edits += 1

                    if line_num % 1000 == 0:
                        print(f"  Processed {line_num} documents, generated {len(sequences)} sequences")

                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON on line {line_num} in {input_file}")
                    pass
                except Exception as e:
                    print(f"Warning: Error processing line {line_num} in {input_file}: {e}")
                    pass

        print(f"\n--- {split_name} Split Summary ---")
        print(f"Total sequences generated: {len(sequences)}")
        print(f"Individual edits processed: {processed_edits}")
        print(f"Documents skipped (no sents_char_pos): {skipped_no_sents}")
        print(f"Documents skipped (no valid edits): {skipped_no_edits}")

        if not sequences:
            print(f"No sequences generated for split '{split_name}'. Skipping.")
            continue

        dataset = Dataset.from_dict({
            'sequence': [s['sequence'] for s in sequences],
            'label': [s['label'] for s in sequences],
            'doc_id': [s['doc_id'] for s in sequences],
            'sent_idx': [s['sent_idx'] for s in sequences],
            'edit_idx': [s['edit_idx'] for s in sequences],
            'original_sentence': [s['original_sentence'] for s in sequences],
            'inappropriate_part': [s['inappropriate_part'] for s in sequences],
            'rewritten_part': [s['rewritten_part'] for s in sequences]
        })

        dataset_splits[split_name] = dataset

    if not dataset_splits:
        print("\nNo datasets were created. Exiting.")
        return

    print(f"\n{'='*80}")
    print("Combined Dataset Statistics")
    print(f"{'='*80}")
    for split_name, dataset in dataset_splits.items():
        print(f"Split '{split_name}': {len(dataset)} examples")

    print(f"\nVocabulary: {vocab}")
    print(f"Note: Each sequence represents ONE edit within a sentence.")

    if args.output_dataset_name:
        print(f"\nPushing dataset to HuggingFace Hub: {args.output_dataset_name}")
        dataset_splits.push_to_hub(args.output_dataset_name)
        print(f"✓ Dataset successfully uploaded to {args.output_dataset_name}")

    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving dataset to local directory: {output_path}")
        dataset_splits.save_to_disk(str(output_path))
        print(f"✓ Dataset saved to {output_path}")

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()

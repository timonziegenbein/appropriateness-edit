import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import math
import json
import difflib
import sys
sys.path.append('/mnt/home/tziegenb/appropriateness-feedback/src/end-to-end')

from utils.model_defs import LanguageModel, PositionalEncoding
from utils.reward_functions import MEAN_HUMAN_LIKE_PPL, STD_HUMAN_LIKE_PPL, Z_SCORE_THRESHOLD

# --- Sequence Generation Logic ---
def generate_sequence_simple(before_revision, edit, tokenizer):
    encoding = tokenizer(before_revision, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    if len(tokens) > 0:
        tags = ['keep'] * len(tokens)
        start_char = edit["start_char_pos"]
        end_char = edit["end_char_pos"]
        
        for i, offset in enumerate(offsets):
            token_start, token_end = offset
            if start_char < token_end and end_char > token_start:
                tags[i] = 'replace'
        return tags
    return None

def generate_sequence_difflib(before_revision, edit, tokenizer):
    encoding = tokenizer(before_revision, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    if len(tokens) > 0:
        tags = ['keep'] * len(tokens)
        start_char = edit["start_char_pos"]
        end_char = edit["end_char_pos"]
        after_edit_text = edit.get("after")

        token_start_index = -1
        token_end_index = -1
        for i, offset in enumerate(offsets):
            token_start, token_end = offset
            if start_char < token_end and end_char > token_start:
                if token_start_index == -1:
                    token_start_index = i
                token_end_index = i
        
        if token_start_index != -1:
            before_edit_tokens = tokens[token_start_index:token_end_index+1]
            
            if not isinstance(after_edit_text, str):
                after_edit_text = ""
            after_edit_tokens = tokenizer.tokenize(after_edit_text)

            edit_tags = []
            matcher = difflib.SequenceMatcher(None, before_edit_tokens, after_edit_tokens)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    edit_tags.extend(['keep_edit'] * (i2 - i1))
                elif tag in ['replace', 'delete']:
                    edit_tags.extend(['replace'] * (i2 - i1))
            
            if len(edit_tags) == (token_end_index - token_start_index + 1):
                tags[token_start_index:token_end_index+1] = edit_tags
        return tags
    return None

# --- Perplexity Calculation ---
def calculate_perplexity_for_sequence(sequence, model, vocab, device, max_len):
    sequence_as_int = [vocab.get(token, 0) for token in sequence]
    
    if len(sequence_as_int) <= 1:
        return float('inf')

    input_seq = sequence_as_int[:-1]
    target_seq = sequence_as_int[1:]
    
    padded_input = np.array(input_seq[:max_len] + [vocab['<pad>']]*(max_len - len(input_seq)) if len(input_seq) < max_len else input_seq[:max_len])
    padded_target = np.array(target_seq[:max_len] + [vocab['<pad>']]*(max_len - len(target_seq)) if len(target_seq) < max_len else target_seq[:max_len])

    inputs = torch.from_numpy(padded_input).long().unsqueeze(0).to(device)
    targets = torch.from_numpy(padded_target).long().unsqueeze(0).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    with torch.no_grad():
        output = model(inputs)
        loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
        perplexity = torch.exp(loss)
        return perplexity.item()

# --- Main Evaluation Function ---
def main():
    parser = argparse.ArgumentParser(description='Evaluate a language model on human edit data.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--perplexity-scores-csv', type=str, required=True, help='Path to the perplexity scores CSV file.')
    parser.add_argument('--input-json', type=str, default='IteraTeR/human_doc_level/train.json', help='Path to the input JSON file.')
    parser.add_argument('--use-keep-edit', action='store_true', help='Use difflib for fine-grained edit tagging.')
    parser.add_argument('--percentile-threshold', type=float, help='The percentile threshold for OOD detection.')
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_keep_edit:
        vocab = {'<pad>': 0, 'keep': 1, 'replace': 2, 'keep_edit': 3}
        generate_sequence_for_edit = generate_sequence_difflib
    else:
        vocab = {'<pad>': 0, 'keep': 1, 'replace': 2}
        generate_sequence_for_edit = generate_sequence_simple

    embedding_dim = 200
    nhead = 2
    nhid = 200
    nlayers = 2
    max_len = 500
    
    model = LanguageModel(len(vocab), embedding_dim, nhead, nhid, nlayers).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # --- Evaluation Loop ---
    in_distribution_count = 0
    out_of_distribution_count = 0
    total_edits = 0

    with open(args.input_json, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "before_revision" in data and "edit_actions" in data:
                    before_revision = data["before_revision"]
                    
                    for edit in data["edit_actions"]:
                        total_edits += 1
                        sequence = generate_sequence_for_edit(before_revision, edit, tokenizer)
                        
                        if sequence:
                            perplexity = calculate_perplexity_for_sequence(sequence, model, vocab, device, max_len)
                            if args.percentile_threshold is not None:
                                if perplexity <= args.percentile_threshold:
                                    in_distribution_count += 1
                                else:
                                    out_of_distribution_count += 1
                            else:
                                if STD_HUMAN_LIKE_PPL > 0:
                                    z_score = (perplexity - MEAN_HUMAN_LIKE_PPL) / STD_HUMAN_LIKE_PPL
                                else:
                                    z_score = 0.0
                                
                                if abs(z_score) <= Z_SCORE_THRESHOLD:
                                    in_distribution_count += 1
                                else:
                                    out_of_distribution_count += 1
            except json.JSONDecodeError:
                pass

    print(f"\n--- Evaluation Results on {args.input_json} ---")
    print(f"Total edits evaluated: {total_edits}")
    print(f"In-distribution edits: {in_distribution_count}")
    print(f"Out-of-distribution edits: {out_of_distribution_count}")
    if total_edits > 0:
        in_distribution_percentage = (in_distribution_count / total_edits) * 100
        print(f"Percentage of in-distribution edits: {in_distribution_percentage:.2f}%")

if __name__ == "__main__":
    main()

import argparse
import json
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import difflib

def generate_sequences(before_revision, edit, tokenizer, tokens, offsets):
    start_char = edit["start_char_pos"]
    end_char = edit["end_char_pos"]
    rewritten_part = edit.get("after")

    token_start_index = -1
    token_end_index = -1
    for i, offset in enumerate(offsets):
        token_start, token_end = offset
        if start_char < token_end and end_char > token_start:
            if token_start_index == -1:
                token_start_index = i
            token_end_index = i
    
    if token_start_index != -1:
        tags = []
        tags.extend(['keep'] * token_start_index)

        before_edit_tokens = tokens[token_start_index:token_end_index+1]
        if not isinstance(rewritten_part, str):
            rewritten_part = ""
        after_edit_tokens = tokenizer.tokenize(rewritten_part)
        
        matcher = difflib.SequenceMatcher(None, before_edit_tokens, after_edit_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                tags.extend(['keep'] * (i2 - i1))
            elif tag == 'delete':
                tags.extend(['del'] * (i2 - i1))
            elif tag == 'replace':
                tags.extend(['replace'] * (i2 - i1))
            elif tag == 'insert':
                tags.extend(['add'] * (j2 - j1))

        tags.extend(['keep'] * (len(tokens) - token_end_index - 1))
        
        return tags
    else:
        return ['keep'] * len(tokens)

def main():
    parser = argparse.ArgumentParser(description='Generate edit sequences from human edit data.')
    parser.add_argument('--output-csv', type=str, required=True, help='Path to the output CSV file.')
    parser.add_argument('--input-json', type=str, default='data/IteraTeR/full_doc_level/train.json', help='Path to the input JSON file.')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    sequences = []

    with open(args.input_json, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "before_revision" in data and "edit_actions" in data:
                    before_revision = data["before_revision"]
                    encoding = tokenizer(before_revision, return_offsets_mapping=True)
                    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
                    offsets = encoding['offset_mapping']
                    
                    if len(tokens) > 0:
                        for edit in data["edit_actions"]:
                            tags = generate_sequences(before_revision, edit, tokenizer, tokens, offsets)
                            sequences.append({
                                "sequence": ",".join(tags),
                                "label": 1
                            })
            except json.JSONDecodeError:
                pass

    df = pd.DataFrame(sequences)
    df.to_csv(args.output_csv, index=False)
    print(f"Edit sequences extracted and saved to {args.output_csv}")

if __name__ == "__main__":
    main()
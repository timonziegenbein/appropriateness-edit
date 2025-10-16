import pandas as pd
import numpy as np
import torch
import sys
import os
import json
import random
import spacy
import signal
import math
from sentence_transformers import SentenceTransformer, util
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def handler(signum, frame):
    raise TimeoutError("Levenshtein distance calculation timed out")

signal.signal(signal.SIGALRM, handler)
def normalized_edit_similarity(m, d):
    # d : edit distance between the two strings
    # m : length of the shorter string
    if m == d:
        return 0.0
    elif d == 0:
        return 1.0
    else:
        # to avoid division by zero
        if m - d == 0:
            return 0.0
        return (1.0 / math.exp(d / (m - d)))

from hirschberg import Hirschberg

def calculate_semantic_similarity_scores():
    """
    Calculates semantic similarity between sentences before and after an edit,
    and the percentage of token changes.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    ss_model = SentenceTransformer('google/embeddinggemma-300m', device=device)
    nlp = spacy.load("en_core_web_sm")
    word_tokenizer = Tokenizer(nlp.vocab)

    results = []
    
    print("Starting to process examples from data/IteraTeR/full_doc_level/train.json...")
    with open("data/IteraTeR/full_doc_level/train.json", 'r') as f:
        for i, line in enumerate(f):
            #if i >= 100:
            #    break
            try:
                data = json.loads(line)
                original_doc = data.get('before_revision', '')

                if not original_doc or 'edit_actions' not in data:
                    continue

                doc = nlp(original_doc)
                sentences = list(doc.sents)

                for edit in data['edit_actions']:
                    if edit.get('type') == 'R' and edit.get('before') and edit.get('after'):
                        start_char = edit.get('start_char_pos')
                        end_char = edit.get('end_char_pos')

                        if start_char is None or end_char is None:
                            continue

                        # Find the sentence for the edit
                        target_sentence = None
                        for sent in sentences:
                            if sent.start_char <= start_char and sent.end_char >= end_char:
                                target_sentence = sent
                                break
                        
                        if not target_sentence:
                            continue

                        sentence_before_text = target_sentence.text

                        # Apply edit within the sentence
                        edit_start_in_sent = start_char - target_sentence.start_char
                        edit_end_in_sent = end_char - target_sentence.start_char
                        
                        sentence_after_text = sentence_before_text[:edit_start_in_sent] + edit['after'] + sentence_before_text[edit_end_in_sent:]

                        # Add prompt instructions
                        document_prompt = "title: none | text: "
                        query_prompt = "task: sentence similarity | query: "
                        
                        sentence_before_with_prompt = query_prompt + sentence_before_text
                        sentence_after_with_prompt = document_prompt + sentence_after_text

                        # Calculate semantic similarity
                        query_embedding = ss_model.encode_query(sentence_before_with_prompt)
                        doc_embedding = ss_model.encode_document([sentence_after_with_prompt])
                        
                        similarities = ss_model.similarity(query_embedding, doc_embedding)
                        ss_score = similarities[0][0].item()

                        # Calculate normalized edit distance
                        tokens_before = [token.text for token in word_tokenizer(sentence_before_text)]
                        tokens_after = [token.text for token in word_tokenizer(sentence_after_text)]
                        
                        signal.alarm(5) # 5 seconds
                        try:
                            Z, W, S = Hirschberg(tokens_before, tokens_after)
                            num_edits = len([s for s in S if s != '<KEEP>']) 
                            m = len(S)
                            normalized_distance = normalized_edit_similarity(m, num_edits)

                            results.append({
                                'semantic_similarity': ss_score,
                                'normalized_edit_distance': normalized_distance
                            })
                        except TimeoutError:
                            print(f"Skipping example (Hirschberg timeout): {sentence_before_text}")
                        finally:
                            signal.alarm(0) # Disable the alarm
            except json.JSONDecodeError:
                print(f"Skipping line {i+1} due to JSON decoding error.")
                continue

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} documents...")

    if results:
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_path = "results/semantic_similarity_classification.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Semantic similarity results saved to {output_path}")
    else:
        print("No valid semantic similarity scores were calculated.")

if __name__ == '__main__':
    calculate_semantic_similarity_scores()

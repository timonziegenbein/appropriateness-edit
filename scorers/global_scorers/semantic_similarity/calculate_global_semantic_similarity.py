import pandas as pd
import numpy as np
import torch
import sys
import os
import json
import random
from sentence_transformers import SentenceTransformer

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def calculate_global_semantic_similarity_scores():
    """
    Calculates semantic similarity between documents before and after ALL edits.
    This is the document-level version of the local semantic similarity calculation.
    Uses before_revision and after_revision fields directly from the IteraTeR dataset.
    """
    output_path = "scorers/global_scorers/semantic_similarity/results/global_semantic_similarity_classification.csv"

    # Check if results already exist
    if os.path.exists(output_path):
        print(f"Results file already exists: {output_path}")
        print("Loading existing results and computing statistics...\n")

        results_df = pd.read_csv(output_path)
        ss_scores = results_df['global_semantic_similarity'].values

        print(f"{'='*80}")
        print("Global Semantic Similarity Score Distribution:")
        print(f"{'='*80}")
        print(f"Total documents: {len(ss_scores)}")
        print(f"Mean: {np.mean(ss_scores):.6f}")
        print(f"Std: {np.std(ss_scores):.6f}")
        print(f"Min: {np.min(ss_scores):.6f}")
        print(f"Max: {np.max(ss_scores):.6f}")
        print(f"\nPercentiles:")
        print(f"  P01: {np.percentile(ss_scores, 1):.6f}")
        print(f"  P05: {np.percentile(ss_scores, 5):.6f}")
        print(f"  P10: {np.percentile(ss_scores, 10):.6f}")
        print(f"  P25: {np.percentile(ss_scores, 25):.6f}")
        print(f"  P50 (median): {np.percentile(ss_scores, 50):.6f}")
        print(f"  P75: {np.percentile(ss_scores, 75):.6f}")
        print(f"  P90: {np.percentile(ss_scores, 90):.6f}")
        print(f"  P95: {np.percentile(ss_scores, 95):.6f}")
        print(f"  P99: {np.percentile(ss_scores, 99):.6f}")

        # Recommend threshold (e.g., P05 means 95% of human edits pass)
        recommended_threshold = np.percentile(ss_scores, 5)
        print(f"\nRECOMMENDED THRESHOLD (P05): {recommended_threshold:.6f}")
        print(f"  (95% of human-edited documents would pass this threshold)")

        return

    # Results don't exist, run full calculation
    print(f"Results file not found. Running full calculation...")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model (same as local scorer)
    ss_model = SentenceTransformer('google/embeddinggemma-300m', device=device)

    results = []

    print("Starting to process examples from datasets/scorers/IteraTeR/full_doc_level/train.json...")
    with open("datasets/scorers/IteraTeR/full_doc_level/train.json", 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                original_doc = data.get('before_revision', '')
                edited_doc = data.get('after_revision', '')

                # Skip if missing required fields
                if not original_doc or not edited_doc:
                    continue

                # Skip if no actual changes were made
                if original_doc == edited_doc:
                    continue

                # Add prompt instructions (same as local scorer)
                document_prompt = "title: none | text: "
                query_prompt = "task: sentence similarity | query: "

                doc_before_with_prompt = query_prompt + original_doc
                doc_after_with_prompt = document_prompt + edited_doc

                # Calculate semantic similarity at document level
                query_embedding = ss_model.encode_query(doc_before_with_prompt)
                doc_embedding = ss_model.encode_document([doc_after_with_prompt])

                similarities = ss_model.similarity(query_embedding, doc_embedding)
                ss_score = similarities[0][0].item()

                # Count edits if available
                num_edits = len(data.get('edit_actions', []))

                results.append({
                    'global_semantic_similarity': ss_score,
                    'num_edits': num_edits,
                    'original_length': len(original_doc),
                    'edited_length': len(edited_doc)
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
        output_path = "results/global_semantic_similarity_classification.csv"
        os.makedirs("results", exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nGlobal semantic similarity results saved to {output_path}")

        # Print statistics
        ss_scores = results_df['global_semantic_similarity'].values
        print(f"\n{'='*80}")
        print("Global Semantic Similarity Score Distribution:")
        print(f"{'='*80}")
        print(f"Total documents: {len(ss_scores)}")
        print(f"Mean: {np.mean(ss_scores):.6f}")
        print(f"Std: {np.std(ss_scores):.6f}")
        print(f"Min: {np.min(ss_scores):.6f}")
        print(f"Max: {np.max(ss_scores):.6f}")
        print(f"\nPercentiles:")
        print(f"  P01: {np.percentile(ss_scores, 1):.6f}")
        print(f"  P05: {np.percentile(ss_scores, 5):.6f}")
        print(f"  P10: {np.percentile(ss_scores, 10):.6f}")
        print(f"  P25: {np.percentile(ss_scores, 25):.6f}")
        print(f"  P50 (median): {np.percentile(ss_scores, 50):.6f}")
        print(f"  P75: {np.percentile(ss_scores, 75):.6f}")
        print(f"  P90: {np.percentile(ss_scores, 90):.6f}")
        print(f"  P95: {np.percentile(ss_scores, 95):.6f}")
        print(f"  P99: {np.percentile(ss_scores, 99):.6f}")

        # Recommend threshold (e.g., P05 means 95% of human edits pass)
        recommended_threshold = np.percentile(ss_scores, 5)
        print(f"\nRECOMMENDED THRESHOLD (P05): {recommended_threshold:.6f}")
        print(f"  (95% of human-edited documents would pass this threshold)")
    else:
        print("No valid global semantic similarity scores were calculated.")

if __name__ == '__main__':
    calculate_global_semantic_similarity_scores()

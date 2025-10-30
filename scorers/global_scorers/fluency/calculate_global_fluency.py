import pandas as pd
import numpy as np
import sys
import os
import json
import random
import weave
from weave.scorers import WeaveFluencyScorerV1

# Set seeds for reproducibility
np.random.seed(0)
random.seed(0)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def calculate_global_fluency_scores():
    """
    Calculates fluency scores for entire documents (before vs after all edits).
    This is the document-level version of the local fluency calculation.
    Uses Weave's built-in WeaveFluencyScorerV1.
    """
    print("Initializing Weave fluency scorer...")
    # Explicitly set device to "cpu" to avoid flash attention issues
    scorer = WeaveFluencyScorerV1(device="cpu")

    results = []

    print("Starting to process examples from datasets/scorers/IteraTeR/full_doc_level/train.json...")
    with open("datasets/scorers/IteraTeR/full_doc_level/train.json", 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                before_revision = data.get('before_revision', '')
                after_revision = data.get('after_revision', '')

                # Skip if missing required fields
                if not before_revision or not after_revision:
                    continue

                # Skip if no actual changes were made
                if before_revision == after_revision:
                    continue

                # Use Weave's fluency scorer on the edited text
                result = scorer.score(output=after_revision)

                # Convert Weave's result to our expected format
                binary_score = 1.0 if result.passed else 0.0
                # Extract fluency score from metadata
                confidence = result.metadata.get('score', binary_score)
                predicted_class = 1 if result.passed else 0

                # Count edits if available
                num_edits = len(data.get('edit_actions', []))

                results.append({
                    'global_fluency_binary': binary_score,
                    'global_fluency_confidence': confidence,
                    'predicted_class': predicted_class,
                    'num_edits': num_edits,
                    'original_length': len(before_revision),
                    'edited_length': len(after_revision)
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
        output_path = "results/global_fluency_classification.csv"
        os.makedirs("results", exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nGlobal fluency results saved to {output_path}")

        # Print statistics
        binary_scores = results_df['global_fluency_binary'].values
        confidence_scores = results_df['global_fluency_confidence'].values

        print(f"\n{'='*80}")
        print("Global Fluency Binary Score Distribution:")
        print(f"{'='*80}")
        print(f"Total documents: {len(binary_scores)}")
        print(f"Fluent (class 1): {np.sum(binary_scores == 1)} ({np.mean(binary_scores == 1)*100:.2f}%)")
        print(f"Non-fluent (class 0): {np.sum(binary_scores == 0)} ({np.mean(binary_scores == 0)*100:.2f}%)")

        print(f"\n{'='*80}")
        print("Global Fluency Confidence Distribution:")
        print(f"{'='*80}")
        print(f"Mean: {np.mean(confidence_scores):.6f}")
        print(f"Std: {np.std(confidence_scores):.6f}")
        print(f"Min: {np.min(confidence_scores):.6f}")
        print(f"Max: {np.max(confidence_scores):.6f}")
        print(f"\nPercentiles:")
        print(f"  P01: {np.percentile(confidence_scores, 1):.6f}")
        print(f"  P05: {np.percentile(confidence_scores, 5):.6f}")
        print(f"  P10: {np.percentile(confidence_scores, 10):.6f}")
        print(f"  P25: {np.percentile(confidence_scores, 25):.6f}")
        print(f"  P50 (median): {np.percentile(confidence_scores, 50):.6f}")
        print(f"  P75: {np.percentile(confidence_scores, 75):.6f}")
        print(f"  P90: {np.percentile(confidence_scores, 90):.6f}")
        print(f"  P95: {np.percentile(confidence_scores, 95):.6f}")
        print(f"  P99: {np.percentile(confidence_scores, 99):.6f}")

        # For fluency, Weave's scorer already outputs binary predictions
        # The threshold is handled internally by WeaveFluencyScorerV1
        print(f"\nNOTE: Weave's WeaveFluencyScorerV1 already outputs binary predictions (0 or 1).")
        print(f"The scorer handles thresholding internally.")
        print(f"No additional threshold needed for this scorer.")

        # Analyze fluent documents only
        fluent_mask = binary_scores == 1
        if np.any(fluent_mask):
            fluent_confidence = confidence_scores[fluent_mask]
            print(f"\n{'='*80}")
            print("Confidence Distribution for Fluent Documents (class 1):")
            print(f"{'='*80}")
            print(f"Count: {len(fluent_confidence)}")
            print(f"Mean: {np.mean(fluent_confidence):.6f}")
            print(f"Std: {np.std(fluent_confidence):.6f}")
            print(f"Min: {np.min(fluent_confidence):.6f}")
            print(f"Max: {np.max(fluent_confidence):.6f}")
    else:
        print("No valid global fluency scores were calculated.")

if __name__ == '__main__':
    calculate_global_fluency_scores()

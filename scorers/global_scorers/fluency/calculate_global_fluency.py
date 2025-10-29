import pandas as pd
import numpy as np
import torch
import sys
import os
import json
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def load_fluency_model(device, model_name="tcapelle/fluency-scorer"):
    """Load the fluency model from HuggingFace."""
    print(f"Loading fluency model from HuggingFace: {model_name}")

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.to(device)
    model.eval()

    print("Fluency model loaded successfully")
    return model, tokenizer


def calculate_global_fluency_scores():
    """
    Calculates fluency scores for entire documents (before vs after all edits).
    This is the document-level version of the local fluency calculation.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model (same as local scorer)
    model, tokenizer = load_fluency_model(device)

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

                # Tokenize the full documents (same format as local scorer)
                inputs = tokenizer(
                    before_revision,
                    after_revision,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,  # May truncate very long documents
                    padding=True
                ).to(device)

                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Get probabilities using softmax
                    probs = torch.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(logits, dim=-1).item()
                    confidence = probs[0][predicted_class].item()

                # Binary score: 1 if fluent (class 1), 0 if non-fluent (class 0)
                binary_score = float(predicted_class)

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

        # For fluency, the model already outputs binary predictions
        # The threshold is inherent in the model (0.5 probability threshold)
        print(f"\nNOTE: The fluency model already outputs binary predictions (0 or 1).")
        print(f"The model uses an inherent threshold of 0.5 probability.")
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

"""
Create an augmented fluency dataset by combining:
1. The existing timonziegenbein/fluency-edits-gec-only dataset
2. New examples from Gemini 2.5 Pro evaluations (from fluency_traces_export.jsonl)

The resulting dataset uses Gemini 2.5 Pro's evaluation as the ground truth for new examples.
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_traces_from_jsonl(traces_file: Path) -> List[Dict[str, Any]]:
    """
    Load and parse the Weave traces JSONL file.

    Returns a list of examples with:
    - original_sentence
    - inappropriate_part
    - rewritten_part
    - expected_score (from Gemini evaluation)
    - description (Gemini's reason)
    - modernbert_score (for comparison/analysis)
    """
    examples = []

    with open(traces_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                trace = json.loads(line.strip())

                # Extract inputs
                inputs = trace.get('inputs', {})
                original_sentence = inputs.get('original_sentence')
                inappropriate_part = inputs.get('inappropriate_part')
                rewritten_part = inputs.get('rewritten_part')

                # Extract ModernBERT output
                modernbert_score = trace.get('output')

                # Extract Gemini feedback
                feedback_list = trace.get('summary', {}).get('weave', {}).get('feedback', [])

                if not feedback_list:
                    logger.warning(f"Line {line_num}: No feedback found, skipping")
                    continue

                # Get the first feedback (should be from Gemini)
                feedback = feedback_list[0]
                payload = feedback.get('payload', {}).get('output', {})

                is_fluent = payload.get('is_fluent')
                reason = payload.get('reason', '')

                # Validate required fields
                if not all([original_sentence, inappropriate_part is not None,
                           rewritten_part is not None, is_fluent is not None]):
                    logger.warning(f"Line {line_num}: Missing required fields, skipping")
                    continue

                # Convert Gemini's boolean to float (1.0 or 0.0)
                expected_score = 1.0 if is_fluent else 0.0

                # Create example
                example = {
                    'original_sentence': original_sentence,
                    'inappropriate_part': inappropriate_part,
                    'rewritten_part': rewritten_part,
                    'expected_score': expected_score,
                    'description': f"Gemini 2.5 Pro: {reason}",
                    'modernbert_score': modernbert_score,
                    'source': 'gemini_grpo_traces'
                }

                examples.append(example)

            except json.JSONDecodeError:
                logger.error(f"Line {line_num}: Invalid JSON, skipping")
                continue
            except Exception as e:
                logger.error(f"Line {line_num}: Error processing trace: {e}")
                continue

    logger.info(f"Loaded {len(examples)} examples from traces file")
    return examples


def deduplicate_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate examples based on (original_sentence, inappropriate_part, rewritten_part).
    Keep the first occurrence of each unique example.
    """
    seen = set()
    deduplicated = []

    for example in examples:
        key = (
            example['original_sentence'],
            example['inappropriate_part'],
            example['rewritten_part']
        )

        if key not in seen:
            seen.add(key)
            deduplicated.append(example)

    logger.info(f"Deduplicated: {len(examples)} -> {len(deduplicated)} examples")
    return deduplicated


def create_augmented_dataset(
    base_dataset_name: str,
    traces_file: Path,
    output_dataset_name: str,
    gemini_test_ratio: float = 0.2,
    gemini_val_ratio: float = 0.1,
    push_to_hub: bool = False
):
    """
    Create an augmented dataset by combining base dataset with Gemini traces.

    Strategy:
    - Preserve the original base dataset splits (train/val/test)
    - Split new Gemini examples and add them to each split to extend them
    - This ensures the test set is properly extended with new examples

    Args:
        base_dataset_name: HuggingFace dataset to extend (e.g., "timonziegenbein/fluency-edits-gec-only")
        traces_file: Path to the JSONL traces export file
        output_dataset_name: Name for the new dataset
        gemini_test_ratio: Ratio of Gemini examples to add to test split
        gemini_val_ratio: Ratio of Gemini examples to add to validation split
        push_to_hub: Whether to push to HuggingFace Hub
    """
    # Load base dataset
    logger.info(f"Loading base dataset: {base_dataset_name}")
    base_dataset = load_dataset(base_dataset_name)

    # Preserve base dataset splits
    base_train = []
    base_val = []
    base_test = []

    for example in base_dataset['train']:
        base_train.append({
            'original_sentence': example['original_sentence'],
            'inappropriate_part': example['inappropriate_part'],
            'rewritten_part': example['rewritten_part'],
            'expected_score': example['expected_score'],
            'description': example.get('description', ''),
            'source': 'gec_base'
        })

    for example in base_dataset['validation']:
        base_val.append({
            'original_sentence': example['original_sentence'],
            'inappropriate_part': example['inappropriate_part'],
            'rewritten_part': example['rewritten_part'],
            'expected_score': example['expected_score'],
            'description': example.get('description', ''),
            'source': 'gec_base'
        })

    for example in base_dataset['test']:
        base_test.append({
            'original_sentence': example['original_sentence'],
            'inappropriate_part': example['inappropriate_part'],
            'rewritten_part': example['rewritten_part'],
            'expected_score': example['expected_score'],
            'description': example.get('description', ''),
            'source': 'gec_base'
        })

    logger.info(f"Base dataset - Train: {len(base_train)}, Val: {len(base_val)}, Test: {len(base_test)}")

    # Load new examples from traces
    logger.info(f"Loading traces from: {traces_file}")
    gemini_examples = load_traces_from_jsonl(traces_file)

    # Deduplicate Gemini examples
    gemini_examples = deduplicate_examples(gemini_examples)
    logger.info(f"Gemini examples after deduplication: {len(gemini_examples)}")

    # Remove 'modernbert_score' field before creating dataset (not needed for training)
    for example in gemini_examples:
        if 'modernbert_score' in example:
            del example['modernbert_score']

    # Split Gemini examples
    import random
    random.seed(42)
    random.shuffle(gemini_examples)

    n_gemini = len(gemini_examples)
    n_gemini_test = int(n_gemini * gemini_test_ratio)
    n_gemini_val = int(n_gemini * gemini_val_ratio)

    gemini_test = gemini_examples[:n_gemini_test]
    gemini_val = gemini_examples[n_gemini_test:n_gemini_test + n_gemini_val]
    gemini_train = gemini_examples[n_gemini_test + n_gemini_val:]

    logger.info(f"Gemini split - Train: {len(gemini_train)}, Val: {len(gemini_val)}, Test: {len(gemini_test)}")

    # Combine base and Gemini examples for each split
    train_examples = base_train + gemini_train
    val_examples = base_val + gemini_val
    test_examples = base_test + gemini_test

    logger.info(f"Final sizes - Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_examples),
        'validation': Dataset.from_list(val_examples),
        'test': Dataset.from_list(test_examples)
    })

    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    for split_name, split_dataset in dataset_dict.items():
        fluent_count = sum(1 for ex in split_dataset if ex['expected_score'] == 1.0)
        non_fluent_count = len(split_dataset) - fluent_count
        print(f"\n{split_name.upper()}:")
        print(f"  Total examples: {len(split_dataset)}")
        print(f"  Fluent (1.0): {fluent_count} ({fluent_count/len(split_dataset)*100:.1f}%)")
        print(f"  Non-fluent (0.0): {non_fluent_count} ({non_fluent_count/len(split_dataset)*100:.1f}%)")

        # Count by source
        sources = {}
        for ex in split_dataset:
            source = ex.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        print(f"  By source: {sources}")
    print("="*80 + "\n")

    # Save locally
    output_dir = Path("fluency_augmented_dataset")
    output_dir.mkdir(exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    logger.info(f"Saved dataset to {output_dir}")

    # Push to hub if requested
    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {output_dataset_name}")
        dataset_dict.push_to_hub(output_dataset_name)
        logger.info(f"Successfully pushed to Hub!")
        print(f"\nDataset available at: https://huggingface.co/datasets/{output_dataset_name}")

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Create augmented fluency dataset from Gemini traces")
    parser.add_argument(
        "--base-dataset",
        type=str,
        default="timonziegenbein/fluency-edits-gec-only",
        help="Base dataset to extend"
    )
    parser.add_argument(
        "--traces-file",
        type=Path,
        default=Path("scorers/local_scorers/fluency/fluency_traces_export.jsonl"),
        help="Path to the JSONL traces export file"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="timonziegenbein/fluency-augmented-edits",
        help="Name for the output dataset on HuggingFace Hub"
    )
    parser.add_argument(
        "--gemini-test-ratio",
        type=float,
        default=0.2,
        help="Ratio of Gemini examples to allocate to test split (default: 0.2)"
    )
    parser.add_argument(
        "--gemini-val-ratio",
        type=float,
        default=0.1,
        help="Ratio of Gemini examples to allocate to validation split (default: 0.1). Remaining go to train."
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to HuggingFace Hub"
    )

    args = parser.parse_args()

    # Validate split ratios
    total_ratio = args.gemini_test_ratio + args.gemini_val_ratio
    if total_ratio >= 1.0:
        raise ValueError(f"gemini_test_ratio + gemini_val_ratio must be < 1.0, got {total_ratio}")

    # Create dataset
    dataset = create_augmented_dataset(
        base_dataset_name=args.base_dataset,
        traces_file=args.traces_file,
        output_dataset_name=args.output_name,
        gemini_test_ratio=args.gemini_test_ratio,
        gemini_val_ratio=args.gemini_val_ratio,
        push_to_hub=args.push_to_hub
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()

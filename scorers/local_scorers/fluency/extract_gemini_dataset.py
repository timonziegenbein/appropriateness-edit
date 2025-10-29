"""
Extract Gemini-only examples from the extended dataset.

Loads timonziegenbein/fluency-edits-gec-only-extended and creates a new dataset
containing only examples with source='gemini_grpo_traces'.
"""

import argparse
from datasets import load_dataset, DatasetDict, Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_gemini_examples(
    extended_dataset_name: str,
    output_dataset_name: str,
    push_to_hub: bool = False
):
    """
    Extract Gemini-labeled examples from extended dataset.

    Args:
        extended_dataset_name: Name of the extended dataset (e.g., "timonziegenbein/fluency-edits-gec-only-extended")
        output_dataset_name: Name for the Gemini-only dataset (e.g., "timonziegenbein/fluency-edits-gemini")
        push_to_hub: Whether to push to HuggingFace Hub
    """
    # Load extended dataset
    logger.info(f"Loading extended dataset: {extended_dataset_name}")
    extended_dataset = load_dataset(extended_dataset_name)

    # Filter each split to keep only Gemini examples
    gemini_splits = {}

    for split_name in extended_dataset.keys():
        logger.info(f"Processing split: {split_name}")
        split_dataset = extended_dataset[split_name]

        # Filter to keep only gemini_grpo_traces
        gemini_examples = split_dataset.filter(
            lambda example: example.get('source') == 'gemini_grpo_traces'
        )

        gemini_splits[split_name] = gemini_examples
        logger.info(f"  {split_name}: {len(split_dataset)} total -> {len(gemini_examples)} Gemini examples")

    # Create new DatasetDict
    gemini_dataset = DatasetDict(gemini_splits)

    # Print statistics
    print("\n" + "="*80)
    print("GEMINI-ONLY DATASET STATISTICS")
    print("="*80)

    for split_name, split_dataset in gemini_dataset.items():
        fluent_count = sum(1 for ex in split_dataset if ex['expected_score'] == 1.0)
        non_fluent_count = len(split_dataset) - fluent_count

        print(f"\n{split_name.upper()}:")
        print(f"  Total examples: {len(split_dataset)}")
        print(f"  Fluent (1.0): {fluent_count} ({fluent_count/len(split_dataset)*100:.1f}%)")
        print(f"  Non-fluent (0.0): {non_fluent_count} ({non_fluent_count/len(split_dataset)*100:.1f}%)")

    total_examples = sum(len(split_dataset) for split_dataset in gemini_dataset.values())
    print(f"\nTOTAL GEMINI EXAMPLES: {total_examples}")
    print("="*80 + "\n")

    # Save locally
    from pathlib import Path
    output_dir = Path("fluency_gemini_only_dataset")
    output_dir.mkdir(exist_ok=True)
    gemini_dataset.save_to_disk(str(output_dir))
    logger.info(f"Saved dataset to {output_dir}")

    # Push to hub if requested
    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {output_dataset_name}")
        gemini_dataset.push_to_hub(output_dataset_name)
        logger.info(f"Successfully pushed to Hub!")
        print(f"\nDataset available at: https://huggingface.co/datasets/{output_dataset_name}")

    return gemini_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Extract Gemini-only examples from extended fluency dataset"
    )
    parser.add_argument(
        "--extended-dataset",
        type=str,
        default="timonziegenbein/fluency-edits-gec-only-extended",
        help="Name of the extended dataset to load"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="timonziegenbein/fluency-edits-gemini",
        help="Name for the Gemini-only dataset on HuggingFace Hub"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to HuggingFace Hub"
    )

    args = parser.parse_args()

    # Extract dataset
    dataset = extract_gemini_examples(
        extended_dataset_name=args.extended_dataset,
        output_dataset_name=args.output_name,
        push_to_hub=args.push_to_hub
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()

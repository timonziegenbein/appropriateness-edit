
import argparse
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import logging
import sys
import time
from textattack.augmentation.recipes import DeletionAugmenter, EmbeddingAugmenter, SwapAugmenter, SynonymInsertionAugmenter
from textattack.augmentation import Augmenter
from textattack.transformations import Transformation
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import random

# Custom Augmenter for word duplication
class WordDuplication(Transformation):
    """
    Transformation that duplicates a word.
    """
    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        words = current_text.words
        for i in indices_to_modify:
            word_to_duplicate = words[i]
            transformed_texts.append(
                current_text.insert_text_after_word_index(i, word_to_duplicate)
            )
        return transformed_texts

def WordDuplicationAugmenter(transformations_per_example=1, **kwargs):
    """
    Returns an augmenter that duplicates a word.
    """
    transformation = WordDuplication()
    return Augmenter(
        transformation=transformation,
        transformations_per_example=transformations_per_example,
    )

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def augment_example_worker(args_tuple):
    """
    Worker function to augment a single example in parallel.

    Args:
        args_tuple: Tuple of (example, augmenter_classes, augmentations_per_example, seed)

    Returns:
        List of augmented examples
    """
    example, augmenter_classes, augmentations_per_example, seed = args_tuple

    # Set random seed for reproducibility (unique per example)
    random.seed(seed)

    augmented_examples = []

    try:
        # Randomly select one augmenter class
        augmenter_class = random.choice(augmenter_classes)
        augmenter = augmenter_class(transformations_per_example=augmentations_per_example)

        # Augment the more fluent text (tgt)
        augmented_tgt = augmenter.augment(example['tgt'])
        for aug_tgt in augmented_tgt:
            augmented_examples.append({'src': aug_tgt, 'tgt': example['tgt'], 'task': 'augmentation'})

        # Augment the less fluent text (src)
        augmented_src = augmenter.augment(example['src'])
        for aug_src in augmented_src:
            augmented_examples.append({'src': aug_src, 'tgt': example['src'], 'task': 'augmentation'})

    except Exception as e:
        # Log error but don't crash the process
        return []

    return augmented_examples


def main(args):
    # Load and preprocess the grammarly/coedit dataset
    coedit_dataset = load_dataset("grammarly/coedit")

    def preprocess_coedit(example):
        src_parts = example['src'].split(':', 1)
        if len(src_parts) > 1:
            example['src'] = src_parts[1].lstrip()
        return example

    coedit_dataset = coedit_dataset.map(preprocess_coedit)

    coedit_combined = concatenate_datasets([coedit_dataset["train"], coedit_dataset["validation"]])

    # Load and preprocess the jhu-clsp/jfleg dataset
    jfleg_val = load_dataset("jhu-clsp/jfleg", split="validation")
    jfleg_test = load_dataset("jhu-clsp/jfleg", split="test")
    jfleg_dataset = concatenate_datasets([jfleg_val, jfleg_test])
    
    jfleg_src = []
    jfleg_tgt = []
    for example in jfleg_dataset:
        for correction in example['corrections']:
            jfleg_src.append(example['sentence'])
            jfleg_tgt.append(correction)
            
    jfleg_prepared = Dataset.from_dict({'src': jfleg_src, 'tgt': jfleg_tgt, 'task': ['jfleg'] * len(jfleg_src)})

    # Combine the datasets
    if args.tasks:
        # Filter by specified tasks
        combined_dataset = concatenate_datasets([coedit_combined, jfleg_prepared])
        dataset = combined_dataset.filter(lambda x: x['task'] in args.tasks)
        logging.info(f"Filtered dataset to tasks: {args.tasks}")
    else:
        # Default: use all relevant tasks
        combined_dataset = concatenate_datasets([coedit_combined, jfleg_prepared])
        dataset = combined_dataset.filter(lambda x: x['task'] in ['gec', 'simplification', 'coherence', 'jfleg'])
        logging.info(f"Using default tasks: ['gec', 'simplification', 'coherence', 'jfleg']")

    # --- Data Augmentation ---
    if args.enable_augmentation:
        logging.info("Starting data augmentation...")
        start_time = time.time()

        augmenter_classes = [
            DeletionAugmenter,
            EmbeddingAugmenter,
            SwapAugmenter,
            SynonymInsertionAugmenter,
            WordDuplicationAugmenter,
        ]

        dataset_to_augment = dataset
        if args.max_augment_samples is not None:
            dataset_to_augment = dataset.select(range(args.max_augment_samples))

        # Determine number of processes
        num_processes = args.num_workers if args.num_workers else max(1, cpu_count() - 1)
        logging.info(f"Using {num_processes} processes for parallel augmentation")

        # Prepare arguments for parallel processing
        # Use index as seed for reproducibility
        process_args = [
            (example, augmenter_classes, args.augmentations_per_example, 42 + idx)
            for idx, example in enumerate(dataset_to_augment)
        ]

        # Process examples in parallel
        augmented_examples = []
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                # Use imap_unordered for better memory efficiency and progress tracking
                results = list(tqdm(
                    pool.imap_unordered(augment_example_worker, process_args, chunksize=10),
                    total=len(process_args),
                    desc="Augmenting examples"
                ))
        else:
            # Single-process fallback
            results = [
                augment_example_worker(args)
                for args in tqdm(process_args, desc="Augmenting examples")
            ]

        # Flatten results
        for example_augmentations in results:
            augmented_examples.extend(example_augmentations)

        end_time = time.time()
        logging.info(f"Data augmentation finished in {end_time - start_time:.2f} seconds.")
        logging.info(f"Created {len(augmented_examples)} new examples.")

        if augmented_examples:
            augmented_dataset = Dataset.from_dict({
                k: [dic[k] for dic in augmented_examples] for k in augmented_examples[0]
            })
            final_dataset = concatenate_datasets([dataset, augmented_dataset])
        else:
            final_dataset = dataset
    else:
        logging.info("Augmentation disabled. Using original dataset only.")
        final_dataset = dataset

    logging.info(f"Total examples in the final dataset: {len(final_dataset)}")

    # IMPORTANT: Split BEFORE creating bidirectional pairs to avoid data leakage
    # This ensures that if (text1, text2) is in train, then (text2, text1) is also in train
    # and neither appears in validation or test sets
    logging.info("Splitting the dataset into train, validation, and test sets...")
    train_test_split = final_dataset.train_test_split(test_size=0.2, seed=42)
    # Create a validation set from the new training set
    train_val_split = train_test_split["train"].train_test_split(test_size=0.1, seed=42)

    base_split_dataset = {
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": train_test_split["test"]
    }
    logging.info("Dataset splitting complete.")

    # Now create bidirectional pairs for each split separately
    logging.info("Creating bidirectional pairs for each split...")
    split_datasets = {}

    for split_name, split_data in base_split_dataset.items():
        processed_examples = []
        for example in split_data:
            # Create both directions: src->tgt (label=1) and tgt->src (label=0)
            processed_examples.append({'text1': example['src'], 'text2': example['tgt'], 'label': 1})
            processed_examples.append({'text1': example['tgt'], 'text2': example['src'], 'label': 0})

        split_datasets[split_name] = Dataset.from_dict({
            k: [dic[k] for dic in processed_examples] for k in processed_examples[0]
        })

        logging.info(f"{split_name}: {len(split_data)} base examples -> {len(processed_examples)} bidirectional pairs")

    split_dataset = DatasetDict(split_datasets)

    # Push the dataset to the Hugging Face Hub
    logging.info(f"Uploading the dataset to {args.dataset_name}...")
    split_dataset.push_to_hub(args.dataset_name)
    logging.info(f"Dataset successfully uploaded to {args.dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and upload the fluency dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset on the Hugging Face Hub.")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Which tasks to include from coedit (e.g., gec simplification coherence jfleg). If not specified, uses all: ['gec', 'simplification', 'coherence', 'jfleg'].")
    parser.add_argument("--enable_augmentation", action="store_true", help="Enable data augmentation (DeletionAugmenter, EmbeddingAugmenter, SwapAugmenter, SynonymInsertionAugmenter, WordDuplicationAugmenter).")
    parser.add_argument("--max_augment_samples", type=int, default=None, help="For debugging, truncate the number of examples to augment.")
    parser.add_argument("--augmentations_per_example", type=int, default=1, help="The number of augmentations to apply to each example.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes for parallel processing (default: cpu_count - 1).")
    args = parser.parse_args()
    main(args)

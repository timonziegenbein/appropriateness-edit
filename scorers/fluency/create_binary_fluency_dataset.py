
import argparse
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import logging
import sys
import time
from textattack.augmentation.recipes import DeletionAugmenter, EmbeddingAugmenter, SwapAugmenter, SynonymInsertionAugmenter
from textattack.augmentation import Augmenter
from textattack.transformations import Transformation
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
    combined_dataset = concatenate_datasets([coedit_combined, jfleg_prepared])
    dataset = combined_dataset.filter(lambda x: x['task'] in ['gec', 'simplification', 'coherence', 'jfleg'])

    # --- Data Augmentation ---
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

    augmented_sentences = []
    for i, example in enumerate(dataset_to_augment):
        if (i + 1) % 100 == 0:
            logging.info(f"Processing example {i+1}/{len(dataset_to_augment)}")

        # Randomly select one augmenter class
        augmenter_class = random.choice(augmenter_classes)
        augmenter = augmenter_class(transformations_per_example=args.augmentations_per_example)

        # Augment the more fluent text (tgt) and label as not fluent
        augmented_tgt = augmenter.augment(example['tgt'])
        for aug_tgt in augmented_tgt:
            augmented_sentences.append({'text': aug_tgt, 'label': 0})

        # Augment the less fluent text (src) and label as not fluent
        augmented_src = augmenter.augment(example['src'])
        for aug_src in augmented_src:
            augmented_sentences.append({'text': aug_src, 'label': 0})

    end_time = time.time()
    logging.info(f"Data augmentation finished in {end_time - start_time:.2f} seconds.")
    logging.info(f"Created {len(augmented_sentences)} new augmented examples.")

    # --- Process Original and Augmented Data ---
    processed_examples = []
    # Add original sentences
    for example in dataset:
        processed_examples.append({'text': example['tgt'], 'label': 1}) # Fluent
        processed_examples.append({'text': example['src'], 'label': 0}) # Not fluent
    
    # Add augmented sentences
    processed_examples.extend(augmented_sentences)

    # Remove duplicates
    seen = set()
    unique_examples = []
    for example in processed_examples:
        if example['text'] not in seen:
            unique_examples.append(example)
            seen.add(example['text'])
    
    logging.info(f"Total examples after removing duplicates: {len(unique_examples)}")

    processed_dataset = Dataset.from_dict({
        k: [dic[k] for dic in unique_examples] for k in unique_examples[0]
    })

    # Create a train/test split
    logging.info("Splitting the dataset into train, validation, and test sets...")
    train_test_split = processed_dataset.train_test_split(test_size=0.2, seed=42)
    # Create a validation set from the new training set
    train_val_split = train_test_split["train"].train_test_split(test_size=0.1, seed=42)

    split_dataset = DatasetDict({
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": train_test_split["test"]
    })
    logging.info("Dataset splitting complete.")

    # Push the dataset to the Hugging Face Hub
    logging.info(f"Uploading the dataset to {args.dataset_name}...")
    split_dataset.push_to_hub(args.dataset_name)
    logging.info(f"Dataset successfully uploaded to {args.dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and upload the binary fluency dataset.")
    parser.add_argument("--dataset_name", type=str, default="binary-fluency-data", help="Name of the dataset on the Hugging Face Hub.")
    parser.add_argument("--max_augment_samples", type=int, default=None, help="For debugging, truncate the number of examples to augment.")
    parser.add_argument("--augmentations_per_example", type=int, default=1, help="The number of augmentations to apply to each example.")
    args = parser.parse_args()
    main(args)

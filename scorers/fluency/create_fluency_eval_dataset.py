"""
Create Fluency Evaluation Dataset with Edit Information

This script processes the fluency-data-augmented dataset to create an evaluation dataset
compatible with evaluate_fluency_scorer.py. It:
1. Loads the fluency-data-augmented dataset
2. Extracts edit information using DirectLatexdiffParser and fuzzy_post_process_edits
3. Filters to keep only examples with exactly 1 edit
4. Creates evaluation examples with original_sentence, inappropriate_part, rewritten_part
5. For insertions/deletions, adds symmetric context so string replacement still works
6. Validates that replacing inappropriate_part with rewritten_part produces text2
7. Assigns expected_score: 1.0 for edits from bad to good, 0.0 for edits from good to bad
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import contextlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Suppress loguru messages from sentsplit before importing
try:
    from loguru import logger as loguru_logger
    loguru_logger.disable("sentsplit")
except ImportError:
    pass

from ops.latexdiff_parser import DirectLatexdiffParser, fuzzy_post_process_edits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout, stderr, and logging output at both Python and OS level."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # Save original file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        # Also suppress logging from other libraries
        old_log_level = logging.root.level
        logging.root.setLevel(logging.CRITICAL)

        # Disable loguru if it's being used by sentsplit
        try:
            from loguru import logger as loguru_logger
            loguru_logger.disable("")
            loguru_disabled = True
        except ImportError:
            loguru_disabled = False

        try:
            # Redirect at Python level
            sys.stdout = devnull
            sys.stderr = devnull

            # Redirect at OS level (for subprocesses like latexdiff)
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)

            yield
        finally:
            # Restore OS-level file descriptors first
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

            # Restore Python-level streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.root.setLevel(old_log_level)

            # Re-enable loguru if it was disabled
            if loguru_disabled:
                try:
                    from loguru import logger as loguru_logger
                    loguru_logger.enable("")
                except ImportError:
                    pass




# Thread-local storage for parser instances (one per worker process)
_thread_local_parsers = {}

def get_or_create_parser(process_id: int) -> 'DirectLatexdiffParser':
    """Get or create a parser instance for this process."""
    if process_id not in _thread_local_parsers:
        with suppress_output():
            _thread_local_parsers[process_id] = DirectLatexdiffParser()
    return _thread_local_parsers[process_id]


def extract_edits(text1: str, text2: str, process_id: int = 0) -> tuple[List[Dict[str, Any]], str]:
    """
    Extract edit actions from text1 to text2 using DirectLatexdiffParser.

    Args:
        text1: Original text
        text2: Revised text
        process_id: Process ID for creating unique output directory

    Returns:
        Tuple of (list of edit dictionaries with 'before' and 'after' keys, before_revision text)
    """
    # Reuse the parser instance for this process
    parser = get_or_create_parser(process_id)

    try:
        # Create process-specific output directory
        output_dir = f'scorers/fluency/temp_output_proc_{process_id}'

        # Parse the diff between text1 and text2
        diff = parser.parse_latex_diff(text1, text2, output_dir)

        if diff is None or not diff.get('edit_actions'):
            return [], text1

        # Add before_revision for fuzzy_post_process_edits
        diff['before_revision'] = text1

        # Post-process edits to ensure they match the original text
        with suppress_output():
            fuzzy_post_process_edits([diff])

        edits = []
        for action in diff['edit_actions']:
            # Extract before and after text
            before_text = action.get('before', '')
            after_text = action.get('after', '')

            # Only include edits where something actually changed
            if before_text or after_text:
                edits.append({
                    'before': before_text,
                    'after': after_text,
                    'start_char_pos': action.get('start_char_pos', 0),
                    'end_char_pos': action.get('end_char_pos', 0)
                })

        return edits

    except Exception as e:
        return [], text1




def add_context_to_edit(text: str, edit: Dict[str, Any], context_words: int = 1) -> tuple[str, str, str]:
    """
    Add surrounding context to an edit to make its position clear, especially for insertions/deletions.
    The SAME context is added to both before and after parts so string replacement still works.

    Args:
        text: The full text containing the edit
        edit: Edit dictionary with 'before', 'after', 'start_char_pos', 'end_char_pos'
        context_words: Number of words to include before and after the edit

    Returns:
        Tuple of (original_sentence, before_with_context, after_with_context)
        Note: original_sentence is ALWAYS the full original text
    """
    before_text = edit.get('before', '') or ''
    after_text = edit.get('after', '') or ''
    start_pos = edit.get('start_char_pos', 0)
    end_pos = edit.get('end_char_pos', 0)

    # If both before and after have content, no need to add context
    if before_text.strip() and after_text.strip():
        return text, before_text, after_text

    # Extract context before the edit
    context_start = max(0, start_pos - 50)  # Look back up to 50 chars
    before_context = text[context_start:start_pos]

    # Extract context after the edit
    context_end = min(len(text), end_pos + 50)  # Look ahead up to 50 chars
    after_context = text[end_pos:context_end]

    # Split context into words
    before_words = before_context.split()
    after_words = after_context.split()

    # Get the requested number of context words
    context_before = ' '.join(before_words[-context_words:]) if before_words else ''
    context_after = ' '.join(after_words[:context_words]) if after_words else ''

    # Build the extended parts
    if context_before:
        context_before += ' '
    if context_after:
        context_after = ' ' + context_after

    # Create the parts with SAME context on both sides
    # This ensures string replacement still works
    before_with_context = context_before + before_text + context_after
    after_with_context = context_before + after_text + context_after

    # ALWAYS return the full original text as the original_sentence
    # This ensures consistency across all edits from the same source text
    return text, before_with_context.strip(), after_with_context.strip()


def apply_edit_to_text(text: str, edit: Dict[str, Any]) -> str:
    """
    Apply an edit to text and return the modified text.

    Args:
        text: The original text
        edit: Edit dictionary with 'before', 'after', 'start_char_pos', 'end_char_pos'

    Returns:
        The text with the edit applied
    """
    before_text = edit.get('before', '') or ''
    after_text = edit.get('after', '') or ''
    start_pos = edit.get('start_char_pos', 0)
    end_pos = edit.get('end_char_pos', 0)

    # Apply the edit: replace text[start_pos:end_pos] with after_text
    edited_text = text[:start_pos] + after_text + text[end_pos:]
    return edited_text


def create_eval_examples(text1: str, text2: str, label: int, example_id: str, process_id: int = 0) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Create evaluation examples from a fluency dataset entry.
    Only returns examples where exactly ONE edit was extracted and applying it produces text2.

    Args:
        text1: First text (could be bad or good depending on label)
        text2: Second text (could be good or bad depending on label)
        label: 1 if text1->text2 is bad->good, 0 if text1->text2 is good->bad
        example_id: Unique identifier for this example
        process_id: Process ID for parallel processing

    Returns:
        Tuple of (list with single evaluation example if valid, dict with filter statistics)
    """
    # Initialize statistics
    stats = {
        'total': 1,
        'no_edits': 0,
        'multiple_edits': 0,
        'empty_parts': 0,
        'edit_mismatch': 0,
        'valid': 0
    }

    # Extract edits from text1 to text2
    edits = extract_edits(text1, text2, process_id)

    if not edits:
        stats['no_edits'] = 1
        return [], stats

    # FILTER: Only use examples with exactly one edit
    if len(edits) != 1:
        stats['multiple_edits'] = 1
        logger.debug(f"Skipping {example_id}: found {len(edits)} edits, need exactly 1")
        return [], stats

    eval_examples = []

    # label=1 means text1 (bad) -> text2 (good), so edits should be fluent (score=1.0)
    # label=0 means text1 (good) -> text2 (bad), so edits should be non-fluent (score=0.0)
    expected_score = 1.0 if label == 1 else 0.0

    for i, edit in enumerate(edits):
        before_text = edit.get('before', '') or ''
        after_text = edit.get('after', '') or ''

        # Skip edits where both parts are empty
        if not before_text.strip() and not after_text.strip():
            stats['empty_parts'] = 1
            logger.debug(f"Skipping edit with both parts empty: {example_id}_edit_{i}")
            continue

        # VALIDATION: Apply the edit to text1 and check if it produces text2
        edited_text = apply_edit_to_text(text1, edit)
        if edited_text.strip() != text2.strip():
            stats['edit_mismatch'] = 1
            logger.debug(f"Skipping {example_id}: applying edit doesn't produce text2")
            logger.debug(f"  Expected: {text2[:100]}...")
            logger.debug(f"  Got:      {edited_text[:100]}...")
            continue

        # Add context if either part is empty (insertion or deletion)
        # This adds the SAME context to both parts so string replacement still works
        if not before_text.strip() or not after_text.strip():
            original_sentence, inappropriate_part, rewritten_part = add_context_to_edit(text1, edit)
        else:
            original_sentence =text1 
            inappropriate_part = before_text
            rewritten_part = after_text

        # Verify that string replacement works
        # Replace the inappropriate part with the rewritten part in original_sentence
        if inappropriate_part not in original_sentence:
            stats['edit_mismatch'] = 1
            logger.debug(f"Skipping {example_id}: inappropriate_part not found in original_sentence")
            continue

        reconstructed = original_sentence.replace(inappropriate_part, rewritten_part, 1)
        if reconstructed.strip() != text2.strip():
            stats['edit_mismatch'] = 1
            logger.debug(f"Skipping {example_id}: string replacement doesn't produce text2")
            logger.debug(f"  Expected: {text2[:100]}...")
            logger.debug(f"  Got:      {reconstructed[:100]}...")
            continue

        stats['valid'] = 1
        eval_examples.append({
            'id': f"{example_id}_edit_{i}",
            'original_sentence': original_sentence,
            'inappropriate_part': inappropriate_part,
            'rewritten_part': rewritten_part,
            'expected_score': expected_score,
            'description': f"{'Fluent' if label == 1 else 'Non-fluent'} edit from fluency-data-augmented"
        })

    return eval_examples, stats


def process_example_worker(args_tuple):
    """
    Worker function for processing a single example in parallel.

    Args:
        args_tuple: Tuple of (example_dict, split_name, idx, process_id)

    Returns:
        Tuple of (list of evaluation examples, statistics dict)
    """
    example, split_name, idx, process_id = args_tuple

    try:
        text1 = example['text1']
        text2 = example['text2']
        label = example['label']

        # Create evaluation examples
        eval_examples, stats = create_eval_examples(
            text1=text1,
            text2=text2,
            label=label,
            example_id=f"{split_name}_{idx}",
            process_id=process_id
        )

        return eval_examples, stats
    except Exception as e:
        # Log error but don't crash the process
        # Return empty list and error stats
        return [], {
            'total': 1,
            'no_edits': 0,
            'multiple_edits': 0,
            'empty_parts': 0,
            'edit_mismatch': 0,
            'valid': 0
        }


def main(args):
    logger.info(f"Loading dataset: {args.dataset_name}")

    # Load the fluency-data-augmented dataset
    dataset = load_dataset(args.dataset_name)

    logger.info(f"Dataset loaded with splits: {list(dataset.keys())}")

    # Process each split
    processed_splits = {}

    for split_name in dataset.keys():
        if args.splits and split_name not in args.splits:
            logger.info(f"Skipping split: {split_name}")
            continue

        logger.info(f"Processing split: {split_name}")
        split_data = dataset[split_name]

        # Limit examples if specified
        if args.max_examples:
            split_data = split_data.select(range(min(args.max_examples, len(split_data))))
            logger.info(f"Limited to {len(split_data)} examples")

        filtered_examples = []

        # Determine number of processes
        num_processes = args.num_workers if args.num_workers else max(1, cpu_count() - 1)
        logger.info(f"Using {num_processes} processes for parallel processing")

        # Prepare arguments for parallel processing
        # Assign process IDs cyclically to distribute work
        process_args = [
            (example, split_name, idx, idx % num_processes)
            for idx, example in enumerate(split_data)
        ]

        # Process examples in parallel
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                # Use imap_unordered for better memory efficiency and progress tracking
                results = list(tqdm(
                    pool.imap_unordered(process_example_worker, process_args, chunksize=10),
                    total=len(process_args),
                    desc=f"Processing {split_name}"
                ))
        else:
            # Single-process fallback
            results = [
                process_example_worker(args)
                for args in tqdm(process_args, desc=f"Processing {split_name}")
            ]

        # Collect evaluation examples and aggregate statistics
        all_eval_examples = []
        aggregated_stats = {
            'total': 0,
            'no_edits': 0,
            'multiple_edits': 0,
            'empty_parts': 0,
            'edit_mismatch': 0,
            'valid': 0
        }

        for eval_examples, stats in results:
            all_eval_examples.extend(eval_examples)
            # Aggregate stats
            for key in aggregated_stats:
                aggregated_stats[key] += stats.get(key, 0)

        logger.info(f"Processed {len(split_data)} examples, created {len(all_eval_examples)} eval examples")

        # Log filtering statistics
        logger.info("\n" + "="*80)
        logger.info(f"Filtering Statistics for {split_name}:")
        logger.info("="*80)
        logger.info(f"  Total examples processed: {aggregated_stats['total']}")
        logger.info(f"  Valid examples: {aggregated_stats['valid']} ({100*aggregated_stats['valid']/aggregated_stats['total']:.1f}%)")
        logger.info(f"  Filtered out:")
        logger.info(f"    No edits found: {aggregated_stats['no_edits']}")
        logger.info(f"    Multiple edits: {aggregated_stats['multiple_edits']}")
        logger.info(f"    Both edit parts empty: {aggregated_stats['empty_parts']}")
        logger.info(f"    Edit/replacement doesn't produce text2: {aggregated_stats['edit_mismatch']}")
        logger.info("="*80 + "\n")

        logger.info(f"Split {split_name}: Created {len(all_eval_examples)} evaluation examples "
                   f"from {len(split_data)} original examples")

        # Create dataset from examples
        if all_eval_examples:
            processed_splits[split_name] = Dataset.from_dict({
                'id': [ex['id'] for ex in all_eval_examples],
                'original_sentence': [ex['original_sentence'] for ex in all_eval_examples],
                'inappropriate_part': [ex['inappropriate_part'] for ex in all_eval_examples],
                'rewritten_part': [ex['rewritten_part'] for ex in all_eval_examples],
                'expected_score': [ex['expected_score'] for ex in all_eval_examples],
                'description': [ex['description'] for ex in all_eval_examples]
            })
        else:
            logger.warning(f"No evaluation examples created for split: {split_name}")

    if not processed_splits:
        logger.error("No evaluation examples were created!")
        return

    # Create DatasetDict
    eval_dataset = DatasetDict(processed_splits)

    # Print statistics
    logger.info("\n" + "="*80)
    logger.info("Dataset Statistics:")
    logger.info("="*80)
    for split_name, split_data in eval_dataset.items():
        fluent_count = sum(1 for ex in split_data if ex['expected_score'] == 1.0)
        non_fluent_count = sum(1 for ex in split_data if ex['expected_score'] == 0.0)
        logger.info(f"\n{split_name}:")
        logger.info(f"  Total examples: {len(split_data)}")
        logger.info(f"  Fluent edits (score=1.0): {fluent_count}")
        logger.info(f"  Non-fluent edits (score=0.0): {non_fluent_count}")
    logger.info("="*80 + "\n")

    # Push to Hub
    if args.output_dataset_name:
        logger.info(f"Pushing dataset to Hub: {args.output_dataset_name}")
        eval_dataset.push_to_hub(args.output_dataset_name)
        logger.info(f"Dataset successfully uploaded to {args.output_dataset_name}")

    # Save locally if specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset to local directory: {output_path}")
        eval_dataset.save_to_disk(str(output_path))
        logger.info(f"Dataset saved to {output_path}")

    # Cleanup temporary directories
    logger.info("Cleaning up temporary directories...")
    import shutil
    temp_base_dir = Path('scorers/fluency')
    for temp_dir in temp_base_dir.glob('temp_output_proc_*'):
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Removed {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not remove {temp_dir}: {e}")
    logger.info("Cleanup complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create fluency evaluation dataset with edit information (filtered to single-edit examples)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="timonziegenbein/fluency-agumented-pairs",
        help="Name of the source dataset on HuggingFace Hub"
    )
    parser.add_argument(
        "--output-dataset-name",
        type=str,
        default=None,
        help="Name for the evaluation dataset on HuggingFace Hub (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to save the evaluation dataset (optional)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Which splits to process (e.g., train test). If not specified, all splits are processed."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process per split (for testing)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel processing (default: cpu_count - 1)"
    )

    args = parser.parse_args()
    main(args)

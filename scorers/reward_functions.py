import torch
import logging
import nltk
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import weave
from functools import lru_cache
import hashlib

from scorers.local_scorers.semantic_similarity.semantic_similarity_scorer import SemanticSimilarityScorer

logger = logging.getLogger(__name__)

from scorers.appropriateness.appropriateness_scorer import AppropriatenessScorer

from ops.completion_processor import process_completion
from ops.prompt_processor import process_prompt
from ops.edit_applier import apply_edits_to_argument

# Cache to avoid duplicate calculations when both reward functions are called
_edit_scores_cache = {}

def _calculate_edit_scores(prompts, completions, semantic_similarity_scorer, human_like_scorer, fluency_scorer):
    """
    Core internal function that calculates all edit scores (both sparse and dense) in a single pass.
    Returns structured data that can be used by both local and dense reward functions.

    Uses a simple cache to avoid duplicate calculations when both reward functions are called
    in the same training step.

    Returns:
        tuple: (sparse_scores, dense_scores, all_perfect_edits, all_original_sentences, all_original_arguments)
    """
    # Create cache key from prompts and completions by hashing them
    # Use hash instead of tuple to avoid weave serialization issues
    cache_key = hashlib.md5((str(prompts) + str(completions)).encode()).hexdigest()

    # Check cache first
    if cache_key in _edit_scores_cache:
        return _edit_scores_cache[cache_key]

    sparse_scores = []
    dense_scores = []
    all_perfect_edits = []
    all_original_sentences = []
    all_original_arguments = []

    for prompt, completion in zip(prompts, completions):
        original_sentences, original_argument = process_prompt(prompt)
        all_original_sentences.append(original_sentences)
        all_original_arguments.append(original_argument)

        if not original_sentences:
            sparse_scores.append(0.0)
            dense_scores.append(0.0)
            all_perfect_edits.append([])
            continue

        sparse_edit_scores = []
        dense_edit_scores = []
        valid_edits = process_completion(completion, original_sentences)
        perfect_edits = []

        for edit in valid_edits:
            sentence_id = edit.get("sentence_id")
            inappropriate_part = edit.get("inappropriate_part")
            rewritten_part = edit.get("rewritten_part")

            # Map sentence_id to original_sentence (sentence_id is 1-indexed)
            if sentence_id is None or sentence_id < 1 or sentence_id > len(original_sentences):
                sparse_edit_scores.append(0.0)
                dense_edit_scores.append(0.0)
                continue

            original_sentence = original_sentences[sentence_id - 1]  # Convert to 0-indexed

            # Validate that the inappropriate_part exists in the original_sentence
            if inappropriate_part not in original_sentence:
                sparse_edit_scores.append(0.0)
                dense_edit_scores.append(0.0)
                continue

            # Calculate scores only for enabled scorers
            # Store scores in a list to only include enabled scorers
            enabled_scores = []

            if human_like_scorer is not None:
                human_like_reward = human_like_scorer.calculate_human_likeness(
                    original_argument, original_sentence, inappropriate_part, rewritten_part
                )
                enabled_scores.append(human_like_reward)

            if semantic_similarity_scorer is not None:
                semantic_similarity_reward, ss_score = semantic_similarity_scorer.calculate_semantic_similarity(
                    original_sentence, inappropriate_part, rewritten_part
                )
                enabled_scores.append(semantic_similarity_reward)

            if fluency_scorer is not None:
                fluency_reward = fluency_scorer.calculate_fluency(
                    original_sentence, inappropriate_part, rewritten_part
                )
                enabled_scores.append(fluency_reward)

            # If no scorers are enabled, treat as passing (edge case)
            if not enabled_scores:
                sparse_edit_scores.append(1.0)
                dense_edit_scores.append(1.0)
                edit["original_sentence"] = original_sentence
                perfect_edits.append(edit)
                continue

            # Sparse score: all enabled scorers must pass (binary)
            if any(score == 0.0 for score in enabled_scores):
                sparse_edit_scores.append(0.0)
            else:
                # This edit passed all checks - it's a perfect edit
                edit["original_sentence"] = original_sentence
                sparse_edit_scores.append(1.0)
                perfect_edits.append(edit)

            # Dense score: average of only the enabled scorers
            dense_score = sum(enabled_scores) / len(enabled_scores)
            dense_edit_scores.append(dense_score)

        sparse_score = sum(sparse_edit_scores) / len(sparse_edit_scores) if sparse_edit_scores else 0.0
        dense_score = sum(dense_edit_scores) / len(dense_edit_scores) if dense_edit_scores else 0.0

        sparse_scores.append(sparse_score)
        dense_scores.append(dense_score)
        all_perfect_edits.append(perfect_edits)

    # Store in cache before returning
    result = (sparse_scores, dense_scores, all_perfect_edits, all_original_sentences, all_original_arguments)
    _edit_scores_cache[cache_key] = result

    # Clear cache if it gets too large (keep only most recent 100 entries)
    if len(_edit_scores_cache) > 100:
        # Remove oldest entries (first 50)
        keys_to_remove = list(_edit_scores_cache.keys())[:50]
        for key in keys_to_remove:
            del _edit_scores_cache[key]

    return result

@weave.op(tracing_sample_rate=0.1)
def dense_local_appropriateness_reward(prompts, completions, semantic_similarity_scorer, human_like_scorer, fluency_scorer, **kwargs):
    """
    Dense local reward that returns the average of binary scores from enabled scorers.
    Only scorers that are not None are included in the calculation.

    This provides partial credit for edits that pass some but not all checks, enabling better
    gradient signals during training. For example, with all 3 scorers enabled:
    - Edit passing all 3 checks: score = 1.0
    - Edit passing 2/3 checks: score = 0.67
    - Edit passing 1/3 checks: score = 0.33
    - Edit passing 0/3 checks: score = 0.0

    With only 2 scorers enabled (e.g., semantic similarity and fluency):
    - Edit passing both checks: score = 1.0
    - Edit passing 1/2 checks: score = 0.5
    - Edit passing 0/2 checks: score = 0.0

    Note: All individual scorers remain binary/sparse - the "density" comes from averaging them
    rather than requiring all to pass. Disabled scorers (None) are ignored, not treated as passing.

    This is a lightweight wrapper around _calculate_edit_scores.
    """
    _, dense_scores, _, _, _ = _calculate_edit_scores(
        prompts, completions, semantic_similarity_scorer, human_like_scorer, fluency_scorer
    )
    return dense_scores

@weave.op(tracing_sample_rate=0.1)
def global_appropriateness_reward(prompts, completions, appropriateness_scorer, semantic_similarity_scorer, human_like_scorer, fluency_scorer, **kwargs):
    scores = []

    # Get perfect edits and processed prompts from _calculate_edit_scores
    _, _, all_perfect_edits, all_original_sentences, all_original_arguments = _calculate_edit_scores(
        prompts, completions, semantic_similarity_scorer, human_like_scorer, fluency_scorer
    )

    for idx in range(len(prompts)):
        original_sentences = all_original_sentences[idx]
        original_argument = all_original_arguments[idx]

        if not original_sentences:
            scores.append(0.0)
            continue

        perfect_edits = all_perfect_edits[idx]
        if not perfect_edits:
            # No perfect edits means the model didn't produce any valid high-quality edits
            # Return 0.0 instead of rewarding based on the original text's appropriateness
            scores.append(0.0)
            continue

        # Apply perfect edits to the argument
        modified_argument = apply_edits_to_argument(perfect_edits, original_sentences, original_argument)

        after_scores = appropriateness_scorer.get_appropriateness_scores(modified_argument)
        inappropriateness_after = after_scores.get('Inappropriateness', 0.0)
        scores.append(1.0 - inappropriateness_after)

    return scores

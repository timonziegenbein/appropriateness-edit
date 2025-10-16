import torch
import logging
import nltk
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import weave

from scorers.semantic_similarity.semantic_similarity_scorer import SemanticSimilarityScorer

logger = logging.getLogger(__name__)

from scorers.appropriateness.appropriateness_scorer import AppropriatenessScorer

from ops.completion_processor import process_completion
from ops.prompt_processor import process_prompt
from ops.edit_applier import apply_edits_to_argument

@weave.op()
def local_appropriateness_reward(prompts, completions, semantic_similarity_scorer, human_like_scorer, fluency_scorer, **kwargs):
    scores = []
    all_perfect_edits = []
    all_original_sentences = []
    all_original_arguments = []

    for prompt, completion in zip(prompts, completions):
        original_sentences, original_argument = process_prompt(prompt)
        all_original_sentences.append(original_sentences)
        all_original_arguments.append(original_argument)

        if not original_sentences:
            scores.append(0.0)
            all_perfect_edits.append([])
            continue

        edit_scores = []
        valid_edits = process_completion(completion, original_sentences)
        perfect_edits = []

        for edit in valid_edits:
            sentence_id = edit.get("sentence_id")
            inappropriate_part = edit.get("inappropriate_part")
            rewritten_part = edit.get("rewritten_part")

            # Map sentence_id to original_sentence (sentence_id is 1-indexed)
            if sentence_id is None or sentence_id < 1 or sentence_id > len(original_sentences):
                edit_scores.append(0.0)
                continue

            original_sentence = original_sentences[sentence_id - 1]  # Convert to 0-indexed

            # Validate that the inappropriate_part exists in the original_sentence
            if inappropriate_part not in original_sentence:
                edit_scores.append(0.0)
                continue

            # Check human-likeness first (fastest check)
            human_like_reward = human_like_scorer.calculate_human_likeness(original_argument, original_sentence, inappropriate_part, rewritten_part)
            if human_like_reward == 0.0:
                edit_scores.append(0.0)
                continue

            # Check semantic similarity second
            semantic_similarity_reward, _ = semantic_similarity_scorer.calculate_semantic_similarity(original_sentence, inappropriate_part, rewritten_part)
            if semantic_similarity_reward == 0.0:
                edit_scores.append(0.0)
                continue

            # Check fluency last (slowest check)
            fluency_reward = fluency_scorer.calculate_fluency(original_sentence, inappropriate_part, rewritten_part)
            if fluency_reward == 0.0:
                edit_scores.append(0.0)
                continue

            # This edit passed all checks - it's a perfect edit
            # Store the resolved original_sentence along with the edit
            edit["original_sentence"] = original_sentence
            edit_scores.append(1.0)
            perfect_edits.append(edit)

        score = sum(edit_scores) / len(edit_scores) if edit_scores else 0.0
        scores.append(score)
        all_perfect_edits.append(perfect_edits)

    return scores, all_perfect_edits, all_original_sentences, all_original_arguments

@weave.op()
def global_appropriateness_reward(prompts, completions, appropriateness_scorer, semantic_similarity_scorer, human_like_scorer, fluency_scorer, **kwargs):
    scores = []

    # Get perfect edits and processed prompts from local_appropriateness_reward
    _, all_perfect_edits, all_original_sentences, all_original_arguments = local_appropriateness_reward(
        prompts, completions, semantic_similarity_scorer, human_like_scorer, fluency_scorer, **kwargs
    )

    for idx in range(len(prompts)):
        original_sentences = all_original_sentences[idx]
        original_argument = all_original_arguments[idx]

        if not original_sentences:
            scores.append(0.0)
            continue

        before_scores = appropriateness_scorer.get_appropriateness_scores(original_argument)

        perfect_edits = all_perfect_edits[idx]
        if not perfect_edits:
            scores.append(1.0 - before_scores.get('Inappropriateness', 0.0))
            continue

        # Apply perfect edits to the argument
        modified_argument = apply_edits_to_argument(perfect_edits, original_sentences, original_argument)

        after_scores = appropriateness_scorer.get_appropriateness_scores(modified_argument)
        inappropriateness_after = after_scores.get('Inappropriateness', 0.0)
        scores.append(1.0 - inappropriateness_after)

    return scores

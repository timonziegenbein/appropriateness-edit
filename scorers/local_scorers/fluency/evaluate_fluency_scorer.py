"""
Weave Evaluation for Fluency Scorer

This script creates a comprehensive evaluation of the fluency scorer using Weights & Biases Weave.
It tests the scorer's ability to:
1. Identify fluent edits (no grammar errors introduced)
2. Reject non-fluent edits (grammar errors introduced)
3. Handle edge cases (empty strings, special characters, etc.)
"""

import sys
import os
from pathlib import Path

# Disable meta tensor usage in transformers to avoid loading issues
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import weave
import torch
import argparse
from typing import List, Dict, Any
from datasets import load_dataset
import logging
import json
from google import genai
import asyncio

from scorers.local_scorers.fluency.fluency_scorer import FluencyScorer

# Import dependencies for TraditionalFluencyScorer
import re
from transformers import pipeline, AutoTokenizer
from ops.latexdiff_parser import DirectLatexdiffParser, fuzzy_post_process_edits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TraditionalFluencyScorer:
    """
    Traditional rule-based fluency scorer using CoLA and grammar synthesis models.
    This scorer is kept in the evaluation script for comparison purposes.
    """

    def __init__(self, device):
        self.device = device
        self.checker, self.corrector = self._load_fluency_models(device)

    def _load_fluency_models(self, device):
        """Loads the fluency models."""
        fluency_checker = pipeline("text-classification", model="textattack/roberta-base-CoLA", device=device)
        fluency_corrector = pipeline("text2text-generation", "pszemraj/flan-t5-large-grammar-synthesis", device=device)
        return fluency_checker, fluency_corrector

    @weave.op()
    def _get_grammar_corrections(self, text):
        """
        Checks grammar and returns a list of corrections and the corrected text.
        """
        sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)
        corrections = []
        full_corrected_text = ""
        current_pos = 0
        parser = DirectLatexdiffParser()

        for sentence in sentences:
            results = self.checker(sentence)
            corrected_sentence = sentence
            if results[0]['label'] != 'LABEL_1' or (results[0]['label'] == 'LABEL_1' and results[0]['score'] < 0.9):
                corrected_sentence = self.corrector(sentence)[0]['generated_text']
                if corrected_sentence.strip() != sentence.strip():
                    diff = parser.parse_latex_diff(sentence, corrected_sentence, 'scorers/fluency/temp_output')
                    if diff is None:
                        logger.warning(f"Could not parse diff for sentence: {sentence}")
                        continue
                    diff['before_revision'] = sentence
                    if diff and diff['edit_actions']:
                        fuzzy_post_process_edits([diff])
                        for action in diff['edit_actions']:
                            corrections.append({
                                'start': current_pos + action['start_char_pos'],
                                'end': current_pos + action['end_char_pos'],
                                'original': action.get('before', ''),
                                'corrected': action.get('after', '')
                            })
            full_corrected_text += corrected_sentence + " "
            current_pos += len(sentence) + 1  # +1 for the space

        return corrections, full_corrected_text.strip()

    @weave.op()
    def calculate_fluency(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> float:
        if not isinstance(original_sentence, str) or len(original_sentence.strip()) == 0:
            logger.warning("Received empty or invalid text in calculate_fluency.")
            return 0.0

        rewritten_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)

        if self.checker is None or self.corrector is None:
            logger.error("Fluency checker or corrector not loaded correctly.")
            return 0.0

        try:
            # Get grammar corrections on the rewritten sentence
            corrections, corrected_text = self._get_grammar_corrections(rewritten_sentence)

            if not corrections:
                return 1.0

            # The user's edit is the replacement string in the rewritten_sentence
            edit_start = original_sentence.find(inappropriate_part) - 2
            edit_end = edit_start + len(rewritten_part) + 3

            for correction in corrections:
                # Check for overlap
                if max(edit_start, correction['start']) <= min(edit_end, correction['end']):
                    return 0.0

            return 1.0

        except Exception as e:
            logger.error(f"Error in calculate_fluency: {e}")
            return 0.0


# Define the evaluation dataset examples
FLUENCY_TEST_CASES = [
    # Test Case 1: Fluent edit - grammatically correct replacement
    {
        "original_sentence": "The quick brown fox jumps over the lazy dog.",
        "inappropriate_part": "quick brown",
        "rewritten_part": "swift red",
        "expected_score": 1.0,
        "description": "Fluent edit: Simple adjective replacement maintaining grammar"
    },

    # Test Case 2: Non-fluent edit - introduces grammar error
    {
        "original_sentence": "She is going to the store.",
        "inappropriate_part": "is going",
        "rewritten_part": "go",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Creates grammatical error ('She go to the store')"
    },

    # Test Case 3: Fluent edit - maintains subject-verb agreement
    {
        "original_sentence": "The students are studying for their exams.",
        "inappropriate_part": "studying for",
        "rewritten_part": "preparing for",
        "expected_score": 1.0,
        "description": "Fluent edit: Synonym replacement preserving grammar"
    },

    # Test Case 4: Non-fluent edit - breaks verb tense consistency
    {
        "original_sentence": "Yesterday, I walked to the park.",
        "inappropriate_part": "walked",
        "rewritten_part": "walk",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Inconsistent tense ('Yesterday, I walk')"
    },

    # Test Case 5: Fluent edit - proper article usage
    {
        "original_sentence": "I need a pencil to write.",
        "inappropriate_part": "a pencil",
        "rewritten_part": "a pen",
        "expected_score": 1.0,
        "description": "Fluent edit: Article agreement maintained"
    },

    # Test Case 6: Non-fluent edit - article-noun mismatch
    {
        "original_sentence": "She bought a car yesterday.",
        "inappropriate_part": "a car",
        "rewritten_part": "an car",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Incorrect article usage ('an car')"
    },

    # Test Case 7: Fluent edit - complete phrase replacement
    {
        "original_sentence": "This argument is completely invalid.",
        "inappropriate_part": "completely invalid",
        "rewritten_part": "not well-supported",
        "expected_score": 1.0,
        "description": "Fluent edit: Complete phrase replacement"
    },

    # Test Case 8: Fluent edit - removing redundancy
    {
        "original_sentence": "In my personal opinion, this is wrong.",
        "inappropriate_part": "In my personal opinion",
        "rewritten_part": "I think",
        "expected_score": 1.0,
        "description": "Fluent edit: Simplification while maintaining grammar"
    },

    # Test Case 9: Non-fluent edit - incomplete sentence
    {
        "original_sentence": "The government should take action immediately.",
        "inappropriate_part": "should take action",
        "rewritten_part": "should",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Creates incomplete meaning"
    },

    # Test Case 10: Fluent edit - formal to informal
    {
        "original_sentence": "The individual is demonstrating poor behavior.",
        "inappropriate_part": "The individual is demonstrating",
        "rewritten_part": "This person shows",
        "expected_score": 1.0,
        "description": "Fluent edit: Tone change with proper grammar"
    },

    # Test Case 11: Fluent edit - proper punctuation
    {
        "original_sentence": "However, I disagree with this statement.",
        "inappropriate_part": "However, I disagree",
        "rewritten_part": "Nevertheless, I disagree",
        "expected_score": 1.0,
        "description": "Fluent edit: Transition word replacement with punctuation"
    },

    # Test Case 12: Non-fluent edit - missing punctuation
    {
        "original_sentence": "First, we need to consider the evidence.",
        "inappropriate_part": "First, we",
        "rewritten_part": "First we",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Missing comma after transition word"
    },

    # Test Case 13: Fluent edit - complex sentence structure
    {
        "original_sentence": "Although the data suggests otherwise, many people believe this claim.",
        "inappropriate_part": "many people believe",
        "rewritten_part": "numerous individuals accept",
        "expected_score": 1.0,
        "description": "Fluent edit: Maintaining complex sentence structure"
    },

    # Test Case 14: Non-fluent edit - introduces word duplication
    {
        "original_sentence": "The student submitted the assignment on time.",
        "inappropriate_part": "submitted the",
        "rewritten_part": "submitted the the",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Word duplication introduced ('the the')"
    },

    # Test Case 15: Fluent edit - fixes word duplication
    {
        "original_sentence": "The student submitted the the assignment on time.",
        "inappropriate_part": "the the",
        "rewritten_part": "the",
        "expected_score": 1.0,
        "description": "Fluent edit: Removes word duplication"
    },

    # Test Case 16: Non-fluent edit - missing article
    {
        "original_sentence": "I need to buy a new computer.",
        "inappropriate_part": "a new",
        "rewritten_part": "new",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Missing article ('I need to buy new computer')"
    },

    # Test Case 17: Fluent edit - adds missing article
    {
        "original_sentence": "I need to buy new computer.",
        "inappropriate_part": "buy new",
        "rewritten_part": "buy a new",
        "expected_score": 1.0,
        "description": "Fluent edit: Adds missing article"
    },

    # Test Case 18: Non-fluent edit - word order error
    {
        "original_sentence": "She always goes to the library.",
        "inappropriate_part": "always goes",
        "rewritten_part": "goes always",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Incorrect word order ('She goes always to')"
    },

    # Test Case 19: Fluent edit - fixes word order
    {
        "original_sentence": "She goes always to the library.",
        "inappropriate_part": "goes always",
        "rewritten_part": "always goes",
        "expected_score": 1.0,
        "description": "Fluent edit: Corrects word order"
    },

    # Test Case 20: Non-fluent edit - subject-verb disagreement
    {
        "original_sentence": "The team is working hard.",
        "inappropriate_part": "is working",
        "rewritten_part": "are working",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Subject-verb disagreement (team is singular)"
    },

    # Test Case 21: Fluent edit - fixes subject-verb disagreement
    {
        "original_sentence": "The team are working hard.",
        "inappropriate_part": "are working",
        "rewritten_part": "is working",
        "expected_score": 1.0,
        "description": "Fluent edit: Corrects subject-verb agreement"
    },

    # Test Case 22: Non-fluent edit - double negative
    {
        "original_sentence": "I don't have any money.",
        "inappropriate_part": "don't have any",
        "rewritten_part": "don't have no",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Double negative ('don't have no')"
    },

    # Test Case 23: Fluent edit - fixes double negative
    {
        "original_sentence": "I don't have no money.",
        "inappropriate_part": "don't have no",
        "rewritten_part": "don't have any",
        "expected_score": 1.0,
        "description": "Fluent edit: Removes double negative"
    },

    # Test Case 24: Non-fluent edit - missing verb
    {
        "original_sentence": "The cat is sleeping on the couch.",
        "inappropriate_part": "is sleeping",
        "rewritten_part": "sleeping",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Missing auxiliary verb ('The cat sleeping')"
    },

    # Test Case 25: Fluent edit - adds missing verb
    {
        "original_sentence": "The cat sleeping on the couch.",
        "inappropriate_part": "cat sleeping",
        "rewritten_part": "cat is sleeping",
        "expected_score": 1.0,
        "description": "Fluent edit: Adds missing auxiliary verb"
    },

    # Test Case 26: Non-fluent edit - wrong preposition
    {
        "original_sentence": "I am good at playing chess.",
        "inappropriate_part": "good at",
        "rewritten_part": "good in",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Wrong preposition ('good in playing')"
    },

    # Test Case 27: Fluent edit - corrects preposition
    {
        "original_sentence": "I am good in playing chess.",
        "inappropriate_part": "good in",
        "rewritten_part": "good at",
        "expected_score": 1.0,
        "description": "Fluent edit: Corrects preposition"
    },

    # Test Case 28: Non-fluent edit - redundant words
    {
        "original_sentence": "He returned back home.",
        "inappropriate_part": "returned back",
        "rewritten_part": "returned back again",
        "expected_score": 0.0,
        "description": "Non-fluent edit: Adds redundancy ('returned back again')"
    }
]


class FluencyScorerModel(weave.Model):
    """Weave Model wrapper for FluencyScorer"""

    device: str

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(device=device)
        # Don't store models as attributes - use a global cache instead
        self._scorer_cache_key = f"fluency_scorer_{device}"

    def _get_or_create_scorer(self):
        """Get scorer from global cache or create it"""
        global _SCORER_CACHE
        if not hasattr(self, '_scorer_cache_key'):
            self._scorer_cache_key = f"fluency_scorer_{self.device}"

        if self._scorer_cache_key not in _SCORER_CACHE:
            logger.error(f"FluencyScorer not found in cache for device: {self.device}")
            logger.error(f"This should have been pre-loaded. Cache keys: {list(_SCORER_CACHE.keys())}")
            raise RuntimeError(
                f"FluencyScorer was not pre-loaded for device {self.device}. "
                "This is a programming error - the scorer should be loaded before evaluation starts."
            )

        return _SCORER_CACHE[self._scorer_cache_key]

    @weave.op()
    def predict(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> Dict[str, Any]:
        """
        Score the fluency of an edit.

        Args:
            original_sentence: The original sentence containing the inappropriate part
            inappropriate_part: The part of the sentence to be replaced
            rewritten_part: The replacement text

        Returns:
            Dictionary containing the fluency score and metadata
        """
        # Get scorer from cache - this avoids Weave trying to serialize it
        scorer = self._get_or_create_scorer()

        score = scorer.calculate_fluency(
            original_sentence=original_sentence,
            inappropriate_part=inappropriate_part,
            rewritten_part=rewritten_part
        )

        # Construct the rewritten sentence for display
        rewritten_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)

        return {
            "fluency_score": score,
            "rewritten_sentence": rewritten_sentence
        }


class ModernBERTFluencyModel(weave.Model):
    """Weave Model wrapper for ModernBERT fine-tuned fluency classifier"""

    model_path: str
    device: str

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path=model_path, device=device)
        self._model_cache_key = f"modernbert_{model_path}_{device}"

    def _get_or_create_model(self):
        """Get model from global cache or create it"""
        global _SCORER_CACHE
        if self._model_cache_key not in _SCORER_CACHE:
            logger.error(f"ModernBERT model not found in cache: {self._model_cache_key}")
            logger.error(f"This should have been pre-loaded. Cache keys: {list(_SCORER_CACHE.keys())}")
            raise RuntimeError(
                f"ModernBERT model was not pre-loaded. "
                "This is a programming error - the model should be loaded before evaluation starts."
            )
        return _SCORER_CACHE[self._model_cache_key]

    @weave.op()
    @torch._dynamo.disable()
    def predict(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> Dict[str, Any]:
        """
        Score the fluency of an edit using ModernBERT.

        Args:
            original_sentence: The original sentence containing the inappropriate part
            inappropriate_part: The part of the sentence to be replaced
            rewritten_part: The replacement text

        Returns:
            Dictionary containing the fluency score and metadata
        """
        # Get model and tokenizer from cache
        model_data = self._get_or_create_model()
        model = model_data['model']
        tokenizer = model_data['tokenizer']

        # Construct the before and after sentences
        text_before = original_sentence
        text_after = original_sentence.replace(inappropriate_part, rewritten_part, 1)

        # Tokenize
        inputs = tokenizer(
            text_before,
            text_after,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        # Convert to fluency score (1 = fluent, 0 = non-fluent)
        fluency_score = float(predicted_class)

        return {
            "fluency_score": fluency_score,
            "rewritten_sentence": text_after
        }


# Global cache to store scorers outside of Weave's serialization scope
_SCORER_CACHE = {}


class RandomFluencyModel(weave.Model):
    """Random baseline that predicts 0.0 or 1.0 with equal probability"""

    seed: int = 42

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        import random
        random.seed(seed)

    @weave.op()
    def predict(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> Dict[str, Any]:
        """
        Randomly predict fluency score.

        Args:
            original_sentence: The original sentence containing the inappropriate part
            inappropriate_part: The part of the sentence to be replaced
            rewritten_part: The replacement text

        Returns:
            Dictionary containing a random fluency score
        """
        import random
        fluency_score = float(random.choice([0, 1]))
        rewritten_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)

        return {
            "fluency_score": fluency_score,
            "rewritten_sentence": rewritten_sentence
        }


class AlwaysFluentModel(weave.Model):
    """Baseline that always predicts 1.0 (fluent)"""

    @weave.op()
    def predict(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> Dict[str, Any]:
        """
        Always predict fluent (1.0).

        Args:
            original_sentence: The original sentence containing the inappropriate part
            inappropriate_part: The part of the sentence to be replaced
            rewritten_part: The replacement text

        Returns:
            Dictionary containing fluency score of 1.0
        """
        rewritten_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)

        return {
            "fluency_score": 1.0,
            "rewritten_sentence": rewritten_sentence
        }


class AlwaysNonFluentModel(weave.Model):
    """Baseline that always predicts 0.0 (non-fluent)"""

    @weave.op()
    def predict(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> Dict[str, Any]:
        """
        Always predict non-fluent (0.0).

        Args:
            original_sentence: The original sentence containing the inappropriate part
            inappropriate_part: The part of the sentence to be replaced
            rewritten_part: The replacement text

        Returns:
            Dictionary containing fluency score of 0.0
        """
        rewritten_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)

        return {
            "fluency_score": 0.0,
            "rewritten_sentence": rewritten_sentence
        }


class LanguageToolFluencyModel(weave.Model):
    """Weave Model wrapper for LanguageTool-based fluency classifier"""

    language: str = "en-US"

    def __init__(self, language: str = "en-US"):
        super().__init__(language=language)
        self._tool_cache_key = f"languagetool_{language}"

    def _get_or_create_tool(self):
        """Get LanguageTool from global cache or create it"""
        global _SCORER_CACHE
        if self._tool_cache_key not in _SCORER_CACHE:
            logger.error(f"LanguageTool not found in cache: {self._tool_cache_key}")
            logger.error(f"This should have been pre-loaded. Cache keys: {list(_SCORER_CACHE.keys())}")
            raise RuntimeError(
                f"LanguageTool was not pre-loaded. "
                "This is a programming error - the tool should be loaded before evaluation starts."
            )
        return _SCORER_CACHE[self._tool_cache_key]

    @weave.op()
    def predict(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> Dict[str, Any]:
        """
        Score the fluency of an edit using LanguageTool.

        Args:
            original_sentence: The original sentence containing the inappropriate part
            inappropriate_part: The part of the sentence to be replaced
            rewritten_part: The replacement text

        Returns:
            Dictionary containing the fluency score and metadata
        """
        # Get LanguageTool from cache
        tool = self._get_or_create_tool()

        # Construct the rewritten sentence
        rewritten_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)

        # Check both sentences for errors
        original_matches = tool.check(original_sentence)
        rewritten_matches = tool.check(rewritten_sentence)

        # Count errors
        original_error_count = len(original_matches)
        rewritten_error_count = len(rewritten_matches)

        # Logic: Edit is fluent if it doesn't increase error count
        # This means: rewritten_error_count <= original_error_count
        fluency_score = 1.0 if rewritten_error_count <= original_error_count else 0.0

        return {
            "fluency_score": fluency_score,
            "rewritten_sentence": rewritten_sentence,
            "original_error_count": original_error_count,
            "rewritten_error_count": rewritten_error_count,
            "error_diff": rewritten_error_count - original_error_count
        }


class LLMFluencyClassifier(weave.Model):
    """
    LLM-based fluency classifier using Google's Gemini API.
    Uses structured prompting to evaluate if an edit maintains sentence fluency.
    """

    model_name: str = "gemini-2.0-flash-exp"
    system_prompt: str = "You are a meticulous language editor. Your task is to evaluate a suggested text modification for its impact on sentence fluency."
    timeout: int = 60  # Timeout in seconds for API calls

    @weave.op()
    async def predict(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> Dict[str, Any]:
        """
        Evaluate fluency of an edit using an LLM.

        Args:
            original_sentence: The original sentence containing the inappropriate part
            inappropriate_part: The part of the sentence to be replaced
            rewritten_part: The replacement text

        Returns:
            Dictionary containing the fluency score and metadata
        """
        # Construct the scoring prompt
        scoring_prompt = f"""**Objective:** Given an original sentence, a specific part to be replaced, and the replacement text, you must determine if the resulting new sentence is **at least as fluent** as the original.

**Definition of Fluency:** A sentence is considered "fluent" if it is grammatically correct, natural-sounding, easy to read, and clear in its meaning. An edit is acceptable if it maintains or improves fluency. An edit is unacceptable if it harms fluency in any way (e.g., makes it ungrammatical, awkward, or less clear).

---
### Instructions
1.  **Reconstruct the Sentence:** Mentally replace the `inappropriate_part` with the `rewritten_part` in the `original_sentence` to create the `New Sentence`.
2.  **Compare:** Carefully compare the `Original Sentence` and the `New Sentence`.
3.  **Evaluate:** Judge whether the `New Sentence` is at least as fluent as the `Original Sentence`.
4.  **Respond:** Provide your answer in a JSON object with two keys:
    * `"is_fluent"`: A boolean (`true` or `false`).
    * `"reason"`: A brief, one-sentence explanation for your decision.

---
### Task
Now, evaluate the following input:

* **Original Sentence:** {original_sentence}
* **Substring to Replace:** {inappropriate_part}
* **Replacement Substring:** {rewritten_part}

**Your JSON Output:**"""

        # Initialize Google GenAI client
        google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1  # Start with 1 second

        for attempt in range(max_retries):
            try:
                # Make API call with structured output and timeout
                response = await asyncio.wait_for(
                    google_client.aio.models.generate_content(
                        model=self.model_name,
                        contents=scoring_prompt,
                        config={
                            "system_instruction": self.system_prompt,
                            "response_mime_type": "application/json"
                        }
                    ),
                    timeout=self.timeout
                )

                # Parse the JSON response
                result = json.loads(response.text)

                # Construct the rewritten sentence for display
                rewritten_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)

                # Convert boolean to binary score (1.0 or 0.0)
                fluency_score = 1.0 if result.get("is_fluent", False) else 0.0

                return {
                    "fluency_score": fluency_score,
                    "rewritten_sentence": rewritten_sentence,
                    "reason": result.get("reason", "No reason provided")
                }

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Timeout after {max_retries} attempts")
                    return {
                        "fluency_score": 0.0,
                        "rewritten_sentence": original_sentence.replace(inappropriate_part, rewritten_part, 1),
                        "reason": f"Error: Timeout after {max_retries} attempts"
                    }

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error on attempt {attempt + 1}/{max_retries}: {e}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Error in LLM fluency evaluation after {max_retries} attempts: {e}")
                    return {
                        "fluency_score": 0.0,
                        "rewritten_sentence": original_sentence.replace(inappropriate_part, rewritten_part, 1),
                        "reason": f"Error: {str(e)}"
                    }


class AccuracyScorer(weave.Scorer):
    """
    Scorer that evaluates whether the fluency prediction matches the expected score.
    Since fluency scores are binary (0.0 or 1.0), this checks exact match accuracy.
    """

    @weave.op()
    def score(self, expected_score: float, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score whether the fluency prediction exactly matches the expected score.

        Args:
            expected_score: The expected fluency score (0.0 or 1.0)
            model_output: Dictionary containing the model's prediction with 'fluency_score' key

        Returns:
            Dictionary with accuracy metrics
        """
        predicted_score = model_output["fluency_score"]
        is_correct = predicted_score == expected_score

        return {
            "correct": is_correct
        }

class PrecisionScorer(weave.Scorer):
    @weave.op()
    def score(self, expected_score: float, model_output: Dict[str, Any]) -> Dict[str, Any]:
        predicted_score = model_output["fluency_score"]
        true_positive = bool(expected_score and predicted_score)
        false_positive = bool(predicted_score and not expected_score)  # Fixed: FP = predicted 1 but expected 0
        return {
            "true_positive": true_positive,
            "false_positive": false_positive
        }

    @weave.op()
    def summarize(self, score_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_true_positive = sum(score["true_positive"] for score in score_rows)
        total_false_positive = sum(score["false_positive"] for score in score_rows)
        denominator = total_true_positive + total_false_positive
        precision = total_true_positive / denominator if denominator > 0 else 0
        return {"precision": precision}

class RecallScorer(weave.Scorer):
    @weave.op()
    def score(self, expected_score: float, model_output: Dict[str, Any]) -> Dict[str, Any]:
        predicted_score = model_output["fluency_score"]
        true_positive = bool(expected_score and predicted_score)
        false_negative = bool(expected_score and not predicted_score)  # Fixed: FN = expected 1 but predicted 0
        return {
            "true_positive": true_positive,
            "false_negative": false_negative
        }

    @weave.op()
    def summarize(self, score_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_true_positive = sum(score["true_positive"] for score in score_rows)
        total_false_negative = sum(score["false_negative"] for score in score_rows)
        denominator = total_true_positive + total_false_negative
        recall = total_true_positive / denominator if denominator > 0 else 0
        return {"recall": recall}


class F1Scorer(weave.Scorer):
    @weave.op()
    def score(self, expected_score: float, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """Compute per-example metrics needed for precision, recall, and F1"""
        predicted_score = model_output["fluency_score"]
        true_positive = bool(expected_score and predicted_score)
        false_positive = bool(predicted_score and not expected_score)
        false_negative = bool(expected_score and not predicted_score)
        return {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative
        }

    @weave.op()
    def summarize(self, score_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics and compute precision, recall, and F1"""
        total_true_positive = sum(score["true_positive"] for score in score_rows)
        total_false_positive = sum(score["false_positive"] for score in score_rows)
        total_false_negative = sum(score["false_negative"] for score in score_rows)

        # Calculate precision
        precision_denominator = total_true_positive + total_false_positive
        precision = total_true_positive / precision_denominator if precision_denominator > 0 else 0

        # Calculate recall
        recall_denominator = total_true_positive + total_false_negative
        recall = total_true_positive / recall_denominator if recall_denominator > 0 else 0

        # Calculate F1
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


def create_evaluation_dataset(test_cases: List[Dict[str, Any]], dataset_name: str = "fluency_test_cases"):
    """Create a Weave dataset from test cases"""

    # Format examples for Weave
    examples = []
    for i, case in enumerate(test_cases):
        example = {
            "id": f"test_case_{i+1}",
            "original_sentence": case["original_sentence"],
            "inappropriate_part": case["inappropriate_part"],
            "rewritten_part": case["rewritten_part"],
            "expected_score": case["expected_score"],
            "description": case["description"]
        }
        examples.append(example)

    # Publish dataset to Weave
    dataset = weave.Dataset(name=dataset_name, rows=examples)
    weave.publish(dataset)

    logger.info(f"Created Weave dataset '{dataset_name}' with {len(examples)} test cases")
    return dataset


async def run_evaluation_async(args):
    """Run the evaluation asynchronously"""
    # Initialize Weave
    weave.init(args.project)
    logger.info(f"Initialized Weave project: {args.project}")

    # Prepare test cases based on data source
    if args.eval_dataset_name:
        # Load from HuggingFace dataset created by create_fluency_eval_dataset.py
        logger.info(f"Loading evaluation dataset from: {args.eval_dataset_name}")
        hf_dataset = load_dataset(args.eval_dataset_name, split=args.eval_dataset_split)

        # Limit examples if specified
        if args.max_eval_examples:
            # Shuffle with seed for reproducible sampling, then select first N
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(min(args.max_eval_examples, len(hf_dataset))))

        # Convert to list of test cases
        all_test_cases = []
        for example in hf_dataset:
            all_test_cases.append({
                "id": example.get("id", "unknown"),
                "original_sentence": example["original_sentence"],
                "inappropriate_part": example["inappropriate_part"],
                "rewritten_part": example["rewritten_part"],
                "expected_score": example["expected_score"],
                "description": example.get("description", "")
            })

        logger.info(f"Loaded {len(all_test_cases)} test cases from {args.eval_dataset_name}/{args.eval_dataset_split}")
    else:
        # Use default test cases
        all_test_cases = FLUENCY_TEST_CASES.copy()

    # Create evaluation dataset
    dataset = create_evaluation_dataset(all_test_cases, "fluency_test_cases")

    # Initialize the model based on the flags
    if args.use_random_baseline:
        # Use random baseline
        logger.info("Initializing RandomFluencyModel...")
        model = RandomFluencyModel(seed=args.random_seed)
        logger.info(f"Initialized RandomFluencyModel with seed: {args.random_seed}")
    elif args.use_always_fluent:
        # Use always-fluent baseline
        logger.info("Initializing AlwaysFluentModel...")
        model = AlwaysFluentModel()
        logger.info("Initialized AlwaysFluentModel (always predicts 1.0)")
    elif args.use_always_nonfluent:
        # Use always-non-fluent baseline
        logger.info("Initializing AlwaysNonFluentModel...")
        model = AlwaysNonFluentModel()
        logger.info("Initialized AlwaysNonFluentModel (always predicts 0.0)")
    elif args.use_languagetool:
        # Use LanguageTool-based classifier
        logger.info("Pre-loading LanguageTool...")
        import language_tool_python
        tool = language_tool_python.LanguageTool(args.languagetool_language)
        cache_key = f"languagetool_{args.languagetool_language}"
        _SCORER_CACHE[cache_key] = tool
        logger.info(f"LanguageTool pre-loaded successfully for language: {args.languagetool_language}")

        # Initialize the model wrapper
        model = LanguageToolFluencyModel(language=args.languagetool_language)
        logger.info(f"Initialized LanguageToolFluencyModel")
    elif args.use_llm:
        # Use LLM-based classifier
        model = LLMFluencyClassifier(
            model_name=args.llm_model_name,
            timeout=args.llm_timeout
        )
        logger.info(f"Initialized LLMFluencyClassifier with model: {args.llm_model_name}, timeout: {args.llm_timeout}s")
    elif args.use_traditional_scorer:
        # Use traditional fluency scorer (rule-based)
        # PRE-LOAD the scorer BEFORE creating the Weave model
        # This ensures it loads in the main thread, not in an async context
        logger.info(f"Pre-loading TraditionalFluencyScorer on device: {args.device}")
        _SCORER_CACHE[f"fluency_scorer_{args.device}"] = TraditionalFluencyScorer(args.device)
        logger.info(f"TraditionalFluencyScorer pre-loaded successfully")

        # Initialize the model
        model = FluencyScorerModel(device=args.device)
        logger.info(f"Initialized FluencyScorerModel (traditional) on device: {args.device}")
    elif not args.modernbert_path or args.modernbert_path == "":
        # Use default ModernBERT-based fluency scorer
        logger.info(f"Pre-loading default FluencyScorer (ModernBERT) on device: {args.device}")
        _SCORER_CACHE[f"fluency_scorer_{args.device}"] = FluencyScorer(args.device)
        logger.info(f"Default FluencyScorer pre-loaded successfully")

        # Initialize the model
        model = FluencyScorerModel(device=args.device)
        logger.info(f"Initialized FluencyScorerModel (ModernBERT) on device: {args.device}")
    elif args.modernbert_path:
        # Use ModernBERT fine-tuned classifier
        logger.info(f"Pre-loading ModernBERT model from: {args.modernbert_path}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Load tokenizer from base model (checkpoint may not have tokenizer files)
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        # Load fine-tuned model weights from checkpoint
        # Use eager attention to avoid flash attention compatibility issues
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            args.modernbert_path,
            attn_implementation="eager"
        )
        bert_model.to(args.device)
        bert_model.eval()

        # Cache the model
        cache_key = f"modernbert_{args.modernbert_path}_{args.device}"
        _SCORER_CACHE[cache_key] = {
            'model': bert_model,
            'tokenizer': tokenizer
        }
        logger.info(f"ModernBERT model pre-loaded successfully")

        # Initialize the model wrapper
        model = ModernBERTFluencyModel(model_path=args.modernbert_path, device=args.device)
        logger.info(f"Initialized ModernBERTFluencyModel from: {args.modernbert_path}")

    # Create evaluation with accuracy, precision, recall, and F1 scorers
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[
            AccuracyScorer(),
            F1Scorer(),
        ],
        name="fluency_scorer_evaluation"
    )

    logger.info("Starting evaluation...")

    # Run evaluation (async) with optional display name
    if args.run_name:
        logger.info(f"Evaluation run name: {args.run_name}")
        results = await evaluation.evaluate(model, __weave={"display_name": args.run_name})
    else:
        results = await evaluation.evaluate(model)

    # Print summary
    print("\n" + "="*80)
    print("FLUENCY SCORER EVALUATION RESULTS")
    print("="*80)
    if args.run_name:
        print(f"\nRun name: {args.run_name}")
    print(f"Total test cases: {len(all_test_cases)}")
    print(f"Device used: {args.device}")

    # Access the results - handle different result formats
    print("\nScorer Results:")

    # Try different ways to access scores based on Weave version
    if hasattr(results, 'scores'):
        for scorer_name, score_value in results.scores.items():
            print(f"  {scorer_name}: {score_value}")
    elif isinstance(results, dict):
        for key, value in results.items():
            if 'scorer' in key.lower() or 'score' in key.lower():
                print(f"  {key}: {value}")
    else:
        # Print the raw results for inspection
        print(f"  Results object: {results}")
        print(f"  Available attributes: {[attr for attr in dir(results) if not attr.startswith('_')]}")

    # Try to get URL
    url = None
    if hasattr(results, 'url'):
        url = results.url
    elif hasattr(results, 'run_url'):
        url = results.run_url

    if url:
        print(f"\nView detailed results in Weave: {url}")
    else:
        print(f"\nView detailed results in Weave: https://wandb.ai/weave/{args.project}")

    print("="*80 + "\n")

    logger.info("Evaluation complete!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Fluency Scorer using Weave")
    parser.add_argument("--project", type=str, default="fluency-scorer-eval",
                       help="Weave project name")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Display name for this evaluation run (shown in Weave UI)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run the scorer on (only used if --use-llm is not set)")

    # Dataset options
    parser.add_argument("--eval-dataset-name", type=str, default=None,
                       help="Name of the evaluation dataset on HuggingFace Hub (created by create_fluency_eval_dataset.py)")
    parser.add_argument("--eval-dataset-split", type=str, default="test",
                       help="Which split of the evaluation dataset to use (default: test)")
    parser.add_argument("--max-eval-examples", type=int, default=None,
                       help="Maximum number of examples to load from eval dataset (default: 1000 for efficiency)")

    # Model options (mutually exclusive)
    # Baseline models
    parser.add_argument("--use-random-baseline", action="store_true",
                       help="Use random baseline (predicts 0 or 1 randomly)")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for random baseline (default: 42)")
    parser.add_argument("--use-always-fluent", action="store_true",
                       help="Use always-fluent baseline (always predicts 1.0)")
    parser.add_argument("--use-always-nonfluent", action="store_true",
                       help="Use always-non-fluent baseline (always predicts 0.0)")
    # Other models
    parser.add_argument("--use-languagetool", action="store_true",
                       help="Use LanguageTool rule-based grammar checker")
    parser.add_argument("--languagetool-language", type=str, default="en-US",
                       help="Language code for LanguageTool (default: en-US)")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM-based fluency classifier (Gemini API)")
    parser.add_argument("--llm-model-name", type=str, default="gemini-2.5-flash",
                       help="Name of the Google Gemini model to use for LLM-based evaluation")
    parser.add_argument("--llm-timeout", type=int, default=60,
                       help="Timeout in seconds for LLM API calls (default: 60)")
    parser.add_argument("--modernbert-path", type=str, default=None,
                       help="Path to fine-tuned ModernBERT model checkpoint. If not specified, uses the default ModernBERT scorer.")
    parser.add_argument("--use-traditional-scorer", action="store_true",
                       help="Use the traditional rule-based FluencyScorer instead of ModernBERT")
    args = parser.parse_args()

    # Run async evaluation
    import asyncio
    results = asyncio.run(run_evaluation_async(args))

    return results


if __name__ == "__main__":
    main()

import os
import sys
import time
import logging
import re
import json
import json_repair
import spacy
from typing import List, Dict, Any, Optional
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer
from trl import GRPOConfig, GRPOTrainer

from prompts.edit_inappropriate_text import create_llm_prompt
from scorers.fluency.fluency_scorer import FluencyScorer
from scorers.appropriateness.appropriateness_scorer import AppropriatenessScorer
from scorers.semantic_similarity.semantic_similarity_scorer import SemanticSimilarityScorer
from scorers.human_like.human_like_scorer import HumanLikeScorer
from ops.completion_processor import process_completion
from ops.prompt_processor import process_prompt
from ops.edit_applier import apply_edits_to_argument
from ops.latexdiff_parser import DirectLatexdiffParser, fuzzy_post_process_edits

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Constants and prompt template
# -----------------------------
MODEL_BASE = "meta-llama/Llama-3.1-8B-Instruct"
# Resolve paths relative to this script's directory
_BASE_DIR = os.path.dirname(__file__)



# Example usage:
# issue_text = "The importance of recycling"
# sentences_text = "recycling is a total joke and a waste of time. only idiots believe it helps the planet, its so obvious. why even bother when big companies pollute way more???"
#
# formatted_prompt = create_llm_prompt(issue_text, sentences_text)
# print(formatted_prompt)

# -----------------------------
# Reward helpers
# -----------------------------





def find_sentence_window(text, start_char, end_char):
    # Simple regex to find sentence boundaries
    sentence_boundaries = [m.end() for m in re.finditer(r'[.!?]', text)]
    
    window_start = 0
    for i in range(len(sentence_boundaries) - 1, -1, -1):
        if sentence_boundaries[i] < start_char:
            window_start = sentence_boundaries[i] + 1
            break
            
    window_end = len(text)
    for i in range(len(sentence_boundaries)):
        if sentence_boundaries[i] >= end_char:
            window_end = sentence_boundaries[i]
            break
            
    return window_start, window_end


_ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
_ppl_model = AutoModelForCausalLM.from_pretrained("gpt2")
_ppl_tokenizer.pad_token = _ppl_tokenizer.eos_token
_PPL_MAX_TOKENS =1024 

def calculate_text_perplexities(texts: List[str]) -> List[float]:
    perplexities = []
    for i, text in enumerate(texts):
        logging.info(f"Computing perplexity for input {i+1}/{len(texts)}")
        if not isinstance(text, str) or len(text.strip()) == 0:
            perplexities.append(None)
            continue
        inputs = _ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=_PPL_MAX_TOKENS).to(_ppl_model.device)
        with torch.no_grad():
            outputs = _ppl_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
    return perplexities
def calculate_text_perplexity(text: str) -> Optional[float]:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None
    return calculate_text_perplexities([text])[0]


# -----------------------------
# Load reward models and utilities
# -----------------------------
_cuda_available = torch.cuda.is_available()
_local_rank = int(os.environ.get("LOCAL_RANK", 0)) if _cuda_available else 0
_device = torch.device(f"cuda:{_local_rank}" if _cuda_available else "cpu")

# Initialize scorers
logger.info("Loading scorers...")
_semantic_similarity_scorer = SemanticSimilarityScorer(_device)
_human_like_scorer = HumanLikeScorer(_device)  # Uses v4 model and P99 threshold by default
_fluency_scorer = FluencyScorer(_device)
_appropriateness_scorer = AppropriatenessScorer(_device)

# BERTScorer for document-level similarity
_bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", rescale_with_baseline=True, lang="en", batch_size=1, device=_device)

# Spacy for sentence segmentation
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Appropriateness classifier for argument-level scores
# -----------------------------
_DIMS = [
    'Inappropriateness',
    'Toxic Emotions',
    'Excessive Intensity',
    'Emotional Deception',
    'Missing Commitment',
    'Missing Seriousness',
    'Missing Openness',
    'Missing Intelligibility',
    'Unclear Meaning',
    'Missing Relevance',
    'Confusing Reasoning',
    'Other Reasons',
    'Detrimental Orthography',
    'Reason Unclassified'
]
_LABEL_MAP = {f"LABEL_{i}": dim for i, dim in enumerate(_DIMS)}
_ANALYSIS_DIMS = [
    "Inappropriateness",
    "Toxic Emotions",
    "Missing Commitment",
    "Missing Intelligibility",
    "Other Reasons",
]

def _predict_dimension_scores(text: str) -> Dict[str, float]:
    try:
        return _appropriateness_scorer.get_appropriateness_scores(text)
    except Exception as e:
        logger.debug(f"Classifier prediction failed: {e}")
    return {}


# -----------------------------
# Similarity helpers (NES)
# -----------------------------
def _levenshtein_distance(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[len(a)][len(b)]

def _normalized_edit_similarity_words(before: str, after: str) -> float:
    before_tokens = before.split()
    after_tokens = after.split()
    dist = _levenshtein_distance(before_tokens, after_tokens)
    denom = max(1, max(len(before_tokens), len(after_tokens)))
    # Similarity: 1 means identical, 0 means completely different (approx)
    return 1.0 - (dist / denom)

# -----------------------------
# Reward computation per edit
# -----------------------------
def score_edit(original_argument: str, edit: Dict[str, Any], baseline_scores: Dict[str, float] | None = None, original_sentence_context: Optional[str] = None) -> Dict[str, Any]:
    reason = edit.get("reason")
    inappropriate_part = edit.get("inappropriate_part")
    rewritten_part = edit.get("rewritten_part")

    # Validity: must have reason, at least one of inappropriate_part or rewritten_part,
    # and if inappropriate_part exists, it must be found in the context
    edit_context = original_sentence_context if original_sentence_context is not None else original_argument
    has_reason = bool(reason)
    has_at_least_one = bool(inappropriate_part) or bool(rewritten_part)
    substring_match = (not inappropriate_part) or (inappropriate_part in edit_context)
    is_well_formed = has_reason and has_at_least_one and substring_match

    logger.info(f"Scoring edit: inappropriate_part='{inappropriate_part}', rewritten_part='{rewritten_part}', reason='{reason}'")
    logger.info(f"is_well_formed: {is_well_formed} (reason={has_reason}, has_content={has_at_least_one}, substring_match={substring_match})")

    semantic_similarity = 0.0
    fluency_score = 0.0
    human_like = 0.0
    app_reward = 0.0
    reason_correct = False
    classifier_true_reason = None

    if is_well_formed:
        # Semantic similarity
        try:
            semantic_similarity, ss_score = _semantic_similarity_scorer.calculate_semantic_similarity(
                original_argument, inappropriate_part, rewritten_part
            )
            logger.info(f"Semantic similarity: binary={semantic_similarity}, score={ss_score}")
        except Exception as e:
            semantic_similarity = 0.0
            logger.error(f"Semantic similarity check failed: {e}")

        # Fluency
        try:
            context = original_sentence_context if original_sentence_context is not None else original_argument
            fluency_score = _fluency_scorer.calculate_fluency(context, inappropriate_part, rewritten_part)
            logger.info(f"Fluency Input: context='{context}', inappropriate_part='{inappropriate_part}', rewritten_part='{rewritten_part}'")
            logger.info(f"Fluency Output: {fluency_score}")
        except Exception as e:
            fluency_score = 0.0
            logger.error(f"Fluency check failed: {e}")

        # Human-like
        try:
            context = original_sentence_context if original_sentence_context is not None else original_argument
            human_like = _human_like_scorer.calculate_human_likeness(
                original_argument, context, inappropriate_part, rewritten_part
            )
            logger.info(f"Human-like score: {human_like}")
        except Exception as e:
            human_like = 0.0
            logger.error(f"Human-like check failed: {e}")

        # Edit-level appropriateness classifier reward (single-edit replacement on original)
        try:
            if original_sentence_context and inappropriate_part in original_sentence_context:
                modified_sentence = original_sentence_context.replace(inappropriate_part, rewritten_part, 1)
                modified_argument = original_argument.replace(original_sentence_context, modified_sentence, 1)
            else:
                modified_argument = original_argument.replace(inappropriate_part, rewritten_part, 1)
            before_scores = baseline_scores if baseline_scores is not None else _appropriateness_scorer.get_appropriateness_scores(original_argument)
            after_scores = _appropriateness_scorer.get_appropriateness_scores(modified_argument)

            # Compute improvement for overall inappropriateness (for app_reward)
            dim_before = before_scores.get("Inappropriateness")
            dim_after = after_scores.get("Inappropriateness")
            if dim_before is not None and dim_after is not None:
                # Sparse reward: 1 if the reason score improves (decreases), else 0
                app_reward = 1.0 if (dim_after < dim_before) else 0.0

            # Evaluate reason correctness: ranking with positive improvement requirement
            # Compute improvement for all 4 main dimensions
            dimension_improvements = {}
            for dim in _ANALYSIS_DIMS[1:]:  # Skip "Inappropriateness", use the 4 main categories
                dim_before_val = before_scores.get(dim)
                dim_after_val = after_scores.get(dim)
                if dim_before_val is not None and dim_after_val is not None:
                    improvement = dim_before_val - dim_after_val  # Positive = better (score decreased)
                    dimension_improvements[dim] = improvement

            # Filter to only dimensions with positive improvement
            positive_improvements = {
                dim: imp for dim, imp in dimension_improvements.items() if imp > 0
            }

            if positive_improvements:
                # The dimension with highest improvement is the "true reason"
                classifier_true_reason = max(positive_improvements, key=positive_improvements.get)
                reason_correct = (reason == classifier_true_reason)
                logger.info(f"Reason evaluation: predicted='{reason}', true='{classifier_true_reason}', correct={reason_correct}")
                logger.info(f"Positive improvements: {positive_improvements}")
            else:
                classifier_true_reason = None
                reason_correct = False
                logger.info(f"Reason evaluation: No positive improvements, reason_correct=False")

            logger.info(f"App Reward: before_score={dim_before}, after_score={dim_after}, app_reward={app_reward}")
        except Exception as e:
            app_reward = 0.0
            reason_correct = False
            classifier_true_reason = None
            logger.error(f"App reward check failed: {e}")


    # Perfect reward: all three main rewards (semantic_similarity, fluency, human_like) are 1.0
    # Note: excludes appropriateness (classifier not reliable for small edits)
    perfect = 1.0 if (is_well_formed and semantic_similarity == 1.0 and fluency_score == 1.0 and human_like == 1.0) else 0.0
    logger.info(f"Perfect score: {perfect} (excludes app reward)")

    return {
        "reason": reason,
        "classifier_true_reason": classifier_true_reason,
        "inappropriate_part": inappropriate_part,
        "rewritten_part": rewritten_part,
        "valid": bool(is_well_formed),
        "reason_correct": bool(reason_correct),
        "rewards": {
            "semantic_similarity": float(semantic_similarity),
            "fluency": float(fluency_score),
            "human_like": float(human_like),
            "app": float(app_reward),
            "perfect": float(perfect),
        },
    }


# -----------------------------
# Load fine-tuned model (base + LoRA)
# -----------------------------
def load_generation_model(checkpoint_root: str, use_base_model_only: bool = False) -> tuple[GRPOTrainer, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.1-8B-Instruct', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        peft_type="LORA",
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    training_args = GRPOConfig(
        output_dir="./temp_output",
        per_device_train_batch_size=2,
        log_completions=True,
        max_completion_length=1024,
        max_prompt_length=2048,
        scale_rewards=False,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        bf16=True,
        label_names=[],
        use_vllm=False,
        vllm_mode="colocate",
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        reward_weights=[0.5, 0.5],
        use_cpu=not torch.cuda.is_available(),
    )

    empty_reward = lambda *args, **kwargs: 0.0

    trainer = GRPOTrainer(
        model=MODEL_BASE,
        reward_funcs=[empty_reward,empty_reward],
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    #  Load model weights
    if not use_base_model_only:
        trainer.model.load_adapter(checkpoint_root, adapter_name="default")

    return trainer, tokenizer



import pandas as pd
from icecream import ic

# -----------------------------
# Main: generate, parse, score, write JSONL
# -----------------------------

def main(checkpoint_root: str, output_jsonl: str, use_base_model_only: bool = False, parse_diff: bool = False, model_name: str = None, split: str = "validation"):
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    logger.info(f"Starting edit prediction on {split} split. Output file: {output_jsonl}")

    if not parse_diff:
        model_load_start = time.time()
        trainer, tokenizer = load_generation_model(checkpoint_root, use_base_model_only)
        logger.info(f"Loaded generation model and tokenizer in {time.time() - model_load_start:.1f}s")

    eval_dataset = load_dataset("timonziegenbein/appropriateness-corpus", split=split)
    eval_dataset = eval_dataset.filter(lambda x: float(x.get("Inappropriateness", 0.0)) >= 0.5)

    # Load the exact validation set used in guided_grpo.py and keep only inappropriate examples
    if parse_diff:
        # Create a mapping from post_id to ground truth scores before we replace eval_dataset
        gt_scores_map = {}
        for ex in eval_dataset:
            post_id = ex.get('post_id')
            if post_id:
                gt_scores_map[post_id] = {dim: ex.get(dim) for dim in _ANALYSIS_DIMS if dim in ex}

        df1 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part1.csv')
        df2 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part2.csv')
        df = pd.concat([df1, df2], ignore_index=True)
        eval_ids = set(eval_dataset['post_id'])
        df = df[df['id'].isin(eval_ids)].sort_values(by=['id']).reset_index(drop=True)

        # Add ground truth scores back to the dataframe
        for dim in _ANALYSIS_DIMS:
            df[dim] = df['id'].apply(lambda pid: gt_scores_map.get(pid, {}).get(dim, None))

        eval_dataset = Dataset.from_pandas(df)

    total_before_filter = len(eval_dataset)
    # Limit to 1 sample for debugging purposes
    #limited_n = min(10, len(eval_dataset))
    #eval_dataset = eval_dataset.select(range(limited_n))
    num_examples = len(eval_dataset)
    logger.info(f"Loaded {split} dataset: {total_before_filter} examples; filtered to inappropriate: {num_examples}")

    # Aggregate metrics
    num_parse_success = 0
    total_edits = 0
    total_valid_edits = 0
    total_perfect_reward_ones = 0
    total_reason_correct = 0
    # Flip counters per dimension (restricted set)
    flips_per_dim: Dict[str, int] = {dim: 0 for dim in _ANALYSIS_DIMS}

    def _trunc(text: str, max_len: int = 160) -> str:
        return text if len(text) <= max_len else text[: max_len - 3] + "..."







    start_all = time.time()
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, example in enumerate(eval_dataset):
            example_start = time.time()
            issue = example.get("issue", "")
            if parse_diff:
                argument = example.get("source", "")
                argument = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", argument, 0, re.MULTILINE)
                argument = re.sub(r"\s\s+", " ", argument, 0, re.MULTILINE)
                rewritten_argument = example.get(model_name, "")
                completion = example.get(model_name, "")
                completion = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", completion, 0, re.MULTILINE)
                completion = re.sub(r"\s\s+", " ", completion, 0, re.MULTILINE)
            else:
                argument = example.get("post_text", "")
                argument = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", argument, 0, re.MULTILINE)
                argument = re.sub(r"\s\s+", " ", argument, 0, re.MULTILINE)

            doc = nlp(argument)
            sentences = [sent.text for sent in doc.sents]
            formatted_sentences = "\n".join([f"Sentence {i+1}: {sentence}" for i, sentence in enumerate(sentences)])

            if not parse_diff:
                prompt_text = create_llm_prompt(
                    issue=issue[:-1] if isinstance(issue, str) and len(issue) > 0 else issue,
                    sentences=formatted_sentences,
                )
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                logger.info(f"Model input: {prompt}")

                inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
                
                # Use trainer.predict
                prediction_output = trainer.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    #do_sample=False,
                    #temperature=1,
                    #top_p=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                completion = tokenizer.decode(
                    prediction_output[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True
                )
                logger.info(f"Model completion: {completion}")

            # Parse completion using ops (same as GRPO)
            scored_edits = []
            parse_ok = False
            all_edits = []

            try:
                if parse_diff:
                    # Check if texts are different
                    if argument.strip() == rewritten_argument.strip():
                        logger.warning(f"Example {idx}: Source and rewritten text are identical, skipping")
                        all_edits = []
                    else:
                        logger.info(f"Example {idx}: Parsing latex diff")
                        logger.debug(f"Source length: {len(argument)}, Rewritten length: {len(rewritten_argument)}")
                        parser = DirectLatexdiffParser()
                        parsed_example = parser.parse_latex_diff(argument, rewritten_argument, "./temp_output")
                        logger.info(f"LaTeX diff returned {len(parsed_example.get('edit_actions', []))} edit actions")
                        parsed_example['before_revision'] = argument
                        data, _ = fuzzy_post_process_edits([parsed_example])
                        # Extract edits from parsed latex diff
                        all_edits = data.get("edits", [])
                        logger.info(f"After fuzzy post-processing: {len(all_edits)} edits extracted")
                else:
                    # Use process_completion from ops (same as GRPO)
                    all_edits = process_completion(completion, sentences)

                if all_edits:
                    parse_ok = True
                    logger.info(f"Extracted {len(all_edits)} edits")
                    baseline_cls_scores = _predict_dimension_scores(argument)

                    # Score each edit
                    for idx_edit, edit in enumerate(all_edits):
                        logger.info(f"Scoring edit {idx_edit + 1}/{len(all_edits)}: {edit}")
                        # Get sentence context from sentence_id
                        sentence_id = edit.get("sentence_id", 0)
                        try:
                            original_sentence = sentences[sentence_id - 1] if 0 < sentence_id <= len(sentences) else None
                        except (IndexError, TypeError):
                            original_sentence = None

                        scored_edit = score_edit(
                            argument,
                            edit,
                            baseline_scores=baseline_cls_scores,
                            original_sentence_context=original_sentence
                        )
                        scored_edit['original_sentence'] = original_sentence or ""
                        scored_edit['sentence_id'] = sentence_id
                        scored_edits.append(scored_edit)

                    logger.info(f"Parsed {len(scored_edits)} edits from completion")

            except Exception as e:
                import traceback
                logger.error(f"Completion processing failed on example {idx}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(f"Scored {len(scored_edits)} edits before exception")
                scored_edits = []

            if parse_ok:
                num_parse_success += 1

            # Cache baseline classifier scores for the original argument for efficiency
            valid_count = sum(1 for e in scored_edits if e.get("valid"))
            perfect_ones = sum(
                1 for e in scored_edits if e.get("rewards", {}).get("perfect") == 1.0
            )
            reason_correct_count = sum(1 for e in scored_edits if e.get("reason_correct"))

            total_edits += len(scored_edits)
            total_valid_edits += valid_count
            total_perfect_reward_ones += perfect_ones
            total_reason_correct += reason_correct_count

            # Build argument after applying edits with perfect reward = 1 (using ops)
            # Filter edits to only those with perfect score
            perfect_edits = [
                {
                    'original_sentence': e.get('original_sentence', ''),
                    'inappropriate_part': e.get('inappropriate_part'),
                    'rewritten_part': e.get('rewritten_part')
                }
                for e in scored_edits
                if e.get("rewards", {}).get("perfect") == 1.0
            ]

            # For parse_diff mode, edits don't have sentence info, so apply directly to full argument
            if parse_diff and perfect_edits:
                argument_after_edits = argument
                for edit in perfect_edits:
                    inappropriate_part = edit.get('inappropriate_part')
                    rewritten_part = edit.get('rewritten_part')
                    if inappropriate_part and inappropriate_part in argument_after_edits:
                        argument_after_edits = argument_after_edits.replace(inappropriate_part, rewritten_part, 1)
                    elif not inappropriate_part and rewritten_part:
                        # Addition - append at the end (this is a simplification)
                        argument_after_edits = argument_after_edits + " " + rewritten_part
                logger.info(f"Applied {len(perfect_edits)} perfect edits directly to argument")
            else:
                # Apply edits using ops.edit_applier (same as GRPO)
                argument_after_edits = apply_edits_to_argument(perfect_edits, sentences, argument)

            # Also compute metrics using ALL valid edits (not just perfect ones)
            all_valid_edits = [
                {
                    'original_sentence': e.get('original_sentence', ''),
                    'inappropriate_part': e.get('inappropriate_part'),
                    'rewritten_part': e.get('rewritten_part')
                }
                for e in scored_edits
                if e.get("valid") == True
            ]

            if parse_diff and all_valid_edits:
                argument_after_all_edits = argument
                for edit in all_valid_edits:
                    inappropriate_part = edit.get('inappropriate_part')
                    rewritten_part = edit.get('rewritten_part')
                    if inappropriate_part and inappropriate_part in argument_after_all_edits:
                        argument_after_all_edits = argument_after_all_edits.replace(inappropriate_part, rewritten_part, 1)
                    elif not inappropriate_part and rewritten_part:
                        argument_after_all_edits = argument_after_all_edits + " " + rewritten_part
                logger.info(f"Applied {len(all_valid_edits)} valid edits (all) directly to argument")
            else:
                argument_after_all_edits = apply_edits_to_argument(all_valid_edits, sentences, argument)

            # Argument-level classifier metrics
            # App.: flip from inappropriate (>0.5) to appropriate (<=0.5)
            scores_before = _predict_dimension_scores(argument)
            scores_after = _predict_dimension_scores(argument_after_edits)
            # Track flips per dimension (>0.5 -> <=0.5)
            flipped = False
            for dim in _ANALYSIS_DIMS:
                before_val = scores_before.get(dim)
                after_val = scores_after.get(dim)
                if before_val is not None and after_val is not None and after_val < 0.5:
                    flips_per_dim[dim] += 1
                    if dim == "Inappropriateness":
                        flipped = True
            # Sim.: BERTScore F1 between original and after-edits
            try:
                _, _, f1_sim = _bert_scorer.score([argument_after_edits], [argument])
                sim_score = float(f1_sim.item())
            except Exception:
                sim_score = 0.0
            # NES.: normalized word-wise edit similarity
            nes_score = _normalized_edit_similarity_words(argument, argument_after_edits)
            # PPL.: perplexity (lower is better)
            try:
                ppl_score = calculate_text_perplexity(argument_after_edits)
            except Exception:
                ppl_score = float("inf")
            # GM.: geometric mean of App., Sim., 1/PPL (App here is 1 if flipped else 0)
            app_bin = 1.0 if flipped else 0.0
            inv_ppl = 0.0 if not np.isfinite(ppl_score) or ppl_score <= 0 else (1.0 / ppl_score)
            try:
                gm_score = float((app_bin * sim_score * inv_ppl) ** (1 / 3)) if app_bin > 0 and sim_score > 0 and inv_ppl > 0 else 0.0
            except Exception:
                gm_score = 0.0

            # Compute metrics for ALL valid edits
            scores_after_all = _predict_dimension_scores(argument_after_all_edits)
            flipped_all = False
            for dim in _ANALYSIS_DIMS:
                before_val = scores_before.get(dim)
                after_val_all = scores_after_all.get(dim)
                if before_val is not None and after_val_all is not None and after_val_all < 0.5:
                    if dim == "Inappropriateness":
                        flipped_all = True
            # Sim.: BERTScore F1 for all edits
            try:
                _, _, f1_sim_all = _bert_scorer.score([argument_after_all_edits], [argument])
                sim_score_all = float(f1_sim_all.item())
            except Exception:
                sim_score_all = 0.0
            # NES.: normalized word-wise edit similarity for all edits
            nes_score_all = _normalized_edit_similarity_words(argument, argument_after_all_edits)
            # PPL.: perplexity for all edits
            try:
                ppl_score_all = calculate_text_perplexity(argument_after_all_edits)
            except Exception:
                ppl_score_all = float("inf")
            # GM.: geometric mean for all edits
            app_bin_all = 1.0 if flipped_all else 0.0
            inv_ppl_all = 0.0 if not np.isfinite(ppl_score_all) or ppl_score_all <= 0 else (1.0 / ppl_score_all)
            try:
                gm_score_all = float((app_bin_all * sim_score_all * inv_ppl_all) ** (1 / 3)) if app_bin_all > 0 and sim_score_all > 0 and inv_ppl_all > 0 else 0.0
            except Exception:
                gm_score_all = 0.0

            # Ground truth scores (if available in dataset)
            gt_scores: Dict[str, float] = {}
            for dim in _ANALYSIS_DIMS:
                if dim in example:
                    try:
                        val = example[dim]
                        # Handle None, NaN, and empty strings
                        if val is not None and val != '' and not (isinstance(val, float) and np.isnan(val)):
                            gt_scores[dim] = float(val)
                    except Exception as e:
                        logger.debug(f"Could not convert ground truth {dim}={example[dim]} to float: {e}")

            # Predicted scores for all ANALYSIS_DIMS, thresholded to {0,1}
            # We compute these for all dimensions, not just those with ground truth
            pred_scores_before: Dict[str, float] = {}
            pred_scores_after: Dict[str, float] = {}
            pred_scores_after_all: Dict[str, float] = {}

            for dim in _ANALYSIS_DIMS:
                # Before edits
                val_before = scores_before.get(dim)
                if val_before is not None:
                    pred_scores_before[dim] = 1.0 if float(val_before) >= 0.5 else 0.0

                # After perfect edits
                val_after = scores_after.get(dim)
                if val_after is not None:
                    pred_scores_after[dim] = 1.0 if float(val_after) >= 0.5 else 0.0

                # After all valid edits
                val_after_all = scores_after_all.get(dim)
                if val_after_all is not None:
                    pred_scores_after_all[dim] = 1.0 if float(val_after_all) >= 0.5 else 0.0

            record = {
                "issue": issue,
                "argument": argument,
                "argument_after_edits": argument_after_edits,
                "argument_after_all_edits": argument_after_all_edits,
                "metrics": {
                    "App": app_bin,
                    "Sim": sim_score,
                    "NES": nes_score,
                    "PPL": ppl_score,
                    "GM": gm_score,
                },
                "metrics_all": {
                    "App": app_bin_all,
                    "Sim": sim_score_all,
                    "NES": nes_score_all,
                    "PPL": ppl_score_all,
                    "GM": gm_score_all,
                },
                "ground_truth_scores": gt_scores,
                "predicted_scores_before": pred_scores_before,
                "predicted_scores_after": pred_scores_after,
                "predicted_scores_after_all": pred_scores_after_all,
                "edits": scored_edits,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")


            # Periodic progress logging
            if idx < 3 or (idx + 1) % 50 == 0 or (idx + 1) == num_examples:
                elapsed = time.time() - example_start
                logger.info(
                    f"[{idx + 1}/{num_examples}] gen={elapsed:.1f}s edits={len(scored_edits)} valid={valid_count} perfect={perfect_ones} reason_correct={reason_correct_count} | issue='{_trunc(issue)}'"
                )
                logger.debug(f"Completion: {_trunc(completion, 240)}")

    # Compute flip percentages per dimension (restricted set)
    flip_percentages = {dim: (flips_per_dim[dim] / max(1, num_examples)) for dim in _ANALYSIS_DIMS}

    logger.info(
        "Finished. Time={:.1f}s | parse_ok={}/{}"
        " | edits={} valid_edits={} perfect={} reason_correct={} ({:.1%}) | App%={:.2%}".format(
            time.time() - start_all,
            num_parse_success,
            num_examples,
            total_edits,
            total_valid_edits,
            total_perfect_reward_ones,
            total_reason_correct,
            total_reason_correct / max(1, total_edits),
            flip_percentages.get("Inappropriateness", 0.0),
        )
    )
    # Log per-dimension flip percentages
    logger.info("Flip percentages per dimension:")
    for dim in _ANALYSIS_DIMS:
        logger.info(f"- {dim}: {flip_percentages[dim]:.2%}")
    logger.info(f"Wrote {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict edits and calculate rewards on a dataset split.")
    parser.add_argument("--checkpoint_root", type=str, help="Path to the checkpoint directory.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--use_base_model_only", action="store_true", help="Use the base model without LoRA.")
    parser.add_argument("--parse_diff", action="store_true", help="Parse diffs instead of generating edits.")
    parser.add_argument("--model_name", type=str, help="The name of the model to evaluate from the dataframe.")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], help="Dataset split to use (default: validation)")
    args = parser.parse_args()

    if not args.use_base_model_only and not args.checkpoint_root and not args.parse_diff:
        parser.error("--checkpoint_root is required unless --use_base_model_only or --parse_diff is specified.")

    main(args.checkpoint_root, args.output_jsonl, args.use_base_model_only, args.parse_diff, args.model_name, args.split)

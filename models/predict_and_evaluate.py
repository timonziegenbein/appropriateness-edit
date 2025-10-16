import os
import time
import logging
import re
import json
import json_repair
import spacy
from typing import List, Dict, Any, Optional
import argparse

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

from ops.prompt_processor import create_llm_prompt
from scorers.fluency.fluency_scorer import compute_fluency_scores
from scorers.appropriateness.appropriateness_scorer import get_appropriateness_scores, load_app_model
from ops.latexdiff_parser import DirectLatexdiffParser, fuzzy_post_process_edits
from scorers.human_like.human_like_scorer import generate_sequence_for_edit, calculate_perplexity_for_sequence

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

# Load all reward models
_ss_model, _app_model, _hl_model, _hl_tokenizer, _fluency_model = load_reward_models(
    os.path.join(_BASE_DIR, "../models/human_like_language_model_v2.pth"),
    _device
)

# Human-like sequence LM
_hl_device = _device
_hl_vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4}
_hl_max_len = 500

# Perplexity threshold for human-like reward
_ppl_threshold = 1.4381

# Semantic similarity scorer
_ss_device = _device
_bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", rescale_with_baseline=True, lang="en", batch_size=1, device=_ss_device)


# Thresholds aligned with guided_grpo.py
SS_F1_MIN_THRESHOLD = 0.6144470572471619
nlp = spacy.load("en_core_web_sm")
# -----------------------------
# Appropriateness classifier for argument-level scores
# -----------------------------
_clf_device = 0 if torch.cuda.is_available() else -1
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
        outputs = _app_model(text)
        # outputs is List[List[{label,score}]] when return_all_scores=True
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], list):
            scores = { _LABEL_MAP.get(item["label"], item["label"]): float(item["score"]) for item in outputs[0] }
            return scores
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
def score_edit(original_argument: str, edit: Dict[str, Any], baseline_scores: Dict[str, float] | None = None, original_sentence_context: Optional[str] = None, app_model=None) -> Dict[str, Any]:
    reason = edit.get("reason")
    inappropriate_part = edit.get("inappropriate_part")
    rewritten_part = edit.get("rewritten_part")

    # Validity (well-formed + substring in original)
    edit_context = original_sentence_context if original_sentence_context is not None else original_argument
    is_well_formed = bool(reason) and bool(inappropriate_part) and bool(rewritten_part) and (inappropriate_part in edit_context)
    logger.info(f"Scoring edit: inappropriate_part='{inappropriate_part}', rewritten_part='{rewritten_part}', reason='{reason}'")
    logger.info(f"is_well_formed: {is_well_formed}")

    semantic_similarity = 0.0
    fluency_score = 0.0
    human_like = 0.0
    app_reward = 0.0

    if is_well_formed:
        # Semantic similarity
        semantic_similarity, ss_score = calculate_semantic_similarity(edit_context, inappropriate_part, rewritten_part, _ss_model, SS_F1_MIN_THRESHOLD)

        # Fluency change
        try:
            context = original_sentence_context if original_sentence_context is not None else original_argument
            scores = compute_fluency_scores(context, inappropriate_part, rewritten_part)
            if scores and len(scores) > 0:
                fluency_score = scores[0]
            logger.info(f"Fluency Input: context='{context}', inappropriate_part='{inappropriate_part}', rewritten_part='{rewritten_part}'")
            logger.info(f"Fluency Output: {fluency_score}")

        except Exception as e:
            fluency_score = 0.0
            logger.error(f"Fluency check failed: {e}")

        # Human-like sequence plausibility
        try:
            start_char = -1
            if original_sentence_context:
                # Find inappropriate_part within the sentence to handle duplicates in the argument
                start_char_in_sentence = original_sentence_context.find(inappropriate_part)
                if start_char_in_sentence != -1:
                    # Find where the sentence starts in the full argument
                    sentence_start_in_argument = original_argument.find(original_sentence_context)
                    if sentence_start_in_argument != -1:
                        start_char = sentence_start_in_argument + start_char_in_sentence
                    else:
                        # This should not happen if original_sentence_context is from original_argument
                        # Fallback to searching in the whole argument
                        start_char = original_argument.find(inappropriate_part)
            else:
                # Fallback for when there is no sentence context
                start_char = original_argument.find(inappropriate_part)

            if start_char != -1:
                end_char = start_char + len(inappropriate_part)
                sequence = generate_sequence_for_edit(original_argument, start_char, end_char, rewritten_part, _hl_tokenizer)
                if sequence:
                    ppl = calculate_perplexity_for_sequence(sequence, _hl_model, _hl_vocab, _hl_device, _hl_max_len)
                    human_like = 1.0 if ppl <= _ppl_threshold else 0.0
                    logger.info(f"Human-like Input: sequence='{sequence}'")
                    logger.info(f"Human-like Output: ppl={ppl}, human_like={human_like}")
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
            before_scores = baseline_scores if baseline_scores is not None else get_appropriateness_scores(original_argument, app_model)
            after_scores = get_appropriateness_scores(modified_argument, app_model)
            dim_name = reason if isinstance(reason, str) else None
            dim_before = before_scores.get("Inappropriateness")
            dim_after = after_scores.get("Inappropriateness")
            if (
                dim_before is not None and dim_after is not None
            ):
                # Sparse reward: 1 if the reason score improves (decreases), else 0
                app_reward = 1.0 if (dim_after < dim_before) else 0.0
            logger.info(f"App Reward Input: modified_argument='{modified_argument}', reason='{dim_name}'")
            logger.info(f"App Reward Output: before_score={dim_before}, after_score={dim_after}, app_reward={app_reward}")
        except Exception as e:
            app_reward = 0.0
            logger.error(f"App reward check failed: {e}")


    overall = 1.0 if (is_well_formed and semantic_similarity == 1.0 and fluency_score == 1.0 and human_like == 1.0) else 0.0
    logger.info(f"Overall score: {overall}")

    return {
        "reason": reason,
        "inappropriate_part": inappropriate_part,
        "rewritten_part": rewritten_part,
        "valid": bool(is_well_formed),
        "rewards": {
            "semantic_similarity": float(semantic_similarity),
            "perplexity": float(fluency_score),
            "human_like": float(human_like),
            "app": float(app_reward),
            "overall": float(overall),
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

def main(checkpoint_root: str, output_jsonl: str, use_base_model_only: bool = False, parse_diff: bool = False, model_name: str = None):
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    logger.info(f"Starting validation edit prediction. Output file: {output_jsonl}")

    if not parse_diff:
        model_load_start = time.time()
        trainer, tokenizer = load_generation_model(checkpoint_root, use_base_model_only)
        logger.info(f"Loaded generation model and tokenizer in {time.time() - model_load_start:.1f}s")

    app_model = load_app_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    eval_dataset = load_dataset("timonziegenbein/appropriateness-corpus", split="validation")
    eval_dataset = eval_dataset.filter(lambda x: float(x.get("Inappropriateness", 0.0)) >= 0.5)

    # Load the exact validation set used in guided_grpo.py and keep only inappropriate examples
    if parse_diff:
        df1 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part1.csv')
        df2 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part2.csv')
        df = pd.concat([df1, df2], ignore_index=True)
        eval_ids = set(eval_dataset['post_id'])
        df = df[df['id'].isin(eval_ids)].sort_values(by=['id']).reset_index(drop=True)
        eval_dataset = Dataset.from_pandas(df)

    total_before_filter = len(eval_dataset)
    # Limit to 1 sample for debugging purposes
    limited_n = min(1, len(eval_dataset))
    eval_dataset = eval_dataset.select(range(limited_n))
    num_examples = len(eval_dataset)
    logger.info(f"Loaded validation dataset: {total_before_filter} examples; filtered to inappropriate and limited: {num_examples}")

    # Aggregate metrics
    num_parse_success = 0
    total_edits = 0
    total_valid_edits = 0
    total_overall_reward_ones = 0
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

            # Parse JSON edits
            edits_list: List[Dict[str, Any]] = []
            scored_edits = []
            rewritten_argument = ""
            parse_ok = False
            data = {}
            try:
                if parse_diff:
                    parser = DirectLatexdiffParser()
                    parsed_example = parser.parse_latex_diff(argument, rewritten_argument, "./temp_output")
                    ic(parsed_example)
                    data, _ = fuzzy_post_process_edits([parsed_example])
                    ic(data)
                else:
                    data = json_repair.loads(completion)
                
                if isinstance(data, dict) and "sentence_edits" in data:
                    sentence_edits = data.get("sentence_edits", [])
                    rewritten_sentences = [edit.get("rewritten_sentence", edit.get("original_sentence")) for edit in sentence_edits]
                    rewritten_argument = " ".join(rewritten_sentences)
                    
                    baseline_cls_scores = _predict_dimension_scores(argument)

                    for sentence_edit in sentence_edits:
                        original_sentence = sentence_edit.get("original_sentence")
                        tracked_changes = sentence_edit.get("tracked_changes", "")
                        
                        parsed_edits = re.findall(r'<del reason=["\'](.*?)?["\']>(.+?)</del><ins>(.*?)</ins>', tracked_changes)
                        for reason, inappropriate_part, rewritten_part in parsed_edits:
                            edit = {
                                "reason": reason,
                                "inappropriate_part": inappropriate_part,
                                "rewritten_part": rewritten_part,
                            }
                            logger.info(f"Extracted edit: {edit}")
                            scored_edit = score_edit(argument, edit, baseline_scores=baseline_cls_scores, original_sentence_context=original_sentence)
                            scored_edit['original_sentence'] = original_sentence
                            scored_edits.append(scored_edit)
                    parse_ok = True
                elif isinstance(data, dict):
                    edits_list = data.get("edits", [])
                    ic(edits_list)
                    baseline_cls_scores = _predict_dimension_scores(argument)
                    scored_edits = [score_edit(argument, e, baseline_scores=baseline_cls_scores) for e in edits_list]
                    ic(scored_edits)
                    parse_ok = True

            except Exception as e:
                logger.debug(f"JSON parse failed on example {idx}: {e}")
                scored_edits = []

            if parse_ok:
                num_parse_success += 1

            # Cache baseline classifier scores for the original argument for efficiency
            valid_count = sum(1 for e in scored_edits if e.get("valid"))
            overall_ones = sum(
                1 for e in scored_edits if e.get("rewards", {}).get("overall") == 1.0
            )

            total_edits += len(scored_edits)
            total_valid_edits += valid_count
            total_overall_reward_ones += overall_ones

            # Build argument after applying edits with overall reward = 1
            argument_after_edits = argument
            if parse_ok and "sentence_edits" in data and isinstance(data["sentence_edits"], list):
                final_sentences = []
                for sentence_edit in data["sentence_edits"]:
                    original_sentence = sentence_edit.get("original_sentence")
                    
                    if not original_sentence:
                        final_sentences.append(sentence_edit.get("rewritten_sentence", ""))
                        continue

                    sentence_after_edits = original_sentence
                    
                    tracked_changes = sentence_edit.get("tracked_changes", "")
                    parsed_edits_in_order = re.findall(r'<del reason=["\'](.*?)?["\']>(.+?)</del><ins>(.*?)</ins>', tracked_changes)

                    for reason, inappropriate_part, rewritten_part in parsed_edits_in_order:
                        found_scored_edit = None
                        for se in scored_edits:
                            if (se.get("original_sentence") == original_sentence and
                                se.get("inappropriate_part") == inappropriate_part and
                                se.get("rewritten_part") == rewritten_part and
                                se.get("reason") == reason):
                                found_scored_edit = se
                                break
                        
                        if found_scored_edit and found_scored_edit.get("rewards", {}).get("overall") == 1.0:
                            if isinstance(inappropriate_part, str) and isinstance(rewritten_part, str) and inappropriate_part in sentence_after_edits:
                                sentence_after_edits = sentence_after_edits.replace(inappropriate_part, rewritten_part, 1)

                    final_sentences.append(sentence_after_edits)
                
                argument_after_edits = " ".join(final_sentences)
            else:
                # Fallback for old format or parse failure
                for e in scored_edits:
                    try:
                        if e.get("rewards", {}).get("overall") == 1.0:
                            old = e.get("inappropriate_part")
                            new = e.get("rewritten_part")
                            if isinstance(old, str) and isinstance(new, str) and old in argument_after_edits:
                                argument_after_edits = argument_after_edits.replace(old, new, 1)
                    except Exception:
                        pass

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

            # Ground truth scores (if available in dataset)
            gt_scores: Dict[str, float] = {}
            for dim in _ANALYSIS_DIMS:
                if dim in example:
                    try:
                        gt_scores[dim] = float(example[dim])
                    except Exception:
                        pass

            # Predicted scores for the same categories as ground truth, thresholded to {0,1}
            pred_scores_before: Dict[str, float] = {}
            for dim in gt_scores.keys():
                val = scores_before.get(dim)
                pred_scores_before[dim] = 1.0 if (val is not None and float(val) >= 0.5) else 0.0

            # Predicted scores for the same categories as ground truth, after edits
            pred_scores_after: Dict[str, float] = {}
            for dim in gt_scores.keys():
                val = scores_after.get(dim)
                pred_scores_after[dim] = 1.0 if (val is not None and float(val) > 0.5) else 0.0

            record = {
                "issue": issue,
                "argument": argument,
                "argument_after_edits": argument_after_edits,
                "metrics": {
                    "App": app_bin,
                    "Sim": sim_score,
                    "NES": nes_score,
                    "PPL": ppl_score,
                    "GM": gm_score,
                },
                "ground_truth_scores": gt_scores,
                "predicted_scores_before": pred_scores_before,
                "predicted_scores_after": pred_scores_after,
                "edits": scored_edits,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")


            # Periodic progress logging
            if idx < 3 or (idx + 1) % 50 == 0 or (idx + 1) == num_examples:
                elapsed = time.time() - example_start
                logger.info(
                    f"[{idx + 1}/{num_examples}] gen={elapsed:.1f}s edits={len(scored_edits)} valid={valid_count} overall1={overall_ones} | issue='{_trunc(issue)}'"
                )
                logger.debug(f"Completion: {_trunc(completion, 240)}")

    # Compute flip percentages per dimension (restricted set)
    flip_percentages = {dim: (flips_per_dim[dim] / max(1, num_examples)) for dim in _ANALYSIS_DIMS}

    logger.info(
        "Finished. Time={:.1f}s | parse_ok={}/{}" 
        " | edits={} valid_edits={} overall1={} | App%={:.2%}".format(
            time.time() - start_all,
            num_parse_success,
            num_examples,
            total_edits,
            total_valid_edits,
            total_overall_reward_ones,
            flip_percentages.get("Inappropriateness", 0.0),
        )
    )
    # Log per-dimension flip percentages
    logger.info("Flip percentages per dimension:")
    for dim in _ANALYSIS_DIMS:
        logger.info(f"- {dim}: {flip_percentages[dim]:.2%}")
    logger.info(f"Wrote {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict validation set edits and calculate rewards.")
    parser.add_argument("--checkpoint_root", type=str, help="Path to the checkpoint directory.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--use_base_model_only", action="store_true", help="Use the base model without LoRA.")
    parser.add_argument("--parse_diff", action="store_true", help="Parse diffs instead of generating edits.")
    parser.add_argument("--model_name", type=str, help="The name of the model to evaluate from the dataframe.")
    args = parser.parse_args()

    if not args.use_base_model_only and not args.checkpoint_root and not args.parse_diff:
        parser.error("--checkpoint_root is required unless --use_base_model_only or --parse_diff is specified.")

    main(args.checkpoint_root, args.output_jsonl, args.use_base_model_only, args.parse_diff, args.model_name)

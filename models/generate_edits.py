"""
Generate edits from a model and save them to a JSONL file.

This script handles the costly edit generation step (model inference) and saves
the raw edits to a file. The edits can then be evaluated with different scorer
configurations using evaluate_edits.py without regenerating them.

Usage:
    # Generate edits from a trained model
    python models/generate_edits.py --checkpoint_root <checkpoint_path> --output_jsonl <output_file.jsonl>

    # Generate edits from base model
    python models/generate_edits.py --use_base_model_only --output_jsonl <output_file.jsonl>

    # Parse existing diffs (e.g., from human edits)
    python models/generate_edits.py --parse_diff --model_name rewrite_40a_60ss --output_jsonl <output_file.jsonl>
"""

import os
import sys
import time
import logging
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import spacy
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from prompts.edit_inappropriate_text import create_llm_prompt
from ops.completion_processor import process_completion
from ops.latexdiff_parser import DirectLatexdiffParser, fuzzy_post_process_edits

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_BASE = "meta-llama/Llama-3.1-8B-Instruct"

# Spacy for sentence segmentation
nlp = spacy.load("en_core_web_sm")


def load_generation_model(checkpoint_root: str, use_base_model_only: bool = False) -> tuple:
    """Load the model for generating edits."""
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
        reward_funcs=[empty_reward, empty_reward],
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    if not use_base_model_only:
        trainer.model.load_adapter(checkpoint_root, adapter_name="default")

    return trainer, tokenizer


def main(
    checkpoint_root: str,
    output_jsonl: str,
    use_base_model_only: bool = False,
    parse_diff: bool = False,
    model_name: str = None,
    split: str = "validation"
):
    """Generate edits and save them to a JSONL file."""
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    logger.info(f"Starting edit generation on {split} split. Output file: {output_jsonl}")

    if not parse_diff:
        model_load_start = time.time()
        trainer, tokenizer = load_generation_model(checkpoint_root, use_base_model_only)
        logger.info(f"Loaded generation model and tokenizer in {time.time() - model_load_start:.1f}s")

    # Load dataset
    eval_dataset = load_dataset("timonziegenbein/appropriateness-corpus", split=split)
    eval_dataset = eval_dataset.filter(lambda x: float(x.get("Inappropriateness", 0.0)) >= 0.5)

    if parse_diff:
        # Load study data for diff parsing
        df1 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part1.csv')
        df2 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part2.csv')
        df = pd.concat([df1, df2], ignore_index=True)
        eval_ids = set(eval_dataset['post_id'])
        df = df[df['id'].isin(eval_ids)].sort_values(by=['id']).reset_index(drop=True)
        eval_dataset = Dataset.from_pandas(df)

    num_examples = len(eval_dataset)
    logger.info(f"Loaded {split} dataset: {num_examples} examples")

    # Statistics
    num_parse_success = 0
    total_edits = 0

    start_all = time.time()
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, example in enumerate(eval_dataset):
            example_start = time.time()
            issue = example.get("issue", "")
            post_id = example.get("post_id" if not parse_diff else "id")

            if parse_diff:
                argument = example.get("source", "")
                argument = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", argument, 0, re.MULTILINE)
                argument = re.sub(r"\s\s+", " ", argument, 0, re.MULTILINE)
                rewritten_argument = example.get(model_name, "")
                completion = rewritten_argument
                completion = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", completion, 0, re.MULTILINE)
                completion = re.sub(r"\s\s+", " ", completion, 0, re.MULTILINE)
            else:
                argument = example.get("post_text", "")
                argument = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", argument, 0, re.MULTILINE)
                argument = re.sub(r"\s\s+", " ", argument, 0, re.MULTILINE)

            # Segment sentences
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

                prediction_output = trainer.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                completion = tokenizer.decode(
                    prediction_output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                logger.info(f"Model completion: {completion}")

            # Parse completion
            parse_ok = False
            all_edits = []

            try:
                if parse_diff:
                    if argument.strip() == rewritten_argument.strip():
                        logger.warning(f"Example {idx}: Source and rewritten text are identical, skipping")
                        all_edits = []
                    else:
                        logger.info(f"Example {idx}: Parsing latex diff")
                        parser = DirectLatexdiffParser()
                        parsed_example = parser.parse_latex_diff(argument, rewritten_argument, "./temp_output")
                        logger.info(f"LaTeX diff returned {len(parsed_example.get('edit_actions', []))} edit actions")
                        parsed_example['before_revision'] = argument
                        data, _ = fuzzy_post_process_edits([parsed_example])
                        all_edits = data.get("edits", [])
                        logger.info(f"After fuzzy post-processing: {len(all_edits)} edits extracted")
                else:
                    all_edits = process_completion(completion, sentences)

                if all_edits:
                    parse_ok = True
                    logger.info(f"Extracted {len(all_edits)} edits")

            except Exception as e:
                import traceback
                logger.error(f"Completion processing failed on example {idx}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                all_edits = []

            if parse_ok:
                num_parse_success += 1

            total_edits += len(all_edits)

            # Save record with raw edits
            record = {
                "post_id": post_id,
                "issue": issue,
                "argument": argument,
                "sentences": sentences,
                "completion": completion,
                "edits": all_edits,
                "metadata": {
                    "parse_success": parse_ok,
                    "num_edits": len(all_edits),
                    "generation_time": time.time() - example_start,
                }
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Progress logging
            if idx < 3 or (idx + 1) % 50 == 0 or (idx + 1) == num_examples:
                elapsed = time.time() - example_start
                logger.info(
                    f"[{idx + 1}/{num_examples}] gen={elapsed:.1f}s edits={len(all_edits)} | issue='{issue[:100]}'"
                )

    logger.info(
        f"Finished. Time={time.time() - start_all:.1f}s | parse_ok={num_parse_success}/{num_examples} | total_edits={total_edits}"
    )
    logger.info(f"Wrote {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edits from a model and save them to JSONL.")
    parser.add_argument("--checkpoint_root", type=str, help="Path to the checkpoint directory.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--use_base_model_only", action="store_true", help="Use the base model without LoRA.")
    parser.add_argument("--parse_diff", action="store_true", help="Parse diffs instead of generating edits.")
    parser.add_argument("--model_name", type=str, help="The name of the model to evaluate from the dataframe (for parse_diff mode).")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], help="Dataset split to use (default: validation)")
    args = parser.parse_args()

    if not args.use_base_model_only and not args.checkpoint_root and not args.parse_diff:
        parser.error("--checkpoint_root is required unless --use_base_model_only or --parse_diff is specified.")

    main(args.checkpoint_root, args.output_jsonl, args.use_base_model_only, args.parse_diff, args.model_name, args.split)

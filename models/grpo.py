import torch
import os
import sys
import argparse
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from peft import LoraConfig, TaskType
import logging
import wandb
import weave

from prompts.edit_inappropriate_text import create_llm_prompt
from scorers.reward_functions import global_appropriateness_reward, dense_local_appropriateness_reward
from scorers.semantic_similarity.semantic_similarity_scorer import SemanticSimilarityScorer
from scorers.human_like.human_like_scorer import HumanLikeScorer
from scorers.fluency.fluency_scorer import FluencyScorer
from scorers.appropriateness.appropriateness_scorer import AppropriatenessScorer

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train a GRPO model.")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.1-8B-Instruct", help="The name of the model to train.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory for the trained model.")
    parser.add_argument("--wandb_project", type=str, default="appropriateness-edit", help="W&B project name for weave tracing.")

    args = parser.parse_args()

    # Initialize wandb first
    run_name = f"grpo-{args.output_dir.split('/')[-1]}"
    wandb.init(project=args.wandb_project, name=run_name)
    logger.info(f"Initialized WandB for project: {args.wandb_project}, run: {run_name}")

    # Initialize Weave for tracing (will use the existing wandb run)
    weave.init(args.wandb_project)
    logger.info(f"Initialized Weave tracing (sharing WandB run)")

    # --- Load Reward Models ---
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    semantic_similarity_scorer = SemanticSimilarityScorer(device)
    logger.info(f"Memory after semantic similarity scorer: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    human_like_scorer = HumanLikeScorer(device)
    logger.info(f"Memory after human-like scorer: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    fluency_scorer = FluencyScorer(device)
    logger.info(f"Memory after fluency scorer: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    appropriateness_scorer = AppropriatenessScorer(device)
    logger.info(f"Memory after appropriateness scorer: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    # --- GRPOTrainer with Outlines ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    def prepare_dataset(batch):
        # Ensure the input is a string
        if not isinstance(batch["prompt"], str):
            return {{"prompt": ""}}
        
        sentences = batch["sentences"]
        
        # Format the sentences with enumeration
        formatted_sentences = "\n".join([f"Sentence {i+1}: {sentence}" for i, sentence in enumerate(sentences)])
        
        prompt_text = create_llm_prompt(
            issue=batch["issue"][:-1] if isinstance(batch["issue"], str) and len(batch["issue"]) > 0 else batch["issue"],
            sentences=formatted_sentences,
        )

        # Apply the chat template
        return {"prompt": tokenizer.apply_chat_template([{"role":"user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)}

    dataset = load_dataset("timonziegenbein/appropriateness-corpus-extension-cleaned", split="train")
    dataset = dataset.rename_column("post_text", "prompt")
    dataset = dataset.map(prepare_dataset, load_from_cache_file=False)

    eval_dataset = load_dataset("timonziegenbein/appropriateness-corpus-cleaned", split="validation")
    eval_dataset = eval_dataset.rename_column("post_text", "prompt")
    eval_dataset = eval_dataset.map(prepare_dataset, load_from_cache_file=False)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        eval_strategy="steps",
        eval_steps=100,
        eval_on_start=True,
        log_completions=True,
        max_completion_length=1024,
        max_prompt_length=2048,
        scale_rewards=False,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        bf16=True,
        label_names=[],
        use_vllm=True,
        vllm_mode="colocate",
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        beta=0.001857,
        disable_dropout=True,
        report_to="wandb",
    )

    peft_config = LoraConfig(
        peft_type="LORA",
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[
            # Global Reward (80% weight) - measures document-level inappropriateness reduction
            lambda prompts, completions, **kwargs: [
                0.5 * score for score in global_appropriateness_reward(
                    prompts,
                    completions,
                    appropriateness_scorer=appropriateness_scorer,
                    semantic_similarity_scorer=semantic_similarity_scorer,
                    human_like_scorer=human_like_scorer,
                    fluency_scorer=fluency_scorer,
                    **kwargs
                )
            ],
            # Dense Local Reward (20% weight) - provides gradient signal for edit quality
            lambda prompts, completions, **kwargs: [
                0.5 * score for score in dense_local_appropriateness_reward(
                    prompts,
                    completions,
                    semantic_similarity_scorer=semantic_similarity_scorer,
                    human_like_scorer=human_like_scorer,
                    fluency_scorer=fluency_scorer,
                    **kwargs
                )
            ]
        ],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    logger.info(f"Memory after loading main model: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
    logger.info(f"Memory reserved: {torch.cuda.memory_reserved(device)/1024**3:.2f} GB")

    trainer.train()

if __name__ == "__main__":
    main()

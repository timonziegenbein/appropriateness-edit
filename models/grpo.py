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
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to resume (optional). If not provided, creates a new run.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from (optional).")

    # Local reward model flags
    parser.add_argument("--use_semantic_similarity", action="store_true", help="Enable semantic similarity scorer in local reward (default: enabled).")
    parser.add_argument("--use_human_like", action="store_true", help="Enable human-like scorer in local reward (default: disabled).")
    parser.add_argument("--use_fluency", action="store_true", help="Enable fluency scorer in local reward (default: enabled).")

    # Training configuration flags
    parser.add_argument("--disable_eval_on_start", action="store_true", help="Disable evaluation at the start of training.")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit", help="Optimizer to use (default: paged_adamw_8bit). Try 'adamw_torch' if getting SIGBUS errors.")

    args = parser.parse_args()

    # Set defaults for scorer flags (semantic_similarity and fluency enabled by default)
    if not any([args.use_semantic_similarity, args.use_human_like, args.use_fluency]):
        # If no flags were set, use defaults
        args.use_semantic_similarity = True
        args.use_fluency = True

    logger.info("Local reward scorers configuration:")
    logger.info(f"  - Semantic Similarity: {'ENABLED' if args.use_semantic_similarity else 'DISABLED'}")
    logger.info(f"  - Human-Like: {'ENABLED' if args.use_human_like else 'DISABLED'}")
    logger.info(f"  - Fluency: {'ENABLED' if args.use_fluency else 'DISABLED'}")

    if not args.use_semantic_similarity and not args.use_human_like and not args.use_fluency:
        logger.warning("WARNING: All local reward scorers are disabled! Local reward will always be 1.0.")

    # Auto-detect the latest checkpoint if resume path points to output directory
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint and os.path.isdir(resume_checkpoint):
        # Check if this is a checkpoint directory (has trainer_state.json)
        if not os.path.exists(os.path.join(resume_checkpoint, "trainer_state.json")):
            # This is likely the output directory, find the latest checkpoint
            import glob
            checkpoints = glob.glob(os.path.join(resume_checkpoint, "checkpoint-*"))
            if checkpoints:
                # Sort by checkpoint number
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
                resume_checkpoint = checkpoints[-1]
                logger.info(f"Auto-detected latest checkpoint: {resume_checkpoint}")
            else:
                logger.warning(f"No checkpoints found in {resume_checkpoint}, starting fresh")
                resume_checkpoint = None
        else:
            logger.info(f"Using checkpoint directory: {resume_checkpoint}")

    args.resume_from_checkpoint = resume_checkpoint

    # Log environment configuration (set by launch scripts)
    logger.info("=" * 80)
    logger.info("Job Isolation Configuration (from environment):")
    logger.info(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
    logger.info(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
    logger.info(f"  TORCHELASTIC_RUN_ID: {os.environ.get('TORCHELASTIC_RUN_ID', 'not set')}")
    logger.info(f"  VLLM_INSTANCE_ID: {os.environ.get('VLLM_INSTANCE_ID', 'not set')}")
    logger.info(f"  TMPDIR: {os.environ.get('TMPDIR', 'not set')}")
    logger.info(f"  XDG_CACHE_HOME: {os.environ.get('XDG_CACHE_HOME', 'not set')}")
    logger.info(f"  TRITON_CACHE_DIR: {os.environ.get('TRITON_CACHE_DIR', 'not set')}")
    logger.info(f"  TORCH_COMPILE_CACHE_DIR: {os.environ.get('TORCH_COMPILE_CACHE_DIR', 'not set')}")
    logger.info(f"  WANDB_DIR: {os.environ.get('WANDB_DIR', 'not set')}")
    logger.info(f"  Model cache: Using shared HuggingFace cache (read-only, no conflicts)")
    logger.info("=" * 80)

    # Check if this is the main process (for distributed training)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    # Initialize wandb only on main process
    if is_main_process:
        run_name = f"grpo-{args.output_dir.split('/')[-1]}"

        # Configure WandB resumption
        if args.wandb_run_id:
            # Explicitly resume a specific WandB run
            wandb.init(
                project=args.wandb_project,
                id=args.wandb_run_id,
                resume="must"
            )
            logger.info(f"Resuming WandB run: {args.wandb_run_id}")
        else:
            # Create a new run
            wandb.init(
                project=args.wandb_project,
                name=run_name
            )
            logger.info(f"Created new WandB run: {run_name} (id: {wandb.run.id})")

        # Initialize Weave for tracing (will use the existing wandb run)
        weave.init(args.wandb_project)
        logger.info(f"Initialized Weave tracing (sharing WandB run)")
    else:
        logger.info(f"Skipping WandB/Weave initialization on worker process (local_rank={local_rank})")

    # --- Load Reward Models ---
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Load scorers conditionally based on flags
    semantic_similarity_scorer = None
    if args.use_semantic_similarity:
        semantic_similarity_scorer = SemanticSimilarityScorer(device)
        logger.info(f"Memory after semantic similarity scorer: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    human_like_scorer = None
    if args.use_human_like:
        human_like_scorer = HumanLikeScorer(device)
        logger.info(f"Memory after human-like scorer: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    fluency_scorer = None
    if args.use_fluency:
        fluency_scorer = FluencyScorer(device)
        logger.info(f"Memory after fluency scorer: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    # Appropriateness scorer is always loaded (used for global reward)
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
        eval_on_start=not args.disable_eval_on_start,
        log_completions=True,
        max_completion_length=1024,
        max_prompt_length=2048,
        scale_rewards=False,
        gradient_accumulation_steps=8,
        optim=args.optimizer,
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
        report_to="wandb" if is_main_process else "none",
        num_train_epochs=2,
        resume_from_checkpoint=args.resume_from_checkpoint
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

    # Log checkpoint resumption info
    if args.resume_from_checkpoint:
        logger.info(f"=" * 80)
        logger.info(f"RESUMING TRAINING FROM CHECKPOINT: {args.resume_from_checkpoint}")
        logger.info(f"=" * 80)
        # Try to read trainer state to show what step we're resuming from
        import json
        trainer_state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
                logger.info(f"Resuming from global step: {trainer_state.get('global_step', 'unknown')}")
                logger.info(f"Resuming from epoch: {trainer_state.get('epoch', 'unknown')}")
                logger.info(f"Best metric so far: {trainer_state.get('best_metric', 'unknown')}")
        else:
            logger.warning(f"trainer_state.json not found in {args.resume_from_checkpoint}")
    else:
        logger.info("Starting training from scratch (no checkpoint)")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

if __name__ == "__main__":
    main()

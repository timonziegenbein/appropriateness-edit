# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a reinforcement learning system for identifying and editing inappropriate text while preserving the author's core message. The system uses GRPO (Generative Reinforcement Policy Optimization) to fine-tune large language models (specifically Llama-3.1-8B-Instruct) on the task of text appropriateness editing.

The model analyzes argumentative text and identifies inappropriate parts based on four dimensions:
- **Toxic Emotions**: Deceptive or overly intense emotional appeals
- **Missing Commitment**: Lack of seriousness or openness to other arguments
- **Missing Intelligibility**: Unclear meaning, irrelevance, or confusing reasoning
- **Other Reasons**: Severe orthographic errors or other issues

## Key Commands

### Training

Train the GRPO model:
```bash
python models/grpo.py --model_name unsloth/Llama-3.1-8B-Instruct --output_dir <output_directory>
```

### Evaluation

Generate predictions and evaluate a trained model:
```bash
python models/predict_and_evaluate.py --checkpoint_root <checkpoint_path> --output_jsonl <output_file.jsonl>
```

Use base model without LoRA:
```bash
python models/predict_and_evaluate.py --use_base_model_only --output_jsonl <output_file.jsonl>
```

### Testing Scorers

Test individual scorer components on example edits:
```bash
python scorers/test_scorers_on_example.py --model-path <model_path> --output-file <output.txt> --percentile-threshold <threshold> --ss-threshold <threshold>
```

### Model Management

Download required models:
```bash
python models/download_models.py
```

## Architecture

### Core Training Pipeline (models/grpo.py)

The GRPO trainer orchestrates the reinforcement learning loop:
- Uses two reward functions: `local_appropriateness_reward` and `global_appropriateness_reward`
- Loads datasets from HuggingFace: `timonziegenbein/appropriateness-corpus-extension-cleaned` (train) and `timonziegenbein/appropriateness-corpus-cleaned` (validation)
- Applies LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Uses vLLM for efficient inference during training
- Implements DR-GRPO loss variant with reward weighting

### Reward System (scorers/reward_functions.py)

**Local Appropriateness Reward**: Evaluates individual edits based on:
1. **Semantic Similarity** (scorers/semantic_similarity/): Ensures edits preserve meaning
2. **Human-Likeness** (scorers/human_like/): Checks if edit patterns match human editing behavior using a sequence-based language model
3. **Fluency** (scorers/fluency/): Verifies grammatical correctness of the edited text
4. Returns 1.0 if all three criteria pass, 0.0 otherwise (sparse binary reward)

**Global Appropriateness Reward**: Measures document-level inappropriateness reduction:
1. Applies all edits to reconstruct the full argument
2. Uses appropriateness classifier (scorers/appropriateness/) to score before/after
3. Returns `1.0 - inappropriateness_score_after` (higher reward for more appropriate text)

### Prompt Engineering (prompts/edit_inappropriate_text.py)

The `create_llm_prompt()` function generates structured prompts that:
- Specify the task of analyzing and editing inappropriate text
- Define the exact JSON output format with `sentence_edits` structure
- Include `tracked_changes` with inline `<del reason="...">` and `<ins>` tags
- Provide clear definitions of inappropriateness categories
- Include a complete example demonstrating the expected format

### Completion Processing (ops/)

**prompt_processor.py**: Extracts sentences and argument text from formatted prompts
**completion_processor.py**: Parses LLM completions using `json_repair` to handle malformed JSON, extracts edits from `<del>` and `<ins>` tags

### Scorer Components

Each scorer is a standalone module with:
- A scorer class (e.g., `SemanticSimilarityScorer`) that can be instantiated with a device
- A `calculate_*()` method that returns a binary (0.0/1.0) reward signal
- Pre-trained models stored in `scorers/<scorer_name>/` directories

**Human-like Scorer**: Uses a custom Transformer-based sequence model that predicts edit operation sequences (keep/del/add/replace) and compares perplexity against a threshold (1.4381). Located in `scorers/human_like/model_defs.py`.

**Semantic Similarity Scorer**: Uses BERTScore with DeBERTa-XLarge-MNLI and a threshold of 0.6144.

**Fluency Scorer**: Binary classifier trained on fluency judgments.

**Appropriateness Scorer**: Multi-label classifier predicting scores for all 14 inappropriateness dimensions.

## Datasets

Training uses the "appropriateness-corpus-extension-cleaned" dataset with pre-segmented sentences. Each example contains:
- `post_text`: Original argumentative text
- `issue`: Topic of the argument
- `sentences`: List of individual sentences

Validation uses examples with `Inappropriateness >= 0.5` from the base "appropriateness-corpus-cleaned" dataset.

## Key Implementation Details

### GRPO Configuration

- Per-device batch size: 2 with gradient accumulation steps: 8 (effective batch size: 16)
- Learning rate: 5e-6 with cosine scheduler
- Beta (KL penalty): 0.001857
- Reward weights: [0.0, 1.0] (only global reward is weighted)
- Uses 8-bit paged AdamW optimizer
- BF16 precision for training efficiency

### LoRA Configuration

- Rank (r): 16
- Alpha: 32
- Dropout: 0.1
- Applied to all linear layers in the causal LM

### Output Format

The model generates JSON with this structure:
```json
{
  "sentence_edits": [
    {
      "original_sentence": "exact original text",
      "rewritten_sentence": "fully rewritten text",
      "tracked_changes": "<del reason=\"Toxic Emotions\">old</del><ins>new</ins>"
    }
  ]
}
```

The `tracked_changes` field uses inline markup to show exactly what was changed and why, enabling granular reward computation.

## Important Notes

- All reward functions return sparse binary rewards (0.0 or 1.0), not continuous scores
- The training uses a dual-reward system: local (per-edit quality) and global (document-level inappropriateness reduction)
- Sentence tokenization uses spaCy's `en_core_web_sm` model in evaluation
- The system expects formatted input with enumerated sentences (e.g., "Sentence 1: ..., Sentence 2: ...")
- Logs are written to `training.log` and console during training

import re
import weave

@weave.op()
def process_prompt(prompt):
    """Processes the prompt and returns the original sentences and argument."""
    # First, cut off everything before "Now complete the task for the following:"
    task_start_match = re.search(r'Now complete the task for the following:(.*)', prompt, re.DOTALL)
    if task_start_match:
        # Extract only the actual task content, not the example
        actual_task_content = task_start_match.group(1).strip()
    else:
        # Fallback: use the entire prompt if the marker isn't found
        actual_task_content = prompt

    # Strip special tokens from the end of the task content
    # These tokens may appear at the end of prompts to indicate where the assistant should respond
    special_tokens = [
        "<|eot_id|>",
        "<|start_header_id|>",
        "assistant",
        "<|end_header_id|>",
    ]

    for token in special_tokens:
        if token in actual_task_content:
            # Find the position and cut off everything from that token onwards
            token_pos = actual_task_content.find(token)
            actual_task_content = actual_task_content[:token_pos].strip()

    # Now extract sentences from the actual task content
    sentences_match = re.search(r'Input Sentences:(.*)', actual_task_content, re.DOTALL)
    if not sentences_match:
        return None, None

    sentences_block = sentences_match.group(1).strip()
    original_sentences = re.findall(r'Sentence \d+: (.*?)(?=Sentence \d+:|$)', sentences_block)
    original_sentences = [s.strip() for s in original_sentences]
    original_argument = " ".join(original_sentences)

    return original_sentences, original_argument
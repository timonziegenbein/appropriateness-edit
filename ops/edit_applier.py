import weave
import nltk
from typing import List, Dict


@weave.op()
def apply_edits_to_argument(perfect_edits: List[Dict], original_sentences: List[str], original_argument: str) -> str:
    """
    Applies a list of perfect edits to the original argument.

    Args:
        perfect_edits: List of edit dictionaries with 'original_sentence', 'inappropriate_part', and 'rewritten_part'
        original_sentences: List of original sentences from the argument
        original_argument: The original argument text

    Returns:
        The modified argument with all perfect edits applied
    """
    if not perfect_edits:
        return original_argument

    # Group edits by sentence
    edits_by_sentence = {}
    for edit in perfect_edits:
        sentence = edit['original_sentence']
        if sentence not in edits_by_sentence:
            edits_by_sentence[sentence] = []
        edits_by_sentence[sentence].append(edit)

    # If no sentences provided, tokenize the argument
    if not original_sentences:
        original_sentences = nltk.sent_tokenize(original_argument)

    modified_sentences = list(original_sentences)

    # Apply edits sentence by sentence
    for j, sentence in enumerate(original_sentences):
        if sentence in edits_by_sentence:
            temp_sentence = modified_sentences[j]
            for edit in edits_by_sentence[sentence]:
                inappropriate_part = edit.get("inappropriate_part")
                rewritten_part = edit.get("rewritten_part")
                if inappropriate_part and rewritten_part and inappropriate_part in temp_sentence:
                    temp_sentence = temp_sentence.replace(inappropriate_part, rewritten_part, 1)
            modified_sentences[j] = temp_sentence

    modified_argument = " ".join(modified_sentences)
    return modified_argument

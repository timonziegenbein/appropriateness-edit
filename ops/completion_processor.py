import json_repair
import re
import logging
import weave

logger = logging.getLogger(__name__)

@weave.op()
def process_completion(completion, original_sentences=None):
    """
    Processes the completion from the LLM and returns a list of edits.

    Args:
        completion: The LLM completion string
        original_sentences: List of original sentences to map sentence_id to actual sentence text

    Returns:
        List of edit dictionaries with sentence_id, inappropriate_part, rewritten_part, and reason
    """
    valid_edits = []
    try:
        data = json_repair.loads(completion)
        if isinstance(data, dict) and "sentence_edits" in data:
            sentence_edits = data.get("sentence_edits", [])

            for sentence_edit in sentence_edits:
                sentence_id = sentence_edit.get("sentence_id")
                tracked_changes = sentence_edit.get("tracked_changes", "")

                if sentence_id is None or not tracked_changes:
                    continue

                # Convert sentence_id to integer if it's a string
                try:
                    sentence_id = int(sentence_id)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid sentence_id: {sentence_id}, skipping edit")
                    continue

                parsed_edits = re.findall(r'<del reason=[""](.*?)?[""]>(.+?)</del><ins>(.*?)</ins>', tracked_changes)

                for reason, inappropriate_part, rewritten_part in parsed_edits:
                    if not all([inappropriate_part, rewritten_part, reason]):
                        continue

                    valid_edits.append({
                        "sentence_id": sentence_id,
                        "inappropriate_part": inappropriate_part,
                        "rewritten_part": rewritten_part,
                        "reason": reason,
                    })

    except Exception as e:
        logger.error(f"Could not parse completion: {completion}, error: {e}")
        pass
    return valid_edits

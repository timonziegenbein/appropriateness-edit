import json_repair
import logging
import weave

logger = logging.getLogger(__name__)

@weave.op()
def process_completion(completion, original_sentences=None):
    """
    Processes the completion from the LLM and returns a list of edits.
    Parses the JSON format with a structured 'edits' array.

    Args:
        completion: The LLM completion string in JSON format.
        original_sentences: List of original sentences (unused, kept for API compatibility).

    Returns:
        List of edit dictionaries with sentence_id, inappropriate_part, rewritten_part, and reason.
    """
    valid_edits = []

    try:
        # Use json_repair to handle potential malformations in the LLM's JSON output.
        data = json_repair.loads(completion)
        if not isinstance(data, dict) or "sentence_edits" not in data:
            logger.warning(f"Completion missing 'sentence_edits' key: {completion[:100]}...")
            return valid_edits

        sentence_edits = data.get("sentence_edits", [])
        logger.info(f"Found {len(sentence_edits)} sentence_edits in completion")

        for sentence_edit in sentence_edits:
            sentence_id = sentence_edit.get("sentence_id")
            edits = sentence_edit.get("edits", [])
            logger.info(f"Processing sentence_id={sentence_id} with {len(edits)} edits")

            if sentence_id is None or not edits:
                logger.warning(f"Skipping sentence_edit: sentence_id={sentence_id}, num_edits={len(edits)}")
                continue

            # Convert sentence_id to integer if it's a string
            try:
                sentence_id = int(sentence_id)
            except (ValueError, TypeError):
                logger.warning(f"Invalid sentence_id: {sentence_id}, skipping edit")
                continue

            for edit in edits:
                inappropriate_part = edit.get("inappropriate_part")
                rewritten_part = edit.get("rewritten_part")
                reason = edit.get("reason")

                # Ensure all required fields are present.
                # Note: rewritten_part can be empty string (deletion), so check for None specifically
                if inappropriate_part is None or rewritten_part is None or not reason:
                    continue

                # Handle cases where the model outputs lists instead of strings
                if isinstance(inappropriate_part, list):
                    if len(inappropriate_part) > 0:
                        logger.warning(f"inappropriate_part is a list, taking first element: {inappropriate_part}")
                        inappropriate_part = str(inappropriate_part[0])
                    else:
                        logger.warning(f"inappropriate_part is an empty list, skipping edit")
                        continue

                if isinstance(rewritten_part, list):
                    if len(rewritten_part) > 0:
                        logger.warning(f"rewritten_part is a list, taking first element: {rewritten_part}")
                        rewritten_part = str(rewritten_part[0])
                    else:
                        logger.warning(f"rewritten_part is an empty list, skipping edit")
                        continue

                if isinstance(reason, list):
                    if len(reason) > 0:
                        logger.warning(f"reason is a list, taking first element: {reason}")
                        reason = str(reason[0])
                    else:
                        logger.warning(f"reason is an empty list, skipping edit")
                        continue

                # Convert all to strings to be safe
                inappropriate_part = str(inappropriate_part)
                rewritten_part = str(rewritten_part)
                reason = str(reason)

                valid_edits.append({
                    "sentence_id": sentence_id,
                    "inappropriate_part": inappropriate_part,
                    "rewritten_part": rewritten_part,
                    "reason": reason,
                })

    except Exception as e:
        logger.error(f"Could not parse completion: {completion[:100]}..., error: {e}")
        # Return whatever was successfully parsed before the error.
        pass

    logger.info(f"Extracted {len(valid_edits)} total valid edits from completion")
    return valid_edits

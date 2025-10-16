import re
import logging
from typing import List
from transformers import pipeline, AutoTokenizer
import weave

from ops.latexdiff_parser import DirectLatexdiffParser, fuzzy_post_process_edits

logger = logging.getLogger(__name__)

class FluencyScorer:
    def __init__(self, device):
        self.device = device
        self.checker, self.corrector = self._load_fluency_models(device)

    def _load_fluency_models(self, device):
        """Loads the fluency models."""
        #fluency_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
        fluency_checker = pipeline("text-classification", model="textattack/roberta-base-CoLA", device=device)
        fluency_corrector = pipeline("text2text-generation", "pszemraj/flan-t5-large-grammar-synthesis", device=device)
        return fluency_checker, fluency_corrector

    @weave.op()
    def _get_grammar_corrections(self, text):
        """
        Checks grammar and returns a list of corrections and the corrected text.
        """
        sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)
        corrections = []
        full_corrected_text = ""
        current_pos = 0
        parser = DirectLatexdiffParser()

        for sentence in sentences:
            results = self.checker(sentence)
            corrected_sentence = sentence
            if results[0]['label'] != 'LABEL_1' or (results[0]['label'] == 'LABEL_1' and results[0]['score'] < 0.9):
                corrected_sentence = self.corrector(sentence)[0]['generated_text']
                if corrected_sentence.strip() != sentence.strip():
                    diff = parser.parse_latex_diff(sentence, corrected_sentence, 'scorers/fluency/temp_output')
                    if diff is None:
                        logger.warning(f"Could not parse diff for sentence: {sentence}")
                        continue
                    diff['before_revision'] = sentence
                    if diff and diff['edit_actions']:
                        fuzzy_post_process_edits([diff])
                        for action in diff['edit_actions']:
                            corrections.append({
                                'start': current_pos + action['start_char_pos'],
                                'end': current_pos + action['end_char_pos'],
                                'original': action.get('before', ''),
                                'corrected': action.get('after', '')
                            })
            full_corrected_text += corrected_sentence + " "
            current_pos += len(sentence) + 1  # +1 for the space

        return corrections, full_corrected_text.strip()

    @weave.op()
    def calculate_fluency(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> float:
        if not isinstance(original_sentence, str) or len(original_sentence.strip()) == 0:
            logger.warning("Received empty or invalid text in calculate_fluency.")
            return 0.0

        rewritten_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)

        if self.checker is None or self.corrector is None:
            logger.error("Fluency checker or corrector not loaded correctly.")
            return 0.0

        try:
            # Get grammar corrections on the rewritten sentence
            corrections, corrected_text = self._get_grammar_corrections(rewritten_sentence)

            if not corrections:
                return 1.0

            # The user's edit is the replacement string in the rewritten_sentence
            edit_start = original_sentence.find(inappropriate_part) - 2
            edit_end = edit_start + len(rewritten_part) + 3

            for correction in corrections:
                # Check for overlap
                if max(edit_start, correction['start']) <= min(edit_end, correction['end']):
                    return 0.0

            return 1.0

        except Exception as e:
            logger.error(f"Error in calculate_fluency: {e}")
            return 0.0

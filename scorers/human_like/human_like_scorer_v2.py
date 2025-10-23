import torch
import torch.nn as nn
import numpy as np
import difflib
import logging
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import weave

from scorers.human_like.model_defs import LanguageModel

logger = logging.getLogger(__name__)

# Updated vocabulary with keep-in-edit token
hl_vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}

class HumanLikeScorerV2:
    """
    Version 2 of HumanLikeScorer with improvements:
    - Operates on sentence level instead of document level
    - Uses 'keep-in-edit' token for unchanged tokens within edit regions
    - Filters edits to only include those within single sentences
    """

    def __init__(self, device, threshold=1.1465, max_len=500):
        self.device = device
        self.threshold = threshold
        self.max_len = max_len
        self.model, self.tokenizer = self._load_hl_model(device)

    def _load_hl_model(self, device):
        """Loads the human-like model and tokenizer."""
        hl_embedding_dim = 200
        hl_nhead = 2
        hl_nhid = 200
        hl_nlayers = 2
        human_like_model = LanguageModel(len(hl_vocab), hl_embedding_dim, hl_nhead, hl_nhid, hl_nlayers).to(device)

        # Try to load v2 model, fallback to v1 if not available
        try:
            human_like_model.load_state_dict(
                torch.load("scorers/human_like/human_like_language_model_v3.pth", map_location=device)
            )
            logger.info("Loaded v3 model (sentence-level with keep-in-edit)")
        except FileNotFoundError:
            logger.warning("v3 model not found, using v2 model")
            human_like_model.load_state_dict(
                torch.load("scorers/human_like/human_like_language_model_v2.pth", map_location=device)
            )

        human_like_model.eval()
        human_like_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        return human_like_model, human_like_tokenizer

    def _calculate_perplexity_for_sequence(self, sequence):
        """Calculate perplexity for a given edit operation sequence."""
        logger.debug("Calculating perplexity for sequence:")
        logger.debug(f"  Input sequence: {sequence}")
        sequence_as_int = [hl_vocab.get(token, 0) for token in sequence]
        logger.debug(f"  Sequence as int: {sequence_as_int}")

        if len(sequence_as_int) <= 1:
            logger.debug("  Sequence is too short, returning inf")
            return float('inf')

        input_seq = sequence_as_int[:-1]
        target_seq = sequence_as_int[1:]
        logger.debug(f"  Input seq: {input_seq}")
        logger.debug(f"  Target seq: {target_seq}")

        padded_input = np.array(input_seq[:self.max_len] + [hl_vocab['<pad>']]*(self.max_len - len(input_seq)) if len(input_seq) < self.max_len else input_seq[:self.max_len])
        padded_target = np.array(target_seq[:self.max_len] + [hl_vocab['<pad>']]*(self.max_len - len(target_seq)) if len(target_seq) < self.max_len else target_seq[:self.max_len])
        logger.debug(f"  Padded input: {padded_input}")
        logger.debug(f"  Padded target: {padded_target}")

        inputs = torch.from_numpy(padded_input).long().unsqueeze(0).to(self.device)
        targets = torch.from_numpy(padded_target).long().unsqueeze(0).to(self.device)
        logger.debug(f"  Inputs tensor shape: {inputs.shape}")
        logger.debug(f"  Targets tensor shape: {targets.shape}")

        criterion = nn.CrossEntropyLoss(ignore_index=hl_vocab['<pad>'])

        with torch.no_grad():
            output = self.model(inputs)
            logger.debug(f"  Output tensor shape: {output.shape}")
            loss = criterion(output.view(-1, len(hl_vocab)), targets.view(-1))
            perplexity = torch.exp(loss)
            logger.debug(f"  Loss: {loss.item()}, Perplexity: {perplexity.item()}")
            return perplexity.item()

    def _get_sentence_boundaries(self, sents_char_pos: List[int]) -> List[Tuple[int, int]]:
        """Convert sents_char_pos array to list of (start, end) tuples."""
        boundaries = []
        prev_pos = 0
        for pos in sents_char_pos:
            boundaries.append((prev_pos, pos))
            prev_pos = pos + 1
        return boundaries

    def _find_sentence_for_edit(self, start_char: int, end_char: int,
                                sentence_boundaries: List[Tuple[int, int]]) -> Optional[int]:
        """
        Find which sentence an edit belongs to.
        Returns sentence index if edit is fully contained, None otherwise.
        """
        for sent_idx, (sent_start, sent_end) in enumerate(sentence_boundaries):
            if start_char >= sent_start and end_char <= sent_end:
                return sent_idx
        return None

    @weave.op()
    def _generate_sequence_for_edit_in_sentence(self, sentence_text: str, start_char_in_sent: int,
                                                 end_char_in_sent: int, rewritten_part: str) -> Optional[List[str]]:
        """
        Generate edit operation sequence for a single edit within a sentence.

        Args:
            sentence_text: The text of the sentence
            start_char_in_sent: Start position of edit relative to sentence start
            end_char_in_sent: End position of edit relative to sentence start
            rewritten_part: The replacement text

        Returns:
            List of operation tags (keep, del, add, replace, keep-in-edit)
        """
        # Tokenize sentence
        encoding = self.tokenizer(sentence_text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offsets = encoding['offset_mapping']

        if len(tokens) == 0:
            return None

        # Initialize all tags as 'keep'
        tags = ['keep'] * len(tokens)

        # Find token range for this edit
        tok_start = -1
        tok_end = -1
        for i, (token_start, token_end) in enumerate(offsets):
            if start_char_in_sent < token_end and end_char_in_sent > token_start:
                if tok_start == -1:
                    tok_start = i
                tok_end = i

        if tok_start == -1:
            return tags  # Edit doesn't align with tokens, return all keeps

        # Get before and after tokens
        before_tokens = tokens[tok_start:tok_end+1]
        if not isinstance(rewritten_part, str):
            rewritten_part = ""
        after_tokens = self.tokenizer.tokenize(rewritten_part)

        # Use difflib to compare
        matcher = difflib.SequenceMatcher(None, before_tokens, after_tokens)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Tokens are kept but inside edit region
                for i in range(i1, i2):
                    tags[tok_start + i] = 'keep-in-edit'
            elif tag == 'delete':
                for i in range(i1, i2):
                    tags[tok_start + i] = 'del'
            elif tag == 'replace':
                for i in range(i1, i2):
                    tags[tok_start + i] = 'replace'
            elif tag == 'insert':
                # Mark position where insertion happens
                if i1 > 0:
                    insert_pos = tok_start + i1 - 1
                else:
                    insert_pos = tok_start
                if insert_pos < len(tags) and tags[insert_pos] == 'keep':
                    tags[insert_pos] = 'add'

        return tags

    @weave.op()
    def calculate_human_likeness(self, original_argument: str, original_sentence: str,
                                inappropriate_part: str, rewritten_part: str,
                                sents_char_pos: Optional[List[int]] = None) -> float:
        """
        Calculate human-likeness reward for an edit (v2 with sentence-level filtering).

        Args:
            original_argument: The full document text
            original_sentence: The sentence containing the edit
            inappropriate_part: The text being replaced
            rewritten_part: The replacement text
            sents_char_pos: Sentence boundary positions (optional, for filtering)

        Returns:
            1.0 if edit is human-like (perplexity <= threshold), 0.0 otherwise
        """
        # Find the edit location
        start_char_in_sentence = original_sentence.find(inappropriate_part)
        if start_char_in_sentence == -1:
            logger.warning("Could not find inappropriate_part in original_sentence")
            return 0.0

        sentence_start_in_argument = original_argument.find(original_sentence)
        if sentence_start_in_argument == -1:
            logger.warning("Could not find original_sentence in original_argument")
            return 0.0

        start_char = sentence_start_in_argument + start_char_in_sentence
        end_char = start_char + len(inappropriate_part)

        # If sentence boundaries provided, check if edit is within a single sentence
        if sents_char_pos:
            sentence_boundaries = self._get_sentence_boundaries(sents_char_pos)
            sent_idx = self._find_sentence_for_edit(start_char, end_char, sentence_boundaries)

            if sent_idx is None:
                logger.debug("Edit spans multiple sentences, filtering out")
                return 0.0

            # Extract sentence and generate sequence
            sent_start, sent_end = sentence_boundaries[sent_idx]
            sentence_text = original_argument[sent_start:sent_end+1]
            start_in_sent = start_char - sent_start
            end_in_sent = end_char - sent_start

            sequence = self._generate_sequence_for_edit_in_sentence(
                sentence_text, start_in_sent, end_in_sent, rewritten_part
            )
        else:
            # Fallback to old document-level approach
            sequence = self._generate_sequence_for_edit_in_sentence(
                original_sentence, start_char_in_sentence,
                start_char_in_sentence + len(inappropriate_part), rewritten_part
            )

        human_like_reward = 0.0
        if sequence:
            perplexity = self._calculate_perplexity_for_sequence(sequence)
            if perplexity <= self.threshold:
                human_like_reward = 1.0

        return human_like_reward

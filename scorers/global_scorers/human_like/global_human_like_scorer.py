import torch
import torch.nn as nn
import numpy as np
import difflib
import logging
from typing import List, Dict
from transformers import AutoTokenizer
import weave

from scorers.local_scorers.human_like.model_defs import LanguageModel

logger = logging.getLogger(__name__)

# V2 vocab with 'keep-in-edit' token
hl_vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}

class GlobalHumanLikeScorer:
    """
    Global human-like scorer that evaluates document-level editing patterns.

    Unlike the local human-like scorer that evaluates individual edits,
    this scorer evaluates the perplexity of the entire document's edit sequence.
    This addresses the issue where many individually human-like edits might
    create an overall editing pattern that is not human-like (e.g., editing
    every single sentence in a document).

    Uses the same model architecture as the local scorer but with a different
    threshold appropriate for document-level patterns.
    """

    def __init__(self, device, model_path="scorers/global_scorers/human_like/models/global_human_like_v1.pth",
                 threshold=2.3567, max_len=1500):
        """
        Initialize GlobalHumanLikeScorer.

        Args:
            device: torch device
            model_path: Path to the trained model checkpoint (same as local scorer)
            threshold: Perplexity threshold for document-level human-like classification
                      (default: 5.0, higher than local threshold since document-level
                      patterns typically have higher perplexity)
            max_len: Maximum sequence length (default: 1500, longer than local to accommodate
                    full document edit sequences)
        """
        self.device = device
        self.threshold = threshold
        self.max_len = max_len
        self.model_path = model_path

        self.model, self.tokenizer = self._load_hl_model(device)
        logger.info(f"Loaded GlobalHumanLikeScorer: model={model_path}, threshold={threshold}")

    def _load_hl_model(self, device):
        """Loads the human-like model and tokenizer."""
        hl_embedding_dim = 200
        hl_nhead = 2
        hl_nhid = 200
        hl_nlayers = 2
        vocab_size = len(hl_vocab)

        # Create model with dropout=0 for inference (no dropout during evaluation)
        human_like_model = LanguageModel(
            vocab_size, hl_embedding_dim, hl_nhead, hl_nhid, hl_nlayers, dropout=0.0
        ).to(device)

        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            human_like_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            human_like_model.load_state_dict(checkpoint)

        human_like_model.eval()
        human_like_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        return human_like_model, human_like_tokenizer

    def _calculate_perplexity_for_sequence(self, sequence):
        """Calculate perplexity for an edit operation sequence."""
        logger.debug("Calculating perplexity for sequence:")
        logger.debug(f"  Input sequence length: {len(sequence)}")
        sequence_as_int = [hl_vocab.get(token, 0) for token in sequence]

        if len(sequence_as_int) <= 1:
            logger.debug("  Sequence is too short, returning inf")
            return float('inf')

        input_seq = sequence_as_int[:-1]
        target_seq = sequence_as_int[1:]

        padded_input = np.array(input_seq[:self.max_len] + [hl_vocab['<pad>']]*(self.max_len - len(input_seq)) if len(input_seq) < self.max_len else input_seq[:self.max_len])
        padded_target = np.array(target_seq[:self.max_len] + [hl_vocab['<pad>']]*(self.max_len - len(target_seq)) if len(target_seq) < self.max_len else target_seq[:self.max_len])

        inputs = torch.from_numpy(padded_input).long().unsqueeze(0).to(self.device)
        targets = torch.from_numpy(padded_target).long().unsqueeze(0).to(self.device)

        criterion = nn.CrossEntropyLoss(ignore_index=hl_vocab['<pad>'])

        with torch.no_grad():
            output = self.model(inputs)
            loss = criterion(output.view(-1, len(hl_vocab)), targets.view(-1))
            perplexity = torch.exp(loss)
            logger.debug(f"  Loss: {loss.item()}, Perplexity: {perplexity.item()}")
            return perplexity.item()

    def _generate_document_edit_sequence(self, original_text: str, edits: List[Dict]):
        """
        Generate a document-level edit sequence from all edits.

        Args:
            original_text: Original document text
            edits: List of edit dictionaries with 'inappropriate_part' and 'rewritten_part'

        Returns:
            List of edit operation tokens representing the document-level pattern
        """
        # Tokenize the original document
        encoding = self.tokenizer(original_text, return_offsets_mapping=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offsets = encoding['offset_mapping']

        # Initialize all tokens as 'keep'
        tags = ['keep'] * len(tokens)

        # Apply each edit to the tag sequence
        for edit in edits:
            inappropriate_part = edit.get('inappropriate_part', '')
            rewritten_part = edit.get('rewritten_part', '')

            if not inappropriate_part:
                continue

            # Find the inappropriate part in the original text
            start_char = original_text.find(inappropriate_part)
            if start_char == -1:
                logger.warning(f"Could not find edit in text: {inappropriate_part[:50]}...")
                continue

            end_char = start_char + len(inappropriate_part)

            # Find token indices that overlap with this edit
            token_start_index = -1
            token_end_index = -1
            for i, offset in enumerate(offsets):
                token_start, token_end = offset
                if start_char < token_end and end_char > token_start:
                    if token_start_index == -1:
                        token_start_index = i
                    token_end_index = i

            if token_start_index == -1:
                continue

            # Determine edit type by comparing before and after tokens
            before_edit_tokens = tokens[token_start_index:token_end_index+1]
            after_edit_tokens = self.tokenizer.tokenize(rewritten_part) if rewritten_part else []

            # Use difflib to get fine-grained edit operations
            matcher = difflib.SequenceMatcher(None, before_edit_tokens, after_edit_tokens)
            current_idx = token_start_index
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    # Tokens that match within an edit region are 'keep-in-edit'
                    for idx in range(current_idx, current_idx + (i2 - i1)):
                        if idx < len(tags):
                            tags[idx] = 'keep-in-edit'
                    current_idx += (i2 - i1)
                elif tag == 'delete':
                    # Mark as deletions
                    for idx in range(current_idx, current_idx + (i2 - i1)):
                        if idx < len(tags):
                            tags[idx] = 'del'
                    current_idx += (i2 - i1)
                elif tag == 'replace':
                    # Mark as replacements
                    for idx in range(current_idx, current_idx + (i2 - i1)):
                        if idx < len(tags):
                            tags[idx] = 'replace'
                    current_idx += (i2 - i1)
                elif tag == 'insert':
                    # Insertions are represented by 'add' tokens
                    # Insert them at the current position
                    for _ in range(j2 - j1):
                        tags.insert(current_idx, 'add')
                        current_idx += 1

        return tags

    @weave.op()
    def calculate_global_human_likeness(self, original_text: str, edits: List[Dict]):
        """
        Calculate document-level human-likeness of all edits combined.

        Args:
            original_text: Original document text
            edits: List of edit dictionaries with 'inappropriate_part' and 'rewritten_part'

        Returns:
            tuple: (binary_score, perplexity) where binary_score is 1.0 if perplexity <= threshold, 0.0 otherwise
        """
        if not edits:
            # No edits is perfectly human-like
            return 1.0, 0.0

        try:
            # Generate document-level edit sequence
            sequence = self._generate_document_edit_sequence(original_text, edits)

            if not sequence:
                logger.warning("Failed to generate edit sequence")
                return 0.0, float('inf')

            # Calculate perplexity
            perplexity = self._calculate_perplexity_for_sequence(sequence)

            # Binary threshold
            binary_score = 1.0 if perplexity <= self.threshold else 0.0

            logger.debug(f"Global human-likeness: perplexity={perplexity:.4f}, threshold={self.threshold}, binary={binary_score}")
            return binary_score, perplexity

        except Exception as e:
            logger.error(f"Global human-likeness calculation failed: {e}")
            return 0.0, float('inf')

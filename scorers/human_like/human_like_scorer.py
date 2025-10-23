import torch
import torch.nn as nn
import numpy as np
import difflib
import logging
from typing import List, Dict
from transformers import AutoTokenizer
import weave

from scorers.human_like.model_defs import LanguageModel

logger = logging.getLogger(__name__)

# V2 vocab with 'keep-in-edit' token
hl_vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}

class HumanLikeScorer:
    def __init__(self, device, model_path="scorers/human_like/models/human_like_v4.pth",
                 threshold=2.452725653648376, max_len=500):
        """
        Initialize HumanLikeScorer.

        Args:
            device: torch device
            model_path: Path to the trained model checkpoint
            threshold: Perplexity threshold for human-like classification (default: 2.4527, P99 from v4)
            max_len: Maximum sequence length
        """
        self.device = device
        self.threshold = threshold
        self.max_len = max_len
        self.model_path = model_path

        self.model, self.tokenizer = self._load_hl_model(device)
        logger.info(f"Loaded HumanLikeScorer: model={model_path}, threshold={threshold}")

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

    @weave.op()
    def _generate_sequence_for_edit(self, before_revision, start_char, end_char, rewritten_part):
        encoding = self.tokenizer(before_revision, return_offsets_mapping=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offsets = encoding['offset_mapping']
        
        if len(tokens) > 0:
            token_start_index = -1
            token_end_index = -1
            for i, offset in enumerate(offsets):
                token_start, token_end = offset
                if start_char < token_end and end_char > token_start:
                    if token_start_index == -1:
                        token_start_index = i
                    token_end_index = i
            
            if token_start_index != -1:
                tags = []
                tags.extend(['keep'] * token_start_index)

                before_edit_tokens = tokens[token_start_index:token_end_index+1]
                if not isinstance(rewritten_part, str):
                    rewritten_part = ""
                after_edit_tokens = self.tokenizer.tokenize(rewritten_part)
                
                matcher = difflib.SequenceMatcher(None, before_edit_tokens, after_edit_tokens)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        tags.extend(['keep'] * (i2 - i1))
                    elif tag == 'delete':
                        tags.extend(['del'] * (i2 - i1))
                    elif tag == 'replace':
                        tags.extend(['replace'] * (i2 - i1))
                    elif tag == 'insert':
                        tags.extend(['add'] * (j2 - j1))

                tags.extend(['keep'] * (len(tokens) - token_end_index - 1))
                
                return tags
            else:
                return ['keep'] * len(tokens)
                
        return None

    @weave.op()
    def calculate_human_likeness(self, original_argument, original_sentence, inappropriate_part, rewritten_part):
        """
        Calculates the human-likeness reward for an edit.
        """
        start_char_in_sentence = original_sentence.find(inappropriate_part)
        sentence_start_in_argument = original_argument.find(original_sentence)
        start_char = sentence_start_in_argument + start_char_in_sentence
        end_char = start_char + len(inappropriate_part)

        sequence = self._generate_sequence_for_edit(original_argument, start_char, end_char, rewritten_part)
        human_like_reward = 0.0
        if sequence:
            perplexity = self._calculate_perplexity_for_sequence(sequence)
            if perplexity <= self.threshold:
                human_like_reward = 1.0
        return human_like_reward
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import weave

logger = logging.getLogger(__name__)


class FluencyScorer:
    """
    ModernBERT-based fluency scorer.
    This is the default scorer that uses a fine-tuned ModernBERT model.
    """

    def __init__(self, device, model_path: str = None):
        """
        Initialize the ModernBERT-based fluency scorer.

        Args:
            device: Device to run the model on (cuda or cpu)
            model_path: Path to the fine-tuned ModernBERT checkpoint.
                       If None, uses the default path.
        """
        self.device = device

        # Default model path - update this to point to your best checkpoint
        if model_path is None:
            model_path = "/mnt/home/tziegenb/appropriateness-edit/scorers/local_scorers/fluency/modernbert_gec_extended_v2/checkpoint-500"

        self.model_path = model_path
        self.model, self.tokenizer = self._load_model(device, model_path)

    def _load_model(self, device, model_path):
        """Load the ModernBERT model and tokenizer."""
        logger.info(f"Loading ModernBERT fluency model from: {model_path}")

        try:
            # Load tokenizer from base model
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

            # Load fine-tuned model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                attn_implementation="eager"  # Use eager attention for compatibility
            )
            model.to(device)
            model.eval()

            logger.info("ModernBERT fluency model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading ModernBERT model: {e}")
            raise RuntimeError(f"Failed to load ModernBERT fluency model from {model_path}: {e}")

    @weave.op()
    def calculate_fluency(self, original_sentence: str, inappropriate_part: str, rewritten_part: str) -> float:
        """
        Calculate fluency score for an edit.

        Args:
            original_sentence: The original sentence
            inappropriate_part: The part to be replaced
            rewritten_part: The replacement text

        Returns:
            1.0 if the edit is fluent, 0.0 otherwise
        """
        if not isinstance(original_sentence, str) or len(original_sentence.strip()) == 0:
            logger.warning("Received empty or invalid text in calculate_fluency.")
            return 0.0

        try:
            # Construct before and after sentences
            text_before = original_sentence
            text_after = original_sentence.replace(inappropriate_part, rewritten_part, 1)

            # Tokenize
            inputs = self.tokenizer(
                text_before,
                text_after,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()

            # Convert to binary score (1 = fluent, 0 = non-fluent)
            return float(predicted_class)

        except Exception as e:
            logger.error(f"Error in calculate_fluency: {e}")
            return 0.0

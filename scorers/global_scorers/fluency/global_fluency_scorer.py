import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import weave

logger = logging.getLogger(__name__)


class GlobalFluencyScorer:
    """
    Global fluency scorer for document-level evaluation.

    Unlike the local fluency scorer that evaluates individual edits,
    this scorer evaluates the overall fluency of the entire edited document.
    This addresses the issue where many locally fluent edits might result
    in a document that lacks overall coherence or fluency.

    Uses the tcapelle/fluency-scorer model from HuggingFace to evaluate
    the full document rather than individual sentence pairs.
    """

    def __init__(self, device, model_name: str = "tcapelle/fluency-scorer", threshold: float = 0.5):
        """
        Initialize the global fluency scorer.

        Args:
            device: Device to run the model on (cuda or cpu)
            model_name: HuggingFace model name for the fluency scorer
                       (default: tcapelle/fluency-scorer)
            threshold: Confidence threshold for fluency classification (default: 0.5)
        """
        self.device = device
        self.threshold = threshold
        self.model_name = model_name

        self.model, self.tokenizer = self._load_model(device, model_name)
        logger.info(f"Initialized GlobalFluencyScorer with model={model_name}, threshold={threshold}")

    def _load_model(self, device, model_name):
        """Load the fluency model from HuggingFace."""
        logger.info(f"Loading fluency model from HuggingFace: {model_name}")

        try:
            # Load tokenizer and model from HuggingFace
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            model.to(device)
            model.eval()

            logger.info("Fluency model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading fluency model: {e}")
            raise RuntimeError(f"Failed to load fluency model from {model_name}: {e}")

    @weave.op()
    def calculate_global_fluency(self, original_text: str, edited_text: str):
        """
        Calculate document-level fluency score.

        Args:
            original_text: The original document text
            edited_text: The edited document text

        Returns:
            tuple: (binary_score, confidence) where binary_score is 1.0 if fluent, 0.0 otherwise
        """
        if not isinstance(original_text, str) or not isinstance(edited_text, str):
            logger.warning("Invalid input text types")
            return 0.0, 0.0

        if len(original_text.strip()) == 0 or len(edited_text.strip()) == 0:
            logger.warning("Empty text provided")
            return 0.0, 0.0

        # If texts are identical, assume fluency is preserved
        if original_text.strip() == edited_text.strip():
            return 1.0, 1.0

        try:
            # Tokenize the full documents
            # The model was trained on sentence pairs, so we use the same format
            # but with full documents instead of individual sentences
            inputs = self.tokenizer(
                original_text,
                edited_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # May need to handle longer documents
                padding=True
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Get probabilities using softmax
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = probs[0][predicted_class].item()

            # Binary score: 1 if fluent (class 1), 0 if non-fluent (class 0)
            binary_score = float(predicted_class)

            logger.debug(f"Global fluency: class={predicted_class}, confidence={confidence:.4f}, binary={binary_score}")
            return binary_score, confidence

        except Exception as e:
            logger.error(f"Error in calculate_global_fluency: {e}")
            return 0.0, 0.0

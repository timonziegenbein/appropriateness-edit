import logging
import weave
from weave.scorers import WeaveFluencyScorerV1

logger = logging.getLogger(__name__)


class GlobalFluencyScorer:
    """
    Global fluency scorer for document-level evaluation.

    Unlike the local fluency scorer that evaluates individual edits,
    this scorer evaluates the overall fluency of the entire edited document.
    This addresses the issue where many locally fluent edits might result
    in a document that lacks overall coherence or fluency.

    Uses Weave's built-in WeaveFluencyScorerV1 to evaluate the full document.
    """

    def __init__(self, device, model_name: str = None, threshold: float = 0.5):
        """
        Initialize the global fluency scorer.

        Args:
            device: Device to run the model on (passed to Weave scorer)
            model_name: Model name (ignored, using Weave's default)
            threshold: Confidence threshold for fluency classification (default: 0.5)
        """
        self.device = device
        self.threshold = threshold

        # Initialize Weave's built-in fluency scorer with explicit device
        # Convert torch device to string format for Weave
        weave_device = str(device)

        self.scorer = WeaveFluencyScorerV1(device=weave_device, threshold=threshold)
        logger.info(f"Initialized GlobalFluencyScorer using Weave's WeaveFluencyScorerV1 on device={weave_device}")

    @weave.op()
    def calculate_global_fluency(self, original_text: str, edited_text: str):
        """
        Calculate document-level fluency score.

        Args:
            original_text: The original document text (for compatibility, not used)
            edited_text: The edited document text to evaluate

        Returns:
            tuple: (binary_score, confidence) where binary_score is 1.0 if fluent, 0.0 otherwise
        """
        if not isinstance(edited_text, str):
            logger.warning("Invalid input text type")
            return 0.0, 0.0

        if len(edited_text.strip()) == 0:
            logger.warning("Empty text provided")
            return 0.0, 0.0

        try:
            # Use Weave's built-in fluency scorer
            result = self.scorer.score(output=edited_text)

            # Convert Weave's result to our expected format
            # result.passed is a boolean, convert to 1.0 or 0.0
            binary_score = 1.0 if result.passed else 0.0

            # Extract fluency score from metadata
            # Weave's scorer returns score in metadata dict
            confidence = result.metadata.get('score', binary_score)

            logger.debug(f"Global fluency: passed={result.passed}, binary={binary_score}, confidence={confidence}")
            return binary_score, confidence

        except Exception as e:
            logger.error(f"Error in calculate_global_fluency: {e}")
            return 0.0, 0.0

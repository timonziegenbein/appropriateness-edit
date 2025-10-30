import logging
from sentence_transformers import SentenceTransformer
import weave

logger = logging.getLogger(__name__)

class GlobalSemanticSimilarityScorer:
    """
    Global semantic similarity scorer that evaluates document-level semantic preservation.

    Unlike the local semantic similarity scorer that operates on individual edits,
    this scorer evaluates the overall semantic similarity between the original
    and edited full arguments using the same SentenceTransformer model.

    This addresses the issue where many local edits that individually preserve
    semantics might cumulatively shift the document's meaning too far.

    Uses the same model as the local scorer but with a different threshold
    appropriate for document-level evaluation.
    """

    def __init__(self, device, threshold=0.652266):
        """
        Initialize the global semantic similarity scorer.

        Args:
            device: torch device for model inference
            threshold: Minimum similarity score to consider documents semantically similar
                      (default: 0.80, higher than local threshold of 0.6144 to be more strict
                      about document-level semantic preservation)
        """
        self.device = device
        self.threshold = threshold
        self.model = self._load_model(device)
        logger.info(f"Initialized GlobalSemanticSimilarityScorer with threshold={threshold}")

    def _load_model(self, device):
        """Load the same semantic similarity model as the local scorer."""
        return SentenceTransformer('google/embeddinggemma-300m', device=device)

    @weave.op()
    def calculate_global_semantic_similarity(self, original_text: str, edited_text: str):
        """
        Calculate document-level semantic similarity.

        Args:
            original_text: Original argument text
            edited_text: Edited argument text (after applying edits)

        Returns:
            tuple: (binary_score, similarity_score) where binary_score is 1.0 if similarity >= threshold, 0.0 otherwise
        """
        if not isinstance(original_text, str) or not isinstance(edited_text, str):
            logger.warning("Invalid input text types")
            return 0.0, 0.0

        if len(original_text.strip()) == 0 or len(edited_text.strip()) == 0:
            logger.warning("Empty text provided")
            return 0.0, 0.0

        # If texts are identical, return perfect score
        if original_text.strip() == edited_text.strip():
            return 1.0, 1.0

        try:
            # Use the same prompts as the local scorer
            document_prompt = "title: none | text: "
            query_prompt = "task: sentence similarity | query: "

            # Original text is the query, edited text is the document
            original_with_prompt = query_prompt + original_text
            edited_with_prompt = document_prompt + edited_text

            # Encode using the same method as local scorer
            query_embedding = self.model.encode_query(original_with_prompt)
            doc_embedding = self.model.encode_document([edited_with_prompt])

            # Calculate similarity
            similarities = self.model.similarity(query_embedding, doc_embedding)
            similarity_score = similarities[0][0].item()

            # Binary threshold: 1.0 if similarity >= threshold, else 0.0
            binary_score = 1.0 if similarity_score >= self.threshold else 0.0

            logger.debug(f"Global semantic similarity: score={similarity_score:.4f}, binary={binary_score}")
            return binary_score, similarity_score

        except Exception as e:
            logger.error(f"Global semantic similarity calculation failed: {e}")
            return 0.0, 0.0

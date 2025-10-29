import logging
from sentence_transformers import SentenceTransformer
import weave

logger = logging.getLogger(__name__)

class SemanticSimilarityScorer:
    def __init__(self, device, threshold=0.6144470572471619):
        self.device = device
        self.threshold = threshold
        self.model = self._load_ss_model(device)

    def _load_ss_model(self, device):
        """Loads the semantic similarity model."""
        return SentenceTransformer('google/embeddinggemma-300m', device=device)

    @weave.op()
    def calculate_semantic_similarity(self, edit_context, inappropriate_part, rewritten_part):
        """Calculates the semantic similarity between two sentences."""

        semantic_similarity = 0.0
        ss_score = 0.0

        sentence_after_edit = edit_context.replace(inappropriate_part, rewritten_part, 1)
        if inappropriate_part == rewritten_part:
            semantic_similarity = 0.0
        else:
            try:
                document_prompt = "title: none | text: "
                query_prompt = "task: sentence similarity | query: "

                sentence_before_with_prompt = query_prompt + edit_context
                sentence_after_with_prompt = document_prompt + sentence_after_edit

                # sentence_before is the query, sentence_after is the document
                query_embedding = self.model.encode_query(sentence_before_with_prompt)
                doc_embedding = self.model.encode_document([sentence_after_with_prompt])

                similarities = self.model.similarity(query_embedding, doc_embedding)
                ss_score = similarities[0][0].item()

                semantic_similarity = 1.0 if self.threshold <= ss_score <= 1.0 else 0.0
            except Exception as e:
                semantic_similarity = 0.0
                logger.error(f"Semantic Similarity failed: {e}")

        return semantic_similarity, ss_score
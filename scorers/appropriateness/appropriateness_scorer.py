import torch
from transformers import pipeline
import weave

DIMS = [
    'Inappropriateness', 'Toxic Emotions', 'Excessive Intensity', 'Emotional Deception',
    'Missing Commitment', 'Missing Seriousness', 'Missing Openness', 'Missing Intelligibility',
    'Unclear Meaning', 'Missing Relevance', 'Confusing Reasoning', 'Other Reasons',
    'Detrimental Orthography', 'Reason Unclassified'
]
LABEL_MAP = {f"LABEL_{i}": dim for i, dim in enumerate(DIMS)}

class AppropriatenessScorer:
    def __init__(self, device):
        self.model = self._load_app_model(device)

    def _load_app_model(self, device):
        """Loads the appropriateness model."""
        app_device = -1
        if isinstance(device, torch.device):
            if device.type == 'cuda':
                app_device = device.index
        elif isinstance(device, str) and 'cuda' in device:
            app_device = int(device.split(':')[-1])
        elif isinstance(device, int):
            app_device = device
        return pipeline("text-classification", model="timonziegenbein/appropriateness-classifier-multilabel", return_all_scores=True, device=app_device)

    @weave.op()
    def get_appropriateness_scores(self, text: str) -> dict:
        """Return classifier scores mapped to human-readable dimension names."""
        try:
            outputs = self.model(text)
            if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], list):
                return { LABEL_MAP.get(item["label"], item["label"]): float(item["score"]) for item in outputs[0] }
        except Exception as e:
            print(f"Classifier prediction failed: {e}")
        return {}
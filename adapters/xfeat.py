import sys
from pathlib import Path

import torch

# Add the cloned XFeat repository to Python's module search path
XFEAT_ROOT = Path(__file__).resolve().parent.parent / "accelerated_features"

if str(XFEAT_ROOT) not in sys.path:
    sys.path.insert(0, str(XFEAT_ROOT))

from modules.xfeat import XFeat


class XFeatExtractor:
    def __init__(self, top_k=1000, detection_threshold=0.05):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = XFeat(
            top_k=top_k,
            detection_threshold=detection_threshold
        )

    def extract(self, image):
        # Expected input: torch.Tensor [C, H, W] or [B, C, H, W]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.inference_mode():
            output = self.model.detectAndCompute(image)

        features = output[0]

        keypoints = features["keypoints"]
        descriptors = features["descriptors"]
        scores = features["scores"]

        return keypoints, descriptors, scores
import sys
from pathlib import Path

import cv2
import torch


XFEAT_ROOT = (
    Path(__file__).resolve().parent.parent
    / "accelerated_features"
)

if str(XFEAT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(XFEAT_ROOT)
    )

from modules.xfeat import XFeat


class XFeatExtractor:

    def __init__(
        self,
        top_k=1000,
        detection_threshold=0.05
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = XFeat(
            top_k=top_k,
            detection_threshold=detection_threshold
        )

    def extract(self, image):

        # -------------------------------------------------
        # NumPy BGR → RGB Tensor
        # -------------------------------------------------

        if isinstance(image, torch.Tensor):

            tensor = image

            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

        else:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            tensor = torch.from_numpy(
                rgb
            ).float()

            tensor = (
                tensor / 255.0
            )

            tensor = tensor.permute(
                2,
                0,
                1
            )

            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        # -------------------------------------------------
        # XFeat inference
        # -------------------------------------------------

        with torch.inference_mode():

            output = (
                self.model.detectAndCompute(
                    tensor
                )
            )

        features = output[0]

        keypoints = features[
            "keypoints"
        ]

        descriptors = features[
            "descriptors"
        ]

        scores = features[
            "scores"
        ]

        return (
            keypoints,
            descriptors,
            scores
        )
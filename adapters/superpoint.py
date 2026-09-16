import sys
import cv2
import numpy as np
import torch

# Import original SuperPoint implementation
sys.path.insert(
    0,
    ".\\SuperPointPretrainedNetwork"
)

from demo_superpoint import SuperPointFrontend


class SuperPointExtractor:

    def __init__(
        self,
        weights_path=".\\SuperPointPretrainedNetwork\\superpoint_v1.pth",
        nms_dist=4,
        conf_thresh=0.015,
        nn_thresh=0.7,
    ):

        # Use CPU for now
        self.device = "cpu"

        self.model = SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=False,
        )

    def extract(self, image):

        # Convert BGR image to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        # SuperPoint expects float32 image in [0, 1]
        image = image.astype(np.float32) / 255.0

        # Run SuperPoint
        points, descriptors, heatmap = self.model.run(image)

        # No keypoints detected
        if descriptors is None:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 256), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        # points = 3 x N
        # [x, y, confidence]
        keypoints = points[:2, :].T.astype(np.float32)

        scores = points[2, :].astype(np.float32)

        # descriptors = 256 x N
        descriptors = descriptors.T.astype(np.float32)

        return keypoints, descriptors, scores
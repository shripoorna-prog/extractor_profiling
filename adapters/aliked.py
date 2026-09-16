import torch
from lightglue import ALIKED


class ALIKEDExtractor:

    def __init__(self, max_num_keypoints=1000):

        self.device = torch.device("cpu")

        self.model = ALIKED(
            max_num_keypoints=max_num_keypoints
        ).to(self.device)

        self.model.eval()

    def extract(self, image):

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.inference_mode():

            features = self.model.extract(image)

        keypoints = features["keypoints"][0]
        descriptors = features["descriptors"][0]

        if "keypoint_scores" in features:
            scores = features["keypoint_scores"][0]
        else:
            scores = torch.ones(
                keypoints.shape[0],
                device=keypoints.device
            )

        return keypoints, descriptors, scores
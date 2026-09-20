import cv2
import torch

from lightglue import ALIKED


class ALIKEDExtractor:

    def __init__(
        self,
        max_num_keypoints=1000
    ):

        self.device = torch.device(
            "cpu"
        )

        self.model = ALIKED(
            max_num_keypoints=max_num_keypoints
        ).to(
            self.device
        )

        self.model.eval()

    def extract(self, image):

        # -------------------------------------------------
        # NumPy BGR → RGB Tensor
        # -------------------------------------------------

        if isinstance(
            image,
            torch.Tensor
        ):

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
        # ALIKED inference
        # -------------------------------------------------

        with torch.inference_mode():

            features = (
                self.model.extract(
                    tensor
                )
            )

        keypoints = features[
            "keypoints"
        ][0]

        descriptors = features[
            "descriptors"
        ][0]

        if "keypoint_scores" in features:

            scores = features[
                "keypoint_scores"
            ][0]

        else:

            scores = torch.ones(
                keypoints.shape[0],
                device=keypoints.device
            )

        return (
            keypoints,
            descriptors,
            scores
        )
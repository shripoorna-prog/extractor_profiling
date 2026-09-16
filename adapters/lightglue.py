import torch
from lightglue import LightGlue


class LightGlueMatcher:

    def __init__(self, feature_type="superpoint"):

        self.device = torch.device("cpu")

        supported_features = {
            "superpoint",
            "aliked",
            "xfeat",
            "sift",
            "disk",
            "doghardnet",
            "dog_affnet_hardnet",
            "keynet_affnet_hardnet",
            "dedodeb",
            "dedodeg",
        }

        if feature_type not in supported_features:
            raise ValueError(
                f"Unsupported LightGlue feature type: {feature_type}"
            )

        print(f"Loading LightGlue for {feature_type}...")

        self.matcher = LightGlue(
            features=feature_type
        ).to(self.device)

        self.matcher.eval()

        print(f"LightGlue loaded on {self.device}")

    def match(
        self,
        keypoints1,
        descriptors1,
        keypoints2,
        descriptors2,
        image_shape
    ):

        if not torch.is_tensor(keypoints1):
            keypoints1 = torch.from_numpy(keypoints1)

        if not torch.is_tensor(descriptors1):
            descriptors1 = torch.from_numpy(descriptors1)

        if not torch.is_tensor(keypoints2):
            keypoints2 = torch.from_numpy(keypoints2)

        if not torch.is_tensor(descriptors2):
            descriptors2 = torch.from_numpy(descriptors2)

        keypoints1 = keypoints1.float().to(self.device)
        descriptors1 = descriptors1.float().to(self.device)

        keypoints2 = keypoints2.float().to(self.device)
        descriptors2 = descriptors2.float().to(self.device)

        keypoints1 = keypoints1.unsqueeze(0)
        descriptors1 = descriptors1.unsqueeze(0)

        keypoints2 = keypoints2.unsqueeze(0)
        descriptors2 = descriptors2.unsqueeze(0)

        height, width = image_shape[:2]

        image_size = torch.tensor(
            [[width, height]],
            dtype=torch.float32,
            device=self.device
        )

        data = {
            "image0": {
                "keypoints": keypoints1,
                "descriptors": descriptors1,
                "image_size": image_size,
            },
            "image1": {
                "keypoints": keypoints2,
                "descriptors": descriptors2,
                "image_size": image_size,
            },
        }

        with torch.inference_mode():

            output = self.matcher(data)

        matches = output["matches"][0]

        return matches.detach().cpu().numpy()
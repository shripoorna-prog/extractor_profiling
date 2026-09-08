import cv2


class ORBExtractor:
    def __init__(self, nfeatures=1000):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures
        )

    def extract(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(
            image,
            None
        )

        return keypoints, descriptors
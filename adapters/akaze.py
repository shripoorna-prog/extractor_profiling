import cv2


class AKAZEExtractor:
    def __init__(self, nfeatures=1000):
        self.akaze = cv2.AKAZE_create()

    def extract(self, image):
        keypoints, descriptors = self.akaze.detectAndCompute(
            image,
            None
        )

        return keypoints, descriptors
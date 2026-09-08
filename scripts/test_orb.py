import cv2
import numpy as np

from adapters.orb import ORBExtractor


# Create a simple test image
image = np.zeros((480, 640), dtype=np.uint8)

cv2.rectangle(image, (100, 100), (300, 300), 255, 3)
cv2.circle(image, (450, 200), 80, 255, 3)
cv2.line(image, (100, 400), (500, 400), 255, 3)


# Create ORB extractor
extractor = ORBExtractor(nfeatures=1000)


# Extract features
keypoints, descriptors = extractor.extract(image)


# Print results
print("Keypoints:", len(keypoints))

if descriptors is not None:
    print("Descriptor shape:", descriptors.shape)
    print("Descriptor dtype:", descriptors.dtype)
else:
    print("Descriptors: None")


# Draw keypoints on the image
output = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# Save visualization
cv2.imwrite("results/orb_keypoints.png", output)

print("Saved visualization to results/orb_keypoints.png")
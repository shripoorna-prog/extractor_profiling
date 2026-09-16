import cv2
import torch

from adapters.xfeat import XFeatExtractor


image_path = "frames/frame_0000.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Convert BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert HWC -> CHW and normalize
image = torch.from_numpy(image).float() / 255.0
image = image.permute(2, 0, 1)


extractor = XFeatExtractor(top_k=1000)

keypoints, descriptors, scores = extractor.extract(image)


print()
print("===== XFEAT ADAPTER TEST =====")
print("Keypoints shape      :", keypoints.shape)
print("Descriptors shape    :", descriptors.shape)
print("Scores shape         :", scores.shape)

print("Keypoints dtype      :", keypoints.dtype)
print("Descriptors dtype    :", descriptors.dtype)
print("Scores dtype         :", scores.dtype)

print()
print("XFeat adapter working successfully!")
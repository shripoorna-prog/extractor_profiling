import torch
import cv2
from kornia.feature import ALIKED

# --------------------------------------------------
# Load image
# --------------------------------------------------

image_path = r".\frames\frame_0000.png"

image = cv2.imread(image_path)

if image is None:
    raise RuntimeError("Could not load image")

print("Original image shape:", image.shape)

# BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to tensor
image = torch.from_numpy(image).float() / 255.0

# HWC -> CHW
image = image.permute(2, 0, 1)

# Add batch dimension
image = image.unsqueeze(0)

print("Image tensor shape:", image.shape)

# --------------------------------------------------
# Load ALIKED
# --------------------------------------------------

print("Loading ALIKED...")

model = ALIKED(
    max_num_keypoints=1000,
    detection_threshold=0.0
)

model.eval()

print("ALIKED loaded successfully")

# --------------------------------------------------
# Run ALIKED
# --------------------------------------------------

with torch.no_grad():
    features = model(image)

# The model returns a list containing one ALIKEDFeatures object
feature = features[0]

# --------------------------------------------------
# Inspect
# --------------------------------------------------

print("\n===== ALIKED FEATURE OUTPUT =====")

print("Keypoints shape      :", feature.keypoints.shape)
print("Descriptors shape    :", feature.descriptors.shape)
print("Keypoint scores shape:", feature.keypoint_scores.shape)

print("\nKeypoints dtype      :", feature.keypoints.dtype)
print("Descriptors dtype    :", feature.descriptors.dtype)
print("Scores dtype         :", feature.keypoint_scores.dtype)

print("\nFirst 5 keypoints:")
print(feature.keypoints[:5])

print("\nFirst 5 scores:")
print(feature.keypoint_scores[:5])

print("\nFirst descriptor:")
print(feature.descriptors[0])
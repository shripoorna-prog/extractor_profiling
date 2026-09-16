import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from adapters.superpoint import SuperPointExtractor


print("===== SUPERPOINT ADAPTER TEST =====")

# Load image
image = cv2.imread(
    str(PROJECT_ROOT / "frames" / "frame_0001.png")
)

if image is None:
    raise FileNotFoundError(
        "Could not load frame_0001.png"
    )

# Create SuperPoint extractor
extractor = SuperPointExtractor()

# Extract features
keypoints, descriptors, scores = extractor.extract(image)

# Display results
print("Keypoints shape      :", keypoints.shape)
print("Descriptors shape    :", descriptors.shape)
print("Scores shape         :", scores.shape)

print("Keypoint dtype       :", keypoints.dtype)
print("Descriptor dtype     :", descriptors.dtype)
print("Scores dtype         :", scores.dtype)

print("\nFirst 5 keypoints:")
print(keypoints[:5])

print("\nFirst 5 scores:")
print(scores[:5])

print("\nSuperPoint adapter working successfully!")
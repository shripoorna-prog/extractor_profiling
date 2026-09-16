import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from adapters.xfeat import XFeatExtractor
from ransac_all import run_ransac


print("======================================")
print(" XFEAT + RANSAC TEST")
print("======================================")


# --------------------------------------------------
# 1. Load images
# --------------------------------------------------

image1 = cv2.imread(
    str(PROJECT_ROOT / "frames" / "frame_0000.png")
)

image2 = cv2.imread(
    str(PROJECT_ROOT / "frames" / "frame_0001.png")
)

if image1 is None or image2 is None:
    raise FileNotFoundError("Could not load input frames.")

print("Images loaded successfully.")


# --------------------------------------------------
# 2. Convert images to XFeat input format
# --------------------------------------------------

image1_tensor = (
    torch.from_numpy(image1)
    .permute(2, 0, 1)
    .float()
    / 255.0
)

image2_tensor = (
    torch.from_numpy(image2)
    .permute(2, 0, 1)
    .float()
    / 255.0
)


# --------------------------------------------------
# 3. Create XFeat extractor
# --------------------------------------------------

extractor = XFeatExtractor(
    top_k=1000,
    detection_threshold=0.05
)


# --------------------------------------------------
# 4. Extract features
# --------------------------------------------------

keypoints1, descriptors1, scores1 = extractor.extract(
    image1_tensor
)

keypoints2, descriptors2, scores2 = extractor.extract(
    image2_tensor
)


# Convert PyTorch tensors to NumPy
keypoints1 = keypoints1.detach().cpu().numpy()
descriptors1 = descriptors1.detach().cpu().numpy()

keypoints2 = keypoints2.detach().cpu().numpy()
descriptors2 = descriptors2.detach().cpu().numpy()


print()
print("Frame 1 keypoints:", len(keypoints1))
print("Frame 2 keypoints:", len(keypoints2))

print("Frame 1 descriptors:", descriptors1.shape)
print("Frame 2 descriptors:", descriptors2.shape)


# --------------------------------------------------
# 5. Descriptor matching
# --------------------------------------------------

print()
print("Matching descriptors...")

matches = []

ratio_threshold = 0.8

for i, descriptor in enumerate(descriptors1):

    # L2 distance between descriptor and all descriptors
    distances = np.linalg.norm(
        descriptors2 - descriptor,
        axis=1
    )

    # Need at least two candidates
    if len(distances) < 2:
        continue

    # Find nearest and second-nearest
    nearest_indices = np.argsort(distances)[:2]

    best_idx = nearest_indices[0]
    second_idx = nearest_indices[1]

    best_distance = distances[best_idx]
    second_distance = distances[second_idx]

    # Lowe ratio test
    if best_distance < ratio_threshold * second_distance:
        matches.append([
            i,
            best_idx
        ])


matches = np.asarray(
    matches,
    dtype=np.int32
)


print("Matches after ratio test:", len(matches))


# --------------------------------------------------
# 6. RANSAC
# --------------------------------------------------

print()
print("Running RANSAC...")

result = run_ransac(
    keypoints1,
    keypoints2,
    matches
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print()
print("======================================")
print(" RANSAC RESULTS")
print("======================================")

print("Total matches:", result["num_matches"])
print("RANSAC inliers:", result["num_inliers"])

print(
    "Inlier ratio:",
    f"{result['inlier_ratio']:.4f}"
)

print(
    "Inlier ratio (%):",
    f"{result['inlier_ratio'] * 100:.2f}"
)

print()

if result["fundamental_matrix"] is not None:
    print("Fundamental matrix:")
    print(result["fundamental_matrix"])
else:
    print("Fundamental matrix: None")

print()
print("XFeat + RANSAC working!")
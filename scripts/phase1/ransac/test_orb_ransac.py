import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from adapters.orb import ORBExtractor
from scripts.phase1.ransac.ransac_all import run_ransac


print("======================================")
print(" ORB + RANSAC TEST")
print("======================================")


# --------------------------------------------------
# Load images
# --------------------------------------------------

image1 = cv2.imread(
    str(PROJECT_ROOT / "frames" / "frame_0000.png")
)

image2 = cv2.imread(
    str(PROJECT_ROOT / "frames" / "frame_0001.png")
)

if image1 is None:
    raise FileNotFoundError(
        "Could not load frame_0000.png"
    )

if image2 is None:
    raise FileNotFoundError(
        "Could not load frame_0001.png"
    )


print("Images loaded successfully.")


# --------------------------------------------------
# Create ORB extractor
# --------------------------------------------------

extractor = ORBExtractor(
    nfeatures=1000
)


# --------------------------------------------------
# Extract features
# --------------------------------------------------

keypoints1, descriptors1 = extractor.extract(
    image1
)

keypoints2, descriptors2 = extractor.extract(
    image2
)


print()
print("Frame 1 keypoints:", len(keypoints1))
print("Frame 2 keypoints:", len(keypoints2))


# --------------------------------------------------
# Convert OpenCV keypoints to coordinates
# --------------------------------------------------

points1 = np.array(
    [kp.pt for kp in keypoints1],
    dtype=np.float32
)

points2 = np.array(
    [kp.pt for kp in keypoints2],
    dtype=np.float32
)


# --------------------------------------------------
# ORB matching
# --------------------------------------------------

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

raw_matches = matcher.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


# --------------------------------------------------
# Ratio test
# --------------------------------------------------

ratio_threshold = 0.8

matches = []

for pair in raw_matches:

    if len(pair) < 2:
        continue

    best_match, second_match = pair

    if best_match.distance < (
        ratio_threshold * second_match.distance
    ):

        matches.append(
            [
                best_match.queryIdx,
                best_match.trainIdx
            ]
        )


matches = np.array(
    matches,
    dtype=np.int32
)


print("Matches after ratio test:", len(matches))


# --------------------------------------------------
# RANSAC
# --------------------------------------------------

result = run_ransac(
    points1,
    points2,
    matches
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("======================================")
print(" RANSAC RESULTS")
print("======================================")

print(
    "Total matches:",
    result["num_matches"]
)

print(
    "RANSAC inliers:",
    result["num_inliers"]
)

print(
    "Inlier ratio:",
    round(
        result["inlier_ratio"],
        4
    )
)

print(
    "Inlier ratio (%):",
    round(
        result["inlier_ratio"] * 100,
        2
    )
)


if result["fundamental_matrix"] is not None:

    print()
    print("Fundamental matrix:")
    print(result["fundamental_matrix"])


print()
print("ORB + RANSAC working!")
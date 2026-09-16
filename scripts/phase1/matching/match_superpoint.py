import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from adapters.superpoint import SuperPointExtractor


print("===== SUPERPOINT MATCHING TEST =====")


# --------------------------------------------------
# 1. Load two frames
# --------------------------------------------------

image1 = cv2.imread(
    str(PROJECT_ROOT / "frames" / "frame_0000.png")
)

image2 = cv2.imread(
    str(PROJECT_ROOT / "frames" / "frame_0001.png")
)

if image1 is None or image2 is None:
    raise FileNotFoundError(
        "Could not load input frames."
    )


# --------------------------------------------------
# 2. Create SuperPoint extractor
# --------------------------------------------------

extractor = SuperPointExtractor()


# --------------------------------------------------
# 3. Extract features
# --------------------------------------------------

kp1, desc1, scores1 = extractor.extract(image1)
kp2, desc2, scores2 = extractor.extract(image2)

print("\nFrame 1:")
print("Keypoints   :", kp1.shape)
print("Descriptors :", desc1.shape)

print("\nFrame 2:")
print("Keypoints   :", kp2.shape)
print("Descriptors :", desc2.shape)


# --------------------------------------------------
# 4. Nearest-neighbor matching
# --------------------------------------------------


ratio_threshold = 0.8

matches = []

for i in range(len(desc1)):

    # Compare descriptor i with all descriptors
    # in frame 2
    difference = desc2 - desc1[i]

    distances = np.linalg.norm(
        difference,
        axis=1
    )

    # Get indices of the two closest descriptors
    nearest_indices = np.argsort(distances)[:2]

    best_index = nearest_indices[0]
    second_index = nearest_indices[1]

    best_distance = distances[best_index]
    second_distance = distances[second_index]

    # Lowe-style ratio
    ratio = best_distance / (second_distance + 1e-8)

    # Keep only confident matches
    if ratio < ratio_threshold:

        matches.append(
            (
                i,
                best_index,
                best_distance,
                ratio
            )
        )


# --------------------------------------------------
# 5. Sort matches by descriptor distance
# --------------------------------------------------

matches.sort(
    key=lambda x: x[2]
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\n===== RATIO TEST RESULTS =====")

print("Ratio threshold :", ratio_threshold)
print("Accepted matches:", len(matches))

print("\n===== BEST 20 MATCHES =====")

for i, (idx1, idx2, distance, ratio) in enumerate(
    matches[:20]
):

    print(
        f"{i + 1:2d}. "
        f"Frame1 KP {idx1:3d} "
        f"-> Frame2 KP {idx2:3d} "
        f"Distance: {distance:.4f} "
        f"Ratio: {ratio:.4f}"
    )

print(
    "\nTotal keypoints in Frame 1:",
    len(desc1)
)

print(
    "Matches after ratio test:",
    len(matches)
)
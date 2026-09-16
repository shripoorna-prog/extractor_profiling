import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from adapters.superpoint import SuperPointExtractor
from adapters.lightglue import LightGlueMatcher


print("======================================")
print(" SUPERPOINT + LIGHTGLUE CPU TEST")
print("======================================")


# --------------------------------------------------
# 1. Load two frames
# --------------------------------------------------

print()
print("Loading images...")

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

print("Original image 1:", image1.shape)
print("Original image 2:", image2.shape)


# --------------------------------------------------
# 2. Resize images
# --------------------------------------------------

image1 = cv2.resize(
    image1,
    (320, 240)
)

image2 = cv2.resize(
    image2,
    (320, 240)
)

print("Test image 1:", image1.shape)
print("Test image 2:", image2.shape)


# --------------------------------------------------
# 3. Create SuperPoint
# --------------------------------------------------

print()
print("Creating SuperPoint...")

superpoint = SuperPointExtractor()

print("SuperPoint created.")


# --------------------------------------------------
# 4. Extract features from frame 1
# --------------------------------------------------

print()
print("Extracting Frame 1...")

start = time.perf_counter()

kp1, desc1, scores1 = superpoint.extract(
    image1
)

extract_time1 = (
    time.perf_counter() - start
) * 1000

print("Frame 1 keypoints:", kp1.shape)
print("Frame 1 descriptors:", desc1.shape)
print(
    "Frame 1 extraction time:",
    round(extract_time1, 2),
    "ms"
)


# --------------------------------------------------
# 5. Extract features from frame 2
# --------------------------------------------------

print()
print("Extracting Frame 2...")

start = time.perf_counter()

kp2, desc2, scores2 = superpoint.extract(
    image2
)

extract_time2 = (
    time.perf_counter() - start
) * 1000

print("Frame 2 keypoints:", kp2.shape)
print("Frame 2 descriptors:", desc2.shape)
print(
    "Frame 2 extraction time:",
    round(extract_time2, 2),
    "ms"
)


# --------------------------------------------------
# 6. Create LightGlue
# --------------------------------------------------

print()
print("Creating LightGlue...")

lightglue = LightGlueMatcher(
    feature_type="superpoint"
)

print("LightGlue created.")


# --------------------------------------------------
# 7. Match the two frames
# --------------------------------------------------

print()
print("Starting LightGlue matching...")

start = time.perf_counter()

matches = lightglue.match(
    kp1,
    desc1,
    kp2,
    desc2,
    image1.shape
)

match_time = (
    time.perf_counter() - start
) * 1000


# --------------------------------------------------
# 8. Results
# --------------------------------------------------

print()
print("======================================")
print(" RESULTS")
print("======================================")

print(
    "Frame 1 keypoints:",
    len(kp1)
)

print(
    "Frame 2 keypoints:",
    len(kp2)
)

print(
    "LightGlue matches:",
    len(matches)
)

print(
    "LightGlue matching time:",
    round(match_time, 2),
    "ms"
)

print()
print("First 10 matches:")

print(matches[:10])

print()
print("======================================")
print(" SUPERPOINT + LIGHTGLUE TEST COMPLETE")
print("======================================")
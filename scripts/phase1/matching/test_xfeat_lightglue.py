import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch

from adapters.xfeat import XFeatExtractor
from adapters.lightglue import LightGlueMatcher


print("======================================")
print(" XFEAT + LIGHTGLUE CPU TEST")
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

if image1 is None or image2 is None:
    raise FileNotFoundError(
        "Could not load test frames."
    )


print()
print("Original image:", image1.shape)


# --------------------------------------------------
# Resize
# --------------------------------------------------

image1 = cv2.resize(
    image1,
    (320, 240)
)

image2 = cv2.resize(
    image2,
    (320, 240)
)

print("Test image:", image1.shape)


# --------------------------------------------------
# Convert to tensors
# --------------------------------------------------

tensor1 = (
    torch.from_numpy(image1)
    .permute(2, 0, 1)
    .float()
    / 255.0
)

tensor2 = (
    torch.from_numpy(image2)
    .permute(2, 0, 1)
    .float()
    / 255.0
)


# --------------------------------------------------
# Create XFeat
# --------------------------------------------------

print()
print("Creating XFeat...")

xfeat = XFeatExtractor(
    top_k=1000,
    detection_threshold=0.05
)

print("XFeat created.")


# --------------------------------------------------
# Extract Frame 1
# --------------------------------------------------

print()
print("Extracting Frame 1...")

start = time.perf_counter()

kp1, desc1, scores1 = xfeat.extract(
    tensor1
)

time1 = (
    time.perf_counter() - start
) * 1000

print(
    "Frame 1 keypoints:",
    kp1.shape
)

print(
    "Frame 1 descriptors:",
    desc1.shape
)

print(
    "Frame 1 extraction:",
    round(time1, 2),
    "ms"
)


# --------------------------------------------------
# Extract Frame 2
# --------------------------------------------------

print()
print("Extracting Frame 2...")

start = time.perf_counter()

kp2, desc2, scores2 = xfeat.extract(
    tensor2
)

time2 = (
    time.perf_counter() - start
) * 1000

print(
    "Frame 2 keypoints:",
    kp2.shape
)

print(
    "Frame 2 descriptors:",
    desc2.shape
)

print(
    "Frame 2 extraction:",
    round(time2, 2),
    "ms"
)


# --------------------------------------------------
# Descriptor information
# --------------------------------------------------

print()
print("Descriptor information:")

print(
    "Descriptor dtype:",
    desc1.dtype
)

print(
    "Descriptor dimension:",
    desc1.shape[-1]
)

print(
    "Descriptor 1 min:",
    desc1.min().item()
)

print(
    "Descriptor 1 max:",
    desc1.max().item()
)

print(
    "Descriptor 1 mean:",
    desc1.mean().item()
)


# --------------------------------------------------
# Create LightGlue
# --------------------------------------------------

print()
print("Creating XFeat LightGlue...")

matcher = LightGlueMatcher(
    feature_type="xfeat"
)

print("XFeat LightGlue created.")


# --------------------------------------------------
# Match
# --------------------------------------------------

print()
print("Starting LightGlue matching...")

start = time.perf_counter()

matches = matcher.match(
    kp1,
    desc1,
    kp2,
    desc2,
    image1.shape
)

matching_time = (
    time.perf_counter() - start
) * 1000


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("======================================")
print(" RESULTS")
print("======================================")

print(
    "Keypoints frame 1:",
    len(kp1)
)

print(
    "Keypoints frame 2:",
    len(kp2)
)

print(
    "LightGlue matches:",
    len(matches)
)

print(
    "Matching time:",
    round(matching_time, 2),
    "ms"
)

print()
print("First 10 matches:")

print(matches[:10])

print()
print("======================================")
print(" XFEAT + LIGHTGLUE TEST COMPLETE")
print("======================================")
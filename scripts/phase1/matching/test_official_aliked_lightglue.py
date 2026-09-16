import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import time

from lightglue import ALIKED, LightGlue


print("=" * 60)
print(" OFFICIAL ALIKED + LIGHTGLUE TEST")
print("=" * 60)


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

img1 = cv2.imread("frames/frame_0000.png")
img2 = cv2.imread("frames/frame_0001.png")

img1 = cv2.resize(img1, (320, 240))
img2 = cv2.resize(img2, (320, 240))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# Convert to tensors
# ---------------------------------------------------------

image1 = torch.from_numpy(img1).float() / 255.0
image2 = torch.from_numpy(img2).float() / 255.0

image1 = image1.permute(2, 0, 1)
image2 = image2.permute(2, 0, 1)


print("\nImages:")
print("Image 1:", image1.shape)
print("Image 2:", image2.shape)
print("Value range:", image1.min().item(), "to", image1.max().item())


# ---------------------------------------------------------
# Create OFFICIAL LightGlue ALIKED
# ---------------------------------------------------------

print("\nCreating official LightGlue ALIKED...")

extractor = ALIKED(
    max_num_keypoints=1000
).eval()

print("Official ALIKED created.")


# ---------------------------------------------------------
# Create LightGlue
# ---------------------------------------------------------

print("\nCreating LightGlue...")

matcher = LightGlue(
    features="aliked"
).eval()

print("LightGlue created.")


# ---------------------------------------------------------
# Extract features
# ---------------------------------------------------------

print("\nExtracting Frame 1...")

start = time.perf_counter()

with torch.inference_mode():
    feats1 = extractor.extract(image1)

time1 = (time.perf_counter() - start) * 1000


print("Frame 1 keypoints:", feats1["keypoints"].shape)
print("Frame 1 descriptors:", feats1["descriptors"].shape)
print("Frame 1 extraction:", f"{time1:.2f} ms")


print("\nExtracting Frame 2...")

start = time.perf_counter()

with torch.inference_mode():
    feats2 = extractor.extract(image2)

time2 = (time.perf_counter() - start) * 1000


print("Frame 2 keypoints:", feats2["keypoints"].shape)
print("Frame 2 descriptors:", feats2["descriptors"].shape)
print("Frame 2 extraction:", f"{time2:.2f} ms")


# ---------------------------------------------------------
# Descriptor diagnostics
# ---------------------------------------------------------

desc1 = feats1["descriptors"]
desc2 = feats2["descriptors"]

kp1 = feats1["keypoints"]
kp2 = feats2["keypoints"]


print("\n" + "=" * 60)
print(" DESCRIPTOR DIAGNOSTICS")
print("=" * 60)

print("\nDescriptor 1:")
print("shape:", desc1.shape)
print("dtype:", desc1.dtype)
print("min:", desc1.min().item())
print("max:", desc1.max().item())
print("mean:", desc1.mean().item())

print("\nDescriptor 2:")
print("shape:", desc2.shape)
print("dtype:", desc2.dtype)
print("min:", desc2.min().item())
print("max:", desc2.max().item())
print("mean:", desc2.mean().item())


# ---------------------------------------------------------
# Descriptor norms
# ---------------------------------------------------------

norm1 = torch.linalg.norm(desc1, dim=-1)
norm2 = torch.linalg.norm(desc2, dim=-1)

print("\nDescriptor norms:")

print(
    "Frame 1:",
    "min =", norm1.min().item(),
    "max =", norm1.max().item(),
    "mean =", norm1.mean().item()
)

print(
    "Frame 2:",
    "min =", norm2.min().item(),
    "max =", norm2.max().item(),
    "mean =", norm2.mean().item()
)


# ---------------------------------------------------------
# Keypoint ranges
# ---------------------------------------------------------

print("\nKeypoint ranges:")

print(
    "Frame 1 X:",
    kp1[..., 0].min().item(),
    "to",
    kp1[..., 0].max().item()
)

print(
    "Frame 1 Y:",
    kp1[..., 1].min().item(),
    "to",
    kp1[..., 1].max().item()
)

print(
    "Frame 2 X:",
    kp2[..., 0].min().item(),
    "to",
    kp2[..., 0].max().item()
)

print(
    "Frame 2 Y:",
    kp2[..., 1].min().item(),
    "to",
    kp2[..., 1].max().item()
)


# ---------------------------------------------------------
# LightGlue matching
# ---------------------------------------------------------

print("\n" + "=" * 60)
print(" RUNNING LIGHTGLUE")
print("=" * 60)

start = time.perf_counter()

with torch.inference_mode():
    matches_output = matcher(
        {
            "image0": feats1,
            "image1": feats2,
        }
    )

match_time = (time.perf_counter() - start) * 1000


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

matches = matches_output["matches"][0]

print("\n" + "=" * 60)
print(" RESULTS")
print("=" * 60)

print("Keypoints frame 1:", kp1.shape[-2])
print("Keypoints frame 2:", kp2.shape[-2])

print("Matches:", matches.shape[0])

print(
    "Matching time:",
    f"{match_time:.2f} ms"
)


# ---------------------------------------------------------
# Matching scores
# ---------------------------------------------------------

if "scores" in matches_output:

    scores = matches_output["scores"][0]

    print("\nMatching scores:")

    if scores.numel() > 0:

        print("shape:", scores.shape)
        print("min:", scores.min().item())
        print("max:", scores.max().item())
        print("mean:", scores.mean().item())

    else:

        print("No scores because there are no matches.")


# ---------------------------------------------------------
# First matches
# ---------------------------------------------------------

if matches.shape[0] > 0:

    print("\nFirst 10 matches:")
    print(matches[:10])

else:

    print("\nWARNING: ZERO MATCHES")


print("\n" + "=" * 60)
print(" TEST COMPLETE")
print("=" * 60)
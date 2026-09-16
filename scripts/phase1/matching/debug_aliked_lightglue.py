import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import time

from adapters.aliked import ALIKEDExtractor
from kornia.feature import LightGlue


print("=" * 55)
print(" ALIKED + LIGHTGLUE DEBUG")
print("=" * 55)


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

img1 = cv2.imread("frames/frame_0000.png")
img2 = cv2.imread("frames/frame_0001.png")

img1 = cv2.resize(img1, (320, 240))
img2 = cv2.resize(img2, (320, 240))

print("\nImages:")
print("Image 1:", img1.shape)
print("Image 2:", img2.shape)


# ---------------------------------------------------------
# BGR -> RGB
# ---------------------------------------------------------

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# Convert to tensors
# ---------------------------------------------------------

tensor1 = torch.from_numpy(img1_rgb).float() / 255.0
tensor2 = torch.from_numpy(img2_rgb).float() / 255.0

tensor1 = tensor1.permute(2, 0, 1).unsqueeze(0)
tensor2 = tensor2.permute(2, 0, 1).unsqueeze(0)

print("\nInput tensors:")
print("Tensor 1:", tensor1.shape)
print("Tensor 2:", tensor2.shape)
print("Tensor dtype:", tensor1.dtype)


# ---------------------------------------------------------
# ALIKED
# ---------------------------------------------------------

print("\nCreating ALIKED...")

extractor = ALIKEDExtractor(
    max_num_keypoints=1000
)

print("ALIKED created.")


# ---------------------------------------------------------
# Extract features
# ---------------------------------------------------------

print("\nExtracting Frame 1...")

start = time.perf_counter()

kp1, desc1, scores1 = extractor.extract(tensor1)

extract_time1 = (time.perf_counter() - start) * 1000

print("Frame 1 keypoints:", kp1.shape)
print("Frame 1 descriptors:", desc1.shape)
print("Frame 1 scores:", scores1.shape)
print(f"Frame 1 extraction: {extract_time1:.2f} ms")


print("\nExtracting Frame 2...")

start = time.perf_counter()

kp2, desc2, scores2 = extractor.extract(tensor2)

extract_time2 = (time.perf_counter() - start) * 1000

print("Frame 2 keypoints:", kp2.shape)
print("Frame 2 descriptors:", desc2.shape)
print("Frame 2 scores:", scores2.shape)
print(f"Frame 2 extraction: {extract_time2:.2f} ms")


# ---------------------------------------------------------
# Descriptor diagnostics
# ---------------------------------------------------------

print("\n" + "=" * 55)
print(" DESCRIPTOR DIAGNOSTICS")
print("=" * 55)

print("\nDescriptor 1:")
print("dtype:", desc1.dtype)
print("shape:", desc1.shape)
print("min:", desc1.min().item())
print("max:", desc1.max().item())
print("mean:", desc1.mean().item())

print("\nDescriptor 2:")
print("dtype:", desc2.dtype)
print("shape:", desc2.shape)
print("min:", desc2.min().item())
print("max:", desc2.max().item())
print("mean:", desc2.mean().item())


# ---------------------------------------------------------
# Descriptor norms
# ---------------------------------------------------------

norm1 = torch.linalg.norm(desc1, dim=1)
norm2 = torch.linalg.norm(desc2, dim=1)

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
# Keypoint diagnostics
# ---------------------------------------------------------

print("\nKeypoint ranges:")

print(
    "Frame 1 X:",
    kp1[:, 0].min().item(),
    "to",
    kp1[:, 0].max().item()
)

print(
    "Frame 1 Y:",
    kp1[:, 1].min().item(),
    "to",
    kp1[:, 1].max().item()
)

print(
    "Frame 2 X:",
    kp2[:, 0].min().item(),
    "to",
    kp2[:, 0].max().item()
)

print(
    "Frame 2 Y:",
    kp2[:, 1].min().item(),
    "to",
    kp2[:, 1].max().item()
)


# ---------------------------------------------------------
# Create raw LightGlue
# ---------------------------------------------------------

print("\n" + "=" * 55)
print(" CREATING RAW ALIKED LIGHTGLUE")
print("=" * 55)

matcher = LightGlue(
    features="aliked"
).to("cpu")

matcher.eval()

print("Raw LightGlue created.")


# ---------------------------------------------------------
# Prepare input
# ---------------------------------------------------------

height, width = img1.shape[:2]

image_size = torch.tensor(
    [[width, height]],
    dtype=torch.float32
)

data = {
    "image0": {
        "keypoints": kp1.unsqueeze(0).float(),
        "descriptors": desc1.unsqueeze(0).float(),
        "image_size": image_size,
    },
    "image1": {
        "keypoints": kp2.unsqueeze(0).float(),
        "descriptors": desc2.unsqueeze(0).float(),
        "image_size": image_size,
    },
}


# ---------------------------------------------------------
# Run LightGlue
# ---------------------------------------------------------

print("\nRunning raw LightGlue...")

start = time.perf_counter()

with torch.inference_mode():
    output = matcher(data)

match_time = (time.perf_counter() - start) * 1000


# ---------------------------------------------------------
# Inspect outputs
# ---------------------------------------------------------

print("\n" + "=" * 55)
print(" LIGHTGLUE INTERNAL OUTPUT")
print("=" * 55)

print("\nOutput keys:")
print(output.keys())


# ---------------------------------------------------------
# Matching scores 0
# ---------------------------------------------------------

matching_scores0 = output["matching_scores0"][0]

print("\nMatching scores 0:")
print("shape:", matching_scores0.shape)
print("dtype:", matching_scores0.dtype)
print("min:", matching_scores0.min().item())
print("max:", matching_scores0.max().item())
print("mean:", matching_scores0.mean().item())
print("median:", torch.median(matching_scores0).item())

print(
    "scores >= 0.001:",
    (matching_scores0 >= 0.001).sum().item()
)

print(
    "scores >= 0.01:",
    (matching_scores0 >= 0.01).sum().item()
)

print(
    "scores >= 0.05:",
    (matching_scores0 >= 0.05).sum().item()
)

print(
    "scores >= 0.1:",
    (matching_scores0 >= 0.1).sum().item()
)

print(
    "scores >= 0.5:",
    (matching_scores0 >= 0.5).sum().item()
)


# ---------------------------------------------------------
# Matching scores 1
# ---------------------------------------------------------

matching_scores1 = output["matching_scores1"][0]

print("\nMatching scores 1:")
print("shape:", matching_scores1.shape)
print("dtype:", matching_scores1.dtype)
print("min:", matching_scores1.min().item())
print("max:", matching_scores1.max().item())
print("mean:", matching_scores1.mean().item())
print("median:", torch.median(matching_scores1).item())

print(
    "scores >= 0.001:",
    (matching_scores1 >= 0.001).sum().item()
)

print(
    "scores >= 0.01:",
    (matching_scores1 >= 0.01).sum().item()
)

print(
    "scores >= 0.05:",
    (matching_scores1 >= 0.05).sum().item()
)

print(
    "scores >= 0.1:",
    (matching_scores1 >= 0.1).sum().item()
)

print(
    "scores >= 0.5:",
    (matching_scores1 >= 0.5).sum().item()
)


# ---------------------------------------------------------
# Best matches before filtering
# ---------------------------------------------------------

matches0 = output["matches0"][0]

valid_matches0 = matches0 >= 0

print("\nMatches0:")
print("Total keypoints:", len(matches0))
print(
    "Valid matches:",
    valid_matches0.sum().item()
)


# ---------------------------------------------------------
# Log assignment
# ---------------------------------------------------------

log_assignment = output["log_assignment"][0]

print("\nLog assignment:")
print("shape:", log_assignment.shape)
print("min:", log_assignment.min().item())
print("max:", log_assignment.max().item())
print("mean:", log_assignment.mean().item())


# ---------------------------------------------------------
# Final matches
# ---------------------------------------------------------

matches = output["matches"][0]

print("\n" + "=" * 55)
print(" FINAL RESULTS")
print("=" * 55)

print("Keypoints frame 1:", len(kp1))
print("Keypoints frame 2:", len(kp2))
print("Final matches:", len(matches))
print(f"Matching time: {match_time:.2f} ms")

if len(matches) > 0:

    print("\nFirst 10 matches:")
    print(matches[:10])

else:

    print("\nWARNING: ZERO FINAL MATCHES")


print("\n" + "=" * 55)
print(" DEBUG COMPLETE")
print("=" * 55)
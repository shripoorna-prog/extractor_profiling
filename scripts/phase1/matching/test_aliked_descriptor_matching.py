import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch

from adapters.aliked import ALIKEDExtractor


print("=" * 55)
print(" ALIKED DESCRIPTOR MATCHING TEST")
print("=" * 55)


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

img1 = cv2.imread("frames/frame_0000.png")
img2 = cv2.imread("frames/frame_0001.png")

img1 = cv2.resize(img1, (320, 240))
img2 = cv2.resize(img2, (320, 240))


# ---------------------------------------------------------
# BGR -> RGB
# ---------------------------------------------------------

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# Convert to tensors
# ---------------------------------------------------------

tensor1 = torch.from_numpy(img1).float() / 255.0
tensor2 = torch.from_numpy(img2).float() / 255.0

tensor1 = tensor1.permute(2, 0, 1).unsqueeze(0)
tensor2 = tensor2.permute(2, 0, 1).unsqueeze(0)


# ---------------------------------------------------------
# ALIKED
# ---------------------------------------------------------

print("\nCreating ALIKED...")

extractor = ALIKEDExtractor(
    max_num_keypoints=1000
)

print("ALIKED created.")


# ---------------------------------------------------------
# Extract
# ---------------------------------------------------------

print("\nExtracting features...")

kp1, desc1, scores1 = extractor.extract(tensor1)
kp2, desc2, scores2 = extractor.extract(tensor2)

print("Frame 1:", kp1.shape, desc1.shape)
print("Frame 2:", kp2.shape, desc2.shape)


# ---------------------------------------------------------
# L2 nearest-neighbor matching
# ---------------------------------------------------------

print("\nRunning descriptor nearest-neighbor matching...")

# torch.cdist computes Euclidean distance between
# every descriptor in frame 1 and every descriptor in frame 2

distances = torch.cdist(desc1, desc2)

# For every descriptor in frame 1,
# find its closest descriptor in frame 2

best_distances, best_indices = torch.min(
    distances,
    dim=1
)


# ---------------------------------------------------------
# Mutual nearest-neighbor check
# ---------------------------------------------------------

reverse_best_indices = torch.argmin(
    distances,
    dim=0
)

matches = []

for i in range(len(kp1)):

    j = best_indices[i].item()

    # Keep only mutual nearest neighbors
    if reverse_best_indices[j].item() == i:
        matches.append((i, j, best_distances[i].item()))


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 55)
print(" RESULTS")
print("=" * 55)

print("Keypoints frame 1:", len(kp1))
print("Keypoints frame 2:", len(kp2))

print("Raw nearest neighbors:", len(best_indices))

print(
    "Mutual nearest-neighbor matches:",
    len(matches)
)


# ---------------------------------------------------------
# Distance statistics
# ---------------------------------------------------------

print("\nNearest-neighbor distance statistics:")

print(
    "Minimum distance:",
    best_distances.min().item()
)

print(
    "Maximum distance:",
    best_distances.max().item()
)

print(
    "Mean distance:",
    best_distances.mean().item()
)

print(
    "Median distance:",
    torch.median(best_distances).item()
)


# ---------------------------------------------------------
# Print first matches
# ---------------------------------------------------------

if len(matches) > 0:

    print("\nFirst 10 mutual matches:")

    for match in matches[:10]:

        i, j, distance = match

        print(
            f"Frame1[{i}] -> Frame2[{j}] "
            f"distance={distance:.6f}"
        )

else:

    print("\nWARNING: No mutual matches found.")


print("\n" + "=" * 55)
print(" TEST COMPLETE")
print("=" * 55)
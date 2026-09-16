import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch

from kornia.feature.aliked.aliked import ALIKED
from kornia.feature import LightGlue


print("======================================")
print(" RAW KORNIA ALIKED + LIGHTGLUE")
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

image1 = cv2.resize(image1, (320, 240))
image2 = cv2.resize(image2, (320, 240))


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

tensor1 = tensor1.unsqueeze(0)
tensor2 = tensor2.unsqueeze(0)


# --------------------------------------------------
# Create RAW ALIKED
# --------------------------------------------------

print()
print("Creating raw ALIKED...")

aliked = ALIKED(
    model_name="aliked-n16",
    max_num_keypoints=-1,
    detection_threshold=0.2,
    nms_radius=2,
)

aliked.eval()

print("ALIKED created.")


# --------------------------------------------------
# Extract
# --------------------------------------------------

print()
print("Extracting features...")

with torch.inference_mode():

    features1 = aliked(tensor1)[0]
    features2 = aliked(tensor2)[0]


kp1 = features1.keypoints
desc1 = features1.descriptors

kp2 = features2.keypoints
desc2 = features2.descriptors


print()
print("Frame 1 keypoints:", kp1.shape)
print("Frame 1 descriptors:", desc1.shape)

print()
print("Frame 2 keypoints:", kp2.shape)
print("Frame 2 descriptors:", desc2.shape)


# --------------------------------------------------
# Descriptor statistics
# --------------------------------------------------

print()
print("Descriptor 1:")
print("min :", desc1.min().item())
print("max :", desc1.max().item())
print("mean:", desc1.mean().item())

print()
print("Descriptor 2:")
print("min :", desc2.min().item())
print("max :", desc2.max().item())
print("mean:", desc2.mean().item())


# --------------------------------------------------
# LightGlue
# --------------------------------------------------

print()
print("Creating LightGlue...")

matcher = LightGlue(
    features="aliked"
)

matcher.eval()

print("LightGlue created.")


# --------------------------------------------------
# Prepare input
# --------------------------------------------------

height, width = image1.shape[:2]

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


# --------------------------------------------------
# Match
# --------------------------------------------------

print()
print("Running LightGlue...")

with torch.inference_mode():

    output = matcher(data)


matches = output["matches"][0]


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("======================================")
print(" RESULTS")
print("======================================")

print(
    "Keypoints 1:",
    len(kp1)
)

print(
    "Keypoints 2:",
    len(kp2)
)

print(
    "Matches:",
    len(matches)
)

print()
print("First 10 matches:")

print(
    matches[:10]
)

print()
print("======================================")
print(" TEST COMPLETE")
print("======================================")
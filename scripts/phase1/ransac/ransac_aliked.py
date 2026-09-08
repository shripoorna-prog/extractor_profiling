import torch
import cv2
import numpy as np
import os
import glob

from kornia.feature import ALIKED

IMAGE_DIR = r"C:\Users\Shripoorna\OneDrive\Desktop\extractor_profiling\frames"

RATIO_TEST = 0.75
RANSAC_THRESHOLD = 3.0
MAX_FEATURES = 1000

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print("Images found:", len(image_paths))

# --------------------------------------------------
# Load ALIKED
# --------------------------------------------------

print("Loading ALIKED...")

device = torch.device("cpu")

model = ALIKED(
    max_num_keypoints=MAX_FEATURES
).eval().to(device)

print("ALIKED loaded successfully.")
print("Device:", device)

# --------------------------------------------------
# Process consecutive image pairs
# --------------------------------------------------

for i in range(len(image_paths) - 1):

    img1 = cv2.imread(image_paths[i])
    img2 = cv2.imread(image_paths[i + 1])

    if img1 is None or img2 is None:
        print(f"Pair {i+1:03d}: Could not read images")
        continue

    # Convert BGR -> RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Convert to tensors
    tensor1 = torch.from_numpy(
        img1_rgb
    ).permute(2, 0, 1).float() / 255.0

    tensor2 = torch.from_numpy(
        img2_rgb
    ).permute(2, 0, 1).float() / 255.0

    tensor1 = tensor1.unsqueeze(0).to(device)
    tensor2 = tensor2.unsqueeze(0).to(device)

    # --------------------------------------------------
    # ALIKED feature extraction
    # --------------------------------------------------

    with torch.no_grad():
        features1 = model(tensor1)[0]
        features2 = model(tensor2)[0]

    kp1 = features1.keypoints.cpu().numpy()
    kp2 = features2.keypoints.cpu().numpy()

    des1 = features1.descriptors.cpu().numpy()
    des2 = features2.descriptors.cpu().numpy()

    if des1 is None or des2 is None:
        print(f"Pair {i+1:03d}: No descriptors")
        continue

    # --------------------------------------------------
    # Match ALIKED descriptors
    # ALIKED descriptors are float descriptors,
    # so use L2 distance.
    # --------------------------------------------------

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(
        des1.astype(np.float32),
        des2.astype(np.float32),
        k=2
    )

    # Lowe ratio test
    good = []

    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < RATIO_TEST * n.distance:
                good.append(m)

    print(
        f"Pair {i+1:03d}: "
        f"{len(good)} good matches",
        end=""
    )

    # Need at least 4 matches for homography
    if len(good) < 4:
        print(" | Not enough matches for RANSAC")
        continue

    # --------------------------------------------------
    # Prepare matched points
    # --------------------------------------------------

    src_pts = np.float32(
        [kp1[m.queryIdx] for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx] for m in good]
    ).reshape(-1, 1, 2)

    # --------------------------------------------------
    # RANSAC
    # --------------------------------------------------

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if mask is None:
        print(" | RANSAC failed")
        continue

    inliers = int(mask.sum())

    inlier_ratio = inliers / len(good)

    print(
        f" | {inliers} RANSAC inliers"
        f" | {inlier_ratio * 100:.2f}% inlier ratio"
    )
import torch
import cv2
import numpy as np
import os
import glob

from kornia.feature import ALIKED


IMAGE_DIR = r"C:\Users\Shripoorna\OneDrive\Desktop\extractor_profiling\frames"
RESULTS_DIR = r"C:\Users\Shripoorna\OneDrive\Desktop\extractor_profiling\results"

RATIO_TEST = 0.75
RANSAC_THRESHOLD = 3.0
MAX_FEATURES = 1000

# Pair 095 means image 95 and image 96
PAIR_NUMBER = 95


# --------------------------------------------------
# Find images
# --------------------------------------------------

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print("Images found:", len(image_paths))

idx = PAIR_NUMBER - 1

img1 = cv2.imread(image_paths[idx])
img2 = cv2.imread(image_paths[idx + 1])

if img1 is None or img2 is None:
    raise RuntimeError("Could not read images")


# --------------------------------------------------
# Load ALIKED
# --------------------------------------------------

print("Loading ALIKED...")

device = torch.device("cpu")

model = ALIKED(
    max_num_keypoints=MAX_FEATURES
).eval().to(device)

print("ALIKED loaded successfully.")


# --------------------------------------------------
# Convert images
# --------------------------------------------------

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

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

print("Keypoints image 1:", len(kp1))
print("Keypoints image 2:", len(kp2))


# --------------------------------------------------
# Match descriptors
# --------------------------------------------------

matcher = cv2.BFMatcher(cv2.NORM_L2)

matches = matcher.knnMatch(
    des1.astype(np.float32),
    des2.astype(np.float32),
    k=2
)

good = []

for pair in matches:

    if len(pair) == 2:

        m, n = pair

        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

print("Good matches:", len(good))


# --------------------------------------------------
# RANSAC
# --------------------------------------------------

src_pts = np.float32(
    [kp1[m.queryIdx] for m in good]
).reshape(-1, 1, 2)

dst_pts = np.float32(
    [kp2[m.trainIdx] for m in good]
).reshape(-1, 1, 2)

H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    RANSAC_THRESHOLD
)

if mask is None:
    raise RuntimeError("RANSAC failed")

mask = mask.ravel()

inlier_matches = [
    good[i] for i in range(len(good))
    if mask[i] == 1
]

outlier_matches = [
    good[i] for i in range(len(good))
    if mask[i] == 0
]

print("RANSAC inliers:", len(inlier_matches))
print("RANSAC outliers:", len(outlier_matches))

ratio = len(inlier_matches) / len(good) * 100

print(f"Inlier ratio: {ratio:.2f}%")


# --------------------------------------------------
# Draw inliers
# --------------------------------------------------

inlier_image = cv2.drawMatches(
    img1,
    [cv2.KeyPoint(float(x), float(y), 3)
     for x, y in kp1],
    img2,
    [cv2.KeyPoint(float(x), float(y), 3)
     for x, y in kp2],
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# --------------------------------------------------
# Draw outliers
# --------------------------------------------------

outlier_image = cv2.drawMatches(
    img1,
    [cv2.KeyPoint(float(x), float(y), 3)
     for x, y in kp1],
    img2,
    [cv2.KeyPoint(float(x), float(y), 3)
     for x, y in kp2],
    outlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# --------------------------------------------------
# Save
# --------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

inlier_path = os.path.join(
    RESULTS_DIR,
    f"aliked_ransac_pair_{PAIR_NUMBER:03d}_inliers.jpg"
)

outlier_path = os.path.join(
    RESULTS_DIR,
    f"aliked_ransac_pair_{PAIR_NUMBER:03d}_outliers.jpg"
)

cv2.imwrite(inlier_path, inlier_image)
cv2.imwrite(outlier_path, outlier_image)

print()
print("Saved:")
print(inlier_path)
print(outlier_path)

print()
print("DONE.")
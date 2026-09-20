from pathlib import Path
import json
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "desk2-circle"
RGB_DIR = DATA_DIR / "rgb"
CALIBRATION_FILE = DATA_DIR / "calibration.json"

IMAGE_1 = RGB_DIR / "frame_00000.png"
IMAGE_2 = RGB_DIR / "frame_00050.png"


# ============================================================
# PARAMETERS
# ============================================================

RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 1.0
CONFIDENCE = 0.999


# ============================================================
# LOAD CALIBRATION
# ============================================================

with open(CALIBRATION_FILE, "r") as f:
    calibration = json.load(f)

K = np.array(
    calibration["K"],
    dtype=np.float64
).reshape(3, 3)


# ============================================================
# LOAD IMAGES
# ============================================================

image1 = cv2.imread(
    str(IMAGE_1),
    cv2.IMREAD_GRAYSCALE
)

image2 = cv2.imread(
    str(IMAGE_2),
    cv2.IMREAD_GRAYSCALE
)

if image1 is None:
    raise FileNotFoundError(f"Could not load {IMAGE_1}")

if image2 is None:
    raise FileNotFoundError(f"Could not load {IMAGE_2}")


# ============================================================
# PRINT BASIC INFORMATION
# ============================================================

print("=" * 60)
print("ESSENTIAL MATRIX TEST")
print("=" * 60)

print()
print(f"Image 1: {IMAGE_1.name}")
print(f"Image 2: {IMAGE_2.name}")

print()
print("Image size:")
print(f"Image 1: {image1.shape}")
print(f"Image 2: {image2.shape}")

print()
print("Camera matrix K:")
print(K)


# ============================================================
# ORB FEATURE EXTRACTION
# ============================================================

orb = cv2.ORB_create(
    nfeatures=2000
)

keypoints1, descriptors1 = orb.detectAndCompute(
    image1,
    None
)

keypoints2, descriptors2 = orb.detectAndCompute(
    image2,
    None
)

print()
print(f"Keypoints image 1: {len(keypoints1)}")
print(f"Keypoints image 2: {len(keypoints2)}")


# ============================================================
# ORB MATCHING
# ============================================================

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

knn_matches = matcher.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


good_matches = []

for pair in knn_matches:

    if len(pair) < 2:
        continue

    best, second = pair

    if best.distance < RATIO_THRESHOLD * second.distance:
        good_matches.append(best)


print()
print(f"Good matches: {len(good_matches)}")


if len(good_matches) < 5:
    raise RuntimeError(
        "Not enough matches to estimate Essential Matrix."
    )


# ============================================================
# CONVERT MATCHES TO POINTS
# ============================================================

points1 = np.float64([
    keypoints1[m.queryIdx].pt
    for m in good_matches
])

points2 = np.float64([
    keypoints2[m.trainIdx].pt
    for m in good_matches
])


# ============================================================
# ESSENTIAL MATRIX
# ============================================================

E, mask = cv2.findEssentialMat(
    points1,
    points2,
    cameraMatrix=K,
    method=cv2.RANSAC,
    prob=CONFIDENCE,
    threshold=RANSAC_THRESHOLD
)


if E is None:
    raise RuntimeError(
        "Essential Matrix estimation failed."
    )


# OpenCV can theoretically return multiple 3x3
# Essential Matrix solutions stacked vertically.
print()
print(f"Essential Matrix shape returned by OpenCV: {E.shape}")

if E.shape[0] > 3:
    print(
        "Multiple Essential Matrix solutions returned. "
        "Using the first 3x3 solution for this diagnostic."
    )
    E = E[:3, :3]


print()
print("Essential Matrix:")
print(E)


# ============================================================
# ESSENTIAL MATRIX RANSAC INLIERS
# ============================================================

if mask is None:
    raise RuntimeError(
        "Essential Matrix RANSAC did not return a mask."
    )

ransac_mask = mask.ravel().astype(bool)

ransac_inliers = int(np.sum(ransac_mask))

inlier_ratio = (
    ransac_inliers / len(good_matches)
    if len(good_matches) > 0
    else 0.0
)


print()
print(
    f"RANSAC inliers: "
    f"{ransac_inliers}/{len(good_matches)}"
)

print(
    f"Inlier ratio: {inlier_ratio:.4f}"
)


# ============================================================
# RECOVER CAMERA POSE
# ============================================================
#
# IMPORTANT:
#
# We give recoverPose:
#   1. ALL good matches
#   2. The RANSAC mask
#
# This lets OpenCV perform the pose recovery using
# the geometrically valid correspondences.
#
# The returned retval is the number of points that
# passed the cheirality check.
# ============================================================

retval, R, t, pose_mask = cv2.recoverPose(
    E,
    points1,
    points2,
    K,
    mask=mask
)


# ============================================================
# PRINT RECOVERED ROTATION
# ============================================================

print()
print("Recovered rotation R:")
print(R)


# ============================================================
# PRINT TRANSLATION DIRECTION
# ============================================================

print()
print("Recovered translation direction t:")
print(t)


print()
print("Translation vector norm:")
print(np.linalg.norm(t))


# ============================================================
# POSE INLIERS
# ============================================================

print()
print("Recovered pose inliers:")
print(retval)


# ============================================================
# POSE INLIER RATIO
# ============================================================

pose_inlier_ratio = (
    retval / ransac_inliers
    if ransac_inliers > 0
    else 0.0
)

print()
print(
    "Pose inlier ratio among RANSAC inliers:"
    f" {pose_inlier_ratio:.4f}"
)


# ============================================================
# ROTATION ANGLE
# ============================================================

rotation_trace = np.trace(R)

cos_angle = (rotation_trace - 1.0) / 2.0

cos_angle = np.clip(
    cos_angle,
    -1.0,
    1.0
)

rotation_angle_rad = np.arccos(
    cos_angle
)

rotation_angle_deg = np.degrees(
    rotation_angle_rad
)

print()
print(
    f"Recovered rotation angle: "
    f"{rotation_angle_deg:.4f} degrees"
)


# ============================================================
# TRANSLATION DIRECTION NORMALIZED
# ============================================================

t_norm = np.linalg.norm(t)

if t_norm > 0:

    t_normalized = t / t_norm

    print()
    print("Normalized translation direction:")
    print(t_normalized)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Good matches                 : {len(good_matches)}")
print(f"Essential RANSAC inliers     : {ransac_inliers}")
print(f"Essential inlier ratio       : {inlier_ratio:.4f}")
print(f"Pose/cheirality inliers      : {retval}")
print(f"Pose inlier ratio             : {pose_inlier_ratio:.4f}")
print(f"Rotation angle               : {rotation_angle_deg:.4f} deg")
print(f"Translation norm             : {np.linalg.norm(t):.4f}")

print()
print("=" * 60)
print("ESSENTIAL MATRIX TEST COMPLETE")
print("=" * 60)
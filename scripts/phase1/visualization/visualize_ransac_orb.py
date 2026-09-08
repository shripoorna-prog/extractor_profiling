import cv2
import numpy as np
import os

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

FRAME_DIR = "frames"
OUTPUT_DIR = "results"

# Visualize this pair
PAIR_NUMBER = 95

# RANSAC settings
RANSAC_THRESHOLD = 3.0
RANSAC_CONFIDENCE = 0.99

# ORB settings
MAX_FEATURES = 1000


# ---------------------------------------------------------
# FIND IMAGES
# ---------------------------------------------------------

image_files = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print("Images found:", len(image_files))

if len(image_files) < 2:
    raise RuntimeError("Need at least 2 images.")


# ---------------------------------------------------------
# SELECT IMAGE PAIR
# ---------------------------------------------------------

idx1 = PAIR_NUMBER - 1
idx2 = PAIR_NUMBER

if idx2 >= len(image_files):
    raise RuntimeError(
        f"Pair {PAIR_NUMBER} does not exist. "
        f"Only {len(image_files) - 1} pairs available."
    )

img1_path = os.path.join(FRAME_DIR, image_files[idx1])
img2_path = os.path.join(FRAME_DIR, image_files[idx2])

print("Image 1:", img1_path)
print("Image 2:", img2_path)


# ---------------------------------------------------------
# LOAD IMAGES
# ---------------------------------------------------------

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    raise RuntimeError("Could not load images.")


# ---------------------------------------------------------
# ORB
# ---------------------------------------------------------

orb = cv2.ORB_create(nfeatures=MAX_FEATURES)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print("ORB keypoints image 1:", len(kp1))
print("ORB keypoints image 2:", len(kp2))


# ---------------------------------------------------------
# MATCHING
# ---------------------------------------------------------

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

print("Raw matches:", len(matches))


# ---------------------------------------------------------
# GOOD MATCHES
# ---------------------------------------------------------

# Keep the same general matching idea:
# use the better half of the matches.

num_good = max(4, int(len(matches) * 0.5))

good_matches = matches[:num_good]

print("Good matches:", len(good_matches))


# ---------------------------------------------------------
# RANSAC
# ---------------------------------------------------------

if len(good_matches) < 4:
    raise RuntimeError("Not enough matches for RANSAC.")


src_pts = np.float32([
    kp1[m.queryIdx].pt for m in good_matches
]).reshape(-1, 1, 2)

dst_pts = np.float32([
    kp2[m.trainIdx].pt for m in good_matches
]).reshape(-1, 1, 2)


H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    RANSAC_THRESHOLD,
    confidence=RANSAC_CONFIDENCE
)

if H is None or mask is None:
    raise RuntimeError("RANSAC could not estimate a homography.")


mask = mask.ravel().astype(bool)

inlier_matches = [
    m for i, m in enumerate(good_matches)
    if mask[i]
]

outlier_matches = [
    m for i, m in enumerate(good_matches)
    if not mask[i]
]


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

inliers = len(inlier_matches)
outliers = len(outlier_matches)

inlier_ratio = inliers / len(good_matches) * 100

print()
print("==============================================")
print("RANSAC VISUALIZATION")
print("==============================================")
print("Good matches :", len(good_matches))
print("RANSAC inliers:", inliers)
print("RANSAC outliers:", outliers)
print("Inlier ratio :", f"{inlier_ratio:.2f}%")
print("==============================================")


# ---------------------------------------------------------
# DRAW INLIERS
# ---------------------------------------------------------

inlier_image = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    inlier_matches,
    None,
    matchColor=(0, 255, 0),
    singlePointColor=None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# ---------------------------------------------------------
# DRAW OUTLIERS
# ---------------------------------------------------------

outlier_image = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    outlier_matches,
    None,
    matchColor=(0, 0, 255),
    singlePointColor=None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

inlier_path = os.path.join(
    OUTPUT_DIR,
    f"ransac_pair_{PAIR_NUMBER:03d}_inliers.jpg"
)

outlier_path = os.path.join(
    OUTPUT_DIR,
    f"ransac_pair_{PAIR_NUMBER:03d}_outliers.jpg"
)

cv2.imwrite(inlier_path, inlier_image)
cv2.imwrite(outlier_path, outlier_image)

print()
print("Saved:")
print(inlier_path)
print(outlier_path)


# ---------------------------------------------------------
# COMBINED VISUALIZATION
# ---------------------------------------------------------

# Resize both images to the same width for easier viewing.

height = max(inlier_image.shape[0], outlier_image.shape[0])

def resize_height(image, target_height):
    scale = target_height / image.shape[0]
    width = int(image.shape[1] * scale)
    return cv2.resize(image, (width, target_height))


inlier_resized = resize_height(inlier_image, height)
outlier_resized = resize_height(outlier_image, height)

combined = np.hstack([
    inlier_resized,
    outlier_resized
])

combined_path = os.path.join(
    OUTPUT_DIR,
    f"ransac_pair_{PAIR_NUMBER:03d}_comparison.jpg"
)

cv2.imwrite(combined_path, combined)

print(combined_path)
print()
print("DONE.")
import cv2
import numpy as np
import os
import glob

IMAGE_DIR = r"C:\Users\Shripoorna\OneDrive\Desktop\extractor_profiling\frames"

RATIO_TEST = 0.75
RANSAC_THRESHOLD = 3.0
MAX_FEATURES = 1000

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print("Images found:", len(image_paths))

orb = cv2.ORB_create(nfeatures=MAX_FEATURES)

for i in range(len(image_paths) - 1):

    img1 = cv2.imread(image_paths[i])
    img2 = cv2.imread(image_paths[i + 1])

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print(f"Pair {i+1:03d}: No descriptors")
        continue

    # Match ORB descriptors
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
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

    # Need at least 4 points for homography
    if len(good) < 4:
        print(" | Not enough matches for RANSAC")
        continue

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    # RANSAC
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
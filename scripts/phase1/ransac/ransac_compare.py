import cv2
import numpy as np
import torch
import time
from pathlib import Path
from kornia.feature import ALIKED


# ============================================================
# SETTINGS
# ============================================================

FRAME_DIR = Path(r"C:\Users\Shripoorna\OneDrive\Desktop\extractor_profiling\frames")

NUM_FRAMES = 100

# ORB
ORB_FEATURES = 1000

# ALIKED
ALIKED_MODEL = "aliked-n16"
ALIKED_MAX_FEATURES = 1000

# Lowe ratio test
RATIO_TEST = 0.75

# RANSAC
RANSAC_REPROJ_THRESHOLD = 3.0
RANSAC_CONFIDENCE = 0.99
RANSAC_MAX_ITERS = 2000


# ============================================================
# LOAD IMAGES
# ============================================================

image_paths = sorted(FRAME_DIR.glob("frame_*.png"))[:NUM_FRAMES]

print("Images found:", len(image_paths))

if len(image_paths) < 2:
    raise RuntimeError("Need at least 2 frames.")


# ============================================================
# LOAD ALIKED
# ============================================================

print("\nLoading ALIKED...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ALIKED(
    max_num_keypoints=ALIKED_MAX_FEATURES,
    detection_threshold=0.0,
    nms_radius=2,
).to(device)

model.eval()

print("ALIKED loaded successfully.")
print("Device:", device)


# ============================================================
# ALIKED FEATURE EXTRACTION
# ============================================================

def extract_aliked(image):
    """
    Returns:
        keypoints: Nx2
        descriptors: Nx128
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(rgb).float() / 255.0

    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)

    feature = output[0]

    keypoints = feature.keypoints.detach().cpu().numpy()
    descriptors = feature.descriptors.detach().cpu().numpy()

    return keypoints, descriptors


# ============================================================
# ORB SETUP
# ============================================================

orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES
)

bf_orb = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# ALIKED MATCHER
# ============================================================

bf_aliked = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False
)


# ============================================================
# RATIO TEST
# ============================================================

def ratio_test(matches, ratio=RATIO_TEST):

    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append(m)

    return good


# ============================================================
# RANSAC
# ============================================================

def calculate_ransac(keypoints1, keypoints2, matches):

    if len(matches) < 4:
        return 0, 0.0

    pts1 = np.float32([
        keypoints1[m.queryIdx]
        for m in matches
    ])

    pts2 = np.float32([
        keypoints2[m.trainIdx]
        for m in matches
    ])

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        RANSAC_REPROJ_THRESHOLD,
        confidence=RANSAC_CONFIDENCE,
        maxIters=RANSAC_MAX_ITERS
    )

    if mask is None:
        return 0, 0.0

    mask = mask.ravel().astype(bool)

    inliers = int(np.sum(mask))

    inlier_ratio = inliers / len(matches)

    return inliers, inlier_ratio


# ============================================================
# MAIN COMPARISON
# ============================================================

orb_times = []
orb_good_matches = []
orb_inliers = []
orb_inlier_ratios = []

aliked_times = []
aliked_good_matches = []
aliked_inliers = []
aliked_inlier_ratios = []


print("\n" + "=" * 70)
print("ORB vs ALIKED — RANSAC GEOMETRIC VERIFICATION")
print("=" * 70)


for i in range(len(image_paths) - 1):

    img1 = cv2.imread(str(image_paths[i]))
    img2 = cv2.imread(str(image_paths[i + 1]))

    if img1 is None or img2 is None:
        print(f"Skipping pair {i + 1}")
        continue


    # ========================================================
    # ORB
    # ========================================================

    start = time.perf_counter()

    kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
    kp2_orb, des2_orb = orb.detectAndCompute(img2, None)

    orb_good = []

    if des1_orb is not None and des2_orb is not None:

        matches = bf_orb.knnMatch(
            des1_orb,
            des2_orb,
            k=2
        )

        orb_good = ratio_test(matches)

    orb_inlier_count, orb_ratio = calculate_ransac(
        np.array([kp.pt for kp in kp1_orb], dtype=np.float32),
        np.array([kp.pt for kp in kp2_orb], dtype=np.float32),
        orb_good
    )

    orb_time = (time.perf_counter() - start) * 1000


    # ========================================================
    # ALIKED
    # ========================================================

    start = time.perf_counter()

    kp1_aliked, des1_aliked = extract_aliked(img1)
    kp2_aliked, des2_aliked = extract_aliked(img2)

    aliked_good = []

    if des1_aliked is not None and des2_aliked is not None:

        matches = bf_aliked.knnMatch(
            des1_aliked.astype(np.float32),
            des2_aliked.astype(np.float32),
            k=2
        )

        aliked_good = ratio_test(matches)

    aliked_inlier_count, aliked_ratio = calculate_ransac(
        kp1_aliked,
        kp2_aliked,
        aliked_good
    )

    aliked_time = (time.perf_counter() - start) * 1000


    # ========================================================
    # STORE
    # ========================================================

    orb_times.append(orb_time)
    orb_good_matches.append(len(orb_good))
    orb_inliers.append(orb_inlier_count)
    orb_inlier_ratios.append(orb_ratio)

    aliked_times.append(aliked_time)
    aliked_good_matches.append(len(aliked_good))
    aliked_inliers.append(aliked_inlier_count)
    aliked_inlier_ratios.append(aliked_ratio)


    print(
        f"Pair {i + 1:03d}: "
        f"ORB {orb_time:7.2f} ms | "
        f"{len(orb_good):4d} good | "
        f"{orb_inlier_count:4d} inliers | "
        f"{orb_ratio * 100:6.2f}%    ||    "
        f"ALIKED {aliked_time:7.2f} ms | "
        f"{len(aliked_good):4d} good | "
        f"{aliked_inlier_count:4d} inliers | "
        f"{aliked_ratio * 100:6.2f}%"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RANSAC COMPARISON")
print("=" * 70)


print("\nORB")
print("-" * 30)

print(
    f"Pairs processed       : {len(orb_times)}"
)

print(
    f"Average time          : {np.mean(orb_times):.2f} ms/pair"
)

print(
    f"Median time           : {np.median(orb_times):.2f} ms/pair"
)

print(
    f"Average good matches  : {np.mean(orb_good_matches):.1f}"
)

print(
    f"Average RANSAC inliers: {np.mean(orb_inliers):.1f}"
)

print(
    f"Average inlier ratio  : {np.mean(orb_inlier_ratios) * 100:.2f}%"
)


print("\nALIKED")
print("-" * 30)

print(
    f"Pairs processed       : {len(aliked_times)}"
)

print(
    f"Average time          : {np.mean(aliked_times):.2f} ms/pair"
)

print(
    f"Median time           : {np.median(aliked_times):.2f} ms/pair"
)

print(
    f"Average good matches  : {np.mean(aliked_good_matches):.1f}"
)

print(
    f"Average RANSAC inliers: {np.mean(aliked_inliers):.1f}"
)

print(
    f"Average inlier ratio  : {np.mean(aliked_inlier_ratios) * 100:.2f}%"
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("GEOMETRIC QUALITY")
print("=" * 70)

print(
    f"ORB    → {np.mean(orb_inliers):.1f} average RANSAC inliers"
)

print(
    f"ALIKED → {np.mean(aliked_inliers):.1f} average RANSAC inliers"
)

print(
    f"\nORB    → {np.mean(orb_inlier_ratios) * 100:.2f}% average inlier ratio"
)

print(
    f"ALIKED → {np.mean(aliked_inlier_ratios) * 100:.2f}% average inlier ratio"
)

print("\nDone!")
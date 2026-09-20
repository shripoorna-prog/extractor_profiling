from pathlib import Path
import json
import csv

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "desk2-circle"
RGB_DIR = DATA_DIR / "rgb"
CALIBRATION_FILE = DATA_DIR / "calibration.json"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "essential_baseline_results.csv"


# ============================================================
# PARAMETERS
# ============================================================

BASE_FRAME = 0

# Temporal gaps to test
FRAME_GAPS = [1, 5, 10, 20, 30, 50]

RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 1.0
CONFIDENCE = 0.999

ORB_FEATURES = 2000


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

with open(CALIBRATION_FILE, "r") as f:
    calibration = json.load(f)

K = np.array(
    calibration["K"],
    dtype=np.float64
).reshape(3, 3)


# ============================================================
# LOAD BASE IMAGE
# ============================================================

base_image_path = RGB_DIR / f"frame_{BASE_FRAME:05d}.png"

image1 = cv2.imread(
    str(base_image_path),
    cv2.IMREAD_GRAYSCALE
)

if image1 is None:
    raise FileNotFoundError(
        f"Could not load {base_image_path}"
    )


# ============================================================
# ORB FEATURE EXTRACTION - BASE IMAGE
# ============================================================

orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES
)

keypoints1, descriptors1 = orb.detectAndCompute(
    image1,
    None
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 75)
print("ESSENTIAL MATRIX BASELINE EVALUATION")
print("=" * 75)

print()
print(f"Base frame: frame_{BASE_FRAME:05d}.png")
print(f"Frame gaps: {FRAME_GAPS}")

print()
print("Camera matrix K:")
print(K)

print()
print(f"Base image keypoints: {len(keypoints1)}")


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# TEST EACH TEMPORAL BASELINE
# ============================================================

for gap in FRAME_GAPS:

    target_frame = BASE_FRAME + gap

    image2_path = RGB_DIR / f"frame_{target_frame:05d}.png"

    print()
    print("-" * 75)
    print(
        f"Testing frame {BASE_FRAME:05d} "
        f"-> frame {target_frame:05d} "
        f"(gap = {gap})"
    )
    print("-" * 75)

    image2 = cv2.imread(
        str(image2_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image2 is None:
        print(f"Could not load {image2_path}")
        continue


    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    keypoints2, descriptors2 = orb.detectAndCompute(
        image2,
        None
    )

    print(f"Keypoints image 2: {len(keypoints2)}")


    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

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


    good_match_count = len(good_matches)

    print(f"Good matches: {good_match_count}")


    # --------------------------------------------------------
    # CHECK MINIMUM MATCHES
    # --------------------------------------------------------

    if good_match_count < 5:

        print("Not enough matches for Essential Matrix.")

        results.append({
            "frame_1": BASE_FRAME,
            "frame_2": target_frame,
            "frame_gap": gap,
            "good_matches": good_match_count,
            "ransac_inliers": 0,
            "ransac_inlier_ratio": 0.0,
            "pose_inliers": 0,
            "pose_inlier_ratio": 0.0,
            "rotation_deg": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
        })

        continue


    # --------------------------------------------------------
    # CONVERT MATCHES TO POINTS
    # --------------------------------------------------------

    points1 = np.float64([
        keypoints1[m.queryIdx].pt
        for m in good_matches
    ])

    points2 = np.float64([
        keypoints2[m.trainIdx].pt
        for m in good_matches
    ])


    # --------------------------------------------------------
    # ESSENTIAL MATRIX
    # --------------------------------------------------------

    E, mask = cv2.findEssentialMat(
        points1,
        points2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=CONFIDENCE,
        threshold=RANSAC_THRESHOLD
    )

    if E is None or mask is None:

        print("Essential Matrix estimation failed.")

        results.append({
            "frame_1": BASE_FRAME,
            "frame_2": target_frame,
            "frame_gap": gap,
            "good_matches": good_match_count,
            "ransac_inliers": 0,
            "ransac_inlier_ratio": 0.0,
            "pose_inliers": 0,
            "pose_inlier_ratio": 0.0,
            "rotation_deg": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
        })

        continue


    # --------------------------------------------------------
    # HANDLE ESSENTIAL MATRIX SHAPE
    # --------------------------------------------------------

    if E.shape[0] > 3:

        print(
            f"Multiple E solutions returned: {E.shape}"
        )

        E = E[:3, :3]


    # --------------------------------------------------------
    # RANSAC INLIERS
    # --------------------------------------------------------

    ransac_mask = mask.ravel().astype(bool)

    ransac_inliers = int(
        np.sum(ransac_mask)
    )

    ransac_ratio = (
        ransac_inliers / good_match_count
    )


    print(
        f"Essential RANSAC inliers: "
        f"{ransac_inliers}/{good_match_count}"
    )

    print(
        f"Essential inlier ratio: "
        f"{ransac_ratio:.4f}"
    )


    # --------------------------------------------------------
    # RECOVER POSE
    # --------------------------------------------------------

    try:

        pose_inliers, R, t, pose_mask = cv2.recoverPose(
            E,
            points1,
            points2,
            K,
            mask=mask
        )

    except cv2.error as error:

        print("recoverPose failed:")
        print(error)

        results.append({
            "frame_1": BASE_FRAME,
            "frame_2": target_frame,
            "frame_gap": gap,
            "good_matches": good_match_count,
            "ransac_inliers": ransac_inliers,
            "ransac_inlier_ratio": ransac_ratio,
            "pose_inliers": 0,
            "pose_inlier_ratio": 0.0,
            "rotation_deg": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
        })

        continue


    # --------------------------------------------------------
    # POSE INLIER RATIO
    # --------------------------------------------------------

    pose_inlier_ratio = (
        pose_inliers / ransac_inliers
        if ransac_inliers > 0
        else 0.0
    )


    # --------------------------------------------------------
    # ROTATION ANGLE
    # --------------------------------------------------------

    rotation_trace = np.trace(R)

    cos_angle = (
        rotation_trace - 1.0
    ) / 2.0

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


    # --------------------------------------------------------
    # TRANSLATION
    # --------------------------------------------------------

    t = t.reshape(3)

    translation_norm = np.linalg.norm(t)

    if translation_norm > 0:

        t = t / translation_norm


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Pose inliers: "
        f"{pose_inliers}"
    )

    print(
        f"Pose inlier ratio: "
        f"{pose_inlier_ratio:.4f}"
    )

    print(
        f"Rotation angle: "
        f"{rotation_angle_deg:.4f} deg"
    )

    print(
        "Translation direction: "
        f"[{t[0]:.4f}, "
        f"{t[1]:.4f}, "
        f"{t[2]:.4f}]"
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({
        "frame_1": BASE_FRAME,
        "frame_2": target_frame,
        "frame_gap": gap,
        "good_matches": good_match_count,
        "ransac_inliers": ransac_inliers,
        "ransac_inlier_ratio": ransac_ratio,
        "pose_inliers": int(pose_inliers),
        "pose_inlier_ratio": pose_inlier_ratio,
        "rotation_deg": rotation_angle_deg,
        "tx": float(t[0]),
        "ty": float(t[1]),
        "tz": float(t[2]),
    })


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [
    "frame_1",
    "frame_2",
    "frame_gap",
    "good_matches",
    "ransac_inliers",
    "ransac_inlier_ratio",
    "pose_inliers",
    "pose_inlier_ratio",
    "rotation_deg",
    "tx",
    "ty",
    "tz",
]


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# FINAL TABLE
# ============================================================

print()
print()
print("=" * 95)
print("FINAL BASELINE RESULTS")
print("=" * 95)

print(
    f"{'Gap':>5} "
    f"{'Matches':>10} "
    f"{'RANSAC':>10} "
    f"{'R.Inlier':>10} "
    f"{'Pose':>10} "
    f"{'P.Inlier':>10} "
    f"{'Rot(deg)':>10}"
)

print("-" * 95)

for result in results:

    print(
        f"{result['frame_gap']:>5} "
        f"{result['good_matches']:>10} "
        f"{result['ransac_inliers']:>10} "
        f"{result['ransac_inlier_ratio']:>10.4f} "
        f"{result['pose_inliers']:>10} "
        f"{result['pose_inlier_ratio']:>10.4f} "
        f"{result['rotation_deg']:>10.4f}"
    )


print()
print("=" * 95)
print("RESULTS SAVED")
print("=" * 95)

print()
print(OUTPUT_FILE)

print()
print("ESSENTIAL MATRIX BASELINE EVALUATION COMPLETE")
print("=" * 95)
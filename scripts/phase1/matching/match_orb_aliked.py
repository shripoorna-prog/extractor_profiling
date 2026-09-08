import cv2
import numpy as np
import glob
import os
import time
import torch
from kornia.feature import ALIKED


# ============================================================
# SETTINGS
# ============================================================

FRAME_DIR = r"C:\Users\Shripoorna\OneDrive\Desktop\extractor_profiling\frames"

# Number of consecutive frame pairs to test
NUM_PAIRS = 99


# ============================================================
# LOAD IMAGES
# ============================================================

image_paths = sorted(
    glob.glob(os.path.join(FRAME_DIR, "*.png"))
)

print("Images found:", len(image_paths))

if len(image_paths) < 2:
    raise RuntimeError("Need at least 2 images.")


# ============================================================
# ORB SETUP
# ============================================================

orb = cv2.ORB_create(
    nfeatures=1000
)


# ============================================================
# ALIKED SETUP
# ============================================================

print("\nLoading ALIKED...")

device = torch.device("cpu")

model = ALIKED(
    max_num_keypoints=1000,
    detection_threshold=0.2
).eval().to(device)

print("ALIKED loaded successfully.")
print("Device:", device)


# ============================================================
# ALIKED IMAGE CONVERSION
# ============================================================

def image_to_tensor(image):
    """
    OpenCV BGR image -> PyTorch RGB tensor
    Shape: [1, 3, H, W]
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(
        image_rgb
    ).permute(2, 0, 1).float() / 255.0

    return tensor.unsqueeze(0)


# ============================================================
# ALIKED FEATURE EXTRACTION
# ============================================================

def get_aliked_features(image):

    tensor = image_to_tensor(image).to(device)

    with torch.no_grad():
        output = model(tensor)

    # ALIKED returns a list containing ALIKEDFeatures
    feature = output[0]

    keypoints = feature.keypoints
    descriptors = feature.descriptors

    return keypoints, descriptors


# ============================================================
# ALIKED MATCHER
# ============================================================

bf_aliked = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False
)


# ============================================================
# RESULTS
# ============================================================

orb_times = []
aliked_times = []

orb_matches = []
aliked_matches = []

orb_good_matches = []
aliked_good_matches = []


print("\n============================================================")
print("ORB vs ALIKED FEATURE MATCHING")
print("============================================================")


# ============================================================
# PROCESS CONSECUTIVE FRAMES
# ============================================================

for i in range(min(NUM_PAIRS, len(image_paths) - 1)):

    img1 = cv2.imread(image_paths[i])
    img2 = cv2.imread(image_paths[i + 1])

    if img1 is None or img2 is None:
        print("Could not read frame:", i)
        continue


    # ========================================================
    # ORB
    # ========================================================

    start = time.perf_counter()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is not None and des2 is not None:

        matches = cv2.BFMatcher(
            cv2.NORM_HAMMING
        ).knnMatch(des1, des2, k=2)

        good = []

        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n

                if m.distance < 0.75 * n.distance:
                    good.append(m)

    else:
        good = []

    orb_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # ALIKED
    # ========================================================

    start = time.perf_counter()

    kp1_a, des1_a = get_aliked_features(img1)
    kp2_a, des2_a = get_aliked_features(img2)

    # Convert tensors to NumPy
    des1_np = des1_a.cpu().numpy().astype(np.float32)
    des2_np = des2_a.cpu().numpy().astype(np.float32)

    matches_a = bf_aliked.knnMatch(
        des1_np,
        des2_np,
        k=2
    )

    good_a = []

    for m_n in matches_a:

        if len(m_n) == 2:

            m, n = m_n

            if m.distance < 0.75 * n.distance:
                good_a.append(m)

    aliked_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    orb_times.append(orb_time)
    aliked_times.append(aliked_time)

    orb_matches.append(
        len(matches) if des1 is not None and des2 is not None else 0
    )

    aliked_matches.append(
        len(matches_a)
    )

    orb_good_matches.append(
        len(good)
    )

    aliked_good_matches.append(
        len(good_a)
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Pair {i+1:03d}: "
        f"ORB {orb_time:.2f} ms | "
        f"{len(good):4d} good matches    "
        f"|    "
        f"ALIKED {aliked_time:.2f} ms | "
        f"{len(good_a):4d} good matches"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("============================================================")
print("FINAL MATCHING COMPARISON")
print("============================================================")


print("\nORB")
print("----------------------------")

print(
    f"Pairs processed      : {len(orb_times)}"
)

print(
    f"Average time         : {np.mean(orb_times):.2f} ms/pair"
)

print(
    f"Median time          : {np.median(orb_times):.2f} ms/pair"
)

print(
    f"Average raw matches  : {np.mean(orb_matches):.1f}"
)

print(
    f"Average good matches : {np.mean(orb_good_matches):.1f}"
)


print("\nALIKED")
print("----------------------------")

print(
    f"Pairs processed      : {len(aliked_times)}"
)

print(
    f"Average time         : {np.mean(aliked_times):.2f} ms/pair"
)

print(
    f"Median time          : {np.median(aliked_times):.2f} ms/pair"
)

print(
    f"Average raw matches  : {np.mean(aliked_matches):.1f}"
)

print(
    f"Average good matches : {np.mean(aliked_good_matches):.1f}"
)


print("\n")
print("============================================================")
print("MATCH QUALITY COMPARISON")
print("============================================================")

print(
    f"ORB average good matches    : "
    f"{np.mean(orb_good_matches):.1f}"
)

print(
    f"ALIKED average good matches : "
    f"{np.mean(aliked_good_matches):.1f}"
)
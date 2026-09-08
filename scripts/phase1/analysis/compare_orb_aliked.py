import os
import time
import cv2
import torch
import numpy as np
from kornia.feature import ALIKED


# ============================================================
# SETTINGS
# ============================================================

FRAME_DIR = r".\frames"
NUM_FRAMES = 100


# ============================================================
# FIND FRAMES
# ============================================================

images = sorted(
    [
        os.path.join(FRAME_DIR, f)
        for f in os.listdir(FRAME_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)

images = images[:NUM_FRAMES]

print("Images found:", len(images))


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

aliked = ALIKED(
    max_num_keypoints=1000,
    detection_threshold=0.0
)

aliked.eval()

device = torch.device("cpu")
aliked = aliked.to(device)

print("ALIKED loaded successfully")
print("Device:", device)


# ============================================================
# STORAGE
# ============================================================

orb_times = []
orb_keypoints = []

aliked_times = []
aliked_keypoints = []


# ============================================================
# PROCESS FRAMES
# ============================================================

print("\n===== ORB vs ALIKED =====")

for i, path in enumerate(images):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(path)

    if image is None:
        print(f"Could not load: {path}")
        continue


    # ========================================================
    # ORB
    # ========================================================

    start = time.perf_counter()

    kp, des = orb.detectAndCompute(image, None)

    end = time.perf_counter()

    orb_time = (end - start) * 1000

    num_orb = len(kp) if kp is not None else 0

    orb_times.append(orb_time)
    orb_keypoints.append(num_orb)


    # ========================================================
    # ALIKED
    # ========================================================

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(rgb).float() / 255.0

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)


    start = time.perf_counter()

    with torch.no_grad():
        features = aliked(tensor)

    end = time.perf_counter()

    aliked_time = (end - start) * 1000

    feature = features[0]

    num_aliked = feature.keypoints.shape[0]

    aliked_times.append(aliked_time)
    aliked_keypoints.append(num_aliked)


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Frame {i + 1:03d}: "
        f"ORB {orb_time:.2f} ms / {num_orb} kp    |    "
        f"ALIKED {aliked_time:.2f} ms / {num_aliked} kp"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print("\nORB")
print(f"Frames processed     : {len(orb_times)}")
print(f"Average time         : {np.mean(orb_times):.2f} ms/frame")
print(f"Median time          : {np.median(orb_times):.2f} ms/frame")
print(f"Min time             : {np.min(orb_times):.2f} ms/frame")
print(f"Max time             : {np.max(orb_times):.2f} ms/frame")
print(f"Average keypoints    : {np.mean(orb_keypoints):.1f}")
print(f"Max keypoints        : {np.max(orb_keypoints)}")


print("\nALIKED")
print(f"Frames processed     : {len(aliked_times)}")
print(f"Average time         : {np.mean(aliked_times):.2f} ms/frame")
print(f"Median time          : {np.median(aliked_times):.2f} ms/frame")
print(f"Min time             : {np.min(aliked_times):.2f} ms/frame")
print(f"Max time             : {np.max(aliked_times):.2f} ms/frame")
print(f"Average keypoints    : {np.mean(aliked_keypoints):.1f}")
print(f"Max keypoints        : {np.max(aliked_keypoints)}")


# ============================================================
# SPEED COMPARISON
# ============================================================

orb_avg = np.mean(orb_times)
aliked_avg = np.mean(aliked_times)

print("\n")
print("=" * 60)
print("SPEED COMPARISON")
print("=" * 60)

print(f"ORB average time    : {orb_avg:.2f} ms")
print(f"ALIKED average time : {aliked_avg:.2f} ms")

print(
    f"\nALIKED is approximately "
    f"{aliked_avg / orb_avg:.1f}x slower than ORB on CPU."
)


# ============================================================
# KEYPOINT COMPARISON
# ============================================================

orb_kp_avg = np.mean(orb_keypoints)
aliked_kp_avg = np.mean(aliked_keypoints)

print("\n")
print("=" * 60)
print("KEYPOINT COMPARISON")
print("=" * 60)

print(f"ORB average keypoints    : {orb_kp_avg:.1f}")
print(f"ALIKED average keypoints : {aliked_kp_avg:.1f}")
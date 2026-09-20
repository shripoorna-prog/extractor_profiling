import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "desk2-circle"

RGB_TIMESTAMPS = DATA_DIR / "rgb_timestamps.csv"
POSES = DATA_DIR / "poses.csv"


def load_rgb_timestamps():
    timestamps = []

    with open(RGB_TIMESTAMPS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamps.append(float(row["timestamp"]))

    return np.array(timestamps, dtype=np.float64)


def load_pose_timestamps():
    timestamps = []

    with open(POSES, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamps.append(float(row["timestamp"]))

    return np.array(timestamps, dtype=np.float64)


def main():

    rgb_times = load_rgb_timestamps()
    pose_times = load_pose_timestamps()

    print("=" * 60)
    print("RGB ↔ VRPN POSE SYNCHRONIZATION CHECK")
    print("=" * 60)

    print(f"\nRGB frames: {len(rgb_times)}")
    print(f"VRPN poses: {len(pose_times)}")

    differences = []

    nearest_pose_indices = []

    for rgb_time in rgb_times:

        index = np.searchsorted(
            pose_times,
            rgb_time
        )

        candidates = []

        if index > 0:
            candidates.append(index - 1)

        if index < len(pose_times):
            candidates.append(index)

        if not candidates:
            continue

        best_index = min(
            candidates,
            key=lambda i: abs(
                pose_times[i] - rgb_time
            )
        )

        difference = abs(
            pose_times[best_index] - rgb_time
        )

        differences.append(difference)
        nearest_pose_indices.append(best_index)

    differences = np.array(
        differences,
        dtype=np.float64
    )

    print("\nSynchronization error:")
    print(
        f"Mean : {differences.mean() * 1000:.3f} ms"
    )
    print(
        f"Median: {np.median(differences) * 1000:.3f} ms"
    )
    print(
        f"Max  : {differences.max() * 1000:.3f} ms"
    )
    print(
        f"P95  : {np.percentile(differences, 95) * 1000:.3f} ms"
    )

    print("\nFirst 10 RGB → nearest VRPN pose matches:")

    for i in range(min(10, len(nearest_pose_indices))):

        pose_index = nearest_pose_indices[i]

        print(
            f"RGB {i:05d} | "
            f"RGB time: {rgb_times[i]:.6f} | "
            f"Pose index: {pose_index:04d} | "
            f"Pose time: {pose_times[pose_index]:.6f} | "
            f"Δt: {differences[i] * 1000:.3f} ms"
        )

    print("\n" + "=" * 60)
    print("SYNC CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
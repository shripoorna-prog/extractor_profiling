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


def nearest_pose_difference(rgb_time, pose_times):

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
        return None

    best_index = min(
        candidates,
        key=lambda i: abs(
            pose_times[i] - rgb_time
        )
    )

    return abs(
        pose_times[best_index] - rgb_time
    )


def main():

    rgb_times = load_rgb_timestamps()
    pose_times = load_pose_timestamps()

    differences = []

    for rgb_time in rgb_times:

        difference = nearest_pose_difference(
            rgb_time,
            pose_times
        )

        if difference is not None:
            differences.append(difference)

    differences = np.array(
        differences,
        dtype=np.float64
    )

    thresholds_ms = [
        5,
        10,
        20,
        50,
        100,
    ]

    print("=" * 60)
    print("RGB ↔ VRPN SYNCHRONIZATION ANALYSIS")
    print("=" * 60)

    print(
        f"\nTotal RGB frames: {len(rgb_times)}"
    )

    print(
        f"Total RGB↔pose associations: "
        f"{len(differences)}"
    )

    print("\nThreshold analysis:")

    print(
        f"{'Threshold':>12} | "
        f"{'Frames kept':>12} | "
        f"{'Percentage':>12}"
    )

    print("-" * 45)

    for threshold_ms in thresholds_ms:

        threshold_seconds = threshold_ms / 1000.0

        kept = np.sum(
            differences <= threshold_seconds
        )

        percentage = (
            kept / len(differences)
        ) * 100.0

        print(
            f"{threshold_ms:>9} ms | "
            f"{kept:>12} | "
            f"{percentage:>10.2f}%"
        )

    print("\nError distribution:")

    print(
        f"Minimum : "
        f"{np.min(differences) * 1000:.3f} ms"
    )

    print(
        f"Median  : "
        f"{np.median(differences) * 1000:.3f} ms"
    )

    print(
        f"Mean    : "
        f"{np.mean(differences) * 1000:.3f} ms"
    )

    print(
        f"P90     : "
        f"{np.percentile(differences, 90) * 1000:.3f} ms"
    )

    print(
        f"P95     : "
        f"{np.percentile(differences, 95) * 1000:.3f} ms"
    )

    print(
        f"P99     : "
        f"{np.percentile(differences, 99) * 1000:.3f} ms"
    )

    print(
        f"Maximum : "
        f"{np.max(differences) * 1000:.3f} ms"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
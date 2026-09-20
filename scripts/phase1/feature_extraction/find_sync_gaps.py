import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "desk2-circle"

RGB_TIMESTAMPS = DATA_DIR / "rgb_timestamps.csv"
POSES = DATA_DIR / "poses.csv"


def load_rgb():
    frames = []
    timestamps = []

    with open(RGB_TIMESTAMPS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frames.append(row["frame"])
            timestamps.append(float(row["timestamp"]))

    return frames, np.array(timestamps)


def load_poses():
    timestamps = []

    with open(POSES, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamps.append(float(row["timestamp"]))

    return np.array(timestamps)


def main():

    frames, rgb_times = load_rgb()
    pose_times = load_poses()

    results = []

    for frame, rgb_time in zip(frames, rgb_times):

        index = np.searchsorted(
            pose_times,
            rgb_time
        )

        candidates = []

        if index > 0:
            candidates.append(index - 1)

        if index < len(pose_times):
            candidates.append(index)

        best_index = min(
            candidates,
            key=lambda i: abs(
                pose_times[i] - rgb_time
            )
        )

        difference_ms = (
            abs(pose_times[best_index] - rgb_time)
            * 1000.0
        )

        results.append(
            (
                difference_ms,
                frame,
                rgb_time,
                best_index,
                pose_times[best_index],
            )
        )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print("=" * 75)
    print("LARGEST RGB ↔ VRPN SYNCHRONIZATION GAPS")
    print("=" * 75)

    print(
        f"\n{'Frame':<18}"
        f"{'Error':>12}"
        f"{'Pose index':>14}"
        f"{'RGB time':>18}"
    )

    print("-" * 65)

    for difference_ms, frame, rgb_time, pose_index, pose_time in results[:30]:

        print(
            f"{frame:<18}"
            f"{difference_ms:>10.3f} ms"
            f"{pose_index:>14}"
            f"{rgb_time:>18.6f}"
        )

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
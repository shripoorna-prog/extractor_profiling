from pathlib import Path
import sys
import time
import csv

import cv2
import numpy as np
import torch
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from adapters.orb import ORBExtractor
from adapters.akaze import AKAZEExtractor
from adapters.xfeat import XFeatExtractor
from adapters.aliked import ALIKEDExtractor
from adapters.superpoint import SuperPointExtractor

from matchers import (
    match_orb,
    match_akaze,
    match_l2,
    match_lightglue,
)


THERMAL_DIR = ROOT / "data" / "desk2-circle" / "thermal"
RESULTS_DIR = ROOT / "results"

MODELS = [
    "ORB",
    "AKAZE",
    "XFeat",
    "ALIKED",
    "SuperPoint",
]


def thermal_to_uint8(image):
    """
    Convert uint16 thermal image to 8-bit using
    percentile normalization.
    """

    image = image.astype(np.float32)

    lo = np.percentile(image, 1)
    hi = np.percentile(image, 99)

    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)

    return (normalized * 255.0).astype(np.uint8)


def normalize_keypoints(kp):

    if kp is None:
        return np.empty((0, 2), dtype=np.float32)

    if torch.is_tensor(kp):
        kp = kp.detach().cpu().numpy()

    kp = np.asarray(kp)

    if kp.ndim == 3:
        kp = kp[0]

    if kp.ndim == 2 and kp.shape[1] == 2:
        return kp.astype(np.float32)

    if kp.ndim == 2 and kp.shape[0] == 2:
        return kp.T.astype(np.float32)

    return np.empty((0, 2), dtype=np.float32)


def normalize_descriptors(desc):

    if desc is None:
        return None

    if torch.is_tensor(desc):
        desc = desc.detach().cpu().numpy()

    desc = np.asarray(desc)

    if desc.ndim == 3:
        desc = desc[0]

    return desc


def extract_features(extractor, image):

    output = extractor.extract(image)

    if len(output) == 2:
        keypoints, descriptors = output
        scores = None
    else:
        keypoints, descriptors, scores = output

    keypoints = normalize_keypoints(keypoints)
    descriptors = normalize_descriptors(descriptors)

    if scores is not None:

        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()

        scores = np.asarray(scores).reshape(-1)

    return keypoints, descriptors, scores


def match_features(
    model_name,
    kp1,
    desc1,
    scores1,
    kp2,
    desc2,
    scores2
):

    if len(kp1) == 0 or len(kp2) == 0:
        return []

    if desc1 is None or desc2 is None:
        return []

    if model_name == "ORB":
        return match_orb(desc1, desc2)

    if model_name == "AKAZE":
        return match_akaze(desc1, desc2)

    if model_name == "XFeat":
        return match_l2(desc1, desc2)

    if model_name == "ALIKED":

        return match_lightglue(
            kp1,
            desc1,
            scores1,
            kp2,
            desc2,
            scores2,
            image_size=(640, 480),
            feature_type="aliked",
        )

    if model_name == "SuperPoint":

        return match_lightglue(
            kp1,
            desc1,
            scores1,
            kp2,
            desc2,
            scores2,
            image_size=(640, 480),
            feature_type="superpoint",
        )

    return []


def enforce_one_to_one(matches):

    used1 = set()
    used2 = set()

    output = []

    for m in matches:

        i = int(m[0])
        j = int(m[1])

        if i in used1 or j in used2:
            continue

        used1.add(i)
        used2.add(j)

        output.append((i, j))

    return output


def repeatability(kp1, kp2, matches):

    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0

    if not matches:
        return 0.0

    return len(matches) / min(len(kp1), len(kp2))


class TrackManager:

    def __init__(self):
        self.active = {}
        self.lengths = []

    def initialize(self, count):

        self.active = {
            i: 1
            for i in range(count)
        }

    def update(self, matches):

        matched_previous = {
            i
            for i, _ in matches
        }

        next_active = {}

        for i, j in matches:

            if i in self.active:
                next_active[j] = self.active[i] + 1
            else:
                next_active[j] = 1

        for feature, length in self.active.items():

            if feature not in matched_previous:
                self.lengths.append(length)

        self.active = next_active

    def finalize(self):

        self.lengths.extend(
            self.active.values()
        )

        if not self.lengths:
            return 0.0, 0

        return (
            float(np.mean(self.lengths)),
            int(max(self.lengths)),
        )


def get_rss_mb():

    process = psutil.Process()

    return (
        process.memory_info().rss
        / (1024 ** 2)
    )


def create_extractor(model_name):

    if model_name == "ORB":
        return ORBExtractor(
            nfeatures=1000
        )

    if model_name == "AKAZE":
        return AKAZEExtractor(
            nfeatures=1000
        )

    if model_name == "XFeat":
        return XFeatExtractor(
            top_k=1000,
            detection_threshold=0.05,
        )

    if model_name == "ALIKED":
        return ALIKEDExtractor(
            max_num_keypoints=1000
        )

    if model_name == "SuperPoint":
        return SuperPointExtractor()

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def profile_model(
    model_name,
    thermal_files
):

    print()
    print("=" * 75)
    print(f"PROFILE: {model_name}")
    print("=" * 75)

    rss_before = get_rss_mb()

    load_start = time.perf_counter()

    extractor = create_extractor(
        model_name
    )

    extractor_load_ms = (
        time.perf_counter()
        - load_start
    ) * 1000

    print(
        f"Extractor load : "
        f"{extractor_load_ms:.2f} ms"
    )

    print(
        "Matcher load   : 0.00 ms"
    )

    keypoint_counts = []
    match_counts = []
    match_ratios = []
    repeatability_values = []

    extraction_times = []
    matching_times = []

    track_manager = TrackManager()

    previous_kp = None
    previous_desc = None
    previous_scores = None

    peak_rss = rss_before

    pair_rows = []

    for frame_idx, path in enumerate(
        thermal_files
    ):

        raw = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if raw is None:

            print(
                f"WARNING: Could not read "
                f"{path}"
            )

            continue

        image = thermal_to_uint8(
            raw
        )

        extract_start = time.perf_counter()

        kp, desc, scores = extract_features(
            extractor,
            image
        )

        extraction_ms = (
            time.perf_counter()
            - extract_start
        ) * 1000

        extraction_times.append(
            extraction_ms
        )

        keypoint_counts.append(
            len(kp)
        )

        peak_rss = max(
            peak_rss,
            get_rss_mb()
        )

        if frame_idx == 0:

            track_manager.initialize(
                len(kp)
            )

            previous_kp = kp
            previous_desc = desc
            previous_scores = scores

            print(
                f"[{frame_idx + 1}/"
                f"{len(thermal_files)}] "
                f"{path.name} "
                f"KP={len(kp)}"
            )

            continue

        match_start = time.perf_counter()

        raw_matches = match_features(
            model_name,
            previous_kp,
            previous_desc,
            previous_scores,
            kp,
            desc,
            scores,
        )

        matches = enforce_one_to_one(
            raw_matches
        )

        matching_ms = (
            time.perf_counter()
            - match_start
        ) * 1000

        matching_times.append(
            matching_ms
        )

        match_count = len(matches)

        denominator = min(
            len(previous_kp),
            len(kp)
        )

        match_ratio = (
            match_count / denominator
            if denominator > 0
            else 0.0
        )

        rep = repeatability(
            previous_kp,
            kp,
            matches
        )

        match_counts.append(
            match_count
        )

        match_ratios.append(
            match_ratio
        )

        repeatability_values.append(
            rep
        )

        track_manager.update(
            matches
        )

        pair_rows.append({
            "model": model_name,
            "frame": frame_idx,
            "previous_frame": frame_idx - 1,
            "keypoints_previous": len(previous_kp),
            "keypoints_current": len(kp),
            "matches": match_count,
            "match_ratio": match_ratio,
            "repeatability": rep,
            "extraction_ms": extraction_ms,
            "matching_ms": matching_ms,
        })

        if (
            frame_idx < 5
            or frame_idx % 25 == 24
            or frame_idx == len(thermal_files) - 1
        ):

            print(
                f"[{frame_idx + 1}/"
                f"{len(thermal_files)}] "
                f"{path.name} | "
                f"KP={len(kp)} | "
                f"M={match_count} | "
                f"R={match_ratio:.3f}"
            )

        previous_kp = kp
        previous_desc = desc
        previous_scores = scores

    average_track, max_track = (
        track_manager.finalize()
    )

    mean_kp = (
        float(np.mean(keypoint_counts))
        if keypoint_counts
        else 0.0
    )

    std_kp = (
        float(np.std(keypoint_counts))
        if keypoint_counts
        else 0.0
    )

    mean_matches = (
        float(np.mean(match_counts))
        if match_counts
        else 0.0
    )

    mean_match_ratio = (
        float(np.mean(match_ratios))
        if match_ratios
        else 0.0
    )

    mean_repeatability = (
        float(np.mean(
            repeatability_values
        ))
        if repeatability_values
        else 0.0
    )

    extraction_mean = (
        float(np.mean(
            extraction_times
        ))
        if extraction_times
        else 0.0
    )

    extraction_p95 = (
        float(np.percentile(
            extraction_times,
            95
        ))
        if extraction_times
        else 0.0
    )

    matching_mean = (
        float(np.mean(
            matching_times
        ))
        if matching_times
        else 0.0
    )

    matching_p95 = (
        float(np.percentile(
            matching_times,
            95
        ))
        if matching_times
        else 0.0
    )

    peak_rss = max(
        peak_rss,
        get_rss_mb()
    )

    print()

    print(
        f"Mean keypoints        : "
        f"{mean_kp:.2f}"
    )

    print(
        f"Std keypoints         : "
        f"{std_kp:.2f}"
    )

    print(
        f"Mean matches          : "
        f"{mean_matches:.2f}"
    )

    print(
        f"Match ratio           : "
        f"{mean_match_ratio:.4f}"
    )

    print(
        f"Repeatability         : "
        f"{mean_repeatability:.4f}"
    )

    print(
        f"Average track length  : "
        f"{average_track:.2f}"
    )

    print(
        f"Max track length      : "
        f"{max_track}"
    )

    print(
        f"Extraction mean       : "
        f"{extraction_mean:.2f} ms"
    )

    print(
        f"Extraction p95        : "
        f"{extraction_p95:.2f} ms"
    )

    print(
        f"Matching mean         : "
        f"{matching_mean:.2f} ms"
    )

    print(
        f"Matching p95         : "
        f"{matching_p95:.2f} ms"
    )

    print(
        f"Peak RSS              : "
        f"{peak_rss:.2f} MB"
    )

    return {
        "model": model_name,
        "keypoints_mean": mean_kp,
        "keypoints_std": std_kp,
        "mean_matches": mean_matches,
        "match_ratio": mean_match_ratio,
        "repeatability": mean_repeatability,
        "average_track_length": average_track,
        "max_track_length": max_track,
        "extraction_mean_ms": extraction_mean,
        "extraction_p95_ms": extraction_p95,
        "matching_mean_ms": matching_mean,
        "matching_p95_ms": matching_p95,
        "peak_rss_mb": peak_rss,
    }, pair_rows


def main(max_frames):

    thermal_files = sorted(
        THERMAL_DIR.glob("*.png")
    )

    if not thermal_files:

        raise RuntimeError(
            f"No thermal PNG files found in "
            f"{THERMAL_DIR}"
        )

    thermal_files = thermal_files[
        :max_frames
    ]

    print()
    print("=" * 75)
    print(
        "PHASE 1 THERMAL "
        "SEQUENCE PROFILING"
    )
    print("=" * 75)

    print()

    print(
        f"Dataset       : "
        f"{ROOT / 'data' / 'desk2-circle'}"
    )

    print(
        f"Thermal frames: "
        f"{len(thermal_files)}"
    )

    print(
        "Frame size    : 640 x 480"
    )

    print(
        "Input format  : uint16 thermal"
    )

    print(
        "Feature input : normalized uint8"
    )

    print()

    print(
        "NOTE: Thermal frames are evaluated "
        "as a separate sensor stream."
    )

    print(
        "NOTE: RGB calibration is NOT used "
        "for thermal images."
    )

    print(
        "NOTE: Essential-Matrix inlier ratio "
        "is not reported for thermal."
    )

    summary_rows = []
    all_pair_rows = []

    for model_name in MODELS:

        summary, pair_rows = profile_model(
            model_name,
            thermal_files
        )

        summary_rows.append(
            summary
        )

        all_pair_rows.extend(
            pair_rows
        )

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    summary_path = (
        RESULTS_DIR
        / "phase1_thermal_summary.csv"
    )

    pair_path = (
        RESULTS_DIR
        / "phase1_thermal_pair_metrics.csv"
    )

    with open(
        summary_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    with open(
        pair_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=all_pair_rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            all_pair_rows
        )

    print()
    print("=" * 125)
    print(
        "FINAL PHASE 1 THERMAL RESULTS"
    )
    print("=" * 125)

    print(
        f"{'Model':<12}"
        f"{'KP':>9}"
        f"{'Match':>10}"
        f"{'Repeat':>10}"
        f"{'Track':>10}"
        f"{'Extract':>12}"
        f"{'Match ms':>12}"
        f"{'RSS MB':>10}"
    )

    print("-" * 125)

    for r in summary_rows:

        print(
            f"{r['model']:<12}"
            f"{r['keypoints_mean']:>9.1f}"
            f"{r['match_ratio']:>10.3f}"
            f"{r['repeatability']:>10.3f}"
            f"{r['average_track_length']:>10.2f}"
            f"{r['extraction_mean_ms']:>12.1f}"
            f"{r['matching_mean_ms']:>12.1f}"
            f"{r['peak_rss_mb']:>10.1f}"
        )

    print("=" * 125)

    print()
    print("Summary CSV:")
    print(summary_path)

    print()
    print("Per-pair CSV:")
    print(pair_path)

    print()
    print(
        "PHASE 1 THERMAL "
        "PROFILING COMPLETE"
    )

    print("=" * 75)


if __name__ == "__main__":

    max_frames = 20

    if "--max-frames" in sys.argv:

        idx = sys.argv.index(
            "--max-frames"
        )

        if idx + 1 < len(sys.argv):

            max_frames = int(
                sys.argv[idx + 1]
            )

    main(max_frames)
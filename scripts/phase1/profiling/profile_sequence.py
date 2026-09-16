import sys
from pathlib import Path
import time
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from adapters.orb import ORBExtractor
from adapters.akaze import AKAZEExtractor
from adapters.xfeat import XFeatExtractor
from adapters.aliked import ALIKEDExtractor
from adapters.superpoint import SuperPointExtractor
from adapters.lightglue import LightGlueMatcher

from scripts.phase1.matching.matchers import (
    match_binary,
    match_l2,
)

from scripts.phase1.ransac.ransac_all import run_ransac


# ==================================================
# CONFIGURATION
# ==================================================

FRAMES_DIR = PROJECT_ROOT / "frames"

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_FRAMES = 20

WARMUP_RUNS = 2


# ==================================================
# LOAD FRAMES
# ==================================================

def load_frames():

    frame_paths = sorted(
        FRAMES_DIR.glob("frame_*.png")
    )

    if len(frame_paths) < NUM_FRAMES:

        raise RuntimeError(
            f"Only {len(frame_paths)} frames found. "
            f"Need at least {NUM_FRAMES}."
        )

    frame_paths = frame_paths[:NUM_FRAMES]

    frames = []

    for path in frame_paths:

        image = cv2.imread(
            str(path)
        )

        if image is None:

            raise RuntimeError(
                f"Could not load {path}"
            )

        frames.append(image)

    return frames


# ==================================================
# EXTRACTION WRAPPERS
# ==================================================

def extract_orb(extractor, image):

    keypoints, descriptors = (
        extractor.extract(image)
    )

    points = np.array(
        [kp.pt for kp in keypoints],
        dtype=np.float32
    )

    return points, descriptors


def extract_akaze(extractor, image):

    keypoints, descriptors = (
        extractor.extract(image)
    )

    points = np.array(
        [kp.pt for kp in keypoints],
        dtype=np.float32
    )

    return points, descriptors


def extract_torch(extractor, image):

    tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    keypoints, descriptors, scores = (
        extractor.extract(tensor)
    )

    keypoints = (
        keypoints
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    descriptors = (
        descriptors
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return keypoints, descriptors


def extract_superpoint(extractor, image):

    keypoints, descriptors, scores = (
        extractor.extract(image)
    )

    return (
        keypoints.astype(np.float32),
        descriptors.astype(np.float32)
    )


# ==================================================
# MATCHING
# ==================================================

def perform_matching(
    name,
    kp1,
    desc1,
    kp2,
    desc2,
    image_shape,
    lightglue=None
):

    if name in ["ORB", "AKAZE"]:

        return match_binary(
            desc1,
            desc2,
            ratio_threshold=0.8
        )

    elif name == "XFeat":

        return match_l2(
            desc1,
            desc2,
            ratio_threshold=0.8
        )

    elif name in ["ALIKED", "SuperPoint"]:

        return lightglue.match(
            kp1,
            desc1,
            kp2,
            desc2,
            image_shape
        )

    else:

        raise ValueError(
            f"Unknown extractor: {name}"
        )


# ==================================================
# MATCH DISTANCE
# ==================================================

def matched_points(
    kp1,
    kp2,
    matches
):

    if matches is None:
        return None, None

    matches = np.asarray(matches)

    if len(matches) == 0:
        return None, None

    indices1 = matches[:, 0].astype(int)
    indices2 = matches[:, 1].astype(int)

    valid = (
        (indices1 >= 0) &
        (indices1 < len(kp1)) &
        (indices2 >= 0) &
        (indices2 < len(kp2))
    )

    indices1 = indices1[valid]
    indices2 = indices2[valid]

    if len(indices1) == 0:
        return None, None

    return (
        kp1[indices1],
        kp2[indices2]
    )


# ==================================================
# REPEATABILITY
# ==================================================

def calculate_repeatability(
    kp1,
    kp2,
    matches,
    distance_threshold=3.0
):

    points1, points2 = matched_points(
        kp1,
        kp2,
        matches
    )

    if points1 is None:

        return 0.0

    distances = np.linalg.norm(
        points1 - points2,
        axis=1
    )

    repeated = np.sum(
        distances <= distance_threshold
    )

    denominator = min(
        len(kp1),
        len(kp2)
    )

    if denominator == 0:

        return 0.0

    return float(
        repeated / denominator
    )


# ==================================================
# TRACK LENGTH
# ==================================================

def update_tracks(
    tracks,
    keypoints,
    matches
):

    if matches is None:
        return tracks

    matches = np.asarray(matches)

    for match in matches:

        idx1 = int(match[0])
        idx2 = int(match[1])

        if idx1 < 0 or idx2 < 0:
            continue

        if idx1 in tracks:

            tracks[idx2] = tracks[idx1] + 1

        else:

            tracks[idx2] = 1

    return tracks


# ==================================================
# PROFILE ONE MODEL
# ==================================================

def profile_model(
    name,
    extractor,
    extract_function,
    frames,
    lightglue=None
):

    print()
    print("======================================")
    print(f"SEQUENCE PROFILING: {name}")
    print("======================================")


    # ------------------------------------------------
    # WARMUP
    # ------------------------------------------------

    print("Warm-up...")

    for i in range(
        min(WARMUP_RUNS, len(frames))
    ):

        extract_function(
            extractor,
            frames[i]
        )


    # ------------------------------------------------
    # EXTRACT ALL FRAMES
    # ------------------------------------------------

    features = []

    extraction_times = []


    print("Extracting frames...")


    for index, frame in enumerate(frames):

        start = time.perf_counter()

        kp, desc = extract_function(
            extractor,
            frame
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        features.append(
            (kp, desc)
        )

        extraction_times.append(
            elapsed
        )

        print(
            f"Frame {index:03d}: "
            f"{len(kp):4d} keypoints | "
            f"{elapsed:8.2f} ms"
        )


    # ------------------------------------------------
    # SEQUENCE METRICS
    # ------------------------------------------------

    match_counts = []

    match_ratios = []

    inlier_ratios = []

    repeatabilities = []

    matching_times = []

    track_lengths = []

    tracks = {}


    print()
    print("Matching consecutive frames...")


    for i in range(
        len(frames) - 1
    ):

        kp1, desc1 = features[i]

        kp2, desc2 = features[i + 1]


        start = time.perf_counter()

        matches = perform_matching(
            name,
            kp1,
            desc1,
            kp2,
            desc2,
            frames[i].shape,
            lightglue
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        matching_times.append(
            elapsed
        )


        # --------------------------------------------
        # Match count
        # --------------------------------------------

        num_matches = len(matches)

        match_counts.append(
            num_matches
        )


        # --------------------------------------------
        # Match ratio
        # --------------------------------------------

        denominator = min(
            len(kp1),
            len(kp2)
        )

        if denominator > 0:

            ratio = (
                num_matches /
                denominator
            )

        else:

            ratio = 0.0

        match_ratios.append(
            ratio
        )


        # --------------------------------------------
        # RANSAC
        # --------------------------------------------

        ransac = run_ransac(
            kp1,
            kp2,
            matches
        )

        inlier_ratios.append(
            ransac["inlier_ratio"]
        )


        # --------------------------------------------
        # Repeatability
        # --------------------------------------------

        repeatability = (
            calculate_repeatability(
                kp1,
                kp2,
                matches
            )
        )

        repeatabilities.append(
            repeatability
        )


        # --------------------------------------------
        # Track length
        # --------------------------------------------

        tracks = update_tracks(
            tracks,
            kp1,
            matches
        )

        current_track_lengths = list(
            tracks.values()
        )

        if current_track_lengths:

            track_lengths.append(
                float(
                    np.mean(
                        current_track_lengths
                    )
                )
            )

        else:

            track_lengths.append(
                0.0
            )


        print(
            f"Pair {i:03d}->{i+1:03d}: "
            f"matches={num_matches:4d} | "
            f"inliers={ransac['num_inliers']:4d} | "
            f"repeatability={repeatability:.4f} | "
            f"time={elapsed:.2f} ms"
        )


    # ==================================================
    # SUMMARY
    # ==================================================

    extraction_mean = float(
        np.mean(extraction_times)
    )

    extraction_p95 = float(
        np.percentile(
            extraction_times,
            95
        )
    )

    matching_mean = float(
        np.mean(matching_times)
    )

    matching_p95 = float(
        np.percentile(
            matching_times,
            95
        )
    )

    keypoints_mean = float(
        np.mean(
            [
                len(x[0])
                for x in features
            ]
        )
    )

    match_mean = float(
        np.mean(match_counts)
    )

    match_ratio_mean = float(
        np.mean(match_ratios)
    )

    inlier_ratio_mean = float(
        np.mean(inlier_ratios)
    )

    repeatability_mean = float(
        np.mean(repeatabilities)
    )

    if track_lengths:

        track_length_mean = float(
            np.mean(track_lengths)
        )

    else:

        track_length_mean = 0.0


    # ==================================================
    # MATCHER
    # ==================================================

    if name in ["ORB", "AKAZE"]:

        matcher_name = "hamming"

    elif name == "XFeat":

        matcher_name = "l2"

    else:

        matcher_name = "lightglue"


    # ==================================================
    # PRINT SUMMARY
    # ==================================================

    print()
    print("--------------------------------------")
    print(f"{name} SUMMARY")
    print("--------------------------------------")

    print(
        "Mean keypoints/frame:",
        f"{keypoints_mean:.2f}"
    )

    print(
        "Mean matches:",
        f"{match_mean:.2f}"
    )

    print(
        "Mean match ratio:",
        f"{match_ratio_mean:.4f}"
    )

    print(
        "Mean inlier ratio:",
        f"{inlier_ratio_mean:.4f}"
    )

    print(
        "Mean repeatability:",
        f"{repeatability_mean:.4f}"
    )

    print(
        "Mean track length:",
        f"{track_length_mean:.2f}"
    )

    print(
        "Extraction mean:",
        f"{extraction_mean:.2f} ms"
    )

    print(
        "Extraction P95:",
        f"{extraction_p95:.2f} ms"
    )

    print(
        "Matching mean:",
        f"{matching_mean:.2f} ms"
    )

    print(
        "Matching P95:",
        f"{matching_p95:.2f} ms"
    )


    return {

        "extractor": name,

        "matcher": matcher_name,

        "frames": len(frames),

        "mean_keypoints_per_frame":
            keypoints_mean,

        "mean_matches":
            match_mean,

        "mean_match_ratio":
            match_ratio_mean,

        "mean_inlier_ratio":
            inlier_ratio_mean,

        "mean_repeatability":
            repeatability_mean,

        "mean_track_length":
            track_length_mean,

        "extraction_mean_ms":
            extraction_mean,

        "extraction_p95_ms":
            extraction_p95,

        "matching_mean_ms":
            matching_mean,

        "matching_p95_ms":
            matching_p95,
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print()
    print("======================================")
    print(" PHASE 1 SEQUENCE PROFILER")
    print("======================================")


    # ------------------------------------------------
    # Load frames
    # ------------------------------------------------

    frames = load_frames()

    print(
        f"Loaded {len(frames)} frames."
    )


    # ------------------------------------------------
    # Create extractors
    # ------------------------------------------------

    print()
    print("Creating extractors...")


    orb = ORBExtractor()

    akaze = AKAZEExtractor()

    xfeat = XFeatExtractor(
        top_k=1000,
        detection_threshold=0.05
    )

    aliked = ALIKEDExtractor(
        max_num_keypoints=1000
    )

    superpoint = SuperPointExtractor()


    # ------------------------------------------------
    # Create matchers
    # ------------------------------------------------

    print()
    print("Creating LightGlue...")


    aliked_lightglue = LightGlueMatcher(
        feature_type="aliked"
    )

    superpoint_lightglue = LightGlueMatcher(
        feature_type="superpoint"
    )


    # ------------------------------------------------
    # Configuration
    # ------------------------------------------------

    models = [

        (
            "ORB",
            orb,
            extract_orb,
            None
        ),

        (
            "AKAZE",
            akaze,
            extract_akaze,
            None
        ),

        (
            "XFeat",
            xfeat,
            extract_torch,
            None
        ),

        (
            "ALIKED",
            aliked,
            extract_torch,
            aliked_lightglue
        ),

        (
            "SuperPoint",
            superpoint,
            extract_superpoint,
            superpoint_lightglue
        ),
    ]


    # ------------------------------------------------
    # Profile
    # ------------------------------------------------

    results = []


    for (
        name,
        extractor,
        extract_function,
        lightglue
    ) in models:

        result = profile_model(
            name,
            extractor,
            extract_function,
            frames,
            lightglue
        )

        results.append(
            result
        )


    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        "phase1_sequence_profile.csv"
    )


    fieldnames = [

        "extractor",

        "matcher",

        "frames",

        "mean_keypoints_per_frame",

        "mean_matches",

        "mean_match_ratio",

        "mean_inlier_ratio",

        "mean_repeatability",

        "mean_track_length",

        "extraction_mean_ms",

        "extraction_p95_ms",

        "matching_mean_ms",

        "matching_p95_ms",
    ]


    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    print()
    print("======================================")
    print(" SEQUENCE PROFILING COMPLETE")
    print("======================================")

    print(
        "Results saved to:"
    )

    print(output_file)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()
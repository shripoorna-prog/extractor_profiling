import sys
from pathlib import Path
import time
import csv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import psutil

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

FRAME1_PATH = PROJECT_ROOT / "frames" / "frame_0000.png"
FRAME2_PATH = PROJECT_ROOT / "frames" / "frame_0001.png"

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Number of timing repetitions
NUM_EXTRACTION_RUNS = 5
NUM_MATCHING_RUNS = 5


# ==================================================
# MEMORY
# ==================================================

PROCESS = psutil.Process(os.getpid())


def get_memory_mb():

    return PROCESS.memory_info().rss / (1024 * 1024)


# ==================================================
# FEATURE EXTRACTION WRAPPERS
# ==================================================

def extract_orb(extractor, image):

    keypoints, descriptors = extractor.extract(image)

    points = np.array(
        [kp.pt for kp in keypoints],
        dtype=np.float32
    )

    return points, descriptors


def extract_akaze(extractor, image):

    keypoints, descriptors = extractor.extract(image)

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
    extractor_name,
    descriptors1,
    descriptors2,
    keypoints1,
    keypoints2,
    image_shape,
    lightglue_matcher=None
):

    if extractor_name in ["ORB", "AKAZE"]:

        return match_binary(
            descriptors1,
            descriptors2,
            ratio_threshold=0.8
        )

    elif extractor_name == "XFeat":

        return match_l2(
            descriptors1,
            descriptors2,
            ratio_threshold=0.8
        )

    elif extractor_name in ["ALIKED", "SuperPoint"]:

        if lightglue_matcher is None:
            raise ValueError(
                f"LightGlue matcher missing for {extractor_name}"
            )

        return lightglue_matcher.match(
            keypoints1,
            descriptors1,
            keypoints2,
            descriptors2,
            image_shape
        )

    else:

        raise ValueError(
            f"Unknown extractor: {extractor_name}"
        )


# ==================================================
# MODEL LOAD TIMING
# ==================================================

def create_extractor(name):

    start = time.perf_counter()

    if name == "ORB":

        extractor = ORBExtractor()

    elif name == "AKAZE":

        extractor = AKAZEExtractor()

    elif name == "XFeat":

        extractor = XFeatExtractor(
            top_k=1000,
            detection_threshold=0.05
        )

    elif name == "ALIKED":

        extractor = ALIKEDExtractor(
            max_num_keypoints=1000
        )

    elif name == "SuperPoint":

        extractor = SuperPointExtractor()

    else:

        raise ValueError(
            f"Unknown extractor: {name}"
        )

    load_time = (
        time.perf_counter() - start
    ) * 1000

    return extractor, load_time


def create_matcher(feature_type):

    start = time.perf_counter()

    matcher = LightGlueMatcher(
        feature_type=feature_type
    )

    load_time = (
        time.perf_counter() - start
    ) * 1000

    return matcher, load_time


# ==================================================
# PERCENTILE
# ==================================================

def percentile(values, p):

    if len(values) == 0:
        return 0.0

    return float(
        np.percentile(
            np.array(values),
            p
        )
    )


# ==================================================
# PROFILE ONE EXTRACTOR
# ==================================================

def profile_extractor(
    name,
    extractor,
    extract_function,
    image1,
    image2,
    extractor_load_ms,
    lightglue_matcher=None,
    lightglue_load_ms=0.0
):

    print()
    print("--------------------------------------")
    print(f"Profiling: {name}")
    print("--------------------------------------")


    # ==================================================
    # INITIAL MEMORY
    # ==================================================

    memory_before = get_memory_mb()


    # ==================================================
    # EXTRACTION TIMING
    # ==================================================

    extraction_times = []

    kp1 = None
    desc1 = None

    kp2 = None
    desc2 = None


    for run in range(NUM_EXTRACTION_RUNS):

        start = time.perf_counter()

        current_kp1, current_desc1 = extract_function(
            extractor,
            image1
        )

        current_time1 = (
            time.perf_counter() - start
        ) * 1000


        start = time.perf_counter()

        current_kp2, current_desc2 = extract_function(
            extractor,
            image2
        )

        current_time2 = (
            time.perf_counter() - start
        ) * 1000


        extraction_times.append(
            (current_time1 + current_time2) / 2
        )


        # Keep final descriptors/keypoints
        kp1 = current_kp1
        desc1 = current_desc1

        kp2 = current_kp2
        desc2 = current_desc2


        print(
            f"Extraction run {run + 1}: "
            f"{extraction_times[-1]:.2f} ms"
        )


    extraction_mean = float(
        np.mean(extraction_times)
    )

    extraction_p95 = percentile(
        extraction_times,
        95
    )


    # ==================================================
    # MATCHING TIMING
    # ==================================================

    matching_times = []

    matches = None


    for run in range(NUM_MATCHING_RUNS):

        start = time.perf_counter()

        current_matches = perform_matching(
            name,
            desc1,
            desc2,
            kp1,
            kp2,
            image1.shape,
            lightglue_matcher
        )

        current_matching_time = (
            time.perf_counter() - start
        ) * 1000

        matching_times.append(
            current_matching_time
        )

        matches = current_matches


        print(
            f"Matching run {run + 1}: "
            f"{current_matching_time:.2f} ms"
        )


    matching_mean = float(
        np.mean(matching_times)
    )

    matching_p95 = percentile(
        matching_times,
        95
    )


    # ==================================================
    # RANSAC
    # ==================================================

    ransac_result = run_ransac(
        kp1,
        kp2,
        matches
    )


    # ==================================================
    # MATCH RATIO
    # ==================================================

    denominator = min(
        len(kp1),
        len(kp2)
    )

    if denominator > 0:

        match_ratio = (
            len(matches) /
            denominator
        )

    else:

        match_ratio = 0.0


    # ==================================================
    # MATCHER NAME
    # ==================================================

    if name in ["ORB", "AKAZE"]:

        matcher_name = "hamming"

    elif name == "XFeat":

        matcher_name = "l2"

    else:

        matcher_name = "lightglue"


    # ==================================================
    # MEMORY
    # ==================================================

    memory_after = get_memory_mb()

    peak_rss_mb = memory_after


    # ==================================================
    # PRINT RESULTS
    # ==================================================

    print()

    print(
        "Keypoints:",
        len(kp1),
        "/",
        len(kp2)
    )

    print(
        "Matches:",
        ransac_result["num_matches"]
    )

    print(
        "Match ratio:",
        f"{match_ratio:.4f}"
    )

    print(
        "RANSAC inliers:",
        ransac_result["num_inliers"]
    )

    print(
        "Inlier ratio:",
        f"{ransac_result['inlier_ratio']:.4f}"
    )

    print(
        "Extraction mean:",
        f"{extraction_mean:.2f} ms"
    )

    print(
        "Extraction p95:",
        f"{extraction_p95:.2f} ms"
    )

    print(
        "Matching mean:",
        f"{matching_mean:.2f} ms"
    )

    print(
        "Matching p95:",
        f"{matching_p95:.2f} ms"
    )

    print(
        "Peak RSS:",
        f"{peak_rss_mb:.2f} MB"
    )

    print(
        "Extractor load:",
        f"{extractor_load_ms:.2f} ms"
    )

    if name in ["ALIKED", "SuperPoint"]:

        print(
            "LightGlue load:",
            f"{lightglue_load_ms:.2f} ms"
        )


    # ==================================================
    # RETURN
    # ==================================================

    return {

        "extractor": name,

        "matcher": matcher_name,

        "keypoints_frame1": len(kp1),

        "keypoints_frame2": len(kp2),

        "matches": ransac_result["num_matches"],

        "match_ratio": match_ratio,

        "ransac_inliers":
            ransac_result["num_inliers"],

        "inlier_ratio":
            ransac_result["inlier_ratio"],

        "extraction_mean_ms":
            extraction_mean,

        "extraction_p95_ms":
            extraction_p95,

        "matching_mean_ms":
            matching_mean,

        "matching_p95_ms":
            matching_p95,

        "peak_rss_mb":
            peak_rss_mb,

        "extractor_load_ms":
            extractor_load_ms,

        "lightglue_load_ms":
            lightglue_load_ms,

        "repeatability":
            "N/A",

        "track_length":
            "N/A",
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print()
    print("======================================")
    print(" PHASE 1 FEATURE EXTRACTOR PROFILER")
    print("======================================")


    # ==================================================
    # LOAD FRAMES
    # ==================================================

    image1 = cv2.imread(
        str(FRAME1_PATH)
    )

    image2 = cv2.imread(
        str(FRAME2_PATH)
    )

    if image1 is None or image2 is None:

        raise FileNotFoundError(
            "Could not load test frames."
        )

    print(
        "Frames loaded successfully."
    )


    # ==================================================
    # CREATE EXTRACTORS
    # ==================================================

    print()
    print("Creating feature extractors...")


    extractors = []

    for name in [
        "ORB",
        "AKAZE",
        "XFeat",
        "ALIKED",
        "SuperPoint"
    ]:

        print(
            f"Loading {name}..."
        )

        extractor, load_time = (
            create_extractor(name)
        )

        extractors.append(
            (
                name,
                extractor,
                load_time
            )
        )

        print(
            f"{name} loaded in "
            f"{load_time:.2f} ms"
        )


    # ==================================================
    # CREATE LIGHTGLUE
    # ==================================================

    print()
    print("Creating LightGlue matchers...")


    aliked_lightglue, aliked_lg_load = (
        create_matcher("aliked")
    )


    superpoint_lightglue, superpoint_lg_load = (
        create_matcher("superpoint")
    )


    # ==================================================
    # MATCHER MAP
    # ==================================================

    matcher_map = {

        "ORB": (
            None,
            0.0
        ),

        "AKAZE": (
            None,
            0.0
        ),

        "XFeat": (
            None,
            0.0
        ),

        "ALIKED": (
            aliked_lightglue,
            aliked_lg_load
        ),

        "SuperPoint": (
            superpoint_lightglue,
            superpoint_lg_load
        ),
    }


    # ==================================================
    # PROFILE
    # ==================================================

    results = []


    for (
        name,
        extractor,
        extractor_load_ms
    ) in extractors:

        matcher, matcher_load_ms = (
            matcher_map[name]
        )


        extract_function_map = {

            "ORB": extract_orb,

            "AKAZE": extract_akaze,

            "XFeat": extract_torch,

            "ALIKED": extract_torch,

            "SuperPoint": extract_superpoint,
        }


        result = profile_extractor(

            name,

            extractor,

            extract_function_map[name],

            image1,

            image2,

            extractor_load_ms,

            matcher,

            matcher_load_ms
        )


        results.append(result)


    # ==================================================
    # SAVE CSV
    # ==================================================

    output_file = (
        OUTPUT_DIR /
        "phase1_profile_v2.csv"
    )


    fieldnames = [

        "extractor",

        "matcher",

        "keypoints_frame1",

        "keypoints_frame2",

        "matches",

        "match_ratio",

        "ransac_inliers",

        "inlier_ratio",

        "extraction_mean_ms",

        "extraction_p95_ms",

        "matching_mean_ms",

        "matching_p95_ms",

        "peak_rss_mb",

        "extractor_load_ms",

        "lightglue_load_ms",

        "repeatability",

        "track_length",
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

        writer.writerows(results)


    # ==================================================
    # COMPLETE
    # ==================================================

    print()
    print("======================================")
    print(" PROFILING COMPLETE")
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
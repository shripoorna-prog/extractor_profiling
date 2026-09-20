from pathlib import Path
import sys
import json
import csv
import time
import argparse

import cv2
import numpy as np
import psutil
import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT ADAPTERS
# ============================================================

from adapters.orb import ORBExtractor
from adapters.akaze import AKAZEExtractor
from adapters.xfeat import XFeatExtractor
from adapters.aliked import ALIKEDExtractor
from adapters.superpoint import SuperPointExtractor

from scripts.phase1.profiling.matchers import (
    match_binary,
    match_l2,
)


# ============================================================
# LIGHTGLUE
# ============================================================

from lightglue import LightGlue


# ============================================================
# PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "desk2-circle"

RGB_DIR = DATA_DIR / "rgb"
CALIBRATION_FILE = DATA_DIR / "calibration.json"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

DEFAULT_MAX_FRAMES = 20

RATIO_THRESHOLD = 0.80

ESSENTIAL_RANSAC_THRESHOLD = 1.0
ESSENTIAL_CONFIDENCE = 0.999

REPEATABILITY_RADIUS = 3.0

ORB_FEATURES = 1000
AKAZE_FEATURES = 1000
XFEAT_TOP_K = 1000
ALIKED_TOP_K = 1000

EXTRACTORS = [
    "ORB",
    "AKAZE",
    "XFeat",
    "ALIKED",
    "SuperPoint",
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Final Phase 1 RGB sequence profiler"
)

parser.add_argument(
    "--max-frames",
    type=int,
    default=DEFAULT_MAX_FRAMES,
    help="Number of RGB frames to process. Use 20 for test, 678 for full run."
)

args = parser.parse_args()

MAX_FRAMES = args.max_frames


# ============================================================
# PROCESS
# ============================================================

PROCESS = psutil.Process()


# ============================================================
# HELPERS
# ============================================================

def rss_mb():
    return PROCESS.memory_info().rss / (1024 * 1024)


def keypoints_to_xy(keypoints):
    """
    Convert OpenCV / NumPy / Torch keypoints to Nx2 NumPy array.
    """

    if keypoints is None:
        return np.empty((0, 2), dtype=np.float32)

    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()

    keypoints = np.asarray(keypoints)

    # OpenCV KeyPoint list
    if len(keypoints) > 0 and isinstance(keypoints[0], cv2.KeyPoint):
        return np.asarray(
            [kp.pt for kp in keypoints],
            dtype=np.float32
        )

    # Remove batch dimension
    if keypoints.ndim == 3 and keypoints.shape[0] == 1:
        keypoints = keypoints[0]

    # Standard Nx2
    if keypoints.ndim == 2 and keypoints.shape[1] == 2:
        return keypoints.astype(np.float32)

    # SuperPoint-style 2xN
    if keypoints.ndim == 2 and keypoints.shape[0] == 2:
        return keypoints.T.astype(np.float32)

    return np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)


def descriptors_to_numpy(descriptors):
    if descriptors is None:
        return None

    if isinstance(descriptors, torch.Tensor):
        descriptors = descriptors.detach().cpu().numpy()

    descriptors = np.asarray(descriptors)

    if descriptors.ndim == 3 and descriptors.shape[0] == 1:
        descriptors = descriptors[0]

    return descriptors


def scores_to_numpy(scores):
    if scores is None:
        return None

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    scores = np.asarray(scores)

    if scores.ndim > 1:
        scores = scores.reshape(-1)

    return scores


def optical_flow_repeatability(
    image1,
    image2,
    keypoints1,
    keypoints2,
    radius=3.0,
):
    """
    Optical-flow based repeatability.

    Denominator:
        all keypoints in frame N.

    Numerator:
        keypoints whose optical-flow predicted location
        has a corresponding keypoint in frame N+1
        within the specified radius.

    Greedy one-to-one assignment is used.
    """

    if len(keypoints1) == 0:
        return 0.0

    if len(keypoints2) == 0:
        return 0.0

    points1 = np.asarray(
        keypoints1,
        dtype=np.float32
    ).reshape(-1, 1, 2)

    points2, status, _ = cv2.calcOpticalFlowPyrLK(
        image1,
        image2,
        points1,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if points2 is None or status is None:
        return 0.0

    tracked = points2.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)

    predicted = tracked[status]

    if len(predicted) == 0:
        return 0.0

    target = np.asarray(
        keypoints2,
        dtype=np.float32
    )

    used = set()
    matched = 0

    for point in predicted:

        distances = np.linalg.norm(
            target - point,
            axis=1
        )

        order = np.argsort(distances)

        for idx in order:

            if idx in used:
                continue

            if distances[idx] <= radius:

                used.add(int(idx))
                matched += 1
                break

    return matched / len(keypoints1)


def essential_ransac_inlier_ratio(
    keypoints1,
    keypoints2,
    matches,
    K,
):
    """
    Required Phase 1 geometric metric.

    Returns:
        Essential Matrix RANSAC inlier count
        Essential Matrix RANSAC inlier ratio
    """

    if matches is None:
        return 0, 0.0

    matches = np.asarray(matches)

    if len(matches) < 5:
        return 0, 0.0

    kp1 = np.asarray(
        keypoints1,
        dtype=np.float64
    )

    kp2 = np.asarray(
        keypoints2,
        dtype=np.float64
    )

    indices1 = matches[:, 0].astype(int)
    indices2 = matches[:, 1].astype(int)

    if (
        np.any(indices1 >= len(kp1))
        or np.any(indices2 >= len(kp2))
    ):
        return 0, 0.0

    points1 = kp1[indices1]
    points2 = kp2[indices2]

    try:

        E, mask = cv2.findEssentialMat(
            points1,
            points2,
            cameraMatrix=K,
            method=cv2.RANSAC,
            prob=ESSENTIAL_CONFIDENCE,
            threshold=ESSENTIAL_RANSAC_THRESHOLD,
        )

    except cv2.error:
        return 0, 0.0

    if E is None or mask is None:
        return 0, 0.0

    mask = mask.reshape(-1).astype(bool)

    inliers = int(np.sum(mask))

    ratio = (
        inliers / len(matches)
        if len(matches) > 0
        else 0.0
    )

    return inliers, ratio


def track_statistics(track_lengths):
    if len(track_lengths) == 0:
        return 0.0, 0

    return (
        float(np.mean(track_lengths)),
        int(np.max(track_lengths)),
    )


class TrackManager:

    def __init__(self):

        self.next_track_id = 0

        self.current_tracks = {}

        self.track_lengths = []

    def update(self, matches, num_current_keypoints):

        matches = np.asarray(
            matches,
            dtype=np.int32
        )

        if matches.size == 0:
            matches = np.empty(
                (0, 2),
                dtype=np.int32
            )

        matched_current = set()

        new_tracks = {}

        # Continue existing tracks
        for previous_idx, current_idx in matches:

            previous_idx = int(previous_idx)
            current_idx = int(current_idx)

            if previous_idx in self.current_tracks:

                track_id = self.current_tracks[
                    previous_idx
                ]

                new_tracks[current_idx] = track_id

                matched_current.add(current_idx)

        # Finalize tracks that disappeared
        continuing_ids = set(
            new_tracks.values()
        )

        previous_ids = set(
            self.current_tracks.values()
        )

        ended_ids = previous_ids - continuing_ids

        for track_id in ended_ids:
            self.track_lengths.append(
                self._track_length(track_id)
            )

        # Start new tracks
        for current_idx in range(
            num_current_keypoints
        ):

            if current_idx in matched_current:
                continue

            track_id = self.next_track_id

            self.next_track_id += 1

            new_tracks[current_idx] = track_id

        self.current_tracks = new_tracks

        # Track lengths are reconstructed separately.
        # This dictionary stores the current-frame association.
        return len(new_tracks)

    def _track_length(self, track_id):
        return self.lengths.get(track_id, 1)

    def initialize(self, num_keypoints):

        self.lengths = {}

        self.current_tracks = {}

        for idx in range(num_keypoints):

            track_id = self.next_track_id

            self.next_track_id += 1

            self.current_tracks[idx] = track_id

            self.lengths[track_id] = 1

    def update_lengths(self, matches, num_current_keypoints):

        if not hasattr(self, "lengths"):
            self.initialize(num_current_keypoints)
            return

        matches = np.asarray(
            matches,
            dtype=np.int32
        )

        if matches.size == 0:
            matches = np.empty(
                (0, 2),
                dtype=np.int32
            )

        previous_tracks = self.current_tracks

        new_tracks = {}

        continued_ids = set()

        for previous_idx, current_idx in matches:

            previous_idx = int(previous_idx)
            current_idx = int(current_idx)

            if previous_idx not in previous_tracks:
                continue

            track_id = previous_tracks[
                previous_idx
            ]

            new_tracks[current_idx] = track_id

            self.lengths[track_id] += 1

            continued_ids.add(track_id)

        previous_ids = set(
            previous_tracks.values()
        )

        ended_ids = previous_ids - continued_ids

        for track_id in ended_ids:

            self.track_lengths.append(
                self.lengths[track_id]
            )

        # Start new tracks
        for current_idx in range(
            num_current_keypoints
        ):

            if current_idx in new_tracks:
                continue

            track_id = self.next_track_id

            self.next_track_id += 1

            new_tracks[current_idx] = track_id

            self.lengths[track_id] = 1

        self.current_tracks = new_tracks

    def finalize(self):

        if hasattr(self, "lengths"):

            for track_id in self.current_tracks.values():

                self.track_lengths.append(
                    self.lengths[track_id]
                )

        return track_statistics(
            self.track_lengths
        )


def enforce_one_to_one(matches):

    if matches is None:
        return np.empty(
            (0, 2),
            dtype=np.int32
        )

    matches = np.asarray(
        matches,
        dtype=np.int32
    )

    if matches.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.int32
        )

    used_a = set()
    used_b = set()

    selected = []

    for a, b in matches:

        a = int(a)
        b = int(b)

        if a in used_a or b in used_b:
            continue

        used_a.add(a)
        used_b.add(b)

        selected.append(
            [a, b]
        )

    return np.asarray(
        selected,
        dtype=np.int32
    )


def lightglue_matches(
    matcher,
    keypoints1,
    descriptors1,
    keypoints2,
    descriptors2,
    image_shape,
):
    """
    Run official LightGlue using raw adapter outputs.
    """

    h, w = image_shape[:2]

    kp1 = torch.as_tensor(
        keypoints1,
        dtype=torch.float32
    )

    kp2 = torch.as_tensor(
        keypoints2,
        dtype=torch.float32
    )

    desc1 = torch.as_tensor(
        descriptors1,
        dtype=torch.float32
    )

    desc2 = torch.as_tensor(
        descriptors2,
        dtype=torch.float32
    )

    if kp1.ndim == 2:
        kp1 = kp1.unsqueeze(0)

    if kp2.ndim == 2:
        kp2 = kp2.unsqueeze(0)

    if desc1.ndim == 2:
        desc1 = desc1.unsqueeze(0)

    if desc2.ndim == 2:
        desc2 = desc2.unsqueeze(0)

    data = {
        "image0": {
            "keypoints": kp1,
            "descriptors": desc1,
            "image_size": torch.tensor(
                [[w, h]],
                dtype=torch.float32
            ),
        },
        "image1": {
            "keypoints": kp2,
            "descriptors": desc2,
            "image_size": torch.tensor(
                [[w, h]],
                dtype=torch.float32
            ),
        },
    }

    with torch.inference_mode():

        output = matcher(data)

    matches = output["matches"]

    if isinstance(matches, list):

        if len(matches) == 0:
            return np.empty(
                (0, 2),
                dtype=np.int32
            )

        matches = matches[0]

    if isinstance(matches, torch.Tensor):

        matches = matches.detach().cpu().numpy()

    matches = np.asarray(
        matches,
        dtype=np.int32
    )

    if matches.ndim == 3:
        matches = matches[0]

    if matches.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.int32
        )

    return enforce_one_to_one(matches)


# ============================================================
# EXTRACTOR FACTORY
# ============================================================

def create_extractor(name):

    if name == "ORB":
        return ORBExtractor(
            nfeatures=ORB_FEATURES
        )

    if name == "AKAZE":
        return AKAZEExtractor(
            nfeatures=AKAZE_FEATURES
        )

    if name == "XFeat":
        return XFeatExtractor(
            top_k=XFEAT_TOP_K
        )

    if name == "ALIKED":
        return ALIKEDExtractor(
            max_num_keypoints=ALIKED_TOP_K
        )

    if name == "SuperPoint":
        return SuperPointExtractor(
            nms_dist=4,
            conf_thresh=0.015,
            nn_thresh=0.7
        )

    raise ValueError(
        f"Unknown extractor: {name}"
    )


# ============================================================
# MATCHER FACTORY
# ============================================================

def create_matcher(name):

    if name == "ALIKED":

        return LightGlue(
            features="aliked"
        ).eval()

    if name == "SuperPoint":

        return LightGlue(
            features="superpoint"
        ).eval()

    return None


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(extractor, image):

    output = extractor.extract(image)

    if len(output) == 2:

        keypoints, descriptors = output

        scores = None

    else:

        keypoints, descriptors, scores = output

    keypoints = keypoints_to_xy(
        keypoints
    )

    descriptors = descriptors_to_numpy(
        descriptors
    )

    scores = scores_to_numpy(
        scores
    )

    return (
        keypoints,
        descriptors,
        scores,
    )


# ============================================================
# MATCH FEATURES
# ============================================================

def match_features(
    model_name,
    matcher,
    kp1,
    desc1,
    kp2,
    desc2,
    image_shape,
):

    if model_name in [
        "ORB",
        "AKAZE",
    ]:

        return match_binary(
            desc1,
            desc2,
            ratio_threshold=RATIO_THRESHOLD,
        )

    if model_name == "XFeat":

        return match_l2(
            desc1,
            desc2,
            ratio_threshold=RATIO_THRESHOLD,
        )

    if model_name in [
        "ALIKED",
        "SuperPoint",
    ]:

        return lightglue_matches(
            matcher,
            kp1,
            desc1,
            kp2,
            desc2,
            image_shape,
        )

    raise ValueError(
        f"Unknown matcher type: {model_name}"
    )


# ============================================================
# LOAD RGB FILES
# ============================================================

frame_paths = sorted(
    RGB_DIR.glob("frame_*.png")
)

if len(frame_paths) == 0:
    raise FileNotFoundError(
        f"No RGB frames found in {RGB_DIR}"
    )

if MAX_FRAMES is not None:

    frame_paths = frame_paths[:MAX_FRAMES]


# ============================================================
# LOAD CALIBRATION
# ============================================================

with open(
    CALIBRATION_FILE,
    "r"
) as f:

    calibration = json.load(f)

K = np.asarray(
    calibration["K"],
    dtype=np.float64
).reshape(3, 3)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("FINAL PHASE 1 RGB SEQUENCE PROFILING")
print("=" * 75)

print()
print(f"Dataset      : {DATA_DIR}")
print(f"RGB frames   : {len(frame_paths)}")
print(f"Frame size   : 960 x 540")
print()
print("Essential Matrix calibration:")
print(K)

print()
print(
    "NOTE: Inlier ratio uses Essential Matrix RANSAC."
)


# ============================================================
# PROFILE ONE MODEL
# ============================================================

def profile_model(model_name):

    print()
    print("=" * 75)
    print(f"PROFILE: {model_name}")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD EXTRACTOR
    # --------------------------------------------------------

    load_start = time.perf_counter()

    extractor = create_extractor(
        model_name
    )

    extractor_load_ms = (
        time.perf_counter() - load_start
    ) * 1000.0


    # --------------------------------------------------------
    # LOAD MATCHER
    # --------------------------------------------------------

    matcher_load_ms = 0.0

    matcher = None

    if model_name in [
        "ALIKED",
        "SuperPoint",
    ]:

        matcher_start = time.perf_counter()

        matcher = create_matcher(
            model_name
        )

        matcher_load_ms = (
            time.perf_counter() - matcher_start
        ) * 1000.0


    total_load_ms = (
        extractor_load_ms +
        matcher_load_ms
    )


    print(
        f"Extractor load : "
        f"{extractor_load_ms:.2f} ms"
    )

    print(
        f"Matcher load   : "
        f"{matcher_load_ms:.2f} ms"
    )


    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    keypoint_counts = []

    match_counts = []

    match_ratios = []

    essential_inlier_counts = []

    essential_inlier_ratios = []

    repeatabilities = []

    extraction_times = []

    matching_times = []

    pair_rows = []


    # --------------------------------------------------------
    # TRACK MANAGER
    # --------------------------------------------------------

    tracker = TrackManager()

    previous_image = None

    previous_keypoints = None

    previous_descriptors = None

    first_frame = True


    # --------------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------------

    for frame_number, frame_path in enumerate(
        frame_paths
    ):

        image = cv2.imread(
            str(frame_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                f"WARNING: Could not load "
                f"{frame_path.name}"
            )

            continue


        # ----------------------------------------------------
        # EXTRACTION
        # ----------------------------------------------------

        extraction_start = time.perf_counter()

        (
            keypoints,
            descriptors,
            scores,
        ) = extract_features(
            extractor,
            image
        )

        extraction_ms = (
            time.perf_counter() -
            extraction_start
        ) * 1000.0

        extraction_times.append(
            extraction_ms
        )

        keypoint_counts.append(
            len(keypoints)
        )


        # ----------------------------------------------------
        # FIRST FRAME
        # ----------------------------------------------------

        if first_frame:

            tracker.initialize(
                len(keypoints)
            )

            previous_image = image
            previous_keypoints = keypoints
            previous_descriptors = descriptors

            first_frame = False

            print(
                f"[{frame_number + 1}/"
                f"{len(frame_paths)}] "
                f"{frame_path.name} "
                f"KP={len(keypoints)}"
            )

            continue


        # ----------------------------------------------------
        # MATCHING
        # ----------------------------------------------------

        matching_start = time.perf_counter()

        matches = match_features(
            model_name,
            matcher,
            previous_keypoints,
            previous_descriptors,
            keypoints,
            descriptors,
            image.shape,
        )

        matching_ms = (
            time.perf_counter() -
            matching_start
        ) * 1000.0

        matching_times.append(
            matching_ms
        )


        match_count = len(matches)

        match_counts.append(
            match_count
        )


        # ----------------------------------------------------
        # MATCH RATIO
        # ----------------------------------------------------

        denominator = min(
            len(previous_keypoints),
            len(keypoints)
        )

        if denominator > 0:

            match_ratio = (
                match_count /
                denominator
            )

        else:

            match_ratio = 0.0

        match_ratios.append(
            match_ratio
        )


        # ----------------------------------------------------
        # ESSENTIAL MATRIX RANSAC
        # ----------------------------------------------------

        (
            essential_inliers,
            essential_ratio,
        ) = essential_ransac_inlier_ratio(
            previous_keypoints,
            keypoints,
            matches,
            K,
        )

        essential_inlier_counts.append(
            essential_inliers
        )

        essential_inlier_ratios.append(
            essential_ratio
        )


        # ----------------------------------------------------
        # REPEATABILITY
        # ----------------------------------------------------

        repeatability = (
            optical_flow_repeatability(
                cv2.cvtColor(
                    previous_image,
                    cv2.COLOR_BGR2GRAY
                ),
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                ),
                previous_keypoints,
                keypoints,
                radius=REPEATABILITY_RADIUS,
            )
        )

        repeatabilities.append(
            repeatability
        )


        # ----------------------------------------------------
        # TRACKS
        # ----------------------------------------------------

        tracker.update_lengths(
            matches,
            len(keypoints)
        )


        # ----------------------------------------------------
        # PER-PAIR CSV ROW
        # ----------------------------------------------------

        pair_rows.append({

            "model": model_name,

            "frame_1": frame_number - 1,

            "frame_2": frame_number,

            "frame_1_name": frame_paths[
                frame_number - 1
            ].name,

            "frame_2_name": frame_path.name,

            "keypoints_1": len(
                previous_keypoints
            ),

            "keypoints_2": len(
                keypoints
            ),

            "matches": match_count,

            "match_ratio": match_ratio,

            "essential_ransac_inliers":
                essential_inliers,

            "essential_inlier_ratio":
                essential_ratio,

            "repeatability":
                repeatability,

            "extraction_ms":
                extraction_ms,

            "matching_ms":
                matching_ms,
        })


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            frame_number < 5
            or
            (frame_number + 1) % 25 == 0
            or
            frame_number == len(frame_paths) - 1
        ):

            print(
                f"[{frame_number + 1}/"
                f"{len(frame_paths)}] "
                f"{frame_path.name} | "
                f"KP={len(keypoints)} | "
                f"M={match_count} | "
                f"E={essential_ratio:.3f}"
            )


        # ----------------------------------------------------
        # NEXT FRAME
        # ----------------------------------------------------

        previous_image = image

        previous_keypoints = keypoints

        previous_descriptors = descriptors


    # --------------------------------------------------------
    # FINAL TRACK STATISTICS
    # --------------------------------------------------------

    average_track_length, max_track_length = (
        tracker.finalize()
    )


    # --------------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------------

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

    mean_inlier_ratio = (
        float(np.mean(
            essential_inlier_ratios
        ))
        if essential_inlier_ratios
        else 0.0
    )

    mean_repeatability = (
        float(np.mean(
            repeatabilities
        ))
        if repeatabilities
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

    peak_rss = rss_mb()


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "model":
            model_name,

        "frames":
            len(frame_paths),

        "mean_keypoints":
            mean_kp,

        "std_keypoints":
            std_kp,

        "mean_matches":
            mean_matches,

        "mean_match_ratio":
            mean_match_ratio,

        "mean_essential_inlier_ratio":
            mean_inlier_ratio,

        "mean_repeatability":
            mean_repeatability,

        "average_track_length":
            average_track_length,

        "max_track_length":
            max_track_length,

        "extractor_load_ms":
            extractor_load_ms,

        "matcher_load_ms":
            matcher_load_ms,

        "total_load_ms":
            total_load_ms,

        "extraction_mean_ms":
            extraction_mean,

        "extraction_p95_ms":
            extraction_p95,

        "matching_mean_ms":
            matching_mean,

        "matching_p95_ms":
            matching_p95,

        "peak_rss_mb":
            peak_rss,
    }


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
        f"Essential inlier ratio: "
        f"{mean_inlier_ratio:.4f}"
    )

    print(
        f"Repeatability         : "
        f"{mean_repeatability:.4f}"
    )

    print(
        f"Average track length  : "
        f"{average_track_length:.2f}"
    )

    print(
        f"Max track length      : "
        f"{max_track_length}"
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


    return summary, pair_rows


# ============================================================
# RUN ALL MODELS
# ============================================================

all_summaries = []

all_pair_rows = []


for model_name in EXTRACTORS:

    summary, pair_rows = profile_model(
        model_name
    )

    all_summaries.append(
        summary
    )

    all_pair_rows.extend(
        pair_rows
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_file = (
    RESULTS_DIR /
    "phase1_rgb_summary.csv"
)

summary_fields = list(
    all_summaries[0].keys()
)

with open(
    summary_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fields
    )

    writer.writeheader()

    writer.writerows(
        all_summaries
    )


# ============================================================
# SAVE PAIR CSV
# ============================================================

pair_file = (
    RESULTS_DIR /
    "phase1_rgb_pair_metrics.csv"
)

pair_fields = list(
    all_pair_rows[0].keys()
)

with open(
    pair_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=pair_fields
    )

    writer.writeheader()

    writer.writerows(
        all_pair_rows
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print()
print("=" * 125)
print("FINAL PHASE 1 RGB RESULTS")
print("=" * 125)

print(
    f"{'Model':<12}"
    f"{'KP':>9}"
    f"{'Match':>10}"
    f"{'E-Inlier':>11}"
    f"{'Repeat':>10}"
    f"{'Track':>9}"
    f"{'Extract':>12}"
    f"{'Match ms':>11}"
    f"{'RSS MB':>11}"
)

print("-" * 125)

for row in all_summaries:

    print(
        f"{row['model']:<12}"
        f"{row['mean_keypoints']:>9.1f}"
        f"{row['mean_match_ratio']:>10.3f}"
        f"{row['mean_essential_inlier_ratio']:>11.3f}"
        f"{row['mean_repeatability']:>10.3f}"
        f"{row['average_track_length']:>9.2f}"
        f"{row['extraction_mean_ms']:>12.1f}"
        f"{row['matching_mean_ms']:>11.1f}"
        f"{row['peak_rss_mb']:>11.1f}"
    )


print()
print("=" * 125)

print()
print("Summary CSV:")
print(summary_file)

print()
print("Per-pair CSV:")
print(pair_file)

print()
print("PHASE 1 RGB PROFILING COMPLETE")
print("=" * 125)
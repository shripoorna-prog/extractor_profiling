import sys
from pathlib import Path

# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# STANDARD IMPORTS
# ============================================================

import os
import time
import json
import subprocess
import statistics

import cv2
import numpy as np
import psutil


# ============================================================
# PATHS
# ============================================================

FRAMES_DIR = PROJECT_ROOT / "frames"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "phase1_isolated_resources.csv"

NUM_FRAMES = 20


# ============================================================
# MEMORY
# ============================================================

def get_rss_mb(process=None):
    """
    Return current process RSS in MB.
    """
    if process is None:
        process = psutil.Process(os.getpid())

    return process.memory_info().rss / (1024 * 1024)


def get_peak_rss_mb(process=None):
    """
    Return peak RSS on Windows when available.
    Otherwise fall back to current RSS.
    """
    if process is None:
        process = psutil.Process(os.getpid())

    try:
        memory_info = process.memory_info()

        if hasattr(memory_info, "peak_wset"):
            return memory_info.peak_wset / (1024 * 1024)

    except Exception:
        pass

    return get_rss_mb(process)


# ============================================================
# LOAD FRAMES
# ============================================================

def load_frames():
    """
    Load the first NUM_FRAMES PNG frames.
    """

    frame_paths = sorted(FRAMES_DIR.glob("frame_*.png"))

    if len(frame_paths) == 0:
        raise FileNotFoundError(
            f"No frames found in: {FRAMES_DIR}"
        )

    frame_paths = frame_paths[:NUM_FRAMES]

    frames = []

    for path in frame_paths:
        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"Could not read frame: {path}"
            )

        frames.append(image)

    return frames


# ============================================================
# PERCENTILE
# ============================================================

def calculate_p95(values):
    """
    Calculate p95 latency.
    """

    if not values:
        return 0.0

    values = sorted(values)

    index = int(0.95 * (len(values) - 1))

    return values[index]


# ============================================================
# CREATE EXTRACTOR
# ============================================================

def create_extractor(model_name):
    """
    Create only the requested extractor.
    """

    if model_name == "ORB":

        from adapters.orb import ORBExtractor

        extractor = ORBExtractor(
            nfeatures=1000
        )

        return extractor, None


    elif model_name == "AKAZE":

        from adapters.akaze import AKAZEExtractor

        extractor = AKAZEExtractor(
            nfeatures=1000
        )

        return extractor, None


    elif model_name == "XFeat":

        from adapters.xfeat import XFeatExtractor

        extractor = XFeatExtractor(
            top_k=1000,
            detection_threshold=0.05
        )

        return extractor, None


    elif model_name == "ALIKED":

        from adapters.aliked import ALIKEDExtractor

        extractor = ALIKEDExtractor(
            max_num_keypoints=1000
        )

        return extractor, "aliked"


    elif model_name == "SuperPoint":

        from adapters.superpoint import SuperPointExtractor

        extractor = SuperPointExtractor()

        return extractor, "superpoint"


    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# CREATE MATCHER
# ============================================================

def create_matcher(model_name):

    """
    Create the matcher required by the model.

    ORB / AKAZE:
        Hamming

    XFeat:
        L2

    ALIKED / SuperPoint:
        LightGlue
    """

    if model_name in ["ORB", "AKAZE", "XFeat"]:

        return None


    if model_name in ["ALIKED", "SuperPoint"]:

        from lightglue import LightGlue
        import torch

        device = torch.device("cpu")

        matcher = LightGlue(
            features=model_name.lower()
        ).to(device)

        matcher.eval()

        return matcher


    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# CHILD PROCESS
# ============================================================

def run_child(model_name):

    print()
    print("=" * 70)
    print(f"ISOLATED CHILD PROCESS: {model_name}")
    print("=" * 70)

    process = psutil.Process(os.getpid())


    # --------------------------------------------------------
    # BASELINE MEMORY
    # --------------------------------------------------------

    baseline_rss = get_rss_mb(process)


    # --------------------------------------------------------
    # EXTRACTOR LOAD
    # --------------------------------------------------------

    start = time.perf_counter()

    extractor, matcher_type = create_extractor(
        model_name
    )

    end = time.perf_counter()

    extractor_load_ms = (
        end - start
    ) * 1000


    rss_after_extractor = get_rss_mb(
        process
    )


    # --------------------------------------------------------
    # MATCHER LOAD
    # --------------------------------------------------------

    matcher_load_ms = 0.0

    if model_name in ["ALIKED", "SuperPoint"]:

        start = time.perf_counter()

        matcher = create_matcher(
            model_name
        )

        end = time.perf_counter()

        matcher_load_ms = (
            end - start
        ) * 1000

    else:

        matcher = None


    rss_after_matcher = get_rss_mb(
        process
    )


    # --------------------------------------------------------
    # LOAD FRAMES
    # --------------------------------------------------------

    frames = load_frames()


    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    print(
        f"Warming up {model_name}..."
    )

    extractor.extract(
        frames[0]
    )


    # --------------------------------------------------------
    # MEMORY AFTER WARMUP
    # --------------------------------------------------------

    rss_after_warmup = get_rss_mb(
        process
    )


    # --------------------------------------------------------
    # EXTRACTION BENCHMARK
    # --------------------------------------------------------

    extraction_latencies = []

    peak_rss_mb = get_peak_rss_mb(
        process
    )


    for frame in frames:

        start = time.perf_counter()

        extractor.extract(
            frame
        )

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        extraction_latencies.append(
            latency_ms
        )


        current_peak = get_peak_rss_mb(
            process
        )

        peak_rss_mb = max(
            peak_rss_mb,
            current_peak
        )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    extraction_mean_ms = (
        statistics.mean(
            extraction_latencies
        )
    )

    extraction_p95_ms = (
        calculate_p95(
            extraction_latencies
        )
    )


    total_load_ms = (
        extractor_load_ms
        + matcher_load_ms
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "model": model_name,

        "frames": len(frames),

        "baseline_rss_mb": round(
            baseline_rss,
            2
        ),

        "rss_after_extractor_mb": round(
            rss_after_extractor,
            2
        ),

        "rss_after_matcher_mb": round(
            rss_after_matcher,
            2
        ),

        "rss_after_warmup_mb": round(
            rss_after_warmup,
            2
        ),

        "peak_rss_mb": round(
            peak_rss_mb,
            2
        ),

        "extractor_load_ms": round(
            extractor_load_ms,
            2
        ),

        "matcher_load_ms": round(
            matcher_load_ms,
            2
        ),

        "total_load_ms": round(
            total_load_ms,
            2
        ),

        "extraction_mean_ms": round(
            extraction_mean_ms,
            2
        ),

        "extraction_p95_ms": round(
            extraction_p95_ms,
            2
        )
    }


    # --------------------------------------------------------
    # PRINT MACHINE-READABLE RESULT
    # --------------------------------------------------------

    print(
        "ISOLATED_RESULT:"
        + json.dumps(result)
    )


# ============================================================
# RUN CHILD PROCESS
# ============================================================

def run_model_in_child(model_name):

    print()
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        model_name
    ]


    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )


    # --------------------------------------------------------
    # PRINT CHILD OUTPUT
    # --------------------------------------------------------

    if completed.stdout:

        print(
            completed.stdout,
            end=""
        )


    if completed.stderr:

        print(
            "CHILD STDERR:"
        )

        print(
            completed.stderr
        )


    # --------------------------------------------------------
    # FIND RESULT LINE
    # --------------------------------------------------------

    result_line = None

    for line in completed.stdout.splitlines():

        line = line.strip()

        if line.startswith(
            "ISOLATED_RESULT:"
        ):

            result_line = line

            break


    # --------------------------------------------------------
    # NO RESULT
    # --------------------------------------------------------

    if result_line is None:

        print()
        print(
            "ERROR: No ISOLATED_RESULT found."
        )

        print()
        print(
            "Child return code:",
            completed.returncode
        )

        return None


    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    json_text = (
        result_line[
            len("ISOLATED_RESULT:")
        :]
        .strip()
    )


    try:

        result = json.loads(
            json_text
        )

    except json.JSONDecodeError as error:

        print()
        print(
            "ERROR: Could not parse JSON."
        )

        print()
        print(
            "Raw JSON:"
        )

        print(
            json_text
        )

        print()
        print(
            "JSON error:",
            error
        )

        return None


    return result


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results):

    if not results:

        return


    import csv


    fieldnames = [
        "model",
        "frames",

        "baseline_rss_mb",
        "rss_after_extractor_mb",
        "rss_after_matcher_mb",
        "rss_after_warmup_mb",
        "peak_rss_mb",

        "extractor_load_ms",
        "matcher_load_ms",
        "total_load_ms",

        "extraction_mean_ms",
        "extraction_p95_ms"
    ]


    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )


# ============================================================
# PRINT FINAL TABLE
# ============================================================

def print_final_table(results):

    print()
    print()
    print("=" * 100)
    print("FINAL ISOLATED RESOURCE RESULTS")
    print("=" * 100)

    if not results:

        print(
            "No results were produced."
        )

        return


    header = (
        f"{'Model':<12}"
        f"{'Peak RSS':>12}"
        f"{'Extractor Load':>17}"
        f"{'Matcher Load':>15}"
        f"{'Total Load':>13}"
        f"{'Extract Mean':>15}"
        f"{'Extract P95':>14}"
    )

    print(header)

    print("-" * 100)


    for result in results:

        print(
            f"{result['model']:<12}"
            f"{result['peak_rss_mb']:>12.2f}"
            f"{result['extractor_load_ms']:>17.2f}"
            f"{result['matcher_load_ms']:>15.2f}"
            f"{result['total_load_ms']:>13.2f}"
            f"{result['extraction_mean_ms']:>15.2f}"
            f"{result['extraction_p95_ms']:>14.2f}"
        )


    print("=" * 100)


# ============================================================
# PARENT PROCESS
# ============================================================

def run_parent():

    print()
    print("=" * 70)
    print("ISOLATED RESOURCE BENCHMARK")
    print("=" * 70)

    print()
    print(
        "Each model will run in a separate Python process."
    )


    models = [
        "ORB",
        "AKAZE",
        "XFeat",
        "ALIKED",
        "SuperPoint"
    ]


    results = []


    for model in models:

        result = run_model_in_child(
            model
        )

        if result is not None:

            results.append(
                result
            )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_results(
        results
    )


    # --------------------------------------------------------
    # FINAL TABLE
    # --------------------------------------------------------

    print_final_table(
        results
    )


    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    if results:

        print()
        print(
            "Results saved to:"
        )

        print(
            OUTPUT_CSV
        )

    else:

        print()
        print(
            "No results were produced."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if (
        len(sys.argv) >= 3
        and sys.argv[1] == "--child"
    ):

        model_name = sys.argv[2]

        run_child(
            model_name
        )

    else:

        run_parent()
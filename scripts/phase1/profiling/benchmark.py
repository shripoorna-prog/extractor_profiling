import os
import time
import statistics

import psutil


PROCESS = psutil.Process(os.getpid())


def get_memory_mb():
    """
    Get current process resident memory (RSS) in MB.
    """

    rss_bytes = PROCESS.memory_info().rss

    return rss_bytes / (1024 * 1024)


def measure_load_time(create_function):
    """
    Measure model/extractor creation time.

    Example:

        extractor, load_time = measure_load_time(
            lambda: ORBExtractor()
        )
    """

    start = time.perf_counter()

    obj = create_function()

    end = time.perf_counter()

    load_time_ms = (end - start) * 1000

    return obj, load_time_ms


def measure_extraction(
    extractor,
    image,
    warmup=1,
    iterations=5
):
    """
    Measure feature extraction latency.

    Returns:
        latencies_ms
        mean_ms
        p95_ms
        peak_rss_mb
    """

    # Warm-up
    for _ in range(warmup):
        extractor.extract(image)

    latencies = []

    peak_rss_mb = get_memory_mb()

    for _ in range(iterations):

        start = time.perf_counter()

        extractor.extract(image)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

        peak_rss_mb = max(
            peak_rss_mb,
            get_memory_mb()
        )

    mean_ms = statistics.mean(latencies)

    p95_ms = calculate_p95(latencies)

    return (
        latencies,
        mean_ms,
        p95_ms,
        peak_rss_mb
    )


def measure_matching(
    match_function,
    warmup=1,
    iterations=5
):
    """
    Measure matching latency.

    match_function should perform one complete
    matching operation.
    """

    for _ in range(warmup):
        match_function()

    latencies = []

    peak_rss_mb = get_memory_mb()

    for _ in range(iterations):

        start = time.perf_counter()

        match_function()

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

        peak_rss_mb = max(
            peak_rss_mb,
            get_memory_mb()
        )

    mean_ms = statistics.mean(latencies)

    p95_ms = calculate_p95(latencies)

    return (
        latencies,
        mean_ms,
        p95_ms,
        peak_rss_mb
    )


def calculate_p95(values):
    """
    Calculate the 95th percentile latency.
    """

    if not values:
        return 0.0

    values = sorted(values)

    index = int(
        0.95 * (len(values) - 1)
    )

    return values[index]
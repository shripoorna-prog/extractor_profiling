import re
import statistics

def parse_file(filename):
    good = []
    inliers = []
    ratios = []

    with open(filename, "r", encoding="utf-16") as f:
        for line in f:
            match = re.search(
                r"Pair\s+\d+:\s+(\d+)\s+good.*?(\d+)\s+RANSAC inliers.*?([\d.]+)%\s+inlier",
                line
            )

            if match:
                good.append(int(match.group(1)))
                inliers.append(int(match.group(2)))
                ratios.append(float(match.group(3)))

    return good, inliers, ratios


orb_good, orb_inliers, orb_ratios = parse_file(
    "orb_ransac_results.txt"
)

aliked_good, aliked_inliers, aliked_ratios = parse_file(
    "aliked_ransac_results.txt"
)

print("=" * 60)
print("ORB vs ALIKED - RANSAC COMPARISON")
print("=" * 60)

print()
print("AVERAGE GOOD MATCHES")
print(f"ORB    : {statistics.mean(orb_good):.2f}")
print(f"ALIKED : {statistics.mean(aliked_good):.2f}")

print()
print("AVERAGE RANSAC INLIERS")
print(f"ORB    : {statistics.mean(orb_inliers):.2f}")
print(f"ALIKED : {statistics.mean(aliked_inliers):.2f}")

print()
print("AVERAGE INLIER RATIO")
print(f"ORB    : {statistics.mean(orb_ratios):.2f}%")
print(f"ALIKED : {statistics.mean(aliked_ratios):.2f}%")

print()
print("MEDIAN INLIER RATIO")
print(f"ORB    : {statistics.median(orb_ratios):.2f}%")
print(f"ALIKED : {statistics.median(aliked_ratios):.2f}%")

print()
print("PAIRS WITH >= 50% INLIER RATIO")

orb_50 = sum(r >= 50 for r in orb_ratios)
aliked_50 = sum(r >= 50 for r in aliked_ratios)

print(f"ORB    : {orb_50}/{len(orb_ratios)}")
print(f"ALIKED : {aliked_50}/{len(aliked_ratios)}")

print()
print("=" * 60)
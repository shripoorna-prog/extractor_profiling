import re
import statistics

FILE = "aliked_ransac_results.txt"

good = []
inliers = []
ratios = []

with open(FILE, "r", encoding="utf-16") as f:
    for line in f:
        m = re.search(
            r"Pair\s+\d+:\s+(\d+)\s+good matches\s+\|\s+(\d+)\s+RANSAC inliers\s+\|\s+([\d.]+)%",
            line
        )

        if m:
            good.append(int(m.group(1)))
            inliers.append(int(m.group(2)))
            ratios.append(float(m.group(3)))

print("=" * 60)
print("ALIKED RANSAC BENCHMARK ANALYSIS")
print("=" * 60)

if not ratios:
    print("Still no data found.")
    print("\nFirst few lines of the saved file:")
    with open(FILE, "r") as f:
        for i, line in enumerate(f):
            print(repr(line))
            if i >= 5:
                break
    exit()

print(f"Pairs analysed       : {len(ratios)}")

print("\nGOOD MATCHES")
print("-" * 60)
print(f"Average              : {statistics.mean(good):.2f}")
print(f"Median               : {statistics.median(good):.2f}")
print(f"Minimum              : {min(good)}")
print(f"Maximum              : {max(good)}")

print("\nRANSAC INLIERS")
print("-" * 60)
print(f"Average              : {statistics.mean(inliers):.2f}")
print(f"Median               : {statistics.median(inliers):.2f}")
print(f"Minimum              : {min(inliers)}")
print(f"Maximum              : {max(inliers)}")

print("\nINLIER RATIO")
print("-" * 60)
print(f"Average              : {statistics.mean(ratios):.2f}%")
print(f"Median               : {statistics.median(ratios):.2f}%")
print(f"Minimum              : {min(ratios):.2f}%")
print(f"Maximum              : {max(ratios):.2f}%")

print("\nROBUSTNESS")
print("-" * 60)

for threshold in [50, 70, 80, 90]:
    count = sum(r >= threshold for r in ratios)
    percentage = count / len(ratios) * 100

    print(
        f"Pairs with inlier ratio >= {threshold}% : "
        f"{count}/{len(ratios)} ({percentage:.2f}%)"
    )

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
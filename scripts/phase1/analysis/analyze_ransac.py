import re
import statistics

# ---------------------------------------------------------
# PASTE YOUR RANSAC OUTPUT INTO THIS STRING
# ---------------------------------------------------------

text = r"""
Pair 001: 669 good matches | 571 RANSAC inliers | 85.35% inlier ratio
Pair 002: 654 good matches | 549 RANSAC inliers | 83.94% inlier ratio
Pair 003: 684 good matches | 654 RANSAC inliers | 95.61% inlier ratio
Pair 004: 685 good matches | 583 RANSAC inliers | 85.11% inlier ratio
Pair 005: 662 good matches | 433 RANSAC inliers | 65.41% inlier ratio
Pair 006: 637 good matches | 490 RANSAC inliers | 76.92% inlier ratio
Pair 007: 657 good matches | 552 RANSAC inliers | 84.02% inlier ratio
Pair 008: 611 good matches | 496 RANSAC inliers | 81.18% inlier ratio
Pair 009: 668 good matches | 561 RANSAC inliers | 83.98% inlier ratio
Pair 010: 653 good matches | 516 RANSAC inliers | 79.02% inlier ratio
Pair 011: 579 good matches | 496 RANSAC inliers | 85.66% inlier ratio
Pair 012: 535 good matches | 424 RANSAC inliers | 79.25% inlier ratio
Pair 013: 517 good matches | 468 RANSAC inliers | 90.52% inlier ratio
Pair 014: 481 good matches | 342 RANSAC inliers | 71.10% inlier ratio
Pair 015: 370 good matches | 261 RANSAC inliers | 70.54% inlier ratio
Pair 016: 298 good matches | 241 RANSAC inliers | 80.87% inlier ratio
Pair 017: 232 good matches | 184 RANSAC inliers | 79.31% inlier ratio
Pair 018: 174 good matches | 143 RANSAC inliers | 82.18% inlier ratio
Pair 019: 126 good matches | 106 RANSAC inliers | 84.13% inlier ratio
Pair 020: 89 good matches | 74 RANSAC inliers | 83.15% inlier ratio
Pair 021: 115 good matches | 103 RANSAC inliers | 89.57% inlier ratio
Pair 022: 90 good matches | 78 RANSAC inliers | 86.67% inlier ratio
Pair 023: 30 good matches | 26 RANSAC inliers | 86.67% inlier ratio
Pair 024: 4 good matches | 4 RANSAC inliers | 100.00% inlier ratio
Pair 025: 1 good matches | Not enough matches for RANSAC
Pair 026: 0 good matches | Not enough matches for RANSAC
Pair 027: 34 good matches | 23 RANSAC inliers | 67.65% inlier ratio
Pair 028: 161 good matches | 118 RANSAC inliers | 73.29% inlier ratio
Pair 029: 225 good matches | 188 RANSAC inliers | 83.56% inlier ratio
Pair 030: 257 good matches | 195 RANSAC inliers | 75.88% inlier ratio
Pair 031: 340 good matches | 278 RANSAC inliers | 81.76% inlier ratio
Pair 032: 309 good matches | 222 RANSAC inliers | 71.84% inlier ratio
Pair 033: 324 good matches | 252 RANSAC inliers | 77.78% inlier ratio
Pair 034: 383 good matches | 282 RANSAC inliers | 73.63% inlier ratio
Pair 035: 385 good matches | 273 RANSAC inliers | 70.91% inlier ratio
Pair 036: 339 good matches | 287 RANSAC inliers | 84.66% inlier ratio
Pair 037: 357 good matches | 256 RANSAC inliers | 71.71% inlier ratio
Pair 038: 340 good matches | 243 RANSAC inliers | 71.47% inlier ratio
Pair 039: 337 good matches | 191 RANSAC inliers | 56.68% inlier ratio
Pair 040: 274 good matches | 185 RANSAC inliers | 67.52% inlier ratio
Pair 041: 165 good matches | 112 RANSAC inliers | 67.88% inlier ratio
Pair 042: 164 good matches | 97 RANSAC inliers | 59.15% inlier ratio
Pair 043: 287 good matches | 206 RANSAC inliers | 71.78% inlier ratio
Pair 044: 383 good matches | 267 RANSAC inliers | 69.71% inlier ratio
Pair 045: 407 good matches | 262 RANSAC inliers | 64.37% inlier ratio
Pair 046: 485 good matches | 303 RANSAC inliers | 62.47% inlier ratio
Pair 047: 537 good matches | 309 RANSAC inliers | 57.54% inlier ratio
Pair 048: 455 good matches | 261 RANSAC inliers | 57.36% inlier ratio
Pair 049: 506 good matches | 343 RANSAC inliers | 67.79% inlier ratio
Pair 050: 474 good matches | 295 RANSAC inliers | 62.24% inlier ratio
Pair 051: 498 good matches | 311 RANSAC inliers | 62.45% inlier ratio
Pair 052: 477 good matches | 364 RANSAC inliers | 76.31% inlier ratio
Pair 053: 386 good matches | 282 RANSAC inliers | 73.06% inlier ratio
Pair 054: 210 good matches | 172 RANSAC inliers | 81.90% inlier ratio
Pair 055: 163 good matches | 135 RANSAC inliers | 82.82% inlier ratio
Pair 056: 106 good matches | 95 RANSAC inliers | 89.62% inlier ratio
Pair 057: 16 good matches | 7 RANSAC inliers | 43.75% inlier ratio
Pair 058: 10 good matches | 9 RANSAC inliers | 90.00% inlier ratio
Pair 059: 7 good matches | 6 RANSAC inliers | 85.71% inlier ratio
Pair 060: 35 good matches | 30 RANSAC inliers | 85.71% inlier ratio
Pair 061: 15 good matches | 13 RANSAC inliers | 86.67% inlier ratio
Pair 062: 31 good matches | 21 RANSAC inliers | 67.74% inlier ratio
Pair 063: 185 good matches | 144 RANSAC inliers | 77.84% inlier ratio
Pair 064: 352 good matches | 278 RANSAC inliers | 78.98% inlier ratio
Pair 065: 457 good matches | 361 RANSAC inliers | 78.99% inlier ratio
Pair 066: 457 good matches | 354 RANSAC inliers | 77.46% inlier ratio
Pair 067: 474 good matches | 398 RANSAC inliers | 83.97% inlier ratio
Pair 068: 479 good matches | 405 RANSAC inliers | 84.55% inlier ratio
Pair 069: 438 good matches | 325 RANSAC inliers | 74.20% inlier ratio
Pair 070: 414 good matches | 272 RANSAC inliers | 65.70% inlier ratio
Pair 071: 383 good matches | 237 RANSAC inliers | 61.88% inlier ratio
Pair 072: 301 good matches | 184 RANSAC inliers | 61.13% inlier ratio
Pair 073: 278 good matches | 159 RANSAC inliers | 57.19% inlier ratio
Pair 074: 201 good matches | 116 RANSAC inliers | 57.71% inlier ratio
Pair 075: 192 good matches | 111 RANSAC inliers | 57.81% inlier ratio
Pair 076: 205 good matches | 157 RANSAC inliers | 76.59% inlier ratio
Pair 077: 245 good matches | 152 RANSAC inliers | 62.04% inlier ratio
Pair 078: 202 good matches | 120 RANSAC inliers | 59.41% inlier ratio
Pair 079: 194 good matches | 113 RANSAC inliers | 58.25% inlier ratio
Pair 080: 213 good matches | 127 RANSAC inliers | 59.62% inlier ratio
Pair 081: 265 good matches | 152 RANSAC inliers | 57.36% inlier ratio
Pair 082: 361 good matches | 251 RANSAC inliers | 69.53% inlier ratio
Pair 083: 399 good matches | 316 RANSAC inliers | 79.20% inlier ratio
Pair 084: 368 good matches | 301 RANSAC inliers | 81.79% inlier ratio
Pair 085: 294 good matches | 236 RANSAC inliers | 80.27% inlier ratio
Pair 086: 399 good matches | 343 RANSAC inliers | 85.96% inlier ratio
Pair 087: 403 good matches | 348 RANSAC inliers | 86.35% inlier ratio
Pair 088: 407 good matches | 384 RANSAC inliers | 94.35% inlier ratio
Pair 089: 394 good matches | 360 RANSAC inliers | 91.37% inlier ratio
Pair 090: 490 good matches | 417 RANSAC inliers | 85.10% inlier ratio
Pair 091: 513 good matches | 475 RANSAC inliers | 92.59% inlier ratio
Pair 092: 485 good matches | 378 RANSAC inliers | 77.94% inlier ratio
Pair 093: 476 good matches | 413 RANSAC inliers | 86.76% inlier ratio
Pair 094: 509 good matches | 391 RANSAC inliers | 76.82% inlier ratio
Pair 095: 591 good matches | 567 RANSAC inliers | 95.94% inlier ratio
Pair 096: 601 good matches | 482 RANSAC inliers | 80.20% inlier ratio
Pair 097: 575 good matches | 536 RANSAC inliers | 93.22% inlier ratio
Pair 098: 546 good matches | 429 RANSAC inliers | 78.57% inlier ratio
Pair 099: 585 good matches | 475 RANSAC inliers | 81.20% inlier ratio
"""

# ---------------------------------------------------------
# EXTRACT PAIR RESULTS
# ---------------------------------------------------------

pattern = re.compile(
    r"Pair\s+(\d+):\s+(\d+)\s+good matches\s+\|\s+"
    r"(\d+)\s+RANSAC inliers\s+\|\s+"
    r"([\d.]+)%\s+inlier ratio"
)

results = pattern.findall(text)

if not results:
    print("No pair results found.")
    print("Make sure you pasted the RANSAC output.")
    exit()

pairs = []

for pair, good, inliers, ratio in results:
    pairs.append({
        "pair": int(pair),
        "good": int(good),
        "inliers": int(inliers),
        "ratio": float(ratio)
    })

# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

good_matches = [x["good"] for x in pairs]
inliers = [x["inliers"] for x in pairs]
ratios = [x["ratio"] for x in pairs]

valid_ratios = ratios

print()
print("=" * 60)
print("RANSAC BENCHMARK ANALYSIS")
print("=" * 60)

print(f"Pairs analysed       : {len(pairs)}")

print()
print("GOOD MATCHES")
print("-" * 60)
print(f"Average              : {statistics.mean(good_matches):.2f}")
print(f"Median               : {statistics.median(good_matches):.2f}")
print(f"Minimum              : {min(good_matches)}")
print(f"Maximum              : {max(good_matches)}")

print()
print("RANSAC INLIERS")
print("-" * 60)
print(f"Average              : {statistics.mean(inliers):.2f}")
print(f"Median               : {statistics.median(inliers):.2f}")
print(f"Minimum              : {min(inliers)}")
print(f"Maximum              : {max(inliers)}")

print()
print("INLIER RATIO")
print("-" * 60)
print(f"Average              : {statistics.mean(valid_ratios):.2f}%")
print(f"Median               : {statistics.median(valid_ratios):.2f}%")
print(f"Minimum              : {min(valid_ratios):.2f}%")
print(f"Maximum              : {max(valid_ratios):.2f}%")

# ---------------------------------------------------------
# ROBUSTNESS
# ---------------------------------------------------------

thresholds = [50, 70, 80, 90]

print()
print("ROBUSTNESS")
print("-" * 60)

for threshold in thresholds:
    count = sum(r >= threshold for r in ratios)
    percentage = count / len(ratios) * 100

    print(
        f"Pairs with inlier ratio >= {threshold}% : "
        f"{count}/{len(ratios)} ({percentage:.2f}%)"
    )

# ---------------------------------------------------------
# LOW MATCH PAIRS
# ---------------------------------------------------------

print()
print("LOWEST INLIER-RATIO PAIRS")
print("-" * 60)

lowest = sorted(pairs, key=lambda x: x["ratio"])[:10]

for x in lowest:
    print(
        f"Pair {x['pair']:03d}: "
        f"{x['good']} good | "
        f"{x['inliers']} inliers | "
        f"{x['ratio']:.2f}%"
    )

# ---------------------------------------------------------
# BEST PAIRS
# ---------------------------------------------------------

print()
print("HIGHEST INLIER-RATIO PAIRS")
print("-" * 60)

highest = sorted(pairs, key=lambda x: x["ratio"], reverse=True)[:10]

for x in highest:
    print(
        f"Pair {x['pair']:03d}: "
        f"{x['good']} good | "
        f"{x['inliers']} inliers | "
        f"{x['ratio']:.2f}%"
    )

print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
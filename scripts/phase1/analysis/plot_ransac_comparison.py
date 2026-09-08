import matplotlib.pyplot as plt
import os

# Results from our ORB vs ALIKED RANSAC benchmark
methods = ["ORB", "ALIKED"]

good_matches = [354.98, 150.38]
ransac_inliers = [273.35, 57.77]
inlier_ratio = [76.21, 35.18]

# Create results folder if needed
os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# 1. Average Good Matches
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(methods, good_matches)
plt.ylabel("Average Good Matches")
plt.title("ORB vs ALIKED - Average Good Matches")
plt.tight_layout()
plt.savefig("results/average_good_matches.png", dpi=300)
plt.show()

# --------------------------------------------------
# 2. Average RANSAC Inliers
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(methods, ransac_inliers)
plt.ylabel("Average RANSAC Inliers")
plt.title("ORB vs ALIKED - Average RANSAC Inliers")
plt.tight_layout()
plt.savefig("results/average_ransac_inliers.png", dpi=300)
plt.show()

# --------------------------------------------------
# 3. Average Inlier Ratio
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(methods, inlier_ratio)
plt.ylabel("Average Inlier Ratio (%)")
plt.title("ORB vs ALIKED - RANSAC Inlier Ratio")
plt.tight_layout()
plt.savefig("results/average_inlier_ratio.png", dpi=300)
plt.show()

print("=" * 60)
print("GRAPHS GENERATED")
print("=" * 60)
print("results/average_good_matches.png")
print("results/average_ransac_inliers.png")
print("results/average_inlier_ratio.png")
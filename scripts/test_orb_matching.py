
#Imports
import cv2
import numpy as np

from adapters.orb import ORBExtractor

#Create Image 1
image1 = np.zeros((480, 640), dtype=np.uint8)

cv2.rectangle(image1, (100, 100), (300, 300), 255, 3)
cv2.circle(image1, (450, 200), 80, 255, 3)
cv2.line(image1, (100, 400), (500, 400), 255, 3)

#Create Image 2

image2 = np.zeros((480, 640), dtype=np.uint8)

cv2.rectangle(image2, (120, 100), (320, 300), 255, 3)
cv2.circle(image2, (470, 200), 80, 255, 3)
cv2.line(image2, (120, 400), (520, 400), 255, 3)

#create our ORB extractor
extractor = ORBExtractor(nfeatures=1000)


#Extract features from both images
keypoints1, descriptors1 = extractor.extract(image1)
keypoints2, descriptors2 = extractor.extract(image2)

print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))

#match 
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

#Get the best 2 matches  
matches = matcher.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)

print("Raw match groups:", len(matches))



#Lowe's Ratio Test
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches:", len(good_matches))
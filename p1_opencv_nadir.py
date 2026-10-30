#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import *
from utils import *

P1_OUT = OUT / "p1"
P1_OUT.mkdir(exist_ok=True)


photos = sorted(f for f in NADIR_DIR.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif"))
print(len(photos), "nadir photos")         # should print 86

SCALE = 0.25                               # work on quarter-size photos: 16x fewer pixels, much faster
img = cv2.imread(str(photos[0]))
img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
gsd = PHOTO_GSD_M / SCALE                  # shrinking the photo makes each pixel cover more ground
show(img, photos[0].name)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # Hue, Saturation, Value
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)    # Lightness, a (green-red), b (blue-yellow)
fig, ax = plt.subplots(2, 3, figsize=(15, 8)) # a grid of 2 rows x 3 columns of pictures
for i, name in enumerate(["H (hue)", "S (saturation)", "V (value)"]):
    ax[0, i].imshow(hsv[:, :, i], cmap="gray")
    ax[0, i].set_title(name)
for i, name in enumerate(["L (lightness)", "a (green-red)", "b (blue-yellow)"]):
    ax[1, i].imshow(lab[:, :, i], cmap="gray")
    ax[1, i].set_title(name)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.show()


# %% Step 2 - three vegetation indices, each with Otsu's automatic threshold --
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for i, name in enumerate(["ExG", "ExGR", "VARI"]):
    index = vegetation_index(img, name)
    index8 = cv2.normalize(index, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(index8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ax[0, i].imshow(index8, cmap="gray")
    ax[0, i].set_title(name + " index")
    ax[1, i].imshow(mask, cmap="gray")
    ax[1, i].set_title(name + " + Otsu mask")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.show()


# %% Step 3 - clean the mask with morphology -----------------------------------
mask = vegetation_mask(img, VEG_INDEX, kernel_size=5)   # threshold + opening + closing (see utils.py)
show(mask, f"{VEG_INDEX} mask after opening and closing", cmap="gray")

# %% Step 4 - count plants with contours, using real-world sizes ---------------
min_area = min_area_px(MIN_PLANT_DIAM_M, gsd)
print(f"ignoring blobs smaller than {min_area:.0f} px  (= a {MIN_PLANT_DIAM_M * 100:.0f} cm plant)")

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
plants = [c for c in contours if cv2.contourArea(c) >= min_area]

out = img.copy()
for c in plants:
    M = cv2.moments(c)                                    # "centre of mass" maths
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cv2.drawContours(out, [c], -1, (0, 0, 255), 2)        # red outline (BGR)
    cv2.circle(out, (cx, cy), 4, (255, 0, 0), -1)         # blue dot at the centre
print("plants counted:", len(plants))
cv2.imwrite(str(P1_OUT / "count_photo0.jpg"), out)
show(out, f"{len(plants)} plants")





# %% Step 5 - find where it fails: the blob-size histogram ---------------------
areas = [cv2.contourArea(c) for c in plants]
plt.hist(areas, bins=40)
plt.xlabel("blob area (px)")
plt.ylabel("number of blobs")
plt.title("Blobs 2-3x the usual size are probably merged plants")
plt.show()

# Run the same counter on 5 photos and look at each result.
for p in photos[:5]:
    im = cv2.resize(cv2.imread(str(p)), None, fx=SCALE, fy=SCALE)
    m = vegetation_mask(im, VEG_INDEX)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = sum(cv2.contourArea(c) >= min_area for c in cs)   # True counts as 1
    print(f"{p.name}: {n} plants")
# Write down the failures you see: shadows, weeds, touching plants, plants cut at the edge.


# %% Bonus - stitch 3 photos yourself -------------------------------------------
imgs = [cv2.resize(cv2.imread(str(p)), None, fx=0.3, fy=0.3) for p in photos[:3]]
stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)    # SCANS mode suits flat, top-down photos
status, pano = stitcher.stitch(imgs)
print("status:", status, "(0 = success; 1 = needs more overlap/photos)")
if status == 0:
    cv2.imwrite(str(P1_OUT / "stitched_3_photos.jpg"), pano)
    show(pano, "3 photos stitched")

























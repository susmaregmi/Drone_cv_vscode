#%%


import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

from config import *          # all paths and settings
from utils import *   
import rasterio        # find_files, show, ...

card = [] 

for folder in sorted(p for p in BASE.iterdir() if p.is_dir()):
    files = [f for f in folder.rglob("*") if f.is_file()]       # rglob = also look in sub-folders
    example = files[0].name if files else "-"
    print(f"{folder.name:22s} {len(files):6d} files   e.g. {example}")

NADIR_DIR = BASE / "images_nadir_RGB"            # the folder section 1 found the 86 photos in
photos = sorted(f for f in NADIR_DIR.rglob("*")  # look in this folder and any subfolders
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif"))  # keep only image files
print(len(photos), "nadir photos")               # should print 86
img = cv2.imread(str(photos[0]))                 # OpenCV wants a text path, so str(...)
print("image shape (height, width, channels):", img.shape)
show(img, photos[0].name)


def dms_to_degrees(dms, ref):
    """GPS is stored as (degrees, minutes, seconds). Convert to one decimal number."""
    d, m, s = [float(v) for v in dms]
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg 



pil_img = Image.open(photos[0])
exif = {ExifTags.TAGS.get(k, k): v for k, v in pil_img.getexif().items()}   # code -> readable name
exif.update({ExifTags.TAGS.get(k, k): v for k, v in pil_img.getexif().get_ifd(0x8769).items()})  # camera details
gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in pil_img.getexif().get_ifd(0x8825).items()}       # GPS block

print("camera:", exif.get("Make"), exif.get("Model"))
print("taken:", exif.get("DateTimeOriginal"))
print("focal length (mm):", exif.get("FocalLength"))
if "GPSLatitude" in gps:
    lat = dms_to_degrees(gps["GPSLatitude"], gps.get("GPSLatitudeRef"))
    lon = dms_to_degrees(gps["GPSLongitude"], gps.get("GPSLongitudeRef"))
    print(f"GPS: {lat:.6f}, {lon:.6f}   altitude above sea level: {gps.get('GPSAltitude')} m")



raw = photos[0].read_bytes()
rel = re.search(rb'RelativeAltitude="([+-]?[0-9.]+)"', raw)
flight_height = float(rel.group(1)) if rel else None
print("flight height above take-off (m):", flight_height)

card += [f"nadir photos: {len(photos)}", f"photo size: {img.shape[1]} x {img.shape[0]} px",
         f"camera: {exif.get('Make')} {exif.get('Model')}", f"focal length mm: {exif.get('FocalLength')}",
         f"flight height m: {flight_height}"]

# Photo GSD = sensor width x flight height / (focal length x image width)
if SENSOR_WIDTH_MM and flight_height and exif.get("FocalLength"):
    photo_gsd = SENSOR_WIDTH_MM * flight_height / (float(exif["FocalLength"]) * img.shape[1])
    print(f"photo GSD = {photo_gsd * 100:.2f} cm per pixel  -> put {photo_gsd:.4f} in PHOTO_GSD_M")
    card.append(f"photo GSD m: {photo_gsd:.4f}")
else:
    print("Set SENSOR_WIDTH_MM in config.py to compute the photo GSD.")




# %% 3. The orthomosaic: size, coordinate system, GSD (never load it whole!) ---
import rasterio
from rasterio.windows import Window

ortho_path = pick_file(ORTHO_FILE, ORTHO_DIR, TIF_PATTERNS, "orthomosaic")
with rasterio.open(ortho_path) as src:          # "with" closes the file automatically
    print("file:", ortho_path.name)
    print("size:", src.width, "x", src.height, "pixels, bands:", src.count, "type:", src.dtypes[0])
    print("coordinate system (CRS):", src.crs)
    print("pixel size (GSD):", src.res, "map units (usually metres)")
    w = min(2000, src.width)
    h = min(2000, src.height)
    window = Window(src.width // 2 - w // 2, src.height // 2 - h // 2, w, h)   # a piece from the MIDDLE
    crop = src.read([1, 2, 3], window=window)    # bands 1-3 = R, G, B (band 4 is often transparency)
    card += [f"orthomosaic: {src.width} x {src.height} px, {src.count} bands, {src.dtypes[0]}",
             f"CRS: {src.crs}", f"GSD_M: {src.res[0]}"]

crop = to_uint8(crop.transpose(1, 2, 0))        # (bands, H, W) -> (H, W, bands)
show(crop, "orthomosaic - centre crop", bgr=False)   # rasterio gives RGB, not BGR

# %% 4. The DEM(s): DSM or DTM? ------------------------------------------------
dem_files = find_files(DEM_DIR, TIF_PATTERNS)
print("DEM files:", [f.name for f in dem_files])
for dem_path in dem_files:
    with rasterio.open(dem_path) as src:
        step = max(1, max(src.width, src.height) // 1500)        # read a smaller version for speed
        dem = src.read(1, masked=True,
                       out_shape=(src.height // step, src.width // step))
        print(f"{dem_path.name}: pixel {src.res}, min {dem.min():.2f}, max {dem.max():.2f}")
        card.append(f"DEM {dem_path.name}: pixel {src.res[0]}, {dem.min():.2f}-{dem.max():.2f} m")
    plt.figure(figsize=(9, 7))
    plt.imshow(dem, cmap="terrain")
    plt.colorbar(label="elevation (m)")
    plt.title(dem_path.name + "  - plants visible = DSM, only the land slope = DTM")
    plt.show()



import laspy

laz_path = pick_file(LAZ_FILE, LAZ_DIR, ("*.laz", "*.las"), "point cloud")
with laspy.open(laz_path) as f:
    hdr = f.header
    print("points:", hdr.point_count)
    print("columns:", list(hdr.point_format.dimension_names))
    print("min x, y, z:", hdr.mins)
    print("max x, y, z:", hdr.maxs)
    sample = next(f.chunk_iterator(500_000))     # only the first 500,000 points
card.append(f"point cloud: {hdr.point_count} points")

x, y, z = np.asarray(sample.x), np.asarray(sample.y), np.asarray(sample.z)
keep = slice(None, None, 10)                     # every 10th point, so plotting is quick
plt.figure(figsize=(8, 8))
plt.scatter(x[keep], y[keep], c=z[keep], s=0.5, cmap="viridis")
plt.colorbar(label="z (m)")
plt.title("point cloud sample seen from above, coloured by height")
plt.axis("equal")
plt.show()





import pandas as pd

gcp_files = find_files(GCP_DIR, ("*",))
print("GCP files:", [f.name for f in gcp_files])
for g in gcp_files:
    if g.suffix.lower() in (".csv", ".txt"):
        table = pd.read_csv(g, sep=None, engine="python")   # sep=None: guess comma/space/tab
        print(table.head())
        print("columns:", list(table.columns), "-> put the right names in GCP_COLS in config.py")
        break



# %% 7. Save the data card ----------------------------------------------------
card_path = OUT / "data_card.txt"
card_path.write_text("\n".join(str(line) for line in card))
print("\n".join(str(line) for line in card))
print("\nSaved to", card_path, "- now copy GSD_M etc. into config.py")
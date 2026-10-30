"""
PROJECT 3 - Plant height from the DEM and point cloud.

When & why: colour images can't tell a tall plant from a short one with the same crown;
elevation can. CHM (canopy height model) = DSM (surface incl. plants) - DTM (bare ground).
Outputs: outputs/chm.tif, outputs/height_plants.csv / .gpkg (one row per plant top).
"""
# %% Setup ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from config import *
from utils import *

# Guess which DEM file is the DSM and which the DTM from their names.
dem_files = find_files(DEM_DIR, TIF_PATTERNS)
dsm_path = DSM_FILE or next((f for f in dem_files if "dsm" in f.name.lower()), None)
dtm_path = DTM_FILE or next((f for f in dem_files if "dtm" in f.name.lower()), None)
if dsm_path is None and len(dem_files) == 1:
    dsm_path = dem_files[0]
    print("Only one DEM file; treating it as a DSM. Check in p0 that plants are visible in it!")
print("DSM:", dsm_path, "\nDTM:", dtm_path)

# %% Step 1 - build the canopy height model (CHM) --------------------------------
with rasterio.open(dsm_path) as s:
    dsm = s.read(1, masked=True).astype(np.float32)     # masked=True hides no-data pixels
    profile = s.profile                                  # size, CRS, transform... reused to save the CHM
    transform, crs = s.transform, s.crs
pixel_m = transform.a                                    # width of one DEM pixel in metres
print("DEM pixel size:", pixel_m, "m, shape:", dsm.shape)

if dtm_path:                                             # CASE A: we have a real ground model
    with rasterio.open(dtm_path) as t:
        if t.shape != dsm.shape or t.transform != transform:
            raise ValueError("DSM and DTM grids differ - resample one to the other first (QGIS: Align Rasters).")
        ground = t.read(1, masked=True).astype(np.float32)
else:                                                    # CASE B: estimate the ground from the DSM
    win_px = int(3 / pixel_m)                            # window wider than the biggest plant (3 m)
    filled = dsm.filled(float(dsm.mean()))               # replace no-data with the average height
    ground = ndimage.grey_opening(filled, size=(win_px, win_px))   # shave off everything that sticks up
    ground = ndimage.uniform_filter(ground, size=win_px)            # smooth the steps
    print("estimated ground with a", win_px, "px opening (approximate!)")

chm = np.clip(np.ma.filled(dsm - ground, 0), 0, None)   # negative heights are noise -> 0

profile.update(dtype="float32", count=1, nodata=0)
with rasterio.open(OUT / "chm.tif", "w", **profile) as dst:
    dst.write(chm.astype(np.float32), 1)
print("saved chm.tif")

plt.figure(figsize=(10, 8))
plt.imshow(chm, cmap="viridis", vmax=np.percentile(chm, 99.5))
plt.colorbar(label="height above ground (m)")
plt.title("Canopy height model")
plt.show()

# %% Step 2 - find plant tops (local maxima) ---------------------------------------
spacing_px = max(1, int(PLANT_SPACING_M / pixel_m))     # spacing in DEM pixels (often coarser than the ortho)
chm_s = ndimage.gaussian_filter(chm, sigma=1)           # light smoothing removes leaf bumps
tops = peak_local_max(chm_s, min_distance=max(1, spacing_px // 2),
                      threshold_abs=MIN_PLANT_HEIGHT_M, exclude_border=False)
print(len(tops), f"plant tops higher than {MIN_PLANT_HEIGHT_M} m")

# %% Step 3 - crown of each plant: watershed on the upside-down height map ------------
markers = np.zeros(chm_s.shape, np.int32)
for i, (y, x) in enumerate(tops, start=1):
    markers[y, x] = i
crowns = watershed(-chm_s, markers, mask=chm_s > MIN_PLANT_HEIGHT_M)   # tops become "valleys" that flood
px_area = pixel_m ** 2

counts = np.bincount(crowns.ravel(), minlength=len(tops) + 1)   # pixels per crown, fast
xs, ys = rasterio.transform.xy(transform, tops[:, 0], tops[:, 1])
table = pd.DataFrame({"top_id": range(1, len(tops) + 1),
                      "x_map": xs, "y_map": ys,
                      "height_m": chm_s[tops[:, 0], tops[:, 1]],
                      "crown_area_m2": counts[1:] * px_area})
table["crown_diam_m"] = 2 * np.sqrt(table.crown_area_m2 / np.pi)   # circle with the same area
table.to_csv(OUT / "height_plants.csv", index=False)
print(table.head())

import geopandas as gpd
gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table.x_map, table.y_map), crs=crs) \
   .to_file(OUT / "height_plants.gpkg", driver="GPKG")

plt.figure(figsize=(10, 8))
plt.imshow(crowns % 20, cmap="tab20")                    # % 20 cycles 20 colours so neighbours differ
plt.plot(tops[:, 1], tops[:, 0], "k.", markersize=3)
plt.title(f"{len(tops)} crowns from the height model")
plt.show()

# %% Step 4 - check 5 plants with the point cloud (more precise heights) --------------
import laspy

laz_path = pick_file(LAZ_FILE, LAZ_DIR, ("*.laz", "*.las"), "point cloud")
check = table.head(5)
radius = 0.3                                              # metres around each plant top
xmin, xmax = check.x_map.min() - radius, check.x_map.max() + radius
ymin, ymax = check.y_map.min() - radius, check.y_map.max() + radius
near_x, near_y, near_z = [], [], []
with laspy.open(laz_path) as f:
    for chunk in f.chunk_iterator(2_000_000):            # read 2 million points at a time
        x, y, z = np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)
        inside = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
        near_x.append(x[inside]); near_y.append(y[inside]); near_z.append(z[inside])
x, y, z = np.concatenate(near_x), np.concatenate(near_y), np.concatenate(near_z)

with rasterio.open(dsm_path) as s:
    for _, p in check.iterrows():
        close = (x - p.x_map) ** 2 + (y - p.y_map) ** 2 < radius ** 2
        if close.sum() < 5:
            print(f"top {p.top_id}: too few points nearby (check the point cloud's CRS matches)")
            continue
        r, c = s.index(p.x_map, p.y_map)                 # map -> DEM pixel
        ground_z = float(np.ma.filled(ground, np.nan)[r, c]) if np.ndim(ground) else np.nan
        z95 = np.percentile(z[close], 95)                # 95th percentile: ignores stray high points
        print(f"top {p.top_id}: CHM {p.height_m:.2f} m | point cloud {z95 - ground_z:.2f} m")

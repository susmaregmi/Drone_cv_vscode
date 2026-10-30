"""
PROJECT 5c - Canopy size + the final inventory table.

When & why: a box overestimates a round crown. For crown AREA we count the green (plant) pixels
inside each YOLO box, then add height from the canopy height model (p3), the row (p5a)
and the species (p5b, if you ran it). One row per plant = the nursery inventory.

Output: outputs/inventory.csv and .gpkg
"""
# %% Setup ------------------------------------------------------------------
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from config import *
from utils import *

ortho_path = pick_file(ORTHO_FILE, ORTHO_DIR, TIF_PATTERNS, "orthomosaic")
rows_file = OUT / "plants_with_rows.csv"
plants = pd.read_csv(rows_file if rows_file.exists() else OUT / "yolo_plants.csv")
if "x1" not in plants.columns:
    raise ValueError("Run p4d_count_field.py (and then p5a_gaps.py) first: the inventory needs YOLO boxes.")
print(len(plants), "plants")

# %% Crown area: plant pixels inside each box ----------------------------------------
areas = []
with rasterio.open(ortho_path) as src:
    pixel_area = abs(src.transform.a * src.transform.e)   # m^2 per orthomosaic pixel
    for _, p in plants.iterrows():
        win = Window(int(p.x1), int(p.y1), int(p.x2 - p.x1) + 1, int(p.y2 - p.y1) + 1)
        rgb = to_uint8(src.read([1, 2, 3], window=win, boundless=True, fill_value=0).transpose(1, 2, 0))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])         # RGB -> BGR for our OpenCV helper
        mask = vegetation_mask(bgr, VEG_INDEX, kernel_size=3)
        areas.append((mask > 0).sum() * pixel_area)
plants["crown_area_m2"] = areas
plants["crown_diam_m"] = 2 * np.sqrt(plants.crown_area_m2 / np.pi)   # circle with the same area
# Note: Otsu inside a tiny box can misjudge a box that is almost all plant. Check a few in QGIS.

# %% Height: highest CHM value inside each plant's box ------------------------------
chm_file = OUT / "chm.tif"
if chm_file.exists():
    heights = []
    with rasterio.open(ortho_path) as src:
        o_t = src.transform
    with rasterio.open(chm_file) as chm:
        for _, p in plants.iterrows():
            # box corners: ortho pixels -> map coordinates -> CHM pixels (the two grids differ)
            mx1, my1 = o_t * (p.x1, p.y1)
            mx2, my2 = o_t * (p.x2, p.y2)
            r1, c1 = chm.index(min(mx1, mx2), max(my1, my2))
            r2, c2 = chm.index(max(mx1, mx2), min(my1, my2))
            win = Window(c1, r1, max(1, c2 - c1 + 1), max(1, r2 - r1 + 1))
            values = chm.read(1, window=win, boundless=True, fill_value=0)
            heights.append(float(values.max()))
    plants["height_m"] = heights
    plants["volume_proxy_m3"] = plants.crown_area_m2 * plants.height_m   # simple canopy volume estimate
else:
    print("no chm.tif yet - run p3_height.py to add heights")

# %% Species (optional) -----------------------------------------------------------
species_file = OUT / "species_predictions.csv"
if species_file.exists():
    plants = plants.merge(pd.read_csv(species_file), on="plant_id", how="left")

# %% Save the inventory -------------------------------------------------------------
keep = [c for c in ["plant_id", "x_map", "y_map", "row", "species", "species_conf",
                    "height_m", "crown_area_m2", "crown_diam_m", "volume_proxy_m3", "confidence"]
        if c in plants.columns]
inventory = plants[keep]
inventory.to_csv(OUT / "inventory.csv", index=False)
print(inventory.describe().round(2))

import geopandas as gpd
with rasterio.open(ortho_path) as src:
    crs = src.crs
gpd.GeoDataFrame(inventory, geometry=gpd.points_from_xy(inventory.x_map, inventory.y_map), crs=crs) \
   .to_file(OUT / "inventory.gpkg", driver="GPKG")
print("saved inventory.csv and inventory.gpkg - this table IS the nursery inventory")

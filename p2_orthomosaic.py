

# %% Setup ------------------------------------------------------------------
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

from config import *
from utils import *

ortho_path = pick_file(ORTHO_FILE, ORTHO_DIR, TIF_PATTERNS, "orthomosaic")
spacing_px = PLANT_SPACING_M / GSD_M                    # plant spacing measured in pixels
min_radius_px = MIN_PLANT_DIAM_M / GSD_M / 2
print(f"plant spacing = {spacing_px:.0f} px, smallest plant radius = {min_radius_px:.1f} px")

# %% Step 1 - cut the map into overlapping 640x640 tiles ------------------------
saved = 0
with rasterio.open(ortho_path) as src:
    rows = tile_starts(src.height, TILE, STEP)           # top edge of every tile row
    cols = tile_starts(src.width, TILE, STEP)            # left edge of every tile column
    for row in rows:
        for col in cols:
            tile = src.read([1, 2, 3], window=Window(col, row, TILE, TILE), boundless=True, fill_value=0)
            tile = to_uint8(tile.transpose(1, 2, 0))     # (bands, H, W) -> (H, W, bands), RGB
            if (tile.sum(axis=2) == 0).mean() > 0.5:     # more than half empty border -> skip
                continue
            name = TILES_DIR / f"tile_{row}_{col}.jpg"   # the NAME remembers where the tile came from
            cv2.imwrite(str(name), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))   # imwrite expects BGR
            saved += 1
print(saved, "tiles saved in", TILES_DIR)



# %% Step 2 - try watershed on ONE tile and look at it -----------------------
# 
tiles = find_files(TILES_DIR, ("*.jpg",))
print(len(tiles), "tiles found")

best_tile, best_n = None, -1

for t in tiles[::max(1, len(tiles) // 50)]:      # check about 50 tiles spread over the map
    im = cv2.imread(str(t))
    if im.std() < 10:                            # blank edge tile? skip
        continue
    m = vegetation_mask(im, VEG_INDEX)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = sum(cv2.contourArea(c) >= min_area_px(MIN_PLANT_DIAM_M, GSD_M) for c in cs)  # plant-sized blobs
    if n > best_n:                               # keep the tile with the most plant-sized blobs
        best_tile, best_n = t, n


with rasterio.open(ortho_path) as src:
    mid_row, mid_col = src.height // 2, src.width // 2      # centre pixel of the whole map

def tile_pos(t):                                            # "tile_14336_15872" -> (14336, 15872)
    _, r, c = t.stem.split("_")
    return int(r), int(c)

best_tile = min(tiles, key=lambda t: abs(tile_pos(t)[0] - mid_row) + abs(tile_pos(t)[1] - mid_col))
print(best_tile.name)


with rasterio.open(ortho_path) as src:
    print("real GSD:", src.res[0], "m/px   config GSD_M:", GSD_M)

tile_bgr = cv2.imread(str(best_tile))          # look at it BEFORE counting

mask = vegetation_mask(tile_bgr, VEG_INDEX)    # build the mask for THIS tile
show(mask, "vegetation mask", cmap="gray")   # white = what the code thinks is plant
peaks, labels = split_plants(mask, tile_bgr, spacing_px, min_radius_px)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_contours = sum(cv2.contourArea(c) >= min_area_px(MIN_PLANT_DIAM_M, GSD_M) for c in contours)
print(f"contours only: {n_contours} plants   watershed: {len(peaks)} plants")

view = tile_bgr.copy()
view[labels == -1] = (0, 0, 255)                        # watershed borders in red
for y, x in peaks:
    cv2.circle(view, (int(x), int(y)), 4, (255, 0, 0), -1)
show(view, f"{len(peaks)} plants (blue dots), borders in red")
# Too many dots on one plant? raise PLANT_SPACING_M. Plants missed? lower it.








# %% Find every target near its surveyed position ------------------------------


import sys
print(sys.executable)

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as reproject_points

from config import *
from utils import *

GCP_COLS = {"name": "id", "x": "x", "y": "y"}

ortho_path = pick_file(ORTHO_FILE, ORTHO_DIR, TIF_PATTERNS, "orthomosaic")
#gcp_path = pick_file(GCP_FILE, GCP_DIR, ("*.csv", "*.txt"), "GCP file")
gcps = pd.read_csv(r"C:\Users\lab_tech_01\OneDrive - Tennessee State University\First Semester\Programming for Data Science\Dataset Practice\drone_cv_vscode\drone_cv_vscode\Real\GCPs\GCPs\GCPs.csv", encoding="utf-8-sig")



gcps.columns = gcps.columns.str.strip()      # also remove any stray spaces around column names
print(list(gcps.columns))                    # should print exactly: ['id', 'x', 'y', 'z']
print(gcps.head())

# Cut a small picture of the first GCP target out of the orthomosaic to use as the template
TEMPLATE_M = 1.0                               # template size in metres (about one target wide)
with rasterio.open(ortho_path) as src:
    x0 = gcps[GCP_COLS["x"]].iloc[0]           # first GCP's x coordinate
    y0 = gcps[GCP_COLS["y"]].iloc[0]           # first GCP's y coordinate
    if GCP_CRS:                                # if GCPs use a different coordinate system...
        xs0, ys0 = reproject_points(GCP_CRS, src.crs, [x0], [y0])
        x0, y0 = xs0[0], ys0[0]                # ...convert to the map's system
    r0, c0 = src.index(x0, y0)                 # map coordinates -> pixel row/column
    t_half = int(TEMPLATE_M / 2 / src.res[0])  # half the template size, in pixels
    win0 = Window(c0 - t_half, r0 - t_half, 2 * t_half, 2 * t_half)
    template = to_uint8(src.read([1, 2, 3], window=win0, boundless=True,
                                 fill_value=0).transpose(1, 2, 0))
    template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)

cv2.imwrite("gcp_template.png", template)      # save it so you can open and look at it
th, tw = template.shape[:2]                    # template height and width in pixels
SEARCH_M = 2.0                                 # look within 2 m of where each GCP should be


# Check every GCP: search near its surveyed position and measure how far off the match is
errors = []
with rasterio.open(ortho_path) as src:
    xs = gcps[GCP_COLS["x"]].to_numpy(float)          # all GCP x coordinates
    ys = gcps[GCP_COLS["y"]].to_numpy(float)          # all GCP y coordinates
    if GCP_CRS:                                       # GCPs in another system?
        xs, ys = reproject_points(GCP_CRS, src.crs, xs, ys)   # convert to the map's system
    half = int(SEARCH_M / src.res[0])                 # search radius in pixels
    for name, x, y in zip(gcps[GCP_COLS["name"]], xs, ys):
        r, c = src.index(x, y)                        # surveyed position -> pixel
        win = Window(c - half, r - half, 2 * half, 2 * half)
        area = to_uint8(src.read([1, 2, 3], window=win, boundless=True,
                                 fill_value=0).transpose(1, 2, 0))
        area = cv2.cvtColor(area, cv2.COLOR_RGB2BGR)
        if area.shape[0] < th or area.shape[1] < tw:  # search area smaller than template? skip
            continue
        res = cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED)
        _, best, _, (bx, by) = cv2.minMaxLoc(res)     # best score and its top-left corner
        found_col = c - half + bx + tw / 2            # centre of the match, in full-map pixels
        found_row = r - half + by + th / 2
        fx, fy = rasterio.transform.xy(src.transform, found_row, found_col)
        err = np.hypot(fx - x, fy - y)                # straight-line distance in metres
        errors.append(err)
        print(f"{name}: match score {best:.2f}, offset {err * 100:.1f} cm")

errors = np.array(errors)
print(f"\nRMSE = {np.sqrt((errors ** 2).mean()) * 100:.1f} cm over {len(errors)} GCPs")
print("Ignore matches with a score below ~0.6: the target probably wasn't found.")



ortho_path = OUT / "my_odm" / "odm_orthophoto" / "odm_orthophoto.tif"
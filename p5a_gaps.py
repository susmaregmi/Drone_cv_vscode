"""
PROJECT 5a - Find gaps (missing plants) in the rows.

When & why: nursery plants stand in rows at a fixed spacing. A missing plant shows up as an
unusually long distance between two neighbours in the same row.
Child version: kids in a line, one step apart. If two kids are 3 steps apart, 2 are missing.

Input : outputs/yolo_plants.csv (or classical_plants.csv if YOLO isn't trained yet)
Output: outputs/plants_with_rows.csv, outputs/gaps.csv / .gpkg
"""
# %% Setup ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

source = OUT / "yolo_plants.csv"
if not source.exists():
    source = OUT / "classical_plants.csv"
plants = pd.read_csv(source)
print("using", source.name, "-", len(plants), "plants")
pts = plants[["x_map", "y_map"]].to_numpy()             # positions in metres
centre = pts.mean(axis=0)

EXTRA_ROTATION_DEG = 0      # set to 90 if the rows come out vertical in the plot below
                            # (in a regular grid the plant columns also form "stripes")

# %% Step 1 - rotate the field so rows run left-right ------------------------------
# Idea: turn the field step by step. At the right angle, all plants of a row share the same
# height on the screen, so a histogram of heights shows sharp spikes (one per row).
def rotation(a):
    return np.array([[np.cos(-a), -np.sin(-a)],
                     [np.sin(-a), np.cos(-a)]])            # rotates points by minus angle a


def stripe_score(a):
    y = ((pts - centre) @ rotation(a).T)[:, 1]             # up-down position after rotating
    bins = np.arange(y.min(), y.max() + ROW_SPACING_M / 6, ROW_SPACING_M / 6)
    counts, _ = np.histogram(y, bins=bins)
    return (counts ** 2).sum()                             # spiky histogram -> big score


angles = np.radians(np.arange(-90, 90, 0.25))              # try every quarter degree
angle = angles[np.argmax([stripe_score(a) for a in angles])] + np.radians(EXTRA_ROTATION_DEG)
print(f"row direction: {np.degrees(angle):.2f} degrees")
rot = rotation(angle)
aligned = (pts - centre) @ rot.T                          # rows are now horizontal

plt.figure(figsize=(12, 6))
plt.scatter(aligned[:, 0], aligned[:, 1], s=3)
plt.title("Rows should be horizontal lines. If they're vertical, set EXTRA_ROTATION_DEG = 90")
plt.axis("equal")
plt.show()

# %% Step 2 - give every plant a row number -------------------------------------------
# Rows repeat every ROW_SPACING_M. Find where the "first" row line sits (its offset),
# then row number = how many row spacings a plant is above that line, rounded.
y = aligned[:, 1]
phase = np.angle(np.exp(2j * np.pi * y / ROW_SPACING_M).mean())   # average position within one spacing
offset = phase / (2 * np.pi) * ROW_SPACING_M
row_id = np.round((y - offset) / ROW_SPACING_M).astype(int)
row_id -= row_id.min()                                    # number rows 0, 1, 2 ...
plants["row"] = row_id
plants.to_csv(OUT / "plants_with_rows.csv", index=False)
print(len(np.unique(row_id)), "rows found; plants per row:", np.bincount(row_id).tolist())

# %% Step 3 - inside each row, look for long gaps between neighbours -----------------
gaps = []
for r in np.unique(row_id):
    idx = np.where(row_id == r)[0]
    idx = idx[np.argsort(aligned[idx, 0])]                # this row's plants, left to right
    xs = aligned[idx, 0]
    d = np.diff(xs)                                       # distance to the next plant
    for i in np.where(d > 1.5 * PLANT_SPACING_M)[0]:
        missing = int(round(d[i] / PLANT_SPACING_M)) - 1
        mid_aligned = (aligned[idx[i]] + aligned[idx[i + 1]]) / 2
        mid_map = mid_aligned @ rot + centre               # rotate back to map coordinates
        gaps.append({"row": r, "missing_plants": max(missing, 1),
                     "gap_length_m": d[i], "x_map": mid_map[0], "y_map": mid_map[1]})
gaps = pd.DataFrame(gaps)
print(f"{len(gaps)} gaps, about {gaps.missing_plants.sum() if len(gaps) else 0} missing plants")
gaps.to_csv(OUT / "gaps.csv", index=False)

if len(gaps):
    import geopandas as gpd
    import rasterio
    from utils import pick_file, TIF_PATTERNS
    with rasterio.open(pick_file(ORTHO_FILE, ORTHO_DIR, TIF_PATTERNS, "orthomosaic")) as src:
        crs = src.crs
    gpd.GeoDataFrame(gaps, geometry=gpd.points_from_xy(gaps.x_map, gaps.y_map), crs=crs) \
       .to_file(OUT / "gaps.gpkg", driver="GPKG")

# Note: a plant missing at the very START or END of a row has no neighbour on one side,
# so this method can't see it. Compare row lengths with the nursery's planting plan for those.

# %% Step 4 - picture: plants coloured by row, gaps as red crosses ----------------------
plt.figure(figsize=(12, 6))
plt.scatter(aligned[:, 0], aligned[:, 1], c=row_id % 10, cmap="tab10", s=4)
if len(gaps):
    g = (gaps[["x_map", "y_map"]].to_numpy() - centre) @ rot.T
    plt.scatter(g[:, 0], g[:, 1], marker="x", c="red", s=60, label="gap")
    plt.legend()
plt.axis("equal")
plt.title("Plants by row and detected gaps - check them on the orthomosaic in QGIS")
plt.show()

"""
PROJECT 4d - Count every plant in the whole orthomosaic with your trained YOLO.

When & why: YOLO sees one 640x640 tile at a time. We predict on every tile, shift each box
back to its place in the full map, and remove the duplicates that overlapping tiles create (NMS).
Outputs: outputs/yolo_plants.csv / .gpkg (one row per plant, with its box and map position).
"""
# %% Setup ------------------------------------------------------------------
import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision
from ultralytics import YOLO

from config import *
from utils import *

model = YOLO(str(RUNS_DIR / "plants" / "weights" / "best.pt"))
ortho_path = pick_file(ORTHO_FILE, ORTHO_DIR, TIF_PATTERNS, "orthomosaic")
tiles = find_files(TILES_DIR, ("*.jpg",))
print(len(tiles), "tiles to predict")

# %% Predict on every tile and shift boxes into full-map pixels ----------------
boxes, scores = [], []
for k, path in enumerate(tiles):
    _, row, col = path.stem.split("_")                   # where this tile sits in the map
    row, col = int(row), int(col)
    result = model.predict(str(path), conf=CONF, imgsz=TILE, verbose=False)[0]
    for (x1, y1, x2, y2), s in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        boxes.append([x1 + col, y1 + row, x2 + col, y2 + row])   # tile offset -> full-map pixels
        scores.append(s)
    if k % 50 == 0:
        print(f"{k}/{len(tiles)} tiles done")
print(len(boxes), "boxes before removing duplicates")
if not boxes:
    raise RuntimeError("YOLO found nothing. Check that best.pt is trained, or lower CONF in config.py.")

# %% Remove duplicates from the tile overlaps (Non-Maximum Suppression) ----------
boxes_t = torch.tensor(boxes, dtype=torch.float32)
scores_t = torch.tensor(scores, dtype=torch.float32)
keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.5).numpy()   # keeps the best of overlapping boxes
boxes = np.array(boxes)[keep]
scores = np.array(scores)[keep]
print(len(boxes), "plants in the whole field")

# %% Save with map coordinates ---------------------------------------------------
cx = (boxes[:, 0] + boxes[:, 2]) / 2                     # box centre in full-map pixels
cy = (boxes[:, 1] + boxes[:, 3]) / 2
with rasterio.open(ortho_path) as src:
    transform, crs = src.transform, src.crs
xs, ys = rasterio.transform.xy(transform, cy, cx)        # (rows, cols) -> map x, y

table = pd.DataFrame({"plant_id": range(1, len(boxes) + 1), "x_map": xs, "y_map": ys,
                      "x1": boxes[:, 0], "y1": boxes[:, 1], "x2": boxes[:, 2], "y2": boxes[:, 3],
                      "confidence": scores})
table.to_csv(OUT / "yolo_plants.csv", index=False)

import geopandas as gpd
gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table.x_map, table.y_map), crs=crs) \
   .to_file(OUT / "yolo_plants.gpkg", driver="GPKG")
print("saved yolo_plants.csv and .gpkg")

# %% Compare with the other counters --------------------------------------------
counts = {"YOLO": len(table)}
for name, file in [("classical (p2)", "classical_plants.csv"), ("height tops (p3)", "height_plants.csv")]:
    if (OUT / file).exists():
        counts[name] = len(pd.read_csv(OUT / file))
print(counts)
print("Now count one block BY HAND in QGIS and add it to this comparison - that's your results table.")

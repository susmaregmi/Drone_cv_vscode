"""
PROJECT 4a - Make automatic "pre-labels" for YOLO from the classical detector.

When & why: drawing hundreds of boxes by hand is slow. The classical detector already finds
most plants, so we save its boxes in YOLO format; you then only CORRECT them
(add misses, delete false ones, fix merged plants) in a labeling tool.

Next steps after running this:
  1. Pick ~50 tiles from outputs/tiles (spread over the field).
  2. Import those tiles + their .txt files from outputs/prelabels into CVAT, Label Studio
     or Roboflow (format: "YOLO"), correct them, and export again as YOLO.
  3. Put the corrected .txt files in outputs/labels_corrected  -> then run p4b_split_dataset.py
"""
# %% Setup ------------------------------------------------------------------
import cv2

from config import *
from utils import *

spacing_px = PLANT_SPACING_M / GSD_M
min_radius_px = MIN_PLANT_DIAM_M / GSD_M / 2
min_area = min_area_px(MIN_PLANT_DIAM_M, GSD_M)
tiles = find_files(TILES_DIR, ("*.jpg",))
print(len(tiles), "tiles")

# %% Write one YOLO .txt file per tile ------------------------------------------
def save_yolo_labels(boxes, tile_w, tile_h, txt_path):
    """YOLO format: one line per plant -> class x_center y_center width height (all 0-1)."""
    with open(txt_path, "w") as f:
        for x, y, w, h in boxes:
            xc, yc = (x + w / 2) / tile_w, (y + h / 2) / tile_h   # centre, divided by tile size
            f.write(f"0 {xc:.6f} {yc:.6f} {w / tile_w:.6f} {h / tile_h:.6f}\n")   # 0 = class "plant"


total = 0
for path in tiles:
    img = cv2.imread(str(path))
    mask = vegetation_mask(img, VEG_INDEX)
    _, labels = split_plants(mask, img, spacing_px, min_radius_px)
    boxes = plant_boxes(labels, min_area)
    save_yolo_labels(boxes, img.shape[1], img.shape[0], PRELABEL_DIR / (path.stem + ".txt"))
    total += len(boxes)

(PRELABEL_DIR / "classes.txt").write_text("plant\n")   # labeling tools read class names from this
print(f"{total} boxes written to {PRELABEL_DIR}")

# %% Look at 3 pre-labelled tiles (green boxes) ----------------------------------
for path in tiles[:: max(1, len(tiles) // 3)][:3]:
    img = cv2.imread(str(path))
    H, W = img.shape[:2]
    for line in open(PRELABEL_DIR / (path.stem + ".txt")):
        _, xc, yc, w, h = map(float, line.split())
        x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)     # back from 0-1 to pixels
        x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    show(img, path.name)

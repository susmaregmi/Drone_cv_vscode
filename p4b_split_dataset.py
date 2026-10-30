"""
PROJECT 4b - Build the YOLO dataset: train / val / test split BY AREA.

When & why: neighbouring tiles overlap and share plants. If they land in both training and
test sets, the test score is falsely high. So we split by position in the field:
top 70 % of tile rows -> train, next 15 % -> val, bottom 15 % -> test.

Input : tiles in outputs/tiles + YOUR corrected labels in outputs/labels_corrected
Output: outputs/yolo_dataset/{images,labels}/{train,val,test} + data.yaml
"""
# %% Setup ------------------------------------------------------------------
import shutil

import numpy as np

from config import *
from utils import *

label_files = find_files(LABELS_DIR, ("*.txt",))
label_files = [f for f in label_files if f.name != "classes.txt"]
if not label_files:
    print("No corrected labels found - using the uncorrected PRE-labels (only for a first test!)")
    label_files = [f for f in find_files(PRELABEL_DIR, ("*.txt",)) if f.name != "classes.txt"]

pairs = []                                            # (image, label, tile row)
for lab in label_files:
    img = TILES_DIR / (lab.stem + ".jpg")
    if img.exists():
        row = int(lab.stem.split("_")[1])            # "tile_1024_512" -> 1024
        pairs.append((img, lab, row))
print(len(pairs), "labelled tiles")

# %% Split by tile row (= position from top to bottom of the field) ---------------
rows = np.array(sorted({r for _, _, r in pairs}))     # every different tile row, top to bottom
n = len(rows)
if n < 3:
    raise ValueError("Labelled tiles come from fewer than 3 tile rows - label tiles from more of the field.")
i1 = min(max(1, round(n * 0.70)), n - 2)              # index where "val" starts (at least 1 row each)
i2 = min(max(i1 + 1, round(n * 0.85)), n - 1)         # index where "test" starts
cut1, cut2 = rows[i1], rows[i2]


def which_split(row):
    if row < cut1:
        return "train"
    if row < cut2:
        return "val"
    return "test"


for split in ["train", "val", "test"]:
    for kind in ["images", "labels"]:
        folder = YOLO_DATA_DIR / kind / split
        shutil.rmtree(folder, ignore_errors=True)    # start clean each time you run this
        folder.mkdir(parents=True)

counts = {"train": 0, "val": 0, "test": 0}
for img, lab, row in pairs:
    split = which_split(row)
    shutil.copy(img, YOLO_DATA_DIR / "images" / split / img.name)
    shutil.copy(lab, YOLO_DATA_DIR / "labels" / split / lab.name)
    counts[split] += 1
print(counts)

# %% Write data.yaml (YOLO reads this to find the data) ---------------------------
yaml_text = f"""path: {YOLO_DATA_DIR.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: plant
"""
(YOLO_DATA_DIR / "data.yaml").write_text(yaml_text)
print(yaml_text)
# .as_posix() writes the path with "/" - YOLO is happiest with forward slashes, even on Windows.

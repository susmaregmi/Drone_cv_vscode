#%%
"""
config.py - ALL settings for every script live here.
Edit this file first. Every other script does `from config import *`.

Why one settings file? You change a path or a number in ONE place,
and every project (p0 ... p5) uses the new value.
"""
from pathlib import Path   # Path handles Windows paths (C:\...) safely

# ---------------------------------------------------------------------------
# 1. WHERE YOUR DATA IS
# ---------------------------------------------------------------------------
# The folder that CONTAINS images_nadir_RGB, orthomosaic, DEM, ...
# r"..." = "raw string": Python reads the backslashes literally (needed on Windows).
BASE = Path(r"C:\Users\lab_tech_01\OneDrive - Tennessee State University\First Semester\Programming for Data Science\Dataset Practice\drone_cv_vscode\drone_cv_vscode\Real")



NADIR_DIR = BASE / "images_nadir_RGB"      # "/" joins folder names, like in a path
OBLIQUE_DIR = BASE / "images_oblique_RGB"
ORTHO_DIR = BASE / "orthomosaic"
DEM_DIR = BASE / "DEM"
LAZ_DIR = BASE / "3DpointcloudLAZ"
GCP_DIR = BASE / "GCPs"


# If a script picks the wrong file, type the exact file path here (else leave None).
ORTHO_FILE = None      # e.g. Path(r"C:\...\orthomosaic\site_ortho.tif")
DSM_FILE = None        # surface model (includes plant tops)
DTM_FILE = None        # terrain model (bare ground)
LAZ_FILE = None
GCP_FILE = None

# Everything the scripts create goes into this folder (made automatically).
OUT = BASE / "outputs"
TILES_DIR = OUT / "tiles"                  # p2: 640x640 pieces of the orthomosaic
PRELABEL_DIR = OUT / "prelabels"           # p4a: automatic YOLO labels (to be corrected)
LABELS_DIR = OUT / "labels_corrected"      # p4b: put YOUR corrected YOLO .txt files here
YOLO_DATA_DIR = OUT / "yolo_dataset"       # p4b: train/val/test folders for YOLO
RUNS_DIR = OUT / "runs"                    # p4c: YOLO training results
CHIPS_DIR = OUT / "chips"                  # p5b: one small image per plant



# ---------------------------------------------------------------------------
# 2. NUMBERS YOU FILL IN AFTER RUNNING p0_explore.py
# ---------------------------------------------------------------------------
GSD_M = 0.02            # orthomosaic: metres of ground per pixel (p0 prints it)
PHOTO_GSD_M = 0.02      # raw nadir photos: metres per pixel (usually close to GSD_M)
SENSOR_WIDTH_MM = None  # camera sensor width from the drone spec sheet (optional, for GSD maths)

MIN_PLANT_DIAM_M = 0.10    # ignore green blobs smaller than this (weeds, noise)
PLANT_SPACING_M = 0.50     # distance between plants in a row (measure it on the map!)
ROW_SPACING_M = 0.60       # distance between rows
MIN_PLANT_HEIGHT_M = 0.30  # p3: ignore bumps lower than this (grass)



# ---------------------------------------------------------------------------
# 3. METHOD CHOICES
# ---------------------------------------------------------------------------
VEG_INDEX = "ExG"      # "ExG", "ExGR" or "VARI" - pick the best one in p1
TILE = 640             # tile size in pixels (YOLO's favourite input size)
STEP = 512             # step between tiles -> 640 - 512 = 128 px overlap
YOLO_MODEL = "yolov8n.pt"   # small pretrained YOLO; downloads automatically
EPOCHS = 100
CONF = 0.25            # YOLO: ignore boxes with confidence below 25 %

# GCP check (gcp_check.py) - column names inside your GCP file
GCP_COLS = {"name": "name", "x": "x", "y": "y"}
GCP_CRS = None             # e.g. "EPSG:4326" if the GCPs are in lat/lon; None = same as orthomosaic
GCP_TEMPLATE = None        # path to a small crop of ONE GCP target (you cut it yourself)

# Stop early with a clear message if BASE still points to a folder that doesn't exist.
if not BASE.exists():
    raise FileNotFoundError(f"BASE folder not found: {BASE}\n"
                            "Open config.py and set BASE to your dataset folder.")

# Make the output folders now, so no script crashes with "folder not found".
for _folder in [OUT, TILES_DIR, PRELABEL_DIR, LABELS_DIR, YOLO_DATA_DIR, RUNS_DIR, CHIPS_DIR]:
    _folder.mkdir(parents=True, exist_ok=True)

# Drone dataset → OpenCV → YOLO → nursery inventory (VS Code version)

These scripts are the code from the roadmap doc, rewritten to run on your own Windows computer in
VS Code. Each `pX_...py` file is one mini-project. Run them in order.

## 1. One-time setup

1. **Install VS Code extensions:** open VS Code → Extensions (Ctrl+Shift+X) → install **Python** and **Jupyter** (both by Microsoft).
2. **Create a clean Python environment** (you already have Anaconda). Open **Anaconda Prompt** and type:

   ```
   conda create -n dronecv python=3.11 -y
   conda activate dronecv
   cd C:\path\to\drone_cv_vscode
   pip install -r requirements.txt
   ```

   - `conda create` makes a separate "box" of packages, so this project can't break your other projects.
   - Python 3.11 because every package here (rasterio, YOLO, TensorFlow) supports it on Windows.
   - The install takes a while (YOLO brings PyTorch; TensorFlow is large). You can drop the `tensorflow` line until you reach p5b.

3. **Open the folder in VS Code:** File → Open Folder → `drone_cv_vscode`.
4. **Choose the environment:** Ctrl+Shift+P → "Python: Select Interpreter" → pick **dronecv**.
5. **Edit `config.py`:** set `BASE` to your dataset folder (the one containing `images_nadir_RGB`, `orthomosaic`, …).

## 2. How to run a script

- **Cell by cell (best for learning):** every `# %%` line starts a cell. Click **Run Cell** above it
  (or Shift+Enter). Results and pictures appear in the **Interactive** window on the right, and variables
  stay in memory, like a Jupyter notebook.
- **Whole file:** click the ▶ Run button (top right). Pictures open in pop-up windows; close each one to continue.
- **Exception:** run `p4c_train_yolo.py` as a whole file (▶), not cell by cell. On Windows YOLO's data loader
  starts helper processes, which needs the `if __name__ == "__main__":` guard in that file.

## 3. The files, in order

| File | Project | What it makes (in `outputs/`) |
| --- | --- | --- |
| `config.py` | settings for everything | — edit this first |
| `utils.py` | shared helpers (show, vegetation mask, watershed, tiling) | — |
| `p0_explore.py` | 0 Explore | `data_card.txt` → copy GSD etc. into config.py |
| `p1_opencv_nadir.py` | 1 OpenCV basics on photos | `p1/` result images |
| `p2_orthomosaic.py` | 2 Tiles, watershed, map points | `tiles/`, `classical_plants.csv/.gpkg` |
| `p3_height.py` | 3 Plant height | `chm.tif`, `height_plants.csv/.gpkg` |
| `p4a_prelabel.py` | 4 YOLO: automatic pre-labels | `prelabels/` (correct them in CVAT / Label Studio / Roboflow) |
| `p4b_split_dataset.py` | 4 YOLO: train/val/test by area | `yolo_dataset/` + `data.yaml` |
| `p4c_train_yolo.py` | 4 YOLO: training | `runs/plants/` (best.pt, results.png, …) |
| `p4d_count_field.py` | 4 YOLO: whole-field count | `yolo_plants.csv/.gpkg` |
| `p5a_gaps.py` | 5 Missing plants | `plants_with_rows.csv`, `gaps.csv/.gpkg` |
| `p5b_species.py` | 5 Species classifier (only if several species) | `chips/`, `species_predictions.csv` |
| `p5c_inventory.py` | 5 Crown size + final table | `inventory.csv/.gpkg` |
| `gcp_check.py` | optional: map accuracy with GCPs | printed RMSE |

Open any `.gpkg` file in QGIS or ArcGIS Pro on top of the orthomosaic to check results visually.

## 4. Differences from the Colab version

| Colab | VS Code (here) |
| --- | --- |
| `drive.mount(...)`, `/content/drive/MyDrive/...` | `BASE = Path(r"C:\...")` in config.py |
| `!pip install ...` | `pip install -r requirements.txt` once, in Anaconda Prompt |
| `cv2_imshow(img)` | `show(img)` from utils.py (matplotlib) |
| free Colab GPU | your computer: CPU unless you have an NVIDIA GPU (training is slow on CPU) |

If training on your computer is too slow, you can train only `p4c` in Colab and copy `best.pt` back to
`outputs/runs/plants/weights/`. All other scripts run fine on a normal laptop.

## 5. Common errors

| Error | Fix |
| --- | --- |
| `BASE folder not found` | set `BASE` in config.py |
| `ModuleNotFoundError` | the wrong interpreter is selected: pick **dronecv** (step 1.4) |
| `No orthomosaic (...) found` | file has another extension, or set `ORTHO_FILE` in config.py |
| Memory error on a big file | the scripts read in windows/tiles; reduce `TILE` or the preview size |
| YOLO: `RuntimeError ... freeze_support` | you ran p4c cell by cell — run the whole file with ▶ |

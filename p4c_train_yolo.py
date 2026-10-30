"""
PROJECT 4c - Train YOLO on your plant tiles.

When & why: a pretrained YOLO already knows edges, textures and shapes from millions of photos;
we only teach it "plant". That's why ~50-100 labelled tiles can already work.

Run this file as a whole script (Run button / F5), NOT cell by cell:
on Windows, YOLO's data loader starts helper processes, which needs the
`if __name__ == "__main__":` line below.

No NVIDIA GPU? Training runs on the CPU and is slow (hours). Then try EPOCHS = 30 first,
or train in Google Colab and copy best.pt back into outputs/runs/plants/weights/.
"""
import torch
from ultralytics import YOLO

from config import *


def main():
    device = 0 if torch.cuda.is_available() else "cpu"   # 0 = first NVIDIA GPU
    print("training on:", "GPU" if device == 0 else "CPU (slow)")

    model = YOLO(YOLO_MODEL)                  # downloads the pretrained weights the first time
    model.train(data=str(YOLO_DATA_DIR / "data.yaml"),
                imgsz=TILE,                   # tiles are 640 x 640
                epochs=EPOCHS,
                batch=8,                      # images per step; lower it if you run out of memory
                device=device,
                workers=2,                    # helper processes that load images
                flipud=0.5, fliplr=0.5,       # random up-down and left-right flips
                degrees=90,                   # random rotation: a top-down photo has no "up"
                patience=20,                  # stop early if 20 epochs bring no improvement
                project=str(RUNS_DIR), name="plants", exist_ok=True)

    # Score the best model on the TEST block it has never seen.
    best = YOLO(str(RUNS_DIR / "plants" / "weights" / "best.pt"))
    metrics = best.val(data=str(YOLO_DATA_DIR / "data.yaml"), split="test", device=device)
    print(f"test mAP@0.5 = {metrics.box.map50:.3f}")
    print(f"test precision = {metrics.box.mp:.3f}, recall = {metrics.box.mr:.3f}")
    print("Now open outputs/runs/plants/ : results.png, confusion_matrix.png, val_batch0_pred.jpg")


if __name__ == "__main__":          # Windows needs this guard for multi-process data loading
    main()

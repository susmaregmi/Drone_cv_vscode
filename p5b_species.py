"""
PROJECT 5b - Classify the species of each plant (two-stage design).

When & why: YOLO finds "plant"; a second, small classifier names the species from a crop
of each plant. Only useful if the field has more than one species - look at the photos first!

Part 1 crops one small image ("chip") per plant into outputs/chips/unsorted.
YOU then sort a few hundred chips into one folder per species, e.g.
    outputs/chips/train/arborvitae/   outputs/chips/train/boxwood/   ...
Part 2 trains a Keras classifier with transfer learning (MobileNetV2).
Part 3 predicts the species of every plant -> outputs/species_predictions.csv
"""
# %% Setup ------------------------------------------------------------------
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from config import *
from utils import *

ortho_path = pick_file(ORTHO_FILE, ORTHO_DIR, TIF_PATTERNS, "orthomosaic")
plants = pd.read_csv(OUT / "yolo_plants.csv")             # needs the boxes x1, y1, x2, y2
UNSORTED = CHIPS_DIR / "unsorted"
TRAIN_DIR = CHIPS_DIR / "train"
UNSORTED.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
CHIP = 224                                                # MobileNetV2's favourite input size

# %% Part 1 - crop one chip per plant (box -> slice, like the license-plate exercise) ----
def read_chip(src, x1, y1, x2, y2, pad=0.1):
    """Read the box (plus 10 % padding) from the orthomosaic and resize it to CHIP x CHIP."""
    w, h = x2 - x1, y2 - y1
    col, row = int(x1 - pad * w), int(y1 - pad * h)
    win = Window(col, row, int(w * (1 + 2 * pad)), int(h * (1 + 2 * pad)))
    rgb = src.read([1, 2, 3], window=win, boundless=True, fill_value=0).transpose(1, 2, 0)
    return cv2.resize(to_uint8(rgb), (CHIP, CHIP))


with rasterio.open(ortho_path) as src:
    for _, p in plants.iterrows():
        chip = read_chip(src, p.x1, p.y1, p.x2, p.y2)
        cv2.imwrite(str(UNSORTED / f"plant_{int(p.plant_id)}.jpg"), cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))
print(len(plants), "chips saved in", UNSORTED)
print("Now sort a few hundred of them into", TRAIN_DIR, "/<species name>/ folders, then run Part 2.")

# %% Part 2 - train a species classifier with transfer learning --------------------------
import keras
from keras import layers

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training", seed=42,     # 80 % to learn
    image_size=(CHIP, CHIP), batch_size=16, label_mode="int")
val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation", seed=42,   # 20 % to check
    image_size=(CHIP, CHIP), batch_size=16, label_mode="int")
species = train_ds.class_names                       # folder names, in alphabetical order
print("species:", species)

base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                      input_shape=(CHIP, CHIP, 3), pooling="avg")
base.trainable = False                               # keep the pretrained features frozen

model = keras.Sequential([
    keras.Input(shape=(CHIP, CHIP, 3)),
    layers.RandomFlip("horizontal_and_vertical"),    # augmentation: top-down chips have no "up"
    layers.RandomRotation(0.25),
    layers.Rescaling(1 / 127.5, offset=-1),          # MobileNetV2 expects pixels from -1 to 1
    base,
    layers.Dropout(0.3),
    layers.Dense(len(species), activation="softmax") # one probability per species
])
model.compile(loss="sparse_categorical_crossentropy",   # labels are 0, 1, 2 ... (not one-hot)
              optimizer="adam", metrics=["accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=15,
                    callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
model.save(OUT / "species_classifier.keras")

# Confusion matrix on the validation chips: which species get mixed up?
from sklearn.metrics import classification_report, confusion_matrix
y_true, y_pred = [], []
for images, labels in val_ds:                        # one loop, so answers and predictions stay in the same order
    y_true += list(labels.numpy())
    y_pred += list(np.argmax(model.predict(images, verbose=0), axis=1))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=species))

# %% Part 3 - predict the species of every plant ----------------------------------------
chips = np.stack([cv2.cvtColor(cv2.imread(str(UNSORTED / f"plant_{int(i)}.jpg")), cv2.COLOR_BGR2RGB)
                  for i in plants.plant_id])
probs = model.predict(chips.astype(np.float32), batch_size=32)
plants["species"] = [species[k] for k in probs.argmax(axis=1)]
plants["species_conf"] = probs.max(axis=1)
plants[["plant_id", "species", "species_conf"]].to_csv(OUT / "species_predictions.csv", index=False)
print(plants.species.value_counts())

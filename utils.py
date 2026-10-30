"""
utils.py - small helper functions that several projects reuse.
Writing them once here keeps every project script short.
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Finding files
# ---------------------------------------------------------------------------
def find_files(folder, patterns=("*",)):
    """Return a sorted list of files in `folder` matching any of the patterns.

    Example: find_files(NADIR_DIR, ("*.jpg", "*.JPG"))
    set(...) removes duplicates: on Windows "*.jpg" also matches ".JPG".
    """
    files = []
    for pattern in patterns:
        files += list(Path(folder).glob(pattern))
    return sorted(set(files))


def pick_file(override, folder, patterns, what="file"):
    """Use the path typed in config.py if there is one, else the first match in `folder`."""
    if override:
        return Path(override)
    files = find_files(folder, patterns)
    if not files:
        raise FileNotFoundError(f"No {what} ({patterns}) found in {folder}. Check config.py.")
    if len(files) > 1:
        print(f"Note: {len(files)} {what}s found, using {files[0].name}. "
              f"Set it in config.py if that's wrong.")
    return files[0]


IMAGE_PATTERNS = ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.tif", "*.TIF")
TIF_PATTERNS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")


# ---------------------------------------------------------------------------
# Showing images (VS Code: the picture opens in a window or the Interactive panel)
# ---------------------------------------------------------------------------
def show(img, title=None, cmap=None, size=(10, 8), bgr=True):
    """Display an image with matplotlib.

    bgr=True  -> the image came from cv2.imread (Blue-Green-Red), so flip it to RGB.
    bgr=False -> the image is already RGB (e.g. from rasterio).
    """
    plt.figure(figsize=size)
    if img.ndim == 3 and bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# Vegetation: separate plants from soil
# ---------------------------------------------------------------------------
def vegetation_index(img_bgr, name="ExG"):
    """Return a float image where plants are bright and soil is dark."""
    b, g, r = cv2.split(img_bgr.astype(np.float32))      # decimals, so maths can go negative
    if name == "ExG":
        return 2 * g - r - b                              # Excess Green
    if name == "ExGR":
        return (2 * g - r - b) - (1.4 * r - g)            # Excess Green minus Excess Red
    if name == "VARI":
        vari = (g - r) / (g + r - b + 1e-6)               # +1e-6 avoids dividing by zero
        return np.clip(vari, -1, 1)                       # cut extreme values
    raise ValueError("VEG_INDEX must be 'ExG', 'ExGR' or 'VARI'")


def vegetation_mask(img_bgr, name="ExG", kernel_size=5):
    """White (255) = plant, black (0) = not plant. Uses Otsu's automatic threshold."""
    index = vegetation_index(img_bgr, name)
    empty = img_bgr.sum(axis=2) == 0                      # pure black = no-data border
    index8 = cv2.normalize(index, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(index8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes
    mask[empty] = 0
    return mask


def min_area_px(diam_m, gsd_m):
    """Pixel area of a circle with diameter `diam_m` metres (area = pi * r^2)."""
    return np.pi * (diam_m / gsd_m / 2) ** 2


# ---------------------------------------------------------------------------
# Splitting touching plants (distance transform + watershed)
# ---------------------------------------------------------------------------
def split_plants(mask, img_bgr, spacing_px, min_radius_px=1.0):
    """Find one seed per plant and grow each plant's region with watershed.

    Returns
      peaks  : array of (y, x) plant centres
      labels : image where each plant has its own number (>= 2), 1 = background, -1 = border
    """
    from skimage.feature import peak_local_max

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)     # centre of each blob = largest value
    peaks = peak_local_max(dist,
                           min_distance=max(1, int(spacing_px // 2)),   # centres at least half a spacing apart
                           threshold_abs=min_radius_px,                # drop tiny blobs
                           labels=(mask > 0).astype(np.int32),
                           exclude_border=False)                       # keep plants at the edge
    markers = np.zeros(mask.shape, np.int32)
    for i, (y, x) in enumerate(peaks, start=2):           # 0 = undecided, 1 = background
        markers[y, x] = i
    markers[mask == 0] = 1
    labels = cv2.watershed(img_bgr, markers)               # needs an 8-bit, 3-channel image
    return peaks, labels


def plant_boxes(labels, min_area=0):
    """One (x, y, w, h) box per watershed region, skipping regions smaller than min_area."""
    from skimage.measure import regionprops

    lab = labels.copy()
    lab[lab < 2] = 0                                       # background and borders -> 0
    boxes = []
    for region in regionprops(lab):
        if region.area < min_area:
            continue
        y1, x1, y2, x2 = region.bbox                       # skimage gives rows first
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
def tile_starts(length, tile, step):
    """Start positions so tiles cover the whole length, including the last edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)                       # one extra tile flush with the edge
    return starts


def to_uint8(rgb):
    """Some orthomosaics are 16-bit. Scale anything that isn't 8-bit to 0-255."""
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32)
    top = np.percentile(rgb[rgb > 0], 99.5) if (rgb > 0).any() else 1
    return np.clip(rgb / top * 255, 0, 255).astype(np.uint8)

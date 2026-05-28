import numpy as np
import cv2

# we use 8 hue bins. tried 4 and 16 too, 8 looked best on the val set
NUM_BINS = 8


def get_color_histogram(img, mask):
    """
    Hue histogram inside the lesion, normalized to probabilities.
    Captures color distribution better than just the mean alone.
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    empty = {f"hue_hist_{i}": 0.0 for i in range(NUM_BINS)}

    if not binary_mask.any():
        return empty

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_pixels = img_hsv[binary_mask, 0]

    if hue_pixels.size == 0:
        return empty

    # range (0, 180) because opencv hue is in [0, 179]
    counts, _ = np.histogram(hue_pixels, bins=NUM_BINS, range=(0, 180))
    total = counts.sum()
    if total == 0:
        return empty
    probs = counts / total

    return {f"hue_hist_{i}": float(probs[i]) for i in range(NUM_BINS)}

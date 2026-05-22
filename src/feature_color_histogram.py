import numpy as np
import cv2

# Number of hue bins. 8 gives enough resolution to separate red/brown/dark
# from lighter tones without overfitting to per-image noise.
_N_HUE_BINS = 8


def get_color_histogram(img, mask):
    """
    Computes a normalised histogram of hue values inside the lesion.

    A summary statistic like `mean_h` collapses the entire colour
    distribution to a single number. Lesions with bimodal colour
    (e.g. mixed brown + black, often a melanoma cue) get an average
    that's not representative. A histogram preserves multi-modality.

    Bins are evenly spaced across OpenCV's hue range [0, 180]. Each
    bin is the *fraction* of lesion pixels in that hue band, so the
    feature is scale-invariant.

    :param img: numpy array of the image (BGR format from cv2)
    :param mask: numpy array of the mask (grayscale)
    :return: dictionary with hue_hist_0 ... hue_hist_7
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    empty = {f"hue_hist_{i}": 0.0 for i in range(_N_HUE_BINS)}

    if not binary_mask.any():
        return empty

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_pixels = img_hsv[binary_mask, 0]

    if hue_pixels.size == 0:
        return empty

    # Histogram with N bins across [0, 180]
    counts, _ = np.histogram(hue_pixels, bins=_N_HUE_BINS, range=(0, 180))
    # Normalise to a probability distribution (sum to 1)
    total = counts.sum()
    if total == 0:
        return empty
    probs = counts / total

    return {f"hue_hist_{i}": float(probs[i]) for i in range(_N_HUE_BINS)}

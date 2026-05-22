import numpy as np
import cv2
from skimage.feature import local_binary_pattern

# LBP parameters. P=8 neighbours at radius 1 is the standard small-scale
# texture descriptor. method='uniform' collapses patterns with more than 2
# transitions into a single bin, giving 10 bins total (P+2) and reducing
# noise sensitivity.
_LBP_P = 8
_LBP_R = 1
_N_LBP_BINS = _LBP_P + 2  # 10


def get_lbp_features(img, mask):
    """
    Computes a normalised Local Binary Pattern (LBP) histogram inside
    the lesion. LBP captures micro-texture (the ABCDE's "D" + irregular
    pigmentation cues) by recording how each pixel compares to its
    neighbours.

    Background pixels are excluded so the histogram describes texture
    inside the lesion, not the surrounding skin.

    :param img: numpy array of the image (BGR format from cv2)
    :param mask: numpy array of the mask (grayscale)
    :return: dictionary with lbp_0 ... lbp_9
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    empty = {f"lbp_{i}": 0.0 for i in range(_N_LBP_BINS)}

    if not binary_mask.any():
        return empty

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crop to lesion bounding box so the LBP computation is faster
    # and so border-of-image artefacts don't contribute
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped_gray = gray[rmin:rmax + 1, cmin:cmax + 1]
    cropped_mask = binary_mask[rmin:rmax + 1, cmin:cmax + 1]

    if cropped_gray.size == 0:
        return empty

    lbp = local_binary_pattern(cropped_gray, _LBP_P, _LBP_R, method="uniform")

    # Keep only LBP values for pixels inside the lesion
    lesion_lbp = lbp[cropped_mask]
    if lesion_lbp.size == 0:
        return empty

    counts, _ = np.histogram(lesion_lbp, bins=_N_LBP_BINS,
                              range=(0, _N_LBP_BINS))
    total = counts.sum()
    if total == 0:
        return empty
    probs = counts / total

    return {f"lbp_{i}": float(probs[i]) for i in range(_N_LBP_BINS)}

import numpy as np
import cv2
from skimage.feature import local_binary_pattern

# LBP with P=8 neighbours and radius 1. With method="uniform" we get 10 bins
# (= P+2). Read about it here: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
P = 8
R = 1
N_BINS = P + 2   # 10


def get_lbp_features(img, mask):
    """
    Local Binary Pattern histogram inside the lesion.

    LBP describes the texture around each pixel. We compute it, take only the
    pixels that fall inside the lesion mask, and return the normalized histogram
    so the values sum to 1.
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    empty = {f"lbp_{i}": 0.0 for i in range(N_BINS)}

    if not binary_mask.any():
        return empty

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # crop to bbox before computing LBP -- it's WAY faster on small images
    # and avoids picking up nonsense from outside the lesion
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped_gray = gray[rmin:rmax+1, cmin:cmax+1]
    cropped_mask = binary_mask[rmin:rmax+1, cmin:cmax+1]

    if cropped_gray.size == 0:
        return empty

    lbp = local_binary_pattern(cropped_gray, P, R, method="uniform")

    lesion_lbp = lbp[cropped_mask]
    if lesion_lbp.size == 0:
        return empty

    counts, _ = np.histogram(lesion_lbp, bins=N_BINS, range=(0, N_BINS))
    total = counts.sum()
    if total == 0:
        return empty
    probs = counts / total

    return {f"lbp_{i}": float(probs[i]) for i in range(N_BINS)}

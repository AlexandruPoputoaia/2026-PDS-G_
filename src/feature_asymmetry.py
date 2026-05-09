import numpy as np
import cv2

def get_asymmetry(mask, img=None):
    """
    Calculates the asymmetry of the lesion by comparing
    the mask to its flipped versions.

    The mask is first cropped to the lesion bounding box so that
    flipping is done across the lesion's own axes, not the image edges.

    A score of 0 = perfectly symmetric
    A higher score = more asymmetric (more likely cancerous)

    :param mask: numpy array of the mask (grayscale)
    :param img: optional image to check size against
    :return: asymmetry score (float)
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    binary_mask = (mask > 0).astype(np.float32)

    if np.sum(binary_mask) == 0:
        return 0.0

    # Crop to the lesion bounding box so flipping is centred on the lesion
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = binary_mask[rmin:rmax + 1, cmin:cmax + 1]

    total_pixels = np.sum(cropped)

    flipped_h = np.fliplr(cropped)
    flipped_v = np.flipud(cropped)

    asym_h = np.sum(np.abs(cropped - flipped_h)) / (2 * total_pixels)
    asym_v = np.sum(np.abs(cropped - flipped_v)) / (2 * total_pixels)

    return (asym_h + asym_v) / 2
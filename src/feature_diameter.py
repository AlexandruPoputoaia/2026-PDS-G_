import numpy as np
import cv2
from skimage import transform


def get_diameter_at_angle(mask, angle):
    """Measure the lesion width after rotating the mask by `angle` degrees."""
    binary_mask = mask > 0

    # skimage.transform.rotate is slow on big masks but it works ¯\_(ツ)_/¯
    rotated = transform.rotate(binary_mask.astype(float), angle, preserve_range=True)
    rotated = rotated > 0.5

    cols = np.sum(rotated, axis=0)
    diameter = np.sum(cols > 0)

    return diameter


def get_all_diameters(mask, img=None):
    """
    Diameter at 0, 45, 90, 135 degrees + ratio of largest to smallest.
    Ratio > 1 means elongated lesion.
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # empty mask -> all zeros, don't crash
    if np.sum(mask > 0) == 0:
        return {
            "diameter_0": 0,
            "diameter_45": 0,
            "diameter_90": 0,
            "diameter_135": 0,
            "diameter_ratio": 0,
        }

    d0 = get_diameter_at_angle(mask, 0)
    d45 = get_diameter_at_angle(mask, 45)
    d90 = get_diameter_at_angle(mask, 90)
    d135 = get_diameter_at_angle(mask, 135)

    diameters = [d0, d45, d90, d135]
    min_d = min(diameters)
    max_d = max(diameters)
    ratio = max_d / min_d if min_d > 0 else 0

    return {
        "diameter_0": d0,
        "diameter_45": d45,
        "diameter_90": d90,
        "diameter_135": d135,
        "diameter_ratio": ratio,
    }

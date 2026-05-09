import numpy as np
import cv2
from skimage import transform

def get_diameter_at_angle(mask, angle):
    """
    Measures the diameter of the lesion at a given angle
    by rotating the mask and measuring the width.
    
    :param mask: numpy array of the mask (grayscale)
    :param angle: angle in degrees
    :return: diameter in pixels
    """
    binary_mask = mask > 0
    
    # Rotate the mask
    rotated = transform.rotate(binary_mask.astype(float), angle, preserve_range=True)
    rotated = rotated > 0.5
    
    # Measure width (number of columns with lesion pixels)
    cols = np.sum(rotated, axis=0)
    diameter = np.sum(cols > 0)
    
    return diameter


def get_all_diameters(mask, img=None):
    """
    Measures the lesion diameter at 4 angles: 0, 45, 90, 135 degrees.
    Also calculates the ratio between the largest and smallest diameter
    which indicates how elongated the lesion is.
    
    :param mask: numpy array of the mask (grayscale)
    :param img: optional image to check size against
    :return: dictionary with diameter measurements
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    if np.sum(mask > 0) == 0:
        return {
            "diameter_0": 0,
            "diameter_45": 0,
            "diameter_90": 0,
            "diameter_135": 0,
            "diameter_ratio": 0
        }
    
    d0 = get_diameter_at_angle(mask, 0)
    d45 = get_diameter_at_angle(mask, 45)
    d90 = get_diameter_at_angle(mask, 90)
    d135 = get_diameter_at_angle(mask, 135)
    
    # Ratio of max to min diameter - high ratio means elongated lesion
    diameters = [d0, d45, d90, d135]
    min_d = min(diameters)
    max_d = max(diameters)
    ratio = max_d / min_d if min_d > 0 else 0
    
    return {
        "diameter_0": d0,
        "diameter_45": d45,
        "diameter_90": d90,
        "diameter_135": d135,
        "diameter_ratio": ratio
    }
import numpy as np
import cv2

def get_lesion_area(mask, img=None):
    """
    Calculates the fraction of the image covered by the lesion.
    
    :param mask: numpy array of the mask image (white = lesion, black = background)
    :param img: optional image to check size against
    :return: float between 0 and 1
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask > 0
    lesion_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    
    return lesion_pixels / total_pixels
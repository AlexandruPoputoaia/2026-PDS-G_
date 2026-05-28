import numpy as np
import cv2


def get_lesion_area(mask, img=None):

    # mask is sometimes a different size than the image

    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary = mask > 0
    return np.sum(binary) / binary.size

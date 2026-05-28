import numpy as np
import cv2
from scipy.ndimage import binary_dilation


def get_relative_color(img, mask):
    """
    Difference between lesion color and the skin around it.
    We use a thin ring just outside the lesion instead of the whole background
    because the background has hair and other stuff that messes it up.
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0

    # dilate the lesion mask and subtract -> ring of pixels around the lesion
    # 10 pixels was a good compromise (5 was too thin, 20 included other stuff)
    dilated = binary_dilation(binary_mask, iterations=10)
    ring_mask = dilated & ~binary_mask

    img_rgb = img[:, :, ::-1]   # BGR -> RGB

    lesion_pixels = img_rgb[binary_mask]
    surrounding_pixels = img_rgb[ring_mask]

    # fallback for lesions at the edge of the image where the ring gets cropped off
    if len(surrounding_pixels) < 10:
        surrounding_pixels = img_rgb[~binary_mask]

    if len(lesion_pixels) == 0 or len(surrounding_pixels) == 0:
        return {"rel_r": 0, "rel_g": 0, "rel_b": 0}

    diff = np.mean(lesion_pixels, axis=0) - np.mean(surrounding_pixels, axis=0)

    return {
        "rel_r": diff[0],
        "rel_g": diff[1],
        "rel_b": diff[2],
    }

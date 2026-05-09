import numpy as np
import cv2
from scipy.ndimage import binary_dilation

# Ring thickness in pixels — wide enough to sample real skin, narrow enough
# to stay close to the lesion and avoid image borders/artifacts
_RING_THICKNESS = 10

def get_relative_color(img, mask):
    """
    Calculates the difference between the lesion color and
    the immediately surrounding skin color.

    Uses a thin ring of pixels just outside the lesion border rather than
    the entire image background, which may contain artifacts or non-skin areas.

    :param img: numpy array of the image (BGR format from cv2)
    :param mask: numpy array of the mask (grayscale)
    :return: dictionary with relative color features
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0

    # Dilate the lesion outward and subtract the lesion itself to get a ring
    dilated = binary_dilation(binary_mask, iterations=_RING_THICKNESS)
    ring_mask = dilated & ~binary_mask

    # Convert to RGB
    img_rgb = img[:, :, ::-1]

    lesion_pixels = img_rgb[binary_mask]
    surrounding_pixels = img_rgb[ring_mask]

    # Fall back to full background if the ring is too small (edge-of-frame lesion)
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
import numpy as np
import cv2
from skimage import morphology


def get_lesion_dimensions(mask, img=None):
    """
    Height and width of the lesion in pixels.

    :param mask: mask array (grayscale)
    :param img: optional original image, used only to check sizes
    :return: (height, width)
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0

    # sum pixels per row / per column
    rows = np.sum(binary_mask, axis=1)
    cols = np.sum(binary_mask, axis=0)

    height = np.sum(rows > 0)   # rows containing any lesion pixel
    width = np.sum(cols > 0)

    return height, width


def get_perimeter(mask, img=None):
    """
    Perimeter via morphological erosion: subtract the eroded mask from
    itself and count the border pixels.

    :param mask: mask array (grayscale)
    :return: number of border pixels
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    eroded = morphology.erosion(binary_mask)
    perimeter = binary_mask & ~eroded
    return np.sum(perimeter)


def get_compactness(mask, img=None):
    """Compactness (circularity). A perfect circle = 1, weird shapes > 1."""
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    area = np.sum(binary_mask)
    perimeter = get_perimeter(mask)   # FIXME: we already have the mask here, why call get_perimeter again

    if perimeter == 0 or area == 0:
        return 0

    return (perimeter ** 2) / (4 * np.pi * area)


# ----- added in week 5 -----
# solidity and extent - extra shape descriptors. We asked the TA whether these
# would help and she said maybe, so we added them.

def get_solidity(mask, img=None):
    '''
    Solidity = lesion_area / convex_hull_area, in [0, 1].
    1.0 means convex (no dents), <1 means the shape has concave bits.
    Melanomas often have notched borders so this should be lower for them in theory.
    '''
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = (mask > 0).astype(np.uint8)
    area = float(np.sum(binary_mask))
    if area == 0:
        return 0.0

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    # if there are multiple contours we want ONE hull around all of them
    all_points = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(all_points)

    # rasterise the hull so the comparison is in pixels
    hull_mask = np.zeros_like(binary_mask)
    cv2.fillPoly(hull_mask, [hull], 1)
    hull_area = float(np.sum(hull_mask))

    if hull_area == 0:
        return 0.0
    # clip in case of weirdness with anti-aliasing -- shouldn't happen but just in case
    return min(area / hull_area, 1.0)


def get_extent(mask, img=None):
    '''Extent = lesion_area / bounding_box_area. 1.0 = fills its box.'''
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    area = float(np.sum(binary_mask))
    if area == 0:
        return 0.0

    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_area = float((rmax - rmin + 1) * (cmax - cmin + 1))

    if bbox_area == 0:
        return 0.0
    return area / bbox_area

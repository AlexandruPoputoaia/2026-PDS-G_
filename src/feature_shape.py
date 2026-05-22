import numpy as np
import cv2
from skimage import morphology

def get_lesion_dimensions(mask, img=None):
    """
    Calculates the height and width of the lesion in pixels.
    
    :param mask: numpy array of the mask (grayscale)
    :param img: optional image to check size against
    :return: tuple (height, width) in pixels
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask > 0
    
    # Sum pixels per row and per column
    rows = np.sum(binary_mask, axis=1)
    cols = np.sum(binary_mask, axis=0)
    
    # Height = number of rows that contain lesion pixels
    height = np.sum(rows > 0)
    # Width = number of columns that contain lesion pixels
    width = np.sum(cols > 0)
    
    return height, width

def get_perimeter(mask, img=None):
    """
    Calculates the perimeter of the lesion using morphological erosion.
    
    :param mask: numpy array of the mask (grayscale)
    :param img: optional image to check size against
    :return: perimeter in pixels
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask > 0
    
    # Erode the mask and subtract to get border pixels
    eroded = morphology.erosion(binary_mask)
    perimeter = binary_mask & ~eroded
    
    return np.sum(perimeter)

def get_compactness(mask, img=None):
    """
    Calculates compactness (circularity) of the lesion.
    A perfect circle has compactness 1, irregular shapes have higher values.
    
    :param mask: numpy array of the mask (grayscale)
    :return: compactness value
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask > 0
    area = np.sum(binary_mask)
    perimeter = get_perimeter(mask)
    
    if perimeter == 0 or area == 0:
        return 0

    return (perimeter ** 2) / (4 * np.pi * area)


def get_solidity(mask, img=None):
    """
    Calculates solidity: lesion area divided by its convex hull area.

    Solidity is 1.0 for a perfectly convex shape (no indentations) and
    drops below 1.0 when the lesion has concave bays or "bite marks" cut
    out of it. Melanomas often have notched borders that lower solidity.

    Both areas are pixel counts (the hull is rasterised) so the ratio
    is guaranteed to lie in [0, 1].

    :param mask: numpy array of the mask (grayscale)
    :param img: optional image to check size against
    :return: solidity in [0, 1]
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    binary_mask = (mask > 0).astype(np.uint8)
    area = float(np.sum(binary_mask))
    if area == 0:
        return 0.0

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    # Convex hull over ALL contour points so multiple components are
    # enclosed by a single hull, matching the standard definition.
    all_points = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(all_points)

    # Rasterise the hull and pixel-count it for a fair comparison
    hull_mask = np.zeros_like(binary_mask)
    cv2.fillPoly(hull_mask, [hull], 1)
    hull_area = float(np.sum(hull_mask))

    if hull_area == 0:
        return 0.0
    return min(area / hull_area, 1.0)


def get_extent(mask, img=None):
    """
    Calculates extent: lesion area divided by its bounding box area.

    Extent is 1.0 for a perfectly rectangular lesion filling its bounding
    box, lower for round or irregular shapes. Together with solidity it
    helps distinguish blob-like from elongated/irregular lesions.

    :param mask: numpy array of the mask (grayscale)
    :param img: optional image to check size against
    :return: extent in [0, 1]
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    area = float(np.sum(binary_mask))
    if area == 0:
        return 0.0

    # Tight bounding box around all lesion pixels
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_area = float((rmax - rmin + 1) * (cmax - cmin + 1))

    if bbox_area == 0:
        return 0.0
    return area / bbox_area
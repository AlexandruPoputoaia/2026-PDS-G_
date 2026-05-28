import numpy as np
import cv2
from skimage import morphology


def get_lesion_dimensions(mask, img=None):
    
    #Height and width of the lesion in pixels.
    #return: (height, width)

    if img is not None and img.shape[:2] != mask.shape[:2]: #sometimes the mask is a different size than the image
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), #resi
                          interpolation=cv2.INTER_NEAREST) 

    binary_mask = mask > 0

    # sum pixels per row / per column

    rows = np.sum(binary_mask, axis=1) #sum of each row, i.e. number of lesion pixels in each row
    cols = np.sum(binary_mask, axis=0) #sum of each column, i.e. number of lesion pixels in each column

    height = np.sum(rows > 0)   #rows containing any lesion pixel
    width = np.sum(cols > 0)    #columns containing any lesion pixel

    return height, width


def get_perimeter(mask, img=None):

    #Perimeter via morphological erosion: subtract the eroded mask from itself and count the border pixels
    #param mask: mask array (grayscale)
    #return: number of border pixels

    if img is not None and img.shape[:2] != mask.shape[:2]: #sometimes the mask is a different size than the image
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0 
    eroded = morphology.erosion(binary_mask) #enrode the mask
    perimeter = binary_mask & ~eroded
    return np.sum(perimeter)


def get_compactness(mask, img=None):
    #Compactness (circularity). A perfect circle = 1, weird shapes > 1.
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    area = np.sum(binary_mask)
    perimeter = get_perimeter(mask)

    if perimeter == 0 or area == 0: #avoid devide by 0
        return 0

    return (perimeter ** 2) / (4 * np.pi * area)

#solidity and extent - extra shape descriptors

def get_solidity(mask, img=None):

    #Solidity = lesion_area / convex_hull_area, in [0, 1].
    # 1.0 means convex (no dents) 
    # <1 means the shape has concave bits.

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

    all_points = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(all_points)

    hull_mask = np.zeros_like(binary_mask)
    cv2.fillPoly(hull_mask, [hull], 1)
    hull_area = float(np.sum(hull_mask))

    if hull_area == 0: #just ijn case
        return 0.0
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

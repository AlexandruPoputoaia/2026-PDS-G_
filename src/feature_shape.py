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
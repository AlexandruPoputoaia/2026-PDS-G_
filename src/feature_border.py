import numpy as np
import cv2
from skimage import morphology

def get_border_irregularity(mask, img=None):
    """
    Calculates how irregular the border of the lesion is.
    
    A smooth circular lesion has a low score.
    An irregular jagged border has a high score.
    This is done by comparing the perimeter to what a smooth
    shape of the same area would have.
    
    :param mask: numpy array of the mask (grayscale)
    :param img: optional image to check size against
    :return: border irregularity score (float)
    """
    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask > 0
    area = np.sum(binary_mask)
    
    if area == 0:
        return 0.0
    
    # Get perimeter using erosion
    eroded = morphology.erosion(binary_mask)
    perimeter_mask = binary_mask & ~eroded
    perimeter = np.sum(perimeter_mask)
    
    # For a perfect circle: perimeter = 2 * sqrt(pi * area)
    # We compare actual perimeter to this ideal
    ideal_perimeter = 2 * np.sqrt(np.pi * area)
    
    return perimeter / ideal_perimeter


def get_border_gradient(mask, img):
    """
    Calculates the average color gradient at the border of the lesion.
    A sharp border has a high gradient (more likely benign).
    A blurry border has a low gradient (more likely malignant).
    
    :param mask: numpy array of the mask (grayscale)
    :param img: numpy array of the image (BGR)
    :return: mean gradient at border (float)
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask > 0
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradient using Sobel
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Get border pixels
    eroded = morphology.erosion(binary_mask)
    border = binary_mask & ~eroded
    
    if np.sum(border) == 0:
        return 0.0
    
    # Return mean gradient at border
    return np.mean(gradient[border])
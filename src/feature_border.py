import numpy as np
import cv2
from skimage import morphology


def get_border_irregularity(mask, img=None):

    #Compares the lesion's perimeter to the perimeter a perfect circle of the same area would have. Higher = more irregular.

    if img is not None and img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    binary_mask = mask > 0
    area = np.sum(binary_mask)
    if area == 0:
        return 0.0

    # perimeter via erosion (same as feature_shape)

    eroded = morphology.erosion(binary_mask)
    perimeter_mask = binary_mask & ~eroded
    perimeter = np.sum(perimeter_mask)

    # a perfect circle would have perimeter = 2*sqrt(pi*area)

    ideal_perimeter = 2 * np.sqrt(np.pi * area)
    return perimeter / ideal_perimeter


# border gradient

def get_border_gradient(mask, img):
    
    # Idea: sharp lesion border = high gradient at border = probably benign
    # blurry/fuzzy border = low gradient = more suspicious

    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    binary_mask = mask > 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sobel in x and y, then magnitude

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)

    # get the border pixels

    eroded = morphology.erosion(binary_mask)
    border = binary_mask & ~eroded
    if np.sum(border) == 0:
        return 0.0
    return np.mean(gradient[border])

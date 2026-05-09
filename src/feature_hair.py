import numpy as np
import cv2

def get_hair_features(img, mask):
    """
    Detects hair in the image using BlackHat filtering.
    
    BlackHat filtering works by:
    1. Applying morphological closing to fill in dark areas (hair)
    2. Subtracting the original image from the closed image
    3. The result highlights dark thin structures (hair)
    
    Features extracted:
    - hair_coverage: ratio of hair pixels to total lesion pixels
    - hair_in_lesion: ratio of hair pixels inside the lesion
    
    :param img: numpy array of the image (BGR format from cv2)
    :param mask: numpy array of the mask (grayscale)
    :return: dictionary with hair features
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask > 0
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create kernel for BlackHat filtering
    # Larger kernel = detects thicker hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    
    # Apply BlackHat filter
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Otsu's method picks the threshold automatically per image,
    # handling variation in brightness and contrast across images
    _, hair_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Total image pixels
    total_pixels = img.shape[0] * img.shape[1]
    
    # Hair pixels in whole image
    hair_pixels_total = np.sum(hair_mask > 0)
    
    # Hair pixels inside lesion only
    hair_in_lesion = np.sum((hair_mask > 0) & binary_mask)
    lesion_pixels = np.sum(binary_mask)
    
    return {
        # Overall hair coverage in the whole image
        "hair_coverage": hair_pixels_total / total_pixels if total_pixels > 0 else 0,
        # Hair coverage specifically inside the lesion
        "hair_in_lesion": hair_in_lesion / lesion_pixels if lesion_pixels > 0 else 0,
    }
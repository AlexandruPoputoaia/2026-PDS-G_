import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

def get_texture_features(img, mask):
    """
    Calculates texture features using Gray Level Co-occurrence Matrix (GLCM).
    GLCM captures how often pairs of pixels with specific values
    occur next to each other.
    
    Features extracted:
    - Contrast: local intensity variation
    - Dissimilarity: similar to contrast but linear
    - Homogeneity: how similar pixels are to their neighbors
    - Energy: uniformity of the texture
    - Correlation: linear dependency of gray levels
    
    :param img: numpy array of the image (BGR format from cv2)
    :param mask: numpy array of the mask (grayscale)
    :return: dictionary with texture features
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask > 0

    if np.sum(binary_mask) == 0:
        return {
            "contrast": 0, "dissimilarity": 0,
            "homogeneity": 0, "energy": 0, "correlation": 0
        }

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crop to lesion bounding box so background pixels don't skew the GLCM
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped_gray = gray[rmin:rmax + 1, cmin:cmax + 1]
    cropped_mask = binary_mask[rmin:rmax + 1, cmin:cmax + 1]

    # Zero out any background pixels remaining in the bounding box
    cropped_gray = cropped_gray.copy()
    cropped_gray[~cropped_mask] = 0

    # Reduce to 32 gray levels for faster computation
    gray_32 = (cropped_gray // 8).astype(np.uint8)

    # Calculate GLCM at 4 angles and average for rotation invariance
    glcm = graycomatrix(gray_32, distances=[1],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)

    def _mean_prop(prop):
        return float(np.mean(graycoprops(glcm, prop)[0, :]))

    # Correlation is 0/0 for uniform regions — guard against NaN
    correlation = _mean_prop('correlation')
    if np.isnan(correlation):
        correlation = 0.0

    return {
        "contrast": _mean_prop('contrast'),
        "dissimilarity": _mean_prop('dissimilarity'),
        "homogeneity": _mean_prop('homogeneity'),
        "energy": _mean_prop('energy'),
        "correlation": correlation,
    }
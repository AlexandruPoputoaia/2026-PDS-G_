import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops


def get_texture_features(img, mask):
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    binary_mask = mask > 0
    if np.sum(binary_mask) == 0:
        return {
            "contrast": 0, "dissimilarity": 0,
            "homogeneity": 0, "energy": 0, "correlation": 0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped_gray = gray[rmin:rmax+1, cmin:cmax+1]
    cropped_mask = binary_mask[rmin:rmax+1, cmin:cmax+1]
    cropped_gray = cropped_gray.copy()
    cropped_gray[~cropped_mask] = 0
    gray_32 = (cropped_gray // 8).astype(np.uint8)

    glcm = graycomatrix(
        gray_32, distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32, symmetric=True, normed=True
    )

    contrast      = float(np.mean(graycoprops(glcm, 'contrast')[0, :]))
    dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')[0, :]))
    homogeneity   = float(np.mean(graycoprops(glcm, 'homogeneity')[0, :]))
    energy        = float(np.mean(graycoprops(glcm, 'energy')[0, :]))
    correlation   = float(np.mean(graycoprops(glcm, 'correlation')[0, :]))

    if np.isnan(correlation):
        correlation = 0.0

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
    }

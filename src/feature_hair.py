import numpy as np
import cv2


def get_hair_features(img, mask):
    """
    Detect hair using BlackHat filter and return two ratios:
      - hair_coverage: hair pixels over the whole image
      - hair_in_lesion: hair pixels inside the lesion

    BlackHat = closing(image) - image  -> highlights dark thin structures.
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # tried kernel sizes 9, 17, 25 -- 17 looked best on our test images
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # otsu picks the threshold per image (some images are much darker)
    _, hair_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    total_pixels = img.shape[0] * img.shape[1]
    hair_pixels_total = np.sum(hair_mask > 0)

    hair_in_lesion = np.sum((hair_mask > 0) & binary_mask)
    lesion_pixels = np.sum(binary_mask)

    return {
        "hair_coverage":  hair_pixels_total / total_pixels if total_pixels > 0 else 0,
        "hair_in_lesion": hair_in_lesion / lesion_pixels  if lesion_pixels > 0 else 0,
    }
